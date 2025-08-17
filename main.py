"""
main.py; FastAPI backend for PDF-based Q&A.

Handles PDF upload, extracts and splits text, embeds chunks using HuggingFace,
stores vectors in-memory with a unique doc_id, and enables chat-based semantic search.
Uses custom modules: pdf_loader, text_splitter, embedder, and vector_store.

Works with a React frontend (running on port 3000). When the user uploads a PDF,
the backend processes it, stores the vectors, and returns a doc_id. This doc_id
is later used to handle chat queries by searching semantically within the uploaded document.
"""

from fastapi import FastAPI, File, UploadFile, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from typing import List, Optional, Dict, Any
import asyncio
import logging
import traceback
import json
import os
from datetime import datetime
import time
from langchain.prompts import PromptTemplate, FewShotPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_pinecone import PineconeVectorStore
from pinecone import Pinecone as PineconeClient, ServerlessSpec
import re


# Import our custom modules
from utils.gemini_embedder import GeminiEmbedder
from loaders.pdf_loader import PDFLoader
from utils_api.pdf_parser import PDFParser
from utils_api.csv_parser import CSVParser

from langchain.text_splitter import RecursiveCharacterTextSplitter

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('app.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="HackerX VDB - Production RAG System",
    description="Robust Document Q&A System with Advanced Error Handling",
    version="2.0.0"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global variables for document processing
document_store = {}
processing_status = {}

# RAG components
embeddings = None  # GoogleGenerativeAIEmbeddings instance
text_splitter = None  # RecursiveCharacterTextSplitter instance
pc: Optional[PineconeClient] = None
pinecone_index = None
pinecone_index_name = os.getenv("PINECONE_INDEX_NAME", "hackerx-rag")
FEW_SHOTS: List[Dict[str, Any]] = []

class QueryRequest(BaseModel):
    query: str
    document_id: Optional[str] = None
    use_context: bool = True
    max_tokens: int = 500

class DocumentResponse(BaseModel):
    document_id: str
    filename: str
    status: str
    message: str
    processing_time: float
    chunks_created: int

class QueryResponse(BaseModel):
    answer: str
    sources: List[Dict[str, Any]]
    confidence: float
    processing_time: float
    query_type: str
    debug: Optional[List[Dict[str, Any]]] = None

class HealthCheck(BaseModel):
    status: str
    timestamp: str
    version: str
    document_count: int
    system_health: str

# Initialize components
async def initialize_system():
    """Initialize the RAG system components"""
    global embeddings, text_splitter, pc, pinecone_index, FEW_SHOTS
    
    try:
        logger.info("Initializing RAG system components...")
        
        # Initialize Gemini embeddings (uses GEMINI_API_KEY or GOOGLE_API_KEY)
        api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
        if not api_key:
            raise ValueError("Missing GEMINI_API_KEY/GOOGLE_API_KEY in environment")

        embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001", google_api_key=api_key)
        logger.info("Gemini embeddings initialized successfully")

        # Initialize text splitter (LangChain)
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len
        )
        logger.info("Recursive text splitter initialized successfully")

        # Initialize Pinecone
        p_api_key = os.getenv("PINECONE_API_KEY")
        if not p_api_key:
            raise ValueError("Missing PINECONE_API_KEY in environment")
        pc = PineconeClient(api_key=p_api_key)
        # Create index if missing (Gemini embeddings ~768 dims)
        existing = [i.name for i in pc.list_indexes().indexes]
        if pinecone_index_name not in existing:
            pc.create_index(
                name=pinecone_index_name,
                dimension=768,
                metric="cosine",
                spec=ServerlessSpec(cloud="aws", region=os.getenv("PINECONE_REGION", "us-east-1")),
            )
        pinecone_index = pc.Index(pinecone_index_name)
        logger.info(f"Pinecone index ready: {pinecone_index_name}")

        # Load few-shot examples from JSONL if available
        FEW_SHOTS = load_few_shots_jsonl("data/train.jsonl", max_examples=16)
        logger.info(f"Loaded {len(FEW_SHOTS)} few-shot examples")
        
        logger.info("RAG system initialization completed")
        return True
        
    except Exception as e:
        logger.error(f"Failed to initialize RAG system: {str(e)}")
        logger.error(traceback.format_exc())
        return False

@app.on_event("startup")
async def startup_event():
    """Initialize system on startup"""
    success = await initialize_system()
    if not success:
        logger.error("Failed to initialize system on startup")

@app.get("/health", response_model=HealthCheck)
async def health_check():
    """System health check endpoint"""
    try:
        return HealthCheck(
            status="healthy",
            timestamp=datetime.now().isoformat(),
            version="2.0.0",
            document_count=len(document_store),
            system_health="operational"
        )
    except Exception as e:
        logger.error(f"Health check failed: {str(e)}")
        return HealthCheck(
            status="unhealthy",
            timestamp=datetime.now().isoformat(),
            version="2.0.0",
            document_count=0,
            system_health=f"error: {str(e)}"
        )

@app.post("/upload", response_model=DocumentResponse)
async def upload_document(
    file: UploadFile = File(...),
    background_tasks: BackgroundTasks = None
):
    """Upload and process documents with comprehensive error handling"""
    
    start_time = time.time()
    document_id = f"doc_{int(time.time())}_{file.filename}"
    
    try:
        # Validate file
        if not file.filename:
            raise HTTPException(status_code=400, detail="No filename provided")
        
        # Check file size (100MB limit)
        if file.size and file.size > 100 * 1024 * 1024:
            raise HTTPException(status_code=400, detail="File size exceeds 100MB limit")
        
        # Create processing status
        processing_status[document_id] = {
            "status": "processing",
            "filename": file.filename,
            "progress": 0,
            "message": "Starting document processing..."
        }
        
        # Process document based on type
        file_extension = file.filename.lower().split('.')[-1]
        
        if file_extension == 'pdf':
            # Process PDF
            pdf_loader = PDFLoader()
            
            # Read file content
            content = await file.read()
            
            # Parse PDF using our loader
            text_content = pdf_loader.load_pdf_from_bytes(content)
            
            # Split into chunks using LangChain splitter (list[str])
            chunks = text_splitter.split_text(text_content)

            if not chunks:
                raise HTTPException(status_code=400, detail="Failed to split document into chunks")

            
            # Store document
            document_store[document_id] = {
                "filename": file.filename,
                "content": text_content,
                "chunks": chunks,
                "upload_time": datetime.now().isoformat(),
                "file_size": len(content),
                "chunk_count": len(chunks)
            }

            # Upsert into Pinecone
            await upsert_chunks_to_pinecone(document_id, chunks)
            
        elif file_extension in ['txt', 'md']:
            # Process text files
            content = await file.read()
            text_content = content.decode('utf-8')
            
            # Split into chunks (list[str])
            chunks = text_splitter.split_text(text_content)
            
            # Store document
            document_store[document_id] = {
                "filename": file.filename,
                "content": text_content,
                "chunks": chunks,
                "upload_time": datetime.now().isoformat(),
                "file_size": len(content),
                "chunk_count": len(chunks)
            }

            # Upsert into Pinecone
            await upsert_chunks_to_pinecone(document_id, chunks)
            
        elif file_extension == 'csv':
            # Process CSV files
            csv_parser = CSVParser()
            content = await file.read()
            
            # Parse CSV
            parsed_data = csv_parser.parse_csv_from_bytes(content)
            
            # Convert to text chunks
            text_content = csv_parser.convert_to_text(parsed_data)
            
            # Split into chunks (list[str])
            chunks = text_splitter.split_text(text_content)
            
            # Store document
            document_store[document_id] = {
                "filename": file.filename,
                "content": text_content,
                "chunks": chunks,
                "upload_time": datetime.now().isoformat(),
                "file_size": len(content),
                "chunk_count": len(chunks),
                "csv_data": parsed_data
            }

            # Upsert into Pinecone
            await upsert_chunks_to_pinecone(document_id, chunks)
            
        else:
            raise HTTPException(
                status_code=400, 
                detail=f"Unsupported file type: {file_extension}. Supported: PDF, TXT, MD, CSV"
            )
        
        # Update processing status
        processing_status[document_id]["status"] = "completed"
        processing_status[document_id]["progress"] = 100
        processing_status[document_id]["message"] = "Document processed successfully"
        
        processing_time = time.time() - start_time
        
        logger.info(f"Document {document_id} processed successfully in {processing_time:.2f}s")
        
        return DocumentResponse(
            document_id=document_id,
            filename=file.filename,
            status="success",
            message="Document processed and stored successfully",
            processing_time=processing_time,
            chunks_created=len(chunks)
        )
        
    except HTTPException:
        # Re-raise HTTP exceptions
        raise
    except Exception as e:
        # Handle unexpected errors
        error_msg = f"Failed to process document: {str(e)}"
        logger.error(error_msg)
        logger.error(traceback.format_exc())
        
        # Update processing status
        processing_status[document_id]["status"] = "failed"
        processing_status[document_id]["message"] = error_msg
        
        raise HTTPException(status_code=500, detail=error_msg)

@app.get("/documents")
async def list_documents():
    """List all uploaded documents"""
    try:
        documents = []
        for doc_id, doc_data in document_store.items():
            documents.append({
                "document_id": doc_id,
                "filename": doc_data["filename"],
                "upload_time": doc_data["upload_time"],
                "file_size": doc_data["file_size"],
                "chunk_count": doc_data["chunk_count"],
                "status": processing_status.get(doc_id, {}).get("status", "unknown")
            })
        
        return {"documents": documents, "total": len(documents)}
        
    except Exception as e:
        logger.error(f"Failed to list documents: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to retrieve documents")

@app.get("/documents/{document_id}/status")
async def get_document_status(document_id: str):
    """Get processing status of a specific document"""
    try:
        if document_id not in processing_status:
            raise HTTPException(status_code=404, detail="Document not found")
        
        return processing_status[document_id]
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get document status: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to retrieve document status")

# few shot prompting prepare to take few shot prompt json from jsonl
import json

few_shot_examples = [
    {"context": "A hospital is defined as any institution established for in-patient care and day care treatment of illness and/or injuries with at least 15 in-patient beds.",
     "question": "How does the policy define a 'Hospital'?",
     "answer": "A hospital is any institution established for in-patient care and day care treatment of illness and/or injuries with at least 15 in-patient beds."},

    {"context": "Pre-existing Disease means any condition, ailment or injury or related condition(s) for which the Insured Person had signs or symptoms, and/or was diagnosed and/or received medical advice/treatment within 48 months prior to the first policy issued by the insurer.",
     "question": "What does 'Pre-existing Disease' mean under this policy?",
     "answer": "A Pre-existing Disease is any condition, ailment or injury that the insured had signs of, was diagnosed with, or received treatment for within 48 months before the first policy was issued."},

    {"context": "The policy covers AYUSH treatments when taken at a government recognized or accredited AYUSH hospital.",
     "question": "Does this policy include coverage for AYUSH treatments?",
     "answer": "Yes, the policy covers AYUSH treatments when taken at a government recognized or accredited AYUSH hospital."},

    {"context": "Organ donor expenses are covered up to ₹2,00,000 per policy year when the transplant is medically necessary and performed at a network hospital.",
     "question": "What is the extent of coverage for organ donation?",
     "answer": "Organ donor expenses are covered up to ₹2,00,000 per policy year when the transplant is medically necessary and performed at a network hospital."},

    {"context": "Preventive health check-ups are covered once every two policy years up to ₹2,000.",
     "question": "Are preventive health check-ups covered?",
     "answer": "Yes, preventive health check-ups are covered once every two policy years up to ₹2,000."},

    {"context": "Maternity expenses are covered after a waiting period of 24 months of continuous coverage.",
     "question": "Is maternity expense covered under this policy?",
     "answer": "Yes, maternity expenses are covered after 24 months of continuous coverage."},

    {"context": "Cataract surgery has a waiting period of 24 months from the policy inception date.",
     "question": "What is the waiting period for Cataract surgery?",
     "answer": "The waiting period for Cataract surgery is 24 months from the policy inception date."},

    {"context": "Maternity benefits become available only after the female insured has been continuously covered for 24 months.",
     "question": "How long must one be continuously covered for maternity benefits?",
     "answer": "Maternity benefits require 24 months of continuous coverage."},

    {"context": "ICU charges are subject to a sub-limit of 2% of sum insured per day.",
     "question": "Are there sub-limits on ICU charges for Plan A?",
     "answer": "Yes, ICU charges have a sub-limit of 2% of sum insured per day."},

    {"context": "Expenses related to the organ donor's hospitalization are excluded unless the recipient is also covered under the same policy.",
     "question": "What expenses are excluded for organ donation procedures?",
     "answer": "Organ donor hospitalization expenses are excluded unless the recipient is also covered under the same policy."},

    {"context": "To claim post-hospitalization expenses, submit the discharge summary, original bills, prescription, and investigation reports within 30 days of discharge.",
     "question": "What documents are needed to claim post-hospitalization expenses?",
     "answer": "You need to submit the discharge summary, original bills, prescription, and investigation reports within 30 days of discharge."},

    {"context": "Claims must be reported within 7 days of hospitalization and all documents must be submitted within 30 days of discharge.",
     "question": "What is the claim filing timeline after discharge?",
     "answer": "Claims must be reported within 7 days of hospitalization and all documents must be submitted within 30 days of discharge."},

    {"context": "A No Claim Discount of 10% is applied for each claim-free year, up to a maximum of 50%.",
     "question": "Does the policy provide a No Claim Discount?",
     "answer": "Yes, a No Claim Discount of 10% is applied for each claim-free year, up to a maximum of 50%."},

    {"context": "One free preventive health check-up is provided every two policy years.",
     "question": "Is there a benefit for annual preventive check-up?",
     "answer": "Yes, one free preventive health check-up is provided every two policy years."},

    {"context": "The grace period for premium payment is 15 days for monthly premiums and 30 days for annual premiums.",
     "question": "What is the grace period for premium payment under this policy?",
     "answer": "The grace period is 15 days for monthly premiums and 30 days for annual premiums."},

    {"context": "Continuous coverage reduces waiting periods for PED by 50% after 3 claim-free years.",
     "question": "How does continuous coverage affect waiting periods?",
     "answer": "Continuous coverage reduces waiting periods for Pre-existing Diseases by 50% after 3 claim-free years."},

    {"context": "An AYUSH Hospital means a hospital established under the AYUSH system of medicine and recognized by the Central/State Government.",
     "question": "How is AYUSH treatment qualified for coverage?",
     "answer": "AYUSH treatment must be received at a hospital recognized by the Central/State Government under the AYUSH system of medicine."},

    {"context": "Inpatient treatment means treatment for which the insured person has to stay in a hospital for more than 24 hours.",
     "question": "What is the definition of inpatient treatment?",
     "answer": "Inpatient treatment means staying in a hospital for more than 24 hours for treatment."},

    {"context": "Organ donor expenses are covered only when the recipient is also an insured under the same policy and the transplant is medically necessary.",
     "question": "What conditions apply for organ donor expenses?",
     "answer": "Organ donor expenses are covered only when the recipient is also insured under the same policy and the transplant is medically necessary."},

    {"context": "Relapse of a disease within 45 days of discharge will be considered part of the same hospitalization.",
     "question": "Are claims related to relapse diseases covered?",
     "answer": "Yes, if the relapse occurs within 45 days of discharge, it's considered part of the same hospitalization."},

    {"context": "Domiciliary hospitalization is covered when the insured cannot be moved to a hospital and treatment is given at home for at least 3 days.",
     "question": "What are the conditions for domiciliary hospitalization coverage?",
     "answer": "Domiciliary hospitalization is covered when the insured cannot be moved to a hospital and receives at least 3 days of treatment at home."},

    {"context": "Dental treatment is covered only when required due to an accident or as part of a hospital treatment.",
     "question": "Are routine dental treatments covered?",
     "answer": "No, dental treatment is only covered when required due to an accident or as part of hospital treatment."},

    {"context": "The policy provides air ambulance coverage up to ₹5,00,000 when medically necessary and pre-approved.",
     "question": "What are the limits for air ambulance services?",
     "answer": "Air ambulance coverage is provided up to ₹5,00,000 when medically necessary and pre-approved."},

    {"context": "Policy portability is allowed from other insurers without losing continuity benefits if applied 45 days before renewal.",
     "question": "What are the conditions for policy portability?",
     "answer": "Policy portability is allowed from other insurers without losing continuity benefits if applied for 45 days before renewal."},

    {"context": "Newborn babies are covered from day 15 to day 90 under the maternity benefit, after which they need separate enrollment.",
     "question": "Are newborns automatically covered under this policy?",
     "answer": "Newborns are covered from day 15 to day 90 under maternity benefit, after which separate enrollment is required."},

    {"context": "The compassionate visit benefit covers travel expenses up to ₹25,000 when the insured is hospitalized for more than 7 days.",
     "question": "What does the compassionate visit benefit cover?",
     "answer": "It covers travel expenses up to ₹25,000 when the insured is hospitalized for more than 7 days."},

    {"context": "Treatment for obesity is excluded unless it's medically necessary due to other covered conditions.",
     "question": "Is bariatric surgery covered under this policy?",
     "answer": "Bariatric surgery is excluded unless medically necessary due to other covered conditions."},

    {"context": "The daily cash allowance is ₹2,000 per day for ICU stays, limited to 10 days per policy year.",
     "question": "What is the daily cash allowance for ICU stays?",
     "answer": "The daily cash allowance is ₹2,000 per day for ICU stays, limited to 10 days per policy year."},

    {"context": "Claims for alternative treatments require prior approval and must be performed by a registered practitioner.",
     "question": "What is needed to claim for alternative medicine treatments?",
     "answer": "Claims for alternative treatments require prior approval and must be performed by a registered practitioner."},

    {"context": "The policy covers COVID-19 treatment like any other illness, subject to standard terms and conditions.",
     "question": "Is COVID-19 treatment covered under this policy?",
     "answer": "Yes, COVID-19 treatment is covered like any other illness, subject to standard terms and conditions."},

    {"context": "Pre-existing disease waiting periods are waived for newborn babies after 90 days of continuous coverage.",
     "question": "What is the waiting period for newborns with congenital diseases?",
     "answer": "Newborns have no waiting period for congenital diseases after 90 days of continuous coverage."},

    {"context": "The policy provides a second medical opinion service from a panel of specialists for critical illnesses.",
     "question": "Does the policy offer second opinion services?",
     "answer": "Yes, the policy provides a second medical opinion service from specialists for critical illnesses."},

    {"context": "Hospice care is covered up to ₹10,000 per policy year when certified as medically necessary.",
     "question": "Is palliative care covered under this policy?",
     "answer": "Yes, hospice care is covered up to ₹10,000 per policy year when certified as medically necessary."},

    {"context": "The restoration benefit automatically reinstates the sum insured by 100% after a claim exceeding 50% of sum insured.",
     "question": "How does the restoration benefit work?",
     "answer": "The sum insured is automatically reinstated by 100% after a claim exceeding 50% of the sum insured."},

    {"context": "Mental health treatment is covered up to ₹50,000 per year when provided at a recognized facility.",
     "question": "Does the policy cover mental health treatment?",
     "answer": "Yes, mental health treatment is covered up to ₹50,000 per year at recognized facilities."},

    {"context": "The policy covers prosthetic devices up to ₹25,000 per item when medically necessary.",
     "question": "Are prosthetic limbs covered under this policy?",
     "answer": "Yes, prosthetic devices are covered up to ₹25,000 per item when medically necessary."},

    {"context": "Emergency ambulance services are covered up to ₹5,000 per event when used for hospital admission.",
     "question": "What is the coverage limit for ambulance services?",
     "answer": "Emergency ambulance services are covered up to ₹5,000 per event when used for hospital admission."},

    {"context": "The policy covers home nursing up to ₹1,000 per day for 15 days post-hospitalization when certified by a doctor.",
     "question": "Is home nursing care covered after hospitalization?",
     "answer": "Yes, home nursing is covered up to ₹1,000 per day for 15 days post-hospitalization when certified by a doctor."},

    {"context": "International second opinion is available for cancer and cardiac conditions through our partner network.",
     "question": "Does the policy provide international second opinions?",
     "answer": "Yes, international second opinion is available for cancer and cardiac conditions through our partner network."},

    {"context": "The policy covers genetic testing up to ₹20,000 when medically necessary for treatment decisions.",
     "question": "Is genetic testing covered under this policy?",
     "answer": "Yes, genetic testing is covered up to ₹20,000 when medically necessary for treatment decisions."},

    {"context": "The policy provides coverage for robotic surgeries when performed at network hospitals.",
     "question": "Are robotic surgeries covered?",
     "answer": "Yes, robotic surgeries are covered when performed at network hospitals."}
]


print(len(few_shot_examples))

from langchain_huggingface import HuggingFaceEndpoint,ChatHuggingFace
api_key = os.getenv('HUGGINGFACEHUB_API_KEY')

llm = HuggingFaceEndpoint(
    repo_id = "google/gemma-2-2b-it",
    task="text-generation",
    huggingfacehub_api_token=api_key,  # ✅ THIS IS CRITICAL!
)

model = ChatHuggingFace(llm = llm)

def build_prompt(few_shot_examples, retrieved_chunks, user_query):
    prompt = "Answer the question based on the context. Use the examples as a guide.\n\n"
    
    # Add few-shot examples
    for ex in few_shot_examples[:40]:  # use top 5 for few-shot
        prompt += f"Context: {ex['context']}\nQuestion: {ex['question']}\nAnswer: {ex['answer']}\n\n"
    
    # Add retrieved chunks
    for chunk in retrieved_chunks[:3]:
        prompt += f"Context: {chunk['content']}\n"
    
    # Add user query
    prompt += f"\nQuestion: {user_query}\nAnswer:"
    
    return prompt


@app.post("/query", response_model=QueryResponse)
async def query_documents(request: QueryRequest):
    """Query documents with advanced RAG processing and error handling"""
    
    start_time = time.time()
    debug_trace: List[Dict[str, Any]] = []
    
    try:
        # Validate request
        if not request.query.strip():
            raise HTTPException(status_code=400, detail="Query cannot be empty")
        
        if len(request.query) > 1000:
            raise HTTPException(status_code=400, detail="Query too long (max 1000 characters)")
        
        # Determine query type for better processing
        query_type = classify_query(request.query)
        debug_trace.append({"step": "classify_query", "query_type": query_type})
        
        # Search for relevant chunks (vector first, then keyword fallback)
        relevant_chunks = await search_documents(request.query, request.document_id, debug_trace)
        
        if not relevant_chunks:
            return QueryResponse(
                answer="I couldn't find relevant information in the uploaded documents to answer your question. Please try rephrasing your query or upload relevant documents.",
                sources=[],
                confidence=0.0,
                processing_time=time.time() - start_time,
                query_type=query_type,
                debug=debug_trace
            )
        
        # Generate answer using RAG
        # Build few-shot + retrieved chunks prompt
        # Build few-shot + retrieved chunks prompt
        prompt = build_prompt(few_shot_examples, relevant_chunks, request.query)
        
        print("=== Full Prompt Sent to LLM ===")
        print(prompt)
        print("=== End of Prompt ===")

# Call LLM directly on the prompt
        llm_response = model.predict(prompt)
        answer = llm_response.strip()

# Calculate confidence based on chunk relevance
        confidence = sum(chunk["score"] for chunk in relevant_chunks[:3]) / min(len(relevant_chunks), 3)


        
        # Prepare sources
        sources = []
        for chunk in relevant_chunks[:3]:  # Top 3 sources
            sources.append({
                "content": chunk["content"][:200] + "..." if len(chunk["content"]) > 200 else chunk["content"],
                "document": chunk["document_id"],
                "relevance_score": chunk["score"]
            })
        
        processing_time = time.time() - start_time
        
        logger.info(f"Query processed successfully in {processing_time:.2f}s")
        
        return QueryResponse(
            answer=answer,
            sources=sources,
            confidence=confidence,
            processing_time=processing_time,
            query_type=query_type,
            debug=debug_trace
        )
        
    except HTTPException:
        raise
    except Exception as e:
        error_msg = f"Failed to process query: {str(e)}"
        logger.error(error_msg)
        logger.error(traceback.format_exc())
        raise HTTPException(status_code=500, detail=error_msg)

async def search_documents(query: str, document_id: Optional[str] = None, debug: Optional[List[Dict[str, Any]]] = None) -> List[Dict]:
    """Search for relevant document chunks (vector search with keyword fallback)"""
    try:
        relevant_chunks: List[Dict[str, Any]] = []
        
        # If specific document requested, search only in that document
        if document_id:
            if document_id not in document_store:
                return []
            documents_to_search = {document_id: document_store[document_id]}
        else:
            # Search in all documents
            documents_to_search = document_store
        
        # Try vector search via Pinecone (optionally filtered by document)
        total_candidates: List[Dict[str, Any]] = []
        try:
            vectorstore = PineconeVectorStore(index=pinecone_index, embedding=embeddings)
            pinecone_filter = {"document_id": document_id} if document_id else None
            docs_with_scores = vectorstore.similarity_search_with_score(query, k=10, filter=pinecone_filter)
            for doc, score in docs_with_scores:
                # smaller distance = better; convert to 0..1 relevance
                rel = 1.0 / (1.0 + float(score))
                total_candidates.append({
                    "content": doc.page_content,
                    "document_id": doc.metadata.get("document_id", "unknown"),
                    "chunk_index": doc.metadata.get("chunk_index", -1),
                    "score": rel
                })
        except Exception as e:
            if debug is not None:
                debug.append({"step": "vector_search_error", "error": str(e)})
            logger.warning(f"Vector search failed: {e}\n")

        if debug is not None:
            debug.append({"step": "vector_search", "candidates": len(total_candidates)})

        # If vector search yielded nothing, fallback to keyword search
        if not total_candidates:
            query_lower = query.lower()
            for doc_id, doc_data in documents_to_search.items():
                for i, chunk in enumerate(doc_data["chunks"]):
                    score = calculate_relevance_score(query_lower, chunk.lower())
                    if score > 0.1:
                        total_candidates.append({
                            "content": chunk,
                            "document_id": doc_id,
                            "chunk_index": i,
                            "score": score
                        })
            if debug is not None:
                debug.append({"step": "keyword_fallback", "candidates": len(total_candidates)})

        # Sort by score desc and return top 10
        total_candidates.sort(key=lambda x: x["score"], reverse=True)
        return total_candidates[:10]
        
    except Exception as e:
        logger.error(f"Search failed: {str(e)}")
        return []

def calculate_relevance_score(query: str, chunk: str) -> float:
    """Calculate relevance score between query and chunk"""
    try:
        # Simple keyword matching score
        query_words = set(query.split())
        chunk_words = set(chunk.split())
        
        if not query_words:
            return 0.0
        
        # Calculate Jaccard similarity
        intersection = len(query_words.intersection(chunk_words))
        union = len(query_words.union(chunk_words))
        
        if union == 0:
            return 0.0
        
        base_score = intersection / union
        
        # Boost score for exact phrase matches
        if query in chunk:
            base_score += 0.3
        
        # Boost score for consecutive word matches
        query_word_list = query.split()
        for i in range(len(chunk_words) - len(query_word_list) + 1):
            if chunk.split()[i:i+len(query_word_list)] == query_word_list:
                base_score += 0.2
                break
        
        return min(base_score, 1.0)
        
    except Exception as e:
        logger.error(f"Relevance calculation failed: {str(e)}")
        return 0.0

async def generate_answer(query: str, chunks: List[Dict], query_type: str, debug: Optional[List[Dict[str, Any]]] = None) -> tuple:
    """Generate answer using LangChain FewShot + Gemini over retrieved context"""
    try:
        # Combine relevant chunks
        context = "\n\n".join([chunk["content"] for chunk in chunks[:5]])
        if debug is not None:
            debug.append({"step": "build_context", "chars": len(context)})

        # Build chain once per request (uses global FEW_SHOTS)
        chain = build_langchain_chain(FEW_SHOTS, query_type)

        # Invoke chain
        answer = chain.invoke({"context": context, "question": query})
        if debug is not None:
            debug.append({"step": "llm_infer", "answer_preview": str(answer)[:120]})
        
        # Calculate confidence based on chunk relevance scores
        confidence = sum(chunk["score"] for chunk in chunks[:3]) / min(len(chunks), 3)
        
        return answer, confidence
        
    except Exception as e:
        logger.error(f"Answer generation failed: {str(e)}")
        if debug is not None:
            debug.append({"step": "llm_error", "error": str(e)})
        return "I apologize, but I encountered an error while generating the answer. Please try again.", 0.0

def build_langchain_chain(examples: List[Dict[str, Any]], query_type: str):
    """Create FewShot prompt + Gemini chat chain."""
    # example format
    example_prompt = PromptTemplate(
        input_variables=["context", "question", "answer"],
        template=(
            "Context:\n{context}\n\n"
            "Q: {question}\n"
            "A: {answer}\n"
        ),
    )

    # Few-shot prompt
    fs_prompt = FewShotPromptTemplate(
        examples=examples or [],
        example_prompt=example_prompt,
        prefix=(
            "You are a domain expert assistant answering questions from insurance policy documents.\n"
            "Follow the answer style shown in the examples.\n"
            "Always ground answers strictly in the provided Context. If the answer is not in context, say you cannot find it.\n"
        ),
        suffix=(
            "Context:\n{context}\n\n"
            "Question: {question}\n"
            "Answer:"
        ),
        input_variables=["context", "question"],
    )

    # LLM
    api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
    llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0.2, google_api_key=api_key)
    return fs_prompt | llm | StrOutputParser()

def create_rag_prompt(query: str, context: str, query_type: str) -> str:
    """Deprecated (kept for reference). We now use build_langchain_chain."""
    return ""

def generate_template_answer(query: str, context: str, query_type: str) -> str:
    """Generate template-based answer (placeholder for LLM integration)"""
    
    # This is a simplified template system
    # In production, replace with actual LLM call
    
    query_lower = query.lower()
    
    if "what is" in query_lower or "define" in query_lower:
        return f"Based on the document, {query} refers to the information contained in the uploaded document. Please review the specific sections for detailed definitions."
    
    elif "how" in query_lower or "process" in query_lower:
        return f"The process for {query} is outlined in the document. Please refer to the relevant sections for step-by-step instructions."
    
    elif "cover" in query_lower or "include" in query_lower:
        return f"Regarding coverage for {query}, please review the policy document sections that detail inclusions and exclusions."
    
    else:
        return f"Based on the document content, here's what I found regarding {query}: [This would be replaced with actual LLM-generated content in production]"

def classify_query(query: str) -> str:
    """Classify query type for better processing"""
    query_lower = query.lower()
    
    if any(word in query_lower for word in ["what is", "define", "meaning", "definition"]):
        return "definition"
    elif any(word in query_lower for word in ["how", "process", "steps", "procedure"]):
        return "process"
    elif any(word in query_lower for word in ["cover", "include", "exclude", "benefit"]):
        return "coverage"
    elif any(word in query_lower for word in ["when", "time", "period", "duration"]):
        return "timeline"
    elif any(word in query_lower for word in ["where", "location", "place"]):
        return "location"
    else:
        return "general"


def load_few_shots_jsonl(path: str, max_examples: int = 16) -> List[Dict[str, Any]]:
    """Load few-shot examples from a JSONL file with keys: context, question, answer."""
    examples: List[Dict[str, Any]] = []
    try:
        if not os.path.exists(path):
            return []
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                    if all(k in obj for k in ("context", "question", "answer")):
                        examples.append({
                            "context": str(obj["context"]),
                            "question": str(obj["question"]),
                            "answer": str(obj["answer"]),
                        })
                except Exception:
                    continue
        # Truncate
        return examples[:max_examples]
    except Exception as e:
        logger.warning(f"Failed to load few-shots from {path}: {e}")
        return []


async def upsert_chunks_to_pinecone(document_id: str, chunks: List[str]) -> None:
    """Upsert document chunks into Pinecone with metadata for filtering."""
    try:
        if not chunks:
            return
        metadatas = [{"document_id": document_id, "chunk_index": i} for i in range(len(chunks))]
        vectorstore = PineconeVectorStore(index=pinecone_index, embedding=embeddings)
        vectorstore.add_texts(texts=chunks, metadatas=metadatas)
        logger.info(f"Upserted {len(chunks)} chunks for {document_id} into Pinecone")
    except Exception as e:
        logger.warning(f"Failed to upsert chunks for {document_id} into Pinecone: {e}")

@app.delete("/documents/{document_id}")
async def delete_document(document_id: str):
    """Delete a specific document"""
    try:
        if document_id not in document_store:
            raise HTTPException(status_code=404, detail="Document not found")
        
        # Remove document
        del document_store[document_id]
        
        # Remove processing status
        if document_id in processing_status:
            del processing_status[document_id]
        
        logger.info(f"Document {document_id} deleted successfully")
        
        return {"message": "Document deleted successfully"}
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to delete document: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to delete document")

@app.delete("/documents")
async def clear_all_documents():
    """Clear all documents (use with caution)"""
    try:
        document_count = len(document_store)
        
        # Clear all documents
        document_store.clear()
        processing_status.clear()
        
        logger.info(f"All {document_count} documents cleared successfully")
        
        return {"message": f"All {document_count} documents cleared successfully"}
        
    except Exception as e:
        logger.error(f"Failed to clear documents: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to clear documents")

@app.get("/stats")
async def get_system_stats():
    """Get system statistics"""
    try:
        total_chunks = sum(doc.get("chunk_count", 0) for doc in document_store.values())
        total_size = sum(doc.get("file_size", 0) for doc in document_store.values())

        return {
            "total_documents": len(document_store),
            "total_chunks": total_chunks,
            "total_size_bytes": total_size,
            "total_size_mb": round(total_size / (1024 * 1024), 2),
            "system_uptime": "active",
            "last_activity": datetime.now().isoformat()
        }
    except Exception as e:
        logger.error(f"Failed to get stats: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to retrieve system stats")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
