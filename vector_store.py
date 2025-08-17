from langchain_pinecone import PineconeVectorStore
from langchain_community.embeddings import HuggingFaceEmbeddings
from pinecone import Pinecone, ServerlessSpec
import os

def get_vector_store(chunks, index_name="hackerrx-doc-index"):
    index_name="hackerrx-doc-index"
    # Get Pinecone API key
    api_key = "pcsk_4gAiit_CqXuB2Y9wM82R8vFT8qnP4s7x4yH6e5GBGrcChLSnzLxykvK3zapCN6jghSEqJ"
    print(f"🔑 Pinecone API Key loaded: {api_key[:4]}...")

    # Initialize Pinecone client
    pc = Pinecone(api_key=api_key)

    # Create index if it doesn't exist
    if index_name not in pc.list_indexes().names():
        pc.create_index(
            name=index_name,
            dimension=384,
            metric="cosine",
            spec=ServerlessSpec(cloud="aws", region="us-east-1")
        )

    index = pc.Index("hackerrx-doc-index")

    # Create embedding model
    embedder = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

    # Store vectors
    vector_store = PineconeVectorStore(
        index=index,
        embedding=embedder,
        text_key="text"
    )

    # Add texts
    vector_store.add_texts(chunks)

    return vector_store



# Sample usage  testing purposes ((*vector databse*))

# if __name__ == "__main__":
#     sample_chunks = [
#         "This is the first sample text.",
#         "Another piece of sample text.",
#         "ChatGPT is great for AI conversations."
#     ]
# vector_store = get_vector_store(sample_chunks)

# result = vector_store.similarity_search("AI chatbot", k=1)
# print(result[0].page_content)  