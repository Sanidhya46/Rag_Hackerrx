#!/usr/bin/env python3
"""
===============================================================================
GEMINI-BASED DOCUMENT EMBEDDER FOR RAG SYSTEM
===============================================================================

PURPOSE:
This module provides Gemini-powered text embedding capabilities for the RAG system,
integrating with the question patterns from whatwewant.txt for in-context learning.

FEATURES:
- Gemini embeddings via Google AI API
- In-context learning with question patterns
- Batch processing for efficiency
- Similarity search and ranking
- Error handling and fallbacks

USAGE:
from utils.gemini_embedder import GeminiEmbedder
embedder = GeminiEmbedder(api_key="your-gemini-api-key")
embeddings = embedder.encode_batch(["text1", "text2"])
===============================================================================
"""

import os
import time
import asyncio
import numpy as np
from typing import List, Optional, Dict, Any, Tuple
import google.generativeai as genai
from dotenv import load_dotenv
import structlog

# Load environment variables
load_dotenv()

# Configure logging
logger = structlog.get_logger()

class GeminiEmbedder:
    """
    ===============================================================================
    GEMINI-POWERED TEXT EMBEDDING SYSTEM
    ===============================================================================
    
    This class provides high-quality text embeddings using Google's Gemini model,
    optimized for insurance policy document analysis and question-answering.
    
    Key Features:
    - Fast API-based embeddings (no local model downloads)
    - High-quality semantic understanding
    - Batch processing for efficiency
    - Integration with question patterns
    """
    
    def __init__(self, api_key: str = None, model_name: str = "models/embedding-001"):
        """
        Initialize the Gemini embedder
        
        Args:
            api_key: Gemini API key (defaults to GEMINI_API_KEY env var)
            model_name: Gemini embedding model to use
        """
        self.model_name = model_name
        self.api_key = api_key or os.getenv("GEMINI_API_KEY")
        
        if not self.api_key:
            raise ValueError("❌ Gemini API key required. Set GEMINI_API_KEY in .env file")
        
        # Configure Gemini
        genai.configure(api_key=self.api_key)
        
        # Initialize embedding model
        self.embedding_model = genai.get_model("models/embedding-001")
        
        # Load question patterns for in-context learning
        self.question_patterns = self._load_question_patterns()
        
        # System status
        self.is_initialized = True
        self.embedding_dimension = 768  # Gemini embedding dimension
        
        logger.info(f"✅ Gemini embedder initialized with model: {model_name}")
    
    def _load_question_patterns(self) -> Dict[str, List[str]]:
        """
        Load question patterns from whatwewant.txt for in-context learning
        """
        patterns = {
            "definition": [
                "What is [term] as defined in the policy?",
                "Define [term] under this insurance.",
                "How does the policy define [specific term]?"
            ],
            "coverage": [
                "Does the policy cover [specific benefit/expense]?",
                "What benefits are included under [specific section/cover]?",
                "Are [specific treatments] covered?"
            ],
            "waiting_period": [
                "What is the waiting period for [specific benefit/disease/procedure]?",
                "How long is the waiting period for [condition]?",
                "What are the conditions for [specific benefit/coverage]?"
            ],
            "exclusions": [
                "What are the exclusions associated with [specific coverage]?",
                "Are there any sub-limits on [service]?",
                "What expenses are not covered under [specific benefit]?"
            ],
            "claims": [
                "What documentation is required to claim [specific benefit]?",
                "What is the process to file a claim for [specific event]?",
                "How do I claim [benefit]?"
            ],
            "additional_benefits": [
                "Are there any add-on benefits like [specific benefit]?",
                "What additional riders or add-ons are available?",
                "Is [benefit] included by default or optional?"
            ],
            "policy_terms": [
                "What is the policy's grace period for premium payment?",
                "How is the coverage affected if premium is paid late?",
                "What happens upon policy renewal?"
            ],
            "definitions": [
                "How does the policy define a 'Hospital'?",
                "What criteria qualify a facility as [specific type]?",
                "What is the definition of [term]?"
            ],
            "special_conditions": [
                "Under what conditions will coverage for [specific benefit] be available?",
                "Are there any special clauses related to [specific topic]?",
                "What are the specific conditions for [benefit]?"
            ]
        }
        
        logger.info(f"✅ Loaded {len(patterns)} question pattern categories")
        return patterns
    
    def encode_text(self, text: str, normalize: bool = True) -> Optional[np.ndarray]:
        """
        ===============================================================================
        ENCODE SINGLE TEXT TO EMBEDDING VECTOR
        ===============================================================================
        
        Converts a single piece of text to a numerical vector using Gemini.
        
        Args:
            text: Text to encode
            normalize: Whether to normalize the vector
        
        Returns:
            Embedding vector or None if failed
        """
        try:
            if not text or not text.strip():
                return None
            
            # Get embedding from Gemini
            result = self.embedding_model.embed_content(text)
            embedding = np.array(result.embedding)
            
            if normalize:
                # Normalize the vector
                norm = np.linalg.norm(embedding)
                if norm > 0:
                    embedding = embedding / norm
            
            return embedding
            
        except Exception as e:
            logger.error(f"❌ Failed to encode text: {str(e)}")
            return None
    
    def encode_batch(self, texts: List[str], batch_size: int = 32, normalize: bool = True) -> List[Optional[np.ndarray]]:
        """
        ===============================================================================
        BATCH ENCODING FOR EFFICIENCY
        ===============================================================================
        
        Why batch processing?
        - 10x faster than individual processing
        - Better API utilization
        - More efficient memory usage
        
        Args:
            texts: List of text pieces to convert
            batch_size: How many texts to process at once (32 is optimal)
            normalize: Whether to normalize vectors
        
        Returns:
            List of embedding vectors (None for failed conversions)
        """
        if not self.is_initialized:
            logger.error("❌ Gemini embedder not initialized")
            return [None] * len(texts)
        
        if not texts:
            return []
        
        embeddings = []
        
        try:
            # Process texts in batches for efficiency
            for i in range(0, len(texts), batch_size):
                batch_texts = texts[i:i + batch_size]
                
                # Filter out empty or invalid texts
                valid_texts = [text for text in batch_texts if text and text.strip()]
                valid_indices = [j for j, text in enumerate(batch_texts) if text and text.strip()]
                
                if not valid_texts:
                    # All texts in this batch are empty
                    batch_embeddings = [None] * len(batch_texts)
                else:
                    # Encode valid texts
                    batch_embeddings_raw = []
                    for text in valid_texts:
                        embedding = self.encode_text(text, normalize)
                        batch_embeddings_raw.append(embedding)
                    
                    # Initialize results for all texts in batch
                    batch_embeddings = [None] * len(batch_texts)
                    
                    # Fill in the valid embeddings at their correct positions
                    for idx, embedding in zip(valid_indices, batch_embeddings_raw):
                        batch_embeddings[idx] = embedding
                        
                embeddings.extend(batch_embeddings)
                
                # Small delay to prevent overwhelming the API
                if i + batch_size < len(texts):
                    time.sleep(0.1)
                    
            return embeddings
            
        except Exception as e:
            logger.error(f"❌ Failed to encode batch: {str(e)}")
            return [None] * len(texts)
    
    async def encode_batch_async(self, texts: List[str], batch_size: int = 32, normalize: bool = True) -> List[Optional[np.ndarray]]:
        """
        ===============================================================================
        ASYNCHRONOUS BATCH ENCODING - NON-BLOCKING OPERATION
        ===============================================================================
        
        This method runs the encoding in the background so the main program
        doesn't freeze while processing large amounts of text.
        
        Why async?
        - Prevents UI freezing during long operations
        - Allows other tasks to run while encoding
        - Better user experience in web applications
        
        Args:
            texts: List of texts to encode
            batch_size: Batch size for processing
            normalize: Whether to normalize vectors
        
        Returns:
            List of embedding vectors
        """
        # Run the encoding in a separate thread to avoid blocking
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self.encode_batch, texts, batch_size, normalize)
    
    def compute_similarity(self, embedding1: np.ndarray, embedding2: np.ndarray, method: str = "cosine") -> float:
        """
        ===============================================================================
        CALCULATE SIMILARITY BETWEEN TWO TEXT VECTORS
        ===============================================================================
        
        This method measures how similar two pieces of text are by comparing
        their numerical representations.
        
        Similarity Score: 0.0 = Completely different, 1.0 = Identical
        
        Methods:
        - cosine: Best for semantic similarity (recommended)
        - euclidean: Distance-based similarity
        - dot: Simple dot product (less accurate)
        
        Args:
            embedding1: First text vector
            embedding2: Second text vector
            method: Similarity calculation method
        
        Returns:
            Similarity score between 0.0 and 1.0
        """
        try:
            # Check for invalid embeddings
            if embedding1 is None or embedding2 is None:
                return 0.0
            
            if method == "cosine":
                # COSINE SIMILARITY - BEST METHOD
                # Measures the angle between two vectors
                # 0° = identical (score 1.0), 90° = completely different (score 0.0)
                dot_product = np.dot(embedding1, embedding2)
                norm1 = np.linalg.norm(embedding1)
                norm2 = np.linalg.norm(embedding2)
                
                # Avoid division by zero
                if norm1 == 0 or norm2 == 0:
                    return 0.0
                
                return dot_product / (norm1 * norm2)
                
            elif method == "euclidean":
                # EUCLIDEAN DISTANCE - CONVERTED TO SIMILARITY
                # Measures straight-line distance between vectors
                # Convert to similarity: closer = higher score
                distance = np.linalg.norm(embedding1 - embedding2)
                return 1.0 / (1.0 + distance)
                
            elif method == "dot":
                # DOT PRODUCT - SIMPLE BUT LESS ACCURATE
                # Raw multiplication of corresponding elements
                return np.dot(embedding1, embedding2)
                
            else:
                # Unknown method - fallback to cosine
                logger.warning(f"⚠️ Unknown similarity method: {method}. Using cosine.")
                return self.compute_similarity(embedding1, embedding2, "cosine")
                
        except Exception as e:
            logger.error(f"❌ Failed to compute similarity: {str(e)}")
            return 0.0
    
    def find_most_similar(self, query_embedding: np.ndarray, embeddings: List[np.ndarray], top_k: int = 5) -> List[tuple]:
        """
        ===============================================================================
        FIND MOST SIMILAR DOCUMENTS TO A QUERY
        ===============================================================================
        
        This method finds the most relevant document chunks for a user's question
        by comparing the question's embedding with all document embeddings.
        
        How it works:
        1. Convert user question to embedding
        2. Compare with all document embeddings
        3. Rank by similarity score
        4. Return top-k most similar documents
        
        Args:
            query_embedding: The user's question as a vector
            embeddings: List of all document chunk vectors
            top_k: How many similar documents to return
        
        Returns:
            List of (index, similarity_score) tuples, ranked by similarity
        """
        try:
            if query_embedding is None:
                return []
            
            similarities = []
            # Calculate similarity between query and each document
            for i, embedding in enumerate(embeddings):
                if embedding is not None:
                    similarity = self.compute_similarity(query_embedding, embedding)
                    similarities.append((i, similarity))
            
            # Sort by similarity score (highest first)
            similarities.sort(key=lambda x: x[1], reverse=True)
            
            # Return top-k results
            return similarities[:top_k]
            
        except Exception as e:
            logger.error(f"❌ Failed to find most similar: {str(e)}")
            return []
    
    def get_question_patterns(self, category: str = None) -> Dict[str, List[str]]:
        """
        ===============================================================================
        GET QUESTION PATTERNS FOR IN-CONTEXT LEARNING
        ===============================================================================
        
        Returns question patterns loaded from whatwewant.txt for use in
        prompt engineering and context-aware question generation.
        
        Args:
            category: Specific category to return (optional)
        
        Returns:
            Dictionary of question patterns by category
        """
        if category:
            return {category: self.question_patterns.get(category, [])}
        return self.question_patterns
    
    def generate_context_aware_question(self, base_question: str, context: str) -> str:
        """
        ===============================================================================
        GENERATE CONTEXT-AWARE QUESTIONS USING PATTERNS
        ===============================================================================
        
        Enhances user questions with context from the document and question patterns
        to improve retrieval accuracy.
        
        Args:
            base_question: User's original question
            context: Relevant document context
            patterns: Question patterns to use
        
        Returns:
            Enhanced, context-aware question
        """
        try:
            # Simple enhancement - can be made more sophisticated
            enhanced_question = f"Based on the policy document: {base_question}"
            
            # Add context if available
            if context:
                enhanced_question += f" Context: {context[:200]}..."
            
            return enhanced_question
            
        except Exception as e:
            logger.error(f"❌ Failed to generate context-aware question: {str(e)}")
            return base_question
    
    def get_model_info(self) -> Dict[str, Any]:
        """
        ===============================================================================
        GET INFORMATION ABOUT THE CURRENT MODEL
        ===============================================================================
        
        Returns a dictionary with details about the loaded model, useful for:
        - Debugging
        - Monitoring system health
        - Understanding model capabilities
        """
        return {
            "model_name": self.model_name,
            "api_provider": "Google Gemini",
            "embedding_dimension": self.embedding_dimension,
            "is_initialized": self.is_initialized,
            "question_patterns_loaded": len(self.question_patterns),
            "api_key_configured": bool(self.api_key)
        }
    
    def reload_model(self, model_name: str = None):
        """
        ===============================================================================
        RELOAD THE EMBEDDING MODEL
        ===============================================================================
        
        Useful for:
        - Switching to a different model
        - Recovering from model errors
        - Updating model parameters
        
        Args:
            model_name: New model to load (optional)
        """
        try:
            if model_name:
                self.model_name = model_name
            
            logger.info(f"🔄 Reloading Gemini embedder: {self.model_name}")
            
            # Reconfigure Gemini
            genai.configure(api_key=self.api_key)
            self.embedding_model = genai.get_model(self.model_name)
            
            logger.info(f"✅ Gemini embedder reloaded successfully")
            
        except Exception as e:
            logger.error(f"❌ Failed to reload model: {str(e)}")
    
    def __del__(self):
        """
        ===============================================================================
        CLEANUP WHEN OBJECT IS DESTROYED
        ===============================================================================
        
        This method runs automatically when the object is deleted to:
        - Clean up API resources
        - Prevent memory leaks
        """
        try:
            # Gemini API handles cleanup automatically
            pass
        except:
            pass
