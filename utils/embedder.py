"""
===============================================================================
DOCUMENT EMBEDDER - TEXT TO VECTOR CONVERSION SYSTEM
===============================================================================

PURPOSE:
This file converts text documents into numerical vectors (embeddings) that can be
used for similarity search and document retrieval in the RAG system.

WHAT IT DOES:
1. Loads pre-trained language models (sentence transformers)
2. Converts text chunks into 384-dimensional vectors
3. Computes similarity between different text pieces
4. Enables fast document search and retrieval

WHY WE NEED IT:
- Text cannot be directly compared for similarity
- Embeddings allow us to find similar documents quickly
- Enables semantic search (meaning-based, not just keyword-based)
- Core component of the RAG (Retrieval-Augmented Generation) system

TECHNICAL DETAILS:
- Uses SentenceTransformers library for state-of-the-art embeddings
- Supports GPU acceleration for faster processing
- Implements fallback models for reliability
- Handles batch processing for efficiency
===============================================================================
"""

import logging
import numpy as np
from typing import List, Dict, Any, Optional
import asyncio
import time
from sentence_transformers import SentenceTransformer
import torch

# Set up logging for this module
logger = logging.getLogger(__name__)

class DocumentEmbedder:
    """
    ===============================================================================
    DOCUMENT EMBEDDER CLASS - CONVERTS TEXT TO NUMERICAL VECTORS
    ===============================================================================
    
    This class handles the conversion of text documents into numerical representations
    (embeddings) that can be used for similarity search and document retrieval.
    
    Key Features:
    - Automatic device detection (CPU/GPU)
    - Fallback model support for reliability
    - Batch processing for efficiency
    - Multiple similarity calculation methods
    - Async support for non-blocking operations
    """
    
    def __init__(self, model_name: str = "all-MiniLM-L6-v2", device: str = "auto"):
        """
        ===============================================================================
        INITIALIZE THE DOCUMENT EMBEDDER
        ===============================================================================
        
        Args:
            model_name: Name of the pre-trained model to use
                       - "all-MiniLM-L6-v2" = Fast, accurate, 384 dimensions
                       - "paraphrase-MiniLM-L3-v2" = Fallback model, 384 dimensions
            device: Where to run the model ('auto', 'cpu', 'cuda')
                   - 'auto' = Automatically detect best available device
                   - 'cuda' = Use NVIDIA GPU if available
                   - 'cpu' = Use CPU only
        """
        self.model_name = model_name
        self.device = self._determine_device(device)  # Auto-detect best device
        self.model = None  # Will hold the loaded model
        self.embedding_dimension = None  # Will store vector size (e.g., 384)
        self.is_initialized = False  # Track if model loaded successfully
        
        # Try to load the model immediately
        try:
            self._initialize_model()
        except Exception as e:
            logger.error(f"Failed to initialize embedding model: {str(e)}")
            self.is_initialized = False
    
    def _determine_device(self, device: str) -> str:
        """
        ===============================================================================
        AUTOMATIC DEVICE DETECTION - FIND BEST COMPUTING RESOURCE
        ===============================================================================
        
        This method automatically detects the best available computing device:
        - NVIDIA GPU (CUDA) = Fastest for ML tasks
        - Apple Silicon (MPS) = Good for Mac users
        - CPU = Fallback option, slower but always available
        
        Returns:
            String indicating which device to use
        """
        if device == "auto":
            # Check for NVIDIA GPU first (fastest)
            if torch.cuda.is_available():
                logger.info("CUDA GPU detected - using for fast processing")
                return "cuda"
            # Check for Apple Silicon GPU
            elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                logger.info("Apple Silicon GPU detected - using for processing")
                return "mps"
            # Fallback to CPU
            else:
                logger.info("No GPU detected - using CPU (slower but reliable)")
                return "cpu"
        return device
    
    def _initialize_model(self):
        """
        ===============================================================================
        LOAD AND INITIALIZE THE PRE-TRAINED LANGUAGE MODEL
        ===============================================================================
        
        This method:
        1. Downloads the pre-trained model (if not already cached)
        2. Loads it into memory
        3. Tests it with a sample text
        4. Sets up fallback options if primary model fails
        
        The model is pre-trained on millions of text examples and can understand
        the meaning of text, not just keywords.
        """
        try:
            logger.info(f"Loading embedding model: {self.model_name} on {self.device}")
            
            # Download and load the pre-trained model
            # This model was trained on millions of text examples
            self.model = SentenceTransformer(self.model_name, device=self.device)
            
            # Test the model with a simple text to ensure it works
            test_embedding = self.model.encode("test", convert_to_tensor=True)
            self.embedding_dimension = test_embedding.shape[0]  # Usually 384
            
            self.is_initialized = True
            logger.info(f"✅ Model loaded successfully! Vector dimension: {self.embedding_dimension}")
            
        except Exception as e:
            logger.error(f"❌ Primary model failed: {str(e)}")
            
            # FALLBACK: Try a simpler, more reliable model
            try:
                logger.info("🔄 Attempting fallback to simpler model...")
                self.model = SentenceTransformer("paraphrase-MiniLM-L3-v2", device="cpu")
                test_embedding = self.model.encode("test", convert_to_tensor=True)
                self.embedding_dimension = test_embedding.shape[0]
                self.is_initialized = True
                logger.info("✅ Fallback model loaded successfully")
            except Exception as fallback_error:
                logger.error(f"❌ Fallback model also failed: {str(fallback_error)}")
                self.is_initialized = False
    
    def encode_text(self, text: str, normalize: bool = True) -> Optional[np.ndarray]:
        """
        ===============================================================================
        CONVERT SINGLE TEXT TO NUMERICAL VECTOR (EMBEDDING)
        ===============================================================================
        
        This method takes a piece of text and converts it into a numerical vector
        that represents the meaning of that text.
        
        Example:
        Input: "What is the waiting period for maternity coverage?"
        Output: [0.23, -0.45, 0.67, ...] (384 numbers representing meaning)
        
        Args:
            text: The text to convert (e.g., a question or document chunk)
            normalize: Whether to make the vector length = 1.0 (recommended)
                      This makes similarity calculations more accurate
        
        Returns:
            numpy array of 384 numbers, or None if failed
        """
        # Check if model is ready
        if not self.is_initialized:
            logger.error("❌ Embedding model not initialized")
            return None
        
        try:
            # Validate input text
            if not text or not text.strip():
                logger.warning("⚠️ Empty text provided for encoding")
                return None
            
            # Convert text to numerical vector using the pre-trained model
            # This is where the "magic" happens - AI understands text meaning
            embedding = self.model.encode(text, convert_to_tensor=False)
            
            if normalize:
                # Normalize vector to unit length (length = 1.0)
                # This makes similarity calculations more accurate
                norm = np.linalg.norm(embedding)  # Calculate vector length
                if norm > 0:
                    embedding = embedding / norm  # Divide by length to normalize
            
            return embedding
            
        except Exception as e:
            logger.error(f"❌ Failed to encode text: {str(e)}")
            return None
    
    def encode_batch(self, texts: List[str], batch_size: int = 32, normalize: bool = True) -> List[Optional[np.ndarray]]:
        """
        ===============================================================================
        CONVERT MULTIPLE TEXTS TO VECTORS (BATCH PROCESSING)
        ===============================================================================
        
        This method processes multiple text pieces at once, which is much faster
        than processing them one by one.
        
        Why batch processing?
        - 10x faster than individual processing
        - Better GPU utilization
        - More efficient memory usage
        
        Args:
            texts: List of text pieces to convert
            batch_size: How many texts to process at once (32 is optimal)
            normalize: Whether to normalize vectors
        
        Returns:
            List of embedding vectors (None for failed conversions)
        """
        # Check if model is ready
        if not self.is_initialized:
            logger.error("❌ Embedding model not initialized")
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
                    # Convert valid texts to embeddings in one operation
                    batch_embeddings_raw = self.model.encode(valid_texts, convert_to_tensor=False)
                    
                    # Initialize results for all texts in batch
                    batch_embeddings = [None] * len(batch_texts)
                    
                    # Fill in the valid embeddings at their correct positions
                    for idx, embedding in zip(valid_indices, batch_embeddings_raw):
                        if normalize:
                            # Normalize each embedding vector
                            norm = np.linalg.norm(embedding)
                            if norm > 0:
                                embedding = embedding / norm
                        batch_embeddings[idx] = embedding
                
                embeddings.extend(batch_embeddings)
                
                # Small delay to prevent overwhelming the system
                if i + batch_size < len(texts):
                    time.sleep(0.01)
            
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
                dot_product = np.dot(embedding1, embedding2)  # Multiply corresponding elements
                norm1 = np.linalg.norm(embedding1)  # Length of first vector
                norm2 = np.linalg.norm(embedding2)  # Length of second vector
                
                # Avoid division by zero
                if norm1 == 0 or norm2 == 0:
                    return 0.0
                
                return dot_product / (norm1 * norm2)
                
            elif method == "euclidean":
                # EUCLIDEAN DISTANCE - CONVERTED TO SIMILARITY
                # Measures straight-line distance between vectors
                # Convert to similarity: closer = higher score
                distance = np.linalg.norm(embedding1 - embedding2)
                return 1.0 / (1.0 + distance)  # Convert distance to similarity
                
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
            "device": self.device,
            "embedding_dimension": self.embedding_dimension,
            "is_initialized": self.is_initialized,
            "cuda_available": torch.cuda.is_available() if torch.cuda.is_available() else False
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
            
            logger.info(f"🔄 Reloading embedding model: {self.model_name}")
            
            # Clean up old model to free memory
            if self.model:
                del self.model
                self.model = None
            
            # Load new model
            self._initialize_model()
            
        except Exception as e:
            logger.error(f"❌ Failed to reload model: {str(e)}")
    
    def __del__(self):
        """
        ===============================================================================
        CLEANUP WHEN OBJECT IS DESTROYED
        ===============================================================================
        
        This method runs automatically when the object is deleted to:
        - Free up GPU memory
        - Clean up model resources
        - Prevent memory leaks
        """
        try:
            if self.model:
                del self.model
        except:
            pass