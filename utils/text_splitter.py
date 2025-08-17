import re
import logging
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords

# Download required NLTK data
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    try:
        nltk.download('punkt', quiet=True)
    except:
        pass

try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    try:
        nltk.download('stopwords', quiet=True)
    except:
        pass

logger = logging.getLogger(__name__)

@dataclass
class TextChunk:
    """Represents a text chunk with metadata"""
    content: str
    start_index: int
    end_index: int
    chunk_id: str
    metadata: Dict[str, Any]
    word_count: int
    sentence_count: int

class DocumentSplitter:
    """
    Intelligent document splitter with multiple strategies and error handling
    """
    
    def __init__(self, 
                 default_chunk_size: int = 1000,
                 default_overlap: int = 200,
                 preserve_sentences: bool = True,
                 smart_boundaries: bool = True):
        """
        Initialize the document splitter
        
        Args:
            default_chunk_size: Default chunk size in characters
            default_overlap: Default overlap between chunks
            preserve_sentences: Whether to preserve sentence boundaries
            smart_boundaries: Whether to use smart boundary detection
        """
        self.default_chunk_size = default_chunk_size
        self.default_overlap = default_overlap
        self.preserve_sentences = preserve_sentences
        self.smart_boundaries = smart_boundaries
        
        # Initialize stopwords for smart boundary detection
        try:
            self.stopwords = set(stopwords.words('english'))
        except:
            self.stopwords = set()
    
    def split_text(self, 
                   text: str, 
                   chunk_size: int = None, 
                   overlap: int = None,
                   strategy: str = "smart") -> List[str]:
        """
        Split text into chunks using specified strategy
        
        Args:
            text: Text to split
            chunk_size: Size of each chunk
            overlap: Overlap between chunks
            strategy: Splitting strategy ('fixed', 'smart', 'semantic', 'sentence')
            
        Returns:
            List of text chunks
        """
        if not text or not text.strip():
            logger.warning("Empty text provided for splitting")
            return []
        
        chunk_size = chunk_size or self.default_chunk_size
        overlap = overlap or self.default_overlap
        
        try:
            if strategy == "fixed":
                return self._split_fixed(text, chunk_size, overlap)
            elif strategy == "smart":
                return self._split_smart(text, chunk_size, overlap)
            elif strategy == "semantic":
                return self._split_semantic(text, chunk_size, overlap)
            elif strategy == "sentence":
                return self._split_by_sentences(text, chunk_size)
            else:
                logger.warning(f"Unknown strategy: {strategy}. Using smart strategy.")
                return self._split_smart(text, chunk_size, overlap)
                
        except Exception as e:
            logger.error(f"Text splitting failed: {str(e)}")
            # Fallback to simple splitting
            return self._split_fixed(text, chunk_size, overlap)
    
    def _split_fixed(self, text: str, chunk_size: int, overlap: int) -> List[str]:
        """Simple fixed-size splitting"""
        try:
            chunks = []
            start = 0
            
            while start < len(text):
                end = start + chunk_size
                chunk = text[start:end]
                
                if chunk.strip():
                    chunks.append(chunk.strip())
                
                start = end - overlap
                if start >= len(text):
                    break
            
            return chunks
            
        except Exception as e:
            logger.error(f"Fixed splitting failed: {str(e)}")
            return [text]
    
    def _split_smart(self, text: str, chunk_size: int, overlap: int) -> List[str]:
        """Smart splitting with boundary detection"""
        try:
            chunks = []
            sentences = self._split_into_sentences(text)
            
            current_chunk = ""
            current_length = 0
            
            for sentence in sentences:
                sentence_length = len(sentence)
                
                # If adding this sentence would exceed chunk size
                if current_length + sentence_length > chunk_size and current_chunk:
                    # Find the chunk
                    chunks.append(current_chunk.strip())
                    
                    # Start new chunk with overlap
                    if overlap > 0:
                        # Find last sentence boundary for overlap
                        overlap_text = self._get_overlap_text(current_chunk, overlap)
                        current_chunk = overlap_text + sentence
                        current_length = len(overlap_text) + sentence_length
                    else:
                        current_chunk = sentence
                        current_length = sentence_length
                else:
                    current_chunk += " " + sentence if current_chunk else sentence
                    current_length += sentence_length
            
            # Add the last chunk
            if current_chunk.strip():
                chunks.append(current_chunk.strip())
            
            return chunks
            
        except Exception as e:
            logger.error(f"Smart splitting failed: {str(e)}")
            return self._split_fixed(text, chunk_size, overlap)
    
    def _split_semantic(self, text: str, chunk_size: int, overlap: int) -> List[str]:
        """Semantic splitting based on content structure"""
        try:
            # Split by paragraphs first
            paragraphs = text.split('\n\n')
            
            chunks = []
            current_chunk = ""
            current_length = 0
            
            for paragraph in paragraphs:
                paragraph = paragraph.strip()
                if not paragraph:
                    continue
                
                paragraph_length = len(paragraph)
                
                # If adding this paragraph would exceed chunk size
                if current_length + paragraph_length > chunk_size and current_chunk:
                    chunks.append(current_chunk.strip())
                    
                    # Start new chunk with overlap
                    if overlap > 0:
                        overlap_text = self._get_overlap_text(current_chunk, overlap)
                        current_chunk = overlap_text + "\n\n" + paragraph
                        current_length = len(overlap_text) + paragraph_length
                    else:
                        current_chunk = paragraph
                        current_length = paragraph_length
                else:
                    current_chunk += "\n\n" + paragraph if current_chunk else paragraph
                    current_length += paragraph_length
            
            # Add the last chunk
            if current_chunk.strip():
                chunks.append(current_chunk.strip())
            
            return chunks
            
        except Exception as e:
            logger.error(f"Semantic splitting failed: {str(e)}")
            return self._split_smart(text, chunk_size, overlap)
    
    def _split_by_sentences(self, text: str, max_chunk_size: int) -> List[str]:
        """Split by sentences while respecting max chunk size"""
        try:
            sentences = self._split_into_sentences(text)
            
            chunks = []
            current_chunk = ""
            current_length = 0
            
            for sentence in sentences:
                sentence_length = len(sentence)
                
                # If adding this sentence would exceed max size
                if current_length + sentence_length > max_chunk_size and current_chunk:
                    chunks.append(current_chunk.strip())
                    current_chunk = sentence
                    current_length = sentence_length
                else:
                    current_chunk += " " + sentence if current_chunk else sentence
                    current_length += sentence_length
            
            # Add the last chunk
            if current_chunk.strip():
                chunks.append(current_chunk.strip())
            
            return chunks
            
        except Exception as e:
            logger.error(f"Sentence splitting failed: {str(e)}")
            return [text]
    
    def _split_into_sentences(self, text: str) -> List[str]:
        """Split text into sentences using NLTK"""
        try:
            # Clean text
            cleaned_text = self._clean_text(text)
            
            # Split into sentences
            sentences = sent_tokenize(cleaned_text)
            
            # Filter out very short sentences
            sentences = [s.strip() for s in sentences if len(s.strip()) > 10]
            
            return sentences
            
        except Exception as e:
            logger.error(f"Sentence tokenization failed: {str(e)}")
            # Fallback to simple period-based splitting
            return [s.strip() for s in text.split('.') if s.strip()]
    
    def _clean_text(self, text: str) -> str:
        """Clean and normalize text"""
        try:
            # Remove extra whitespace
            text = re.sub(r'\s+', ' ', text)
            
            # Normalize line breaks
            text = re.sub(r'\n+', '\n', text)
            
            # Remove special characters that might interfere with sentence splitting
            text = re.sub(r'[^\w\s\.\!\?\,\;\:\-\(\)\[\]\{\}]', ' ', text)
            
            return text.strip()
            
        except Exception as e:
            logger.error(f"Text cleaning failed: {str(e)}")
            return text
    
    def _get_overlap_text(self, text: str, overlap_size: int) -> str:
        """Get overlap text from the end of a chunk"""
        try:
            if overlap_size >= len(text):
                return text
            
            # Try to find a sentence boundary within overlap
            overlap_text = text[-overlap_size:]
            
            # Find the first sentence boundary
            sentences = self._split_into_sentences(overlap_text)
            if sentences:
                return sentences[0]
            
            return overlap_text
            
        except Exception as e:
            logger.error(f"Overlap text extraction failed: {str(e)}")
            return text[-overlap_size:] if len(text) > overlap_size else text
    
    def split_with_metadata(self, 
                           text: str, 
                           chunk_size: int = None, 
                           overlap: int = None,
                           strategy: str = "smart") -> List[TextChunk]:
        """
        Split text with detailed metadata
        
        Args:
            text: Text to split
            chunk_size: Size of each chunk
            overlap: Overlap between chunks
            strategy: Splitting strategy
            
        Returns:
            List of TextChunk objects with metadata
        """
        try:
            raw_chunks = self.split_text(text, chunk_size, overlap, strategy)
            
            chunks_with_metadata = []
            current_index = 0
            
            for i, chunk in enumerate(raw_chunks):
                # Calculate word and sentence counts
                word_count = len(word_tokenize(chunk)) if chunk else 0
                sentence_count = len(self._split_into_sentences(chunk))
                
                # Create metadata
                metadata = {
                    "chunk_index": i,
                    "strategy": strategy,
                    "chunk_size": chunk_size,
                    "overlap": overlap,
                    "original_length": len(text),
                    "chunk_ratio": len(chunk) / len(text) if text else 0
                }
                
                # Create TextChunk object
                text_chunk = TextChunk(
                    content=chunk,
                    start_index=current_index,
                    end_index=current_index + len(chunk),
                    chunk_id=f"chunk_{i}_{hash(chunk) % 10000}",
                    metadata=metadata,
                    word_count=word_count,
                    sentence_count=sentence_count
                )
                
                chunks_with_metadata.append(text_chunk)
                current_index += len(chunk) - overlap if overlap > 0 else len(chunk)
            
            return chunks_with_metadata
            
        except Exception as e:
            logger.error(f"Metadata splitting failed: {str(e)}")
            # Return simple chunks without metadata
            return [TextChunk(
                content=chunk,
                start_index=0,
                end_index=len(chunk),
                chunk_id=f"chunk_{i}",
                metadata={},
                word_count=len(chunk.split()),
                sentence_count=1
            ) for i, chunk in enumerate(raw_chunks)]
    
    def optimize_chunk_size(self, text: str, target_chunks: int = 10) -> int:
        """
        Optimize chunk size to achieve target number of chunks
        
        Args:
            text: Text to analyze
            target_chunks: Target number of chunks
            
        Returns:
            Optimized chunk size
        """
        try:
            if not text:
                return self.default_chunk_size
            
            # Start with default size
            chunk_size = self.default_chunk_size
            
            # Binary search for optimal chunk size
            min_size = 100
            max_size = len(text)
            
            for _ in range(10):  # Max 10 iterations
                chunks = self.split_text(text, chunk_size, self.default_overlap)
                
                if len(chunks) == target_chunks:
                    break
                elif len(chunks) > target_chunks:
                    # Too many chunks, increase size
                    min_size = chunk_size
                    chunk_size = (chunk_size + max_size) // 2
                else:
                    # Too few chunks, decrease size
                    max_size = chunk_size
                    chunk_size = (min_size + chunk_size) // 2
                
                if max_size - min_size < 100:
                    break
            
            return chunk_size
            
        except Exception as e:
            logger.error(f"Chunk size optimization failed: {str(e)}")
            return self.default_chunk_size
    
    def get_splitting_stats(self, text: str, chunk_size: int = None, overlap: int = None) -> Dict[str, Any]:
        """
        Get statistics about text splitting
        
        Args:
            text: Text to analyze
            chunk_size: Chunk size to use
            overlap: Overlap size to use
            
        Returns:
            Dictionary with splitting statistics
        """
        try:
            chunk_size = chunk_size or self.default_chunk_size
            overlap = overlap or self.default_overlap
            
            chunks = self.split_text(text, chunk_size, overlap)
            
            if not chunks:
                return {
                    "total_chunks": 0,
                    "avg_chunk_size": 0,
                    "min_chunk_size": 0,
                    "max_chunk_size": 0,
                    "total_characters": 0,
                    "efficiency": 0.0
                }
            
            chunk_sizes = [len(chunk) for chunk in chunks]
            total_chars = sum(chunk_sizes)
            
            # Calculate efficiency (how well chunks utilize the target size)
            efficiency = sum(min(size, chunk_size) for size in chunk_sizes) / (len(chunks) * chunk_size)
            
            return {
                "total_chunks": len(chunks),
                "avg_chunk_size": total_chars / len(chunks),
                "min_chunk_size": min(chunk_sizes),
                "max_chunk_size": max(chunk_sizes),
                "total_characters": total_chars,
                "efficiency": efficiency,
                "chunk_size": chunk_size,
                "overlap": overlap
            }
            
        except Exception as e:
            logger.error(f"Statistics calculation failed: {str(e)}")
            return {}
