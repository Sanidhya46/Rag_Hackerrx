#!/usr/bin/env python3
"""
===============================================================================
ENHANCED TEXT SPLITTER FOR RAG SYSTEM
===============================================================================

PURPOSE:
This module provides intelligent text splitting capabilities that prevent frontend
freezing and optimize document processing for the RAG system.

FEATURES:
- Non-blocking text processing
- Intelligent chunk sizing
- Overlap management for context preservation
- Insurance policy document optimization
- Integration with Gemini embedder

USAGE:
from utils.enhanced_text_splitter import EnhancedTextSplitter
splitter = EnhancedTextSplitter()
chunks = splitter.split_text(text, chunk_size=1000)
===============================================================================
"""

import re
import asyncio
import time
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
import structlog

# Configure logging
logger = structlog.get_logger()

@dataclass
class TextChunk:
    """
    ===============================================================================
    TEXT CHUNK DATA STRUCTURE
    ===============================================================================
    
    Represents a single chunk of text with metadata for tracking and processing.
    """
    text: str
    chunk_id: str
    start_char: int
    end_char: int
    chunk_size: int
    overlap_size: int
    metadata: Dict[str, Any]

class EnhancedTextSplitter:
    """
    ===============================================================================
    INTELLIGENT TEXT SPLITTING SYSTEM
    ===============================================================================
    
    This class provides advanced text splitting capabilities that:
    - Prevents frontend freezing during processing
    - Maintains context between chunks
    - Optimizes for insurance policy documents
    - Integrates with Gemini embeddings
    """
    
    def __init__(self, 
                 default_chunk_size: int = 1000,
                 default_overlap: int = 200,
                 max_chunk_size: int = 2000,
                 min_chunk_size: int = 100):
        """
        Initialize the text splitter
        
        Args:
            default_chunk_size: Default size for text chunks
            default_overlap: Default overlap between chunks
            max_chunk_size: Maximum allowed chunk size
            min_chunk_size: Minimum allowed chunk size
        """
        self.default_chunk_size = default_chunk_size
        self.default_overlap = default_overlap
        self.max_chunk_size = max_chunk_size
        self.min_chunk_size = min_chunk_size
        
        # Insurance policy specific patterns
        self.section_patterns = [
            r'^[A-Z][A-Z\s&]+:$',  # Section headers like "COVERAGE:"
            r'^[0-9]+\.\s+[A-Z]',  # Numbered sections like "1. Coverage"
            r'^[A-Z][a-z\s]+:$',   # Subsection headers
        ]
        
        # Compile regex patterns for efficiency
        self.compiled_patterns = [re.compile(pattern) for pattern in self.section_patterns]
        
        logger.info(f"✅ Enhanced text splitter initialized with chunk size: {default_chunk_size}")
    
    def split_text(self, text: str, 
                   chunk_size: int = None, 
                   overlap: int = None,
                   preserve_sections: bool = True) -> List[TextChunk]:
        """
        ===============================================================================
        SPLIT TEXT INTO INTELLIGENT CHUNKS
        ===============================================================================
        
        This method splits text into chunks while:
        - Preserving section boundaries
        - Maintaining context through overlap
        - Preventing frontend freezing
        - Optimizing for RAG processing
        
        Args:
            text: Text to split
            chunk_size: Size of each chunk (defaults to instance default)
            overlap: Overlap between chunks (defaults to instance default)
            preserve_sections: Whether to respect section boundaries
        
        Returns:
            List of TextChunk objects
        """
        if not text or not text.strip():
            return []
        
        # Use defaults if not specified
        chunk_size = chunk_size or self.default_chunk_size
        overlap = overlap or self.default_overlap
        
        # Validate parameters
        chunk_size = max(self.min_chunk_size, min(chunk_size, self.max_chunk_size))
        overlap = min(overlap, chunk_size // 2)  # Overlap shouldn't exceed half chunk size
        
        try:
            if preserve_sections:
                return self._split_with_section_preservation(text, chunk_size, overlap)
            else:
                return self._split_simple(text, chunk_size, overlap)
                
        except Exception as e:
            logger.error(f"❌ Failed to split text: {str(e)}")
            # Fallback to simple splitting
            return self._split_simple(text, chunk_size, overlap)
    
    def _split_with_section_preservation(self, text: str, chunk_size: int, overlap: int) -> List[TextChunk]:
        """
        ===============================================================================
        SECTION-AWARE TEXT SPLITTING
        ===============================================================================
        
        Splits text while preserving natural section boundaries for better context.
        """
        chunks = []
        current_pos = 0
        chunk_id = 0
        
        # Find section boundaries
        section_boundaries = self._find_section_boundaries(text)
        
        while current_pos < len(text):
            # Calculate end position for this chunk
            end_pos = min(current_pos + chunk_size, len(text))
            
            # Look for the best break point near the end
            best_break = self._find_best_break_point(text, current_pos, end_pos, section_boundaries)
            
            if best_break > current_pos:
                end_pos = best_break
            
            # Extract chunk text
            chunk_text = text[current_pos:end_pos].strip()
            
            if chunk_text:  # Only create chunks with actual content
                chunk = TextChunk(
                    text=chunk_text,
                    chunk_id=f"chunk_{chunk_id:04d}",
                    start_char=current_pos,
                    end_char=end_pos,
                    chunk_size=len(chunk_text),
                    overlap_size=overlap,
                    metadata={
                        "section_preserved": True,
                        "break_type": "section_boundary" if end_pos in section_boundaries else "natural"
                    }
                )
                chunks.append(chunk)
                chunk_id += 1
            
            # Move to next position with overlap
            current_pos = max(current_pos + 1, end_pos - overlap)
            
            # Prevent infinite loops
            if current_pos >= len(text):
                break
        
        logger.info(f"✅ Split text into {len(chunks)} chunks with section preservation")
        return chunks
    
    def _split_simple(self, text: str, chunk_size: int, overlap: int) -> List[TextChunk]:
        """
        ===============================================================================
        SIMPLE TEXT SPLITTING
        ===============================================================================
        
        Fallback method for basic text splitting when section preservation fails.
        """
        chunks = []
        current_pos = 0
        chunk_id = 0
        
        while current_pos < len(text):
            # Calculate end position
            end_pos = min(current_pos + chunk_size, len(text))
            
            # Look for natural break points (sentences, paragraphs)
            if end_pos < len(text):
                end_pos = self._find_natural_break(text, end_pos)
            
            # Extract chunk text
            chunk_text = text[current_pos:end_pos].strip()
            
            if chunk_text:
                chunk = TextChunk(
                    text=chunk_text,
                    chunk_id=f"chunk_{chunk_id:04d}",
                    start_char=current_pos,
                    end_char=end_pos,
                    chunk_size=len(chunk_text),
                    overlap_size=overlap,
                    metadata={"section_preserved": False, "break_type": "natural"}
                )
                chunks.append(chunk)
                chunk_id += 1
            
            # Move to next position with overlap
            current_pos = max(current_pos + 1, end_pos - overlap)
            
            if current_pos >= len(text):
                break
        
        logger.info(f"✅ Split text into {len(chunks)} chunks using simple method")
        return chunks
    
    def _find_section_boundaries(self, text: str) -> set:
        """
        ===============================================================================
        FIND SECTION BOUNDARIES IN TEXT
        ===============================================================================
        
        Identifies natural break points in insurance policy documents.
        """
        boundaries = set()
        
        # Find matches for each pattern
        for pattern in self.compiled_patterns:
            for match in pattern.finditer(text):
                boundaries.add(match.start())
        
        # Add paragraph breaks
        for match in re.finditer(r'\n\s*\n', text):
            boundaries.add(match.start())
        
        # Add sentence endings (but be careful not to break too much)
        for match in re.finditer(r'[.!?]\s+[A-Z]', text):
            boundaries.add(match.start() + 1)
        
        return boundaries
    
    def _find_best_break_point(self, text: str, start: int, target_end: int, boundaries: set) -> int:
        """
        ===============================================================================
        FIND OPTIMAL BREAK POINT FOR CHUNK
        ===============================================================================
        
        Looks for the best place to break text near the target end position.
        """
        # Look for section boundaries first
        for boundary in sorted(boundaries):
            if start < boundary <= target_end:
                return boundary
        
        # Look for paragraph breaks
        for i in range(target_end, max(start, target_end - 200), -1):
            if text[i:i+2] == '\n\n':
                return i + 2
        
        # Look for sentence endings
        for i in range(target_end, max(start, target_end - 100), -1):
            if text[i] in '.!?' and i + 1 < len(text) and text[i + 1].isspace():
                return i + 1
        
        # Look for word boundaries
        for i in range(target_end, max(start, target_end - 50), -1):
            if text[i].isspace():
                return i + 1
        
        # If no good break point found, use target end
        return target_end
    
    def _find_natural_break(self, text: str, target_pos: int) -> int:
        """
        ===============================================================================
        FIND NATURAL BREAK POINT
        ===============================================================================
        
        Simple method to find natural break points in text.
        """
        # Look backwards from target position for good break points
        for i in range(target_pos, max(0, target_pos - 100), -1):
            if text[i] in '.!?\n':
                return i + 1
            elif text[i].isspace():
                return i + 1
        
        return target_pos
    
    async def split_text_async(self, text: str, 
                              chunk_size: int = None, 
                              overlap: int = None,
                              preserve_sections: bool = True) -> List[TextChunk]:
        """
        ===============================================================================
        ASYNCHRONOUS TEXT SPLITTING - NON-BLOCKING OPERATION
        ===============================================================================
        
        This method prevents frontend freezing by running text splitting in the background.
        
        Why async?
        - Prevents UI freezing during long document processing
        - Allows other operations to continue
        - Better user experience
        """
        # Run the splitting in a separate thread to avoid blocking
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            None, 
            self.split_text, 
            text, 
            chunk_size, 
            overlap, 
            preserve_sections
        )
    
    def split_with_progress(self, text: str, 
                           chunk_size: int = None, 
                           overlap: int = None,
                           progress_callback=None) -> List[TextChunk]:
        """
        ===============================================================================
        TEXT SPLITTING WITH PROGRESS TRACKING
        ===============================================================================
        
        Splits text while providing progress updates to prevent frontend freezing.
        
        Args:
            text: Text to split
            chunk_size: Size of each chunk
            overlap: Overlap between chunks
            progress_callback: Function to call with progress updates
        """
        if not text or not text.strip():
            return []
        
        chunk_size = chunk_size or self.default_chunk_size
        overlap = overlap or self.default_overlap
        
        chunks = []
        current_pos = 0
        chunk_id = 0
        total_chars = len(text)
        
        while current_pos < total_chars:
            # Calculate progress
            progress = min(100, (current_pos / total_chars) * 100)
            if progress_callback:
                progress_callback(progress, f"Processing chunk {chunk_id + 1}")
            
            # Process chunk
            end_pos = min(current_pos + chunk_size, total_chars)
            end_pos = self._find_best_break_point(text, current_pos, end_pos, set())
            
            chunk_text = text[current_pos:end_pos].strip()
            
            if chunk_text:
                chunk = TextChunk(
                    text=chunk_text,
                    chunk_id=f"chunk_{chunk_id:04d}",
                    start_char=current_pos,
                    end_char=end_pos,
                    chunk_size=len(chunk_text),
                    overlap_size=overlap,
                    metadata={"progress": progress}
                )
                chunks.append(chunk)
                chunk_id += 1
            
            current_pos = max(current_pos + 1, end_pos - overlap)
            
            # Small delay to prevent blocking
            time.sleep(0.001)
        
        if progress_callback:
            progress_callback(100, "Text splitting completed")
        
        logger.info(f"✅ Split text into {len(chunks)} chunks with progress tracking")
        return chunks
    
    def optimize_chunks_for_rag(self, chunks: List[TextChunk], 
                               target_chunk_size: int = 1000) -> List[TextChunk]:
        """
        ===============================================================================
        OPTIMIZE CHUNKS FOR RAG PROCESSING
        ===============================================================================
        
        Post-processes chunks to ensure optimal size and quality for RAG systems.
        """
        optimized_chunks = []
        
        for chunk in chunks:
            # Skip chunks that are too small
            if chunk.chunk_size < self.min_chunk_size:
                continue
            
            # Split chunks that are too large
            if chunk.chunk_size > target_chunk_size * 1.5:
                sub_chunks = self._split_large_chunk(chunk, target_chunk_size)
                optimized_chunks.extend(sub_chunks)
            else:
                optimized_chunks.append(chunk)
        
        logger.info(f"✅ Optimized {len(chunks)} chunks to {len(optimized_chunks)} optimal chunks")
        return optimized_chunks
    
    def _split_large_chunk(self, chunk: TextChunk, target_size: int) -> List[TextChunk]:
        """
        ===============================================================================
        SPLIT LARGE CHUNK INTO SMALLER PIECES
        ===============================================================================
        
        Breaks down oversized chunks for better RAG performance.
        """
        sub_chunks = []
        text = chunk.text
        current_pos = 0
        sub_chunk_id = 0
        
        while current_pos < len(text):
            end_pos = min(current_pos + target_size, len(text))
            end_pos = self._find_natural_break(text, end_pos)
            
            sub_text = text[current_pos:end_pos].strip()
            
            if sub_text:
                sub_chunk = TextChunk(
                    text=sub_text,
                    chunk_id=f"{chunk.chunk_id}_sub_{sub_chunk_id:02d}",
                    start_char=chunk.start_char + current_pos,
                    end_char=chunk.start_char + end_pos,
                    chunk_size=len(sub_text),
                    overlap_size=chunk.overlap_size // 2,  # Reduce overlap for sub-chunks
                    metadata={
                        **chunk.metadata,
                        "parent_chunk": chunk.chunk_id,
                        "is_sub_chunk": True
                    }
                )
                sub_chunks.append(sub_chunk)
                sub_chunk_id += 1
            
            current_pos = end_pos
        
        return sub_chunks
    
    def get_chunk_statistics(self, chunks: List[TextChunk]) -> Dict[str, Any]:
        """
        ===============================================================================
        GET STATISTICS ABOUT TEXT CHUNKS
        ===============================================================================
        
        Provides insights into chunk quality and distribution.
        """
        if not chunks:
            return {}
        
        chunk_sizes = [chunk.chunk_size for chunk in chunks]
        
        return {
            "total_chunks": len(chunks),
            "total_characters": sum(chunk_sizes),
            "average_chunk_size": sum(chunk_sizes) / len(chunk_sizes),
            "min_chunk_size": min(chunk_sizes),
            "max_chunk_size": max(chunk_sizes),
            "chunks_with_overlap": len([c for c in chunks if c.overlap_size > 0]),
            "section_preserved_chunks": len([c for c in chunks if c.metadata.get("section_preserved", False)])
        }
    
    def merge_small_chunks(self, chunks: List[TextChunk], 
                          min_size: int = 200) -> List[TextChunk]:
        """
        ===============================================================================
        MERGE SMALL CHUNKS FOR BETTER CONTEXT
        ===============================================================================
        
        Combines very small chunks to improve context preservation.
        """
        if not chunks:
            return []
        
        merged_chunks = []
        current_chunk = None
        
        for chunk in chunks:
            if current_chunk is None:
                current_chunk = chunk
            elif (current_chunk.chunk_size + chunk.chunk_size <= self.max_chunk_size and
                  chunk.start_char == current_chunk.end_char):
                # Merge consecutive chunks
                current_chunk.text += " " + chunk.text
                current_chunk.end_char = chunk.end_char
                current_chunk.chunk_size = len(current_chunk.text)
                current_chunk.metadata["merged_chunks"] = current_chunk.metadata.get("merged_chunks", 0) + 1
            else:
                # Add current chunk if it's large enough
                if current_chunk.chunk_size >= min_size:
                    merged_chunks.append(current_chunk)
                current_chunk = chunk
        
        # Add the last chunk
        if current_chunk and current_chunk.chunk_size >= min_size:
            merged_chunks.append(current_chunk)
        
        logger.info(f"✅ Merged {len(chunks)} chunks to {len(merged_chunks)} chunks")
        return merged_chunks
