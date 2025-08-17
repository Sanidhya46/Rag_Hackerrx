#!/usr/bin/env python3
"""
===============================================================================
PDF LOADER FOR RAG SYSTEM
===============================================================================

PURPOSE:
This module provides PDF loading capabilities for the RAG system,
integrating with our custom PDF parser and enhanced text splitter.

FEATURES:
- Custom PDF parsing (no LangChain dependency)
- Integration with enhanced text splitter
- Error handling and validation
- Support for various PDF formats

USAGE:
from loaders.pdf_loader import PDFLoader
loader = PDFLoader()
text_content = loader.load_pdf_from_bytes(pdf_bytes)
===============================================================================
"""

import os
import logging
from typing import Optional, Dict, Any
from utils_api.pdf_parser import PDFParser  # Make sure this has a .parse() method

# Configure logging
logger = logging.getLogger(__name__)

class PDFLoader:
    """
    ===============================================================================
    PDF LOADER FOR RAG SYSTEM
    ===============================================================================
    
    This class provides PDF loading capabilities that integrate with our
    custom PDF parser and enhanced text splitter.
    """
    
    def __init__(self):
        """Initialize the PDF loader"""
        self.pdf_parser = PDFParser()
        logger.info("✅ PDF loader initialized successfully")
    
    def load_pdf_from_bytes(self, pdf_bytes: bytes) -> Optional[str]:
        """
        Load PDF content from bytes
        
        Args:
            pdf_bytes: PDF file content as bytes
        
        Returns:
            Extracted text content or None if failed
        """
        try:
            if not pdf_bytes:
                logger.error("❌ No PDF bytes provided")
                return None

            # Validate PDF first
            if not self.validate_pdf(pdf_bytes):
                logger.error("❌ Invalid PDF file format")
                return None

            # Parse PDF using our custom parser
            text_content = self.pdf_parser.parse_pdf_from_bytes(pdf_bytes) # Make sure parse() exists
            
            if text_content and text_content.strip():
                logger.info(f"✅ PDF loaded successfully, extracted {len(text_content)} characters")
                return text_content
            else:
                logger.warning("⚠️ PDF loaded but no text content extracted")
                return None
                
        except Exception as e:
            logger.error(f"❌ Failed to load PDF: {str(e)}")
            return None
    
    def load_pdf_from_file(self, file_path: str) -> Optional[str]:
        """
        Load PDF content from file path
        
        Args:
            file_path: Path to PDF file
        
        Returns:
            Extracted text content or None if failed
        """
        try:
            if not os.path.exists(file_path):
                logger.error(f"❌ PDF file not found: {file_path}")
                return None
            
            with open(file_path, 'rb') as file:
                pdf_bytes = file.read()
            
            return self.load_pdf_from_bytes(pdf_bytes)
            
        except Exception as e:
            logger.error(f"❌ Failed to load PDF from file {file_path}: {str(e)}")
            return None
    
    def get_pdf_info(self, pdf_bytes: bytes) -> Dict[str, Any]:
        """
        Get basic information about PDF
        
        Args:
            pdf_bytes: PDF file content as bytes
        
        Returns:
            Dictionary with PDF information
        """
        try:
            info = {
                "file_size_bytes": len(pdf_bytes),
                "file_size_mb": round(len(pdf_bytes) / (1024 * 1024), 2),
                "has_content": False,
                "content_length": 0
            }
            
            text_content = self.load_pdf_from_bytes(pdf_bytes)
            if text_content:
                info["has_content"] = True
                info["content_length"] = len(text_content)
            
            return info
            
        except Exception as e:
            logger.error(f"❌ Failed to get PDF info: {str(e)}")
            return {"error": str(e)}
    
    def validate_pdf(self, pdf_bytes: bytes) -> bool:
        """
        Validate PDF file
        
        Args:
            pdf_bytes: PDF file content as bytes
        
        Returns:
            True if valid PDF, False otherwise
        """
        try:
            if len(pdf_bytes) < 4:
                return False
            
            if pdf_bytes[:4] != b'%PDF':
                return False
            
            if len(pdf_bytes) < 100:
                return False
            
            return True
            
        except Exception as e:
            logger.error(f"❌ PDF validation failed: {str(e)}")
            return False
