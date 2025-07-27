"""
Document loader module that handles different file types (PDF, DOCX, CSV) using a unified interface.
Optimized for fast processing with minimal torch dependencies.
"""
from pathlib import Path
from typing import List, Union
import re
import logging
import os

from langchain.schema import Document as LangChainDocument
from langchain.text_splitter import RecursiveCharacterTextSplitter

from src.config import CHUNK_SIZE, CHUNK_OVERLAP

# Set up logging with reduced verbosity
logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger(__name__)

# Suppress docling verbose logging
logging.getLogger("docling").setLevel(logging.ERROR)
logging.getLogger("docling.models.factories").setLevel(logging.ERROR)
logging.getLogger("docling.pipeline").setLevel(logging.ERROR)


class DocumentLoader:
    """Unified document loader for multiple file formats with lazy loading."""
    
    def __init__(self, chunk_size: int = CHUNK_SIZE, chunk_overlap: int = CHUNK_OVERLAP):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self._converter = None  # Lazy load to improve startup time
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap
        )
    
    @property
    def converter(self):
        """Lazy load DocumentConverter to improve startup performance."""
        if self._converter is None:
            try:
                from docling.document_converter import DocumentConverter
                self._converter = DocumentConverter()
                logger.info("DocumentConverter loaded successfully")
            except ImportError as e:
                logger.error(f"Failed to import DocumentConverter: {e}")
                raise ImportError("docling package is required for document processing")
        return self._converter

    def clean_text(self, text: str) -> str:
        """Clean extracted text by removing noise and normalizing spacing."""
        text = re.sub(r'\n\s*', ' ', text)  # Remove newlines and leading whitespace
        text = re.sub(r'\s+', ' ', text).strip()  # Normalize spacing
        text = re.sub(r'\$ (\d)', r'$\1', text)  # Fix currency formatting
        return text

    def load_single(self, file_path: Union[str, Path]) -> List[LangChainDocument]:
        """
        Load and chunk a single document file.
        
        Args:
            file_path: Path to the document file
            
        Returns:
            List of LangChain Document objects containing chunked text
        """
        file_path = Path(file_path)
        logger.info(f"Loading file: {file_path}")
        
        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")
            
        if not file_path.is_file():
            raise ValueError(f"Not a file: {file_path}")
            
        # Extract text using Docling with optimized settings
        try:
            # Suppress docling output during processing
            old_level = logging.getLogger().level
            logging.getLogger().setLevel(logging.ERROR)
            
            try:
                result = self.converter.convert(file_path)
                raw_text = result.document.export_to_text()
            finally:
                # Restore logging level
                logging.getLogger().setLevel(old_level)
            
            if not raw_text:
                raise ValueError(f"No text extracted from {file_path}")
                
            # Fast text cleaning and chunking
            clean_text = self.clean_text(raw_text)
            chunks = self.text_splitter.split_text(clean_text)
            
            if not chunks:
                raise ValueError(f"No chunks generated from {file_path}")
            
            # Create LangChain documents with metadata (optimized)
            documents = [
                LangChainDocument(
                    page_content=chunk,
                    metadata={
                        "source": str(file_path),
                        "file_type": file_path.suffix.lower(),
                        "file_name": file_path.name,
                        "chunk_index": i
                    }
                ) for i, chunk in enumerate(chunks)
            ]
            
            logger.warning(f"Processed {file_path.name}: {len(documents)} chunks")
            return documents
            
        except Exception as e:
            logger.error(f"Error processing {file_path}: {str(e)}")
            raise

    def load_batch(self, file_paths: List[Union[str, Path]]) -> List[LangChainDocument]:
        """
        Load and chunk multiple document files with optimized processing.
        
        Args:
            file_paths: List of paths to document files
            
        Returns:
            Combined list of LangChain Document objects from all files
        """
        if not file_paths:
            raise ValueError("No files provided")
            
        all_documents = []
        errors = []
        
        # Process files with minimal logging
        for i, file_path in enumerate(file_paths, 1):
            try:
                documents = self.load_single(file_path)
                all_documents.extend(documents)
            except Exception as e:
                logger.error(f"Error loading {Path(file_path).name}: {str(e)}")
                errors.append((file_path, str(e)))
                continue
                
        if errors and not all_documents:
            error_msg = "\n".join([f"{Path(path).name}: {error}" for path, error in errors[:3]])
            raise Exception(f"Failed to process files:\n{error_msg}")
            
        if not all_documents:
            raise ValueError("No documents were successfully processed")
            
        logger.warning(f"Batch complete: {len(file_paths)} files → {len(all_documents)} chunks")
        return all_documents 