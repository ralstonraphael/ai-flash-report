"""
Document loader module that handles different file types (PDF, DOCX, CSV) using a unified interface.
"""
from pathlib import Path
from typing import List, Union
import re
import logging

from langchain.schema import Document as LangChainDocument
from langchain.text_splitter import RecursiveCharacterTextSplitter
from docling.document_converter import DocumentConverter

from src.config import CHUNK_SIZE, CHUNK_OVERLAP

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DocumentLoader:
    """Unified document loader for multiple file formats."""
    
    def __init__(self, chunk_size: int = CHUNK_SIZE, chunk_overlap: int = CHUNK_OVERLAP):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.converter = DocumentConverter()
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap
        )

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
            
        # Extract text using Docling
        try:
            logger.info(f"Converting {file_path} using Docling")
            result = self.converter.convert(file_path)
            raw_text = result.document.export_to_text()
            
            if not raw_text:
                raise ValueError(f"No text extracted from {file_path}")
                
            logger.info(f"Cleaning text from {file_path}")
            clean_text = self.clean_text(raw_text)
            
            # Split into chunks
            logger.info(f"Splitting text into chunks (size={self.chunk_size}, overlap={self.chunk_overlap})")
            chunks = self.text_splitter.split_text(clean_text)
            
            if not chunks:
                raise ValueError(f"No chunks generated from {file_path}")
            
            # Create LangChain documents with metadata
            documents = []
            for i, chunk in enumerate(chunks):
                doc = LangChainDocument(
                    page_content=chunk,
                    metadata={
                        "source": str(file_path),
                        "file_type": file_path.suffix.lower(),
                        "file_name": file_path.name,
                        "chunk_index": i
                    }
                )
                documents.append(doc)
            
            logger.info(f"Successfully processed {file_path}: {len(documents)} chunks created")
            return documents
            
        except Exception as e:
            logger.error(f"Error processing {file_path}: {str(e)}")
            raise

    def load_batch(self, file_paths: List[Union[str, Path]]) -> List[LangChainDocument]:
        """
        Load and chunk multiple document files.
        
        Args:
            file_paths: List of paths to document files
            
        Returns:
            Combined list of LangChain Document objects from all files
        """
        if not file_paths:
            raise ValueError("No files provided")
            
        logger.info(f"Processing batch of {len(file_paths)} files")
        all_documents = []
        errors = []
        
        for file_path in file_paths:
            try:
                documents = self.load_single(file_path)
                all_documents.extend(documents)
            except Exception as e:
                logger.error(f"Error loading {file_path}: {str(e)}")
                errors.append((file_path, str(e)))
                continue
                
        if errors:
            error_msg = "\n".join([f"{path}: {error}" for path, error in errors])
            raise Exception(f"Errors occurred while processing files:\n{error_msg}")
            
        if not all_documents:
            raise ValueError("No documents were successfully processed")
            
        logger.info(f"Successfully processed {len(file_paths)} files, created {len(all_documents)} chunks")
        return all_documents 