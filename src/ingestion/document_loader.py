"""
Document loader module that handles different file types (PDF, DOCX, CSV) using a unified interface.
Optimized for fast processing with minimal torch dependencies and isolated processing.
"""
from pathlib import Path
from typing import List, Union
import re
import logging
import os
import subprocess
import tempfile
import json
import sys

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
    """Unified document loader for multiple file formats with isolated torch processing."""
    
    def __init__(self, chunk_size: int = CHUNK_SIZE, chunk_overlap: int = CHUNK_OVERLAP):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self._converter = None  # Lazy load to improve startup time
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap
        )
        self._use_isolated_processing = True  # Use subprocess by default
        self._fallback_available = None  # Cache fallback availability
    
    def _check_fallback_available(self) -> bool:
        """Check if fallback PDF processing is available."""
        if self._fallback_available is None:
            try:
                import PyPDF2
                self._fallback_available = True
                logger.info("PyPDF2 fallback available")
            except ImportError:
                self._fallback_available = False
                logger.warning("PyPDF2 not available for fallback")
        return self._fallback_available
    
    def _extract_text_fallback(self, file_path: Path) -> str:
        """Fallback text extraction using PyPDF2 for PDFs."""
        if file_path.suffix.lower() != '.pdf':
            raise Exception("Fallback only supports PDF files")
            
        try:
            import PyPDF2
            
            with open(file_path, 'rb') as file:
                pdf_reader = PyPDF2.PdfReader(file)
                text_parts = []
                
                for page_num in range(len(pdf_reader.pages)):
                    page = pdf_reader.pages[page_num]
                    text_parts.append(page.extract_text())
                
                raw_text = '\n'.join(text_parts)
                if not raw_text.strip():
                    raise Exception("No text extracted from PDF")
                    
                logger.warning(f"✅ Fallback extraction successful for {file_path.name}")
                return raw_text
                
        except ImportError:
            raise Exception("PyPDF2 not available for fallback processing")
        except Exception as e:
            raise Exception(f"Fallback processing failed: {str(e)}")
    
    def _extract_text_isolated(self, file_path: Path) -> str:
        """Extract text using isolated subprocess to avoid torch conflicts."""
        try:
            # Create a temporary Python script for isolated processing
            script_content = f'''
import sys
import os
import logging

# Suppress all torch-related warnings and errors
os.environ["PYTHONWARNINGS"] = "ignore"
os.environ["TORCH_LOGS"] = "ERROR"
logging.getLogger().setLevel(logging.ERROR)

try:
    from docling.document_converter import DocumentConverter
    converter = DocumentConverter()
    result = converter.convert(r"{file_path}")
    text = result.document.export_to_text()
    print("SUCCESS:" + text)
except Exception as e:
    print("ERROR:" + str(e))
'''
            
            # Write script to temporary file
            with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
                f.write(script_content)
                script_path = f.name
            
            try:
                # Run the script in isolated process
                result = subprocess.run(
                    [sys.executable, script_path],
                    capture_output=True,
                    text=True,
                    timeout=120,  # 2 minute timeout
                    env={**os.environ, 'PYTHONWARNINGS': 'ignore'}
                )
                
                if result.returncode == 0 and result.stdout.startswith("SUCCESS:"):
                    return result.stdout[8:]  # Remove "SUCCESS:" prefix
                else:
                    error_msg = result.stderr or result.stdout
                    if "ERROR:" in result.stdout:
                        error_msg = result.stdout.split("ERROR:", 1)[1]
                    raise Exception(f"Subprocess failed: {error_msg}")
                    
            finally:
                # Clean up temporary script
                try:
                    os.unlink(script_path)
                except:
                    pass
                    
        except subprocess.TimeoutExpired:
            raise Exception("Document processing timed out after 2 minutes")
        except Exception as e:
            logger.error(f"Isolated processing failed: {e}")
            raise
    
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
        Load and chunk a single document file with isolated processing and fallback.
        
        Args:
            file_path: Path to the document file
            
        Returns:
            List of LangChain Document objects containing chunked text
        """
        file_path = Path(file_path)
        logger.warning(f"Processing: {file_path.name}")
        
        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")
            
        if not file_path.is_file():
            raise ValueError(f"Not a file: {file_path}")
            
        # Extract text with multiple fallback strategies
        raw_text = None
        error_messages = []
        
        # Strategy 1: Isolated processing
        if self._use_isolated_processing:
            try:
                raw_text = self._extract_text_isolated(file_path)
            except Exception as e:
                error_messages.append(f"Isolated processing: {str(e)}")
                logger.warning(f"Isolated processing failed: {e}")
        
        # Strategy 2: Direct processing
        if not raw_text:
            try:
                raw_text = self._extract_text_direct(file_path)
            except Exception as e:
                error_messages.append(f"Direct processing: {str(e)}")
                logger.warning(f"Direct processing failed: {e}")
        
        # Strategy 3: Fallback processing (PDF only)
        if not raw_text and file_path.suffix.lower() == '.pdf' and self._check_fallback_available():
            try:
                raw_text = self._extract_text_fallback(file_path)
            except Exception as e:
                error_messages.append(f"Fallback processing: {str(e)}")
                logger.warning(f"Fallback processing failed: {e}")
        
        if not raw_text:
            combined_errors = "; ".join(error_messages)
            raise Exception(f"All processing methods failed for {file_path.name}: {combined_errors}")
            
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
        
        logger.warning(f"✅ {file_path.name}: {len(documents)} chunks")
        return documents
    
    def _extract_text_direct(self, file_path: Path) -> str:
        """Direct text extraction with torch suppression."""
        # Suppress docling output during processing
        old_level = logging.getLogger().level
        logging.getLogger().setLevel(logging.ERROR)
        
        try:
            result = self.converter.convert(file_path)
            raw_text = result.document.export_to_text()
            return raw_text
        finally:
            # Restore logging level
            logging.getLogger().setLevel(old_level)

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
                logger.error(f"❌ {Path(file_path).name}: {str(e)}")
                errors.append((file_path, str(e)))
                continue
                
        if errors and not all_documents:
            error_msg = "\n".join([f"{Path(path).name}: {error}" for path, error in errors[:3]])
            raise Exception(f"Failed to process files:\n{error_msg}")
            
        if not all_documents:
            raise ValueError("No documents were successfully processed")
            
        logger.warning(f"Batch complete: {len(file_paths)} files → {len(all_documents)} chunks")
        return all_documents 