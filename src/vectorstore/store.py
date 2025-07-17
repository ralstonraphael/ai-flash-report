"""
Vector store module for document embeddings and retrieval using ChromaDB.
"""
from pathlib import Path
from typing import List, Optional, Union
import re
import uuid
import logging
import warnings
import shutil
import sys
import os

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Filter deprecation warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

from langchain.schema import Document as LangChainDocument
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings
import chromadb
from chromadb.config import Settings

from src.config import VECTORSTORE_PATH

class VectorStore:
    """Manages document embeddings and retrieval using ChromaDB."""
    
    def __init__(self, embedding_function: Optional[OpenAIEmbeddings] = None,
                 persist_dir: Union[str, Path] = VECTORSTORE_PATH):
        """
        Initialize vector store with embedding function and persistence directory.
        
        Args:
            embedding_function: OpenAI embeddings instance (will create if None)
            persist_dir: Directory to persist ChromaDB files
        """
        self.embedding_function = embedding_function or OpenAIEmbeddings()
        self.persist_dir = str(persist_dir)
        self.current_collection = None
        
        # Configure ChromaDB with standard settings
        self.client = chromadb.Client(Settings(
            anonymized_telemetry=False,
            allow_reset=True
        ))

    def _clean_collection_name(self, name: str) -> str:
        """Clean collection name to be compatible with ChromaDB."""
        return re.sub(r'[^a-zA-Z0-9_\-]', '_', name)

    def create_collection(self, documents: List[LangChainDocument],
                        collection_name: str) -> Chroma:
        """
        Create a new vector store collection from documents.
        
        Args:
            documents: List of LangChain documents to add
            collection_name: Name for the collection
            
        Returns:
            Chroma vector store instance
        """
        if not documents:
            raise ValueError("No documents provided")
        
        # Clean collection name
        clean_name = self._clean_collection_name(collection_name)
        logger.info(f"Creating collection '{clean_name}' with {len(documents)} documents")
        
        try:
            # Create ChromaDB collection
            self.current_collection = Chroma.from_documents(
                documents=documents,
                embedding=self.embedding_function,
                client=self.client,
                collection_name=clean_name
            )
            
            logger.info(f"Successfully created collection '{clean_name}'")
            return self.current_collection
            
        except Exception as e:
            logger.error(f"Failed to create collection: {str(e)}")
            raise

    def load_collection(self, collection_name: str) -> Chroma:
        """
        Load an existing vector store collection.
        
        Args:
            collection_name: Name of collection to load
            
        Returns:
            ChromaDB collection instance
        """
        collection = Chroma(
            client=self.client,
            embedding_function=self.embedding_function,
            collection_name=self._clean_collection_name(collection_name)
        )
        self.current_collection = collection
        logger.info(f"Loaded collection: {collection_name}")
        return collection

    def _deduplicate_documents(self, documents: List[LangChainDocument]) -> List[LangChainDocument]:
        """
        Remove duplicate documents based on content hash.
        
        Args:
            documents: List of documents to deduplicate
            
        Returns:
            List of unique documents
        """
        unique_ids = set()
        unique_docs = []
        
        for doc in documents:
            doc_id = str(uuid.uuid5(uuid.NAMESPACE_DNS, doc.page_content))
            if doc_id not in unique_ids:
                unique_ids.add(doc_id)
                unique_docs.append(doc)
                
        return unique_docs

    def get_mmr_retriever(self, k: int = 5, lambda_mult: float = 0.5):
        """
        Get a retriever using Maximal Marginal Relevance search.
        
        Args:
            k: Number of documents to retrieve
            lambda_mult: Trade-off between relevance and diversity (0 to 1)
            
        Returns:
            MMR retriever instance
        """
        if not self.current_collection:
            raise ValueError("No collection loaded. Call create_collection() or load_collection() first.")
            
        return self.current_collection.as_retriever(
            search_type="mmr",
            search_kwargs={
                "k": k,
                "lambda_mult": lambda_mult,
                "fetch_k": k * 2  # Fetch more candidates for better diversity
            }
        )

    def query_collection(self, query: str, k: int = 5,
                        lambda_mult: float = 0.5) -> List[str]:
        """
        Query using MMR search.
        
        Args:
            query: Search query string
            k: Number of results to return
            lambda_mult: Trade-off between relevance and diversity
            
        Returns:
            List of relevant text chunks
        """
        if not self.current_collection:
            raise ValueError("No collection loaded. Call create_collection() or load_collection() first.")
            
        logger.info(f"Querying collection with: {query}")
        retriever = self.get_mmr_retriever(k=k, lambda_mult=lambda_mult)
        
        # Use get_relevant_documents() for older ChromaDB versions
        results = retriever.get_relevant_documents(query)
        
        logger.info(f"Found {len(results)} relevant documents")
        return [doc.page_content for doc in results] 