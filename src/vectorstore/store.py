"""
Vector store module for document embeddings and retrieval using ChromaDB.
"""
# SQLite override for Streamlit Cloud compatibility
__import__('pysqlite3')
import sys
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')

from pathlib import Path
from typing import List, Optional, Union, Dict, Any
import re
import uuid
import logging
import warnings
import shutil
import sys
import os
import numpy as np

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Filter deprecation warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

from langchain.schema import Document as LangChainDocument
from langchain_openai import OpenAIEmbeddings
import chromadb
from chromadb.config import Settings

from src.config import VECTORSTORE_PATH

class VectorStore:
    """Manages document embeddings and retrieval using ChromaDB Cloud."""
    
    def __init__(self, embedding_function: Optional[OpenAIEmbeddings] = None,
                 persist_dir: Union[str, Path] = VECTORSTORE_PATH):
        """
        Initialize vector store with embedding function.
        
        Args:
            embedding_function: OpenAI embeddings instance (will create if None)
            persist_dir: Not used for Cloud client
        """
        self.embedding_function = embedding_function or OpenAIEmbeddings()
        self.current_collection = None
        
        # Configure ChromaDB Cloud client
        try:
            logger.info("Initializing ChromaDB Cloud client...")
            self.client = chromadb.CloudClient(
                api_key='ck-GVHq1VVybwzVMkFdYMChCYkEATbBYqysk8t9oi4g48AR',
                tenant='06518e9c-e9ed-40d4-b45e-65c3aca2a6cd',
                database='Ai flash Report Generator',
                settings=Settings(
                    anonymized_telemetry=False,
                    allow_reset=True
                )
            )
            logger.info("Successfully initialized ChromaDB Cloud client")
        except Exception as e:
            logger.error(f"Failed to initialize ChromaDB Cloud client: {str(e)}")
            raise

    def _clean_collection_name(self, name: str) -> str:
        """Clean collection name to be compatible with ChromaDB."""
        return re.sub(r'[^a-zA-Z0-9_\-]', '_', name)

    def create_collection(self, documents: List[LangChainDocument],
                        collection_name: str):
        """
        Create a new vector store collection from documents using native ChromaDB API.
        
        Args:
            documents: List of LangChain documents to add
            collection_name: Name for the collection
            
        Returns:
            ChromaDB collection instance
        """
        if not documents:
            raise ValueError("No documents provided")
        
        # Clean collection name
        clean_name = self._clean_collection_name(collection_name)
        logger.info(f"Creating collection '{clean_name}' with {len(documents)} documents")
        
        try:
            # Get or create collection using native API
            collection = self.client.get_or_create_collection(name=clean_name)
            
            # Prepare documents for ChromaDB
            texts = [doc.page_content for doc in documents]
            metadatas = [dict(doc.metadata) for doc in documents]  # Convert to dict
            ids = [str(uuid.uuid4()) for _ in documents]
            
            # Generate embeddings
            logger.info("Generating embeddings...")
            embeddings = self.embedding_function.embed_documents(texts)
            
            # Convert to numpy array for ChromaDB
            embeddings_array = np.array(embeddings)
            
            # Add documents to collection
            logger.info("Adding documents to collection...")
            collection.add(
                embeddings=embeddings_array.tolist(),
                documents=texts,
                metadatas=metadatas,  # type: ignore
                ids=ids
            )
            
            self.current_collection = collection
            logger.info(f"Successfully created collection '{clean_name}' with {len(documents)} documents")
            return collection
            
        except Exception as e:
            logger.error(f"Failed to create collection: {str(e)}")
            raise

    def load_collection(self, collection_name: str):
        """
        Load an existing vector store collection.
        
        Args:
            collection_name: Name of collection to load
            
        Returns:
            ChromaDB collection instance
        """
        collection = self.client.get_collection(name=self._clean_collection_name(collection_name))
        self.current_collection = collection
        logger.info(f"Loaded collection: {collection_name}")
        return collection

    def query_collection(self, query: str, k: int = 5) -> List[str]:
        """
        Query using similarity search.
        
        Args:
            query: Search query string
            k: Number of results to return
            
        Returns:
            List of relevant text chunks
        """
        if not self.current_collection:
            raise ValueError("No collection loaded. Call create_collection() or load_collection() first.")
            
        logger.info(f"Querying collection with: {query}")
        
        # Generate query embedding
        query_embedding = self.embedding_function.embed_query(query)
        
        # Query collection
        results = self.current_collection.query(
            query_embeddings=[query_embedding],
            n_results=k
        )
        
        # Extract documents from results
        documents_result = results.get('documents', [])
        documents = documents_result[0] if documents_result else []
        
        logger.info(f"Found {len(documents)} relevant documents")
        return documents

    def get_mmr_retriever(self, k: int = 5, lambda_mult: float = 0.5):
        """
        Get a retriever using Maximal Marginal Relevance search.
        Note: This is a simplified implementation since native ChromaDB doesn't have MMR.
        
        Args:
            k: Number of documents to retrieve
            lambda_mult: Trade-off between relevance and diversity (0 to 1)
            
        Returns:
            Custom retriever object
        """
        if not self.current_collection:
            raise ValueError("No collection loaded. Call create_collection() or load_collection() first.")
            
        class ChromaRetriever:
            def __init__(self, vectorstore, k, lambda_mult):
                self.vectorstore = vectorstore
                self.k = k
                self.lambda_mult = lambda_mult
            
            def get_relevant_documents(self, query):
                # For now, use regular similarity search
                # In a full implementation, you'd implement MMR here
                documents = self.vectorstore.query_collection(query, self.k)
                return [type('Document', (), {'page_content': doc})() for doc in documents]
        
        return ChromaRetriever(self, k, lambda_mult) 