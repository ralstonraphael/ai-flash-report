"""
Vector store module for document embeddings and retrieval using ChromaDB Cloud.
"""
from pathlib import Path
from typing import List, Optional, Union, Dict, Any
import re
import uuid
import logging
import warnings
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
    """Manages document embeddings and retrieval using ChromaDB Cloud ONLY."""
    
    def __init__(self, embedding_function: Optional[OpenAIEmbeddings] = None,
                 persist_dir: Union[str, Path] = VECTORSTORE_PATH):
        """
        Initialize vector store with ChromaDB Cloud client.
        
        Args:
            embedding_function: OpenAI embeddings instance (will create if None)
            persist_dir: Not used for Cloud client
        """
        self.embedding_function = embedding_function or OpenAIEmbeddings()
        self.current_collection = None
        
        # Configure ChromaDB Cloud client - NO FALLBACK
        logger.info("Initializing ChromaDB Cloud client...")
        logger.info("🌩️ USING CLOUD CLIENT ONLY - NO LOCAL SQLITE")
        
        try:
            self.client = chromadb.CloudClient(
                api_key='ck-GVHq1VVybwzVMkFdYMChCYkEATbBYqysk8t9oi4g48AR',
                tenant='06518e9c-e9ed-40d4-b45e-65c3aca2a6cd',
                database='Ai flash Report Generator',
                settings=Settings(
                    anonymized_telemetry=False,
                    allow_reset=True
                )
            )
            
            # Verify client type
            client_type = type(self.client)
            client_class = self.client.__class__.__name__
            
            logger.info("✅ Successfully initialized ChromaDB Cloud client")
            logger.info(f"🔍 CLIENT TYPE: {client_type}")
            logger.info(f"🏷️ CLIENT CLASS: {client_class}")
            
            # Verify it's actually a CloudClient
            if client_class != "CloudClient":
                raise ValueError(f"❌ Expected CloudClient, got {client_class}")
                
            # Test connection
            logger.info("🔗 Testing Cloud client connection...")
            try:
                # Try to list collections to verify connection
                collections = self.client.list_collections()
                logger.info(f"✅ Cloud connection verified. Found {len(collections)} existing collections")
            except Exception as conn_error:
                logger.error(f"❌ Cloud connection test failed: {str(conn_error)}")
                raise
                
        except Exception as e:
            logger.error(f"❌ CRITICAL: Failed to initialize ChromaDB Cloud client: {str(e)}")
            logger.error("🚫 NO FALLBACK - This app requires ChromaDB Cloud")
            raise RuntimeError(f"ChromaDB Cloud initialization failed: {str(e)}")

    def _clean_collection_name(self, name: str) -> str:
        """Clean collection name to be compatible with ChromaDB."""
        return re.sub(r'[^a-zA-Z0-9_\-]', '_', name)

    def create_collection(self, documents: List[LangChainDocument],
                        collection_name: str):
        """
        Create a new vector store collection from documents using native ChromaDB Cloud API.
        
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
        logger.info(f"🔄 Creating collection '{clean_name}' with {len(documents)} documents")
        
        # Verify we're still using CloudClient
        if self.client.__class__.__name__ != "CloudClient":
            raise RuntimeError(f"❌ Client type changed to {self.client.__class__.__name__}! Expected CloudClient")
        
        try:
            # Get or create collection using native Cloud API
            logger.info("🌩️ Creating collection using ChromaDB Cloud API...")
            collection = self.client.get_or_create_collection(name=clean_name)
            
            collection_type = type(collection)
            logger.info(f"✅ Collection created/retrieved: {collection_type}")
            
            # Prepare documents for ChromaDB
            texts = [doc.page_content for doc in documents]
            metadatas = [dict(doc.metadata) for doc in documents]  # Convert to dict
            ids = [str(uuid.uuid4()) for _ in documents]
            
            # Generate embeddings
            logger.info("🔮 Generating embeddings...")
            embeddings = self.embedding_function.embed_documents(texts)
            
            # Convert to numpy array for ChromaDB
            embeddings_array = np.array(embeddings)
            
            # Add documents to collection using Cloud API
            logger.info("📤 Adding documents to Cloud collection...")
            collection.add(
                embeddings=embeddings_array.tolist(),
                documents=texts,
                metadatas=metadatas,  # type: ignore
                ids=ids
            )
            
            self.current_collection = collection
            logger.info(f"✅ Successfully created Cloud collection '{clean_name}' with {len(documents)} documents")
            return collection
            
        except Exception as e:
            logger.error(f"❌ Failed to create Cloud collection: {str(e)}")
            logger.error("🔍 Error details:")
            logger.error(f"  - Client type: {type(self.client)}")
            logger.error(f"  - Client class: {self.client.__class__.__name__}")
            raise

    def load_collection(self, collection_name: str):
        """
        Load an existing vector store collection from ChromaDB Cloud.
        
        Args:
            collection_name: Name of collection to load
            
        Returns:
            ChromaDB collection instance
        """
        # Verify we're still using CloudClient
        if self.client.__class__.__name__ != "CloudClient":
            raise RuntimeError(f"❌ Client type changed to {self.client.__class__.__name__}! Expected CloudClient")
            
        collection = self.client.get_collection(name=self._clean_collection_name(collection_name))
        self.current_collection = collection
        logger.info(f"✅ Loaded Cloud collection: {collection_name}")
        return collection

    def query_collection(self, query: str, k: int = 5) -> List[str]:
        """
        Query using similarity search on ChromaDB Cloud.
        
        Args:
            query: Search query string
            k: Number of results to return
            
        Returns:
            List of relevant text chunks
        """
        if not self.current_collection:
            raise ValueError("No collection loaded. Call create_collection() or load_collection() first.")
            
        logger.info(f"🔍 Querying Cloud collection with: {query}")
        
        # Generate query embedding
        query_embedding = self.embedding_function.embed_query(query)
        
        # Query collection using Cloud API
        results = self.current_collection.query(
            query_embeddings=[query_embedding],
            n_results=k
        )
        
        # Extract documents from results
        documents_result = results.get('documents', [])
        documents = documents_result[0] if documents_result else []
        
        logger.info(f"✅ Found {len(documents)} relevant documents from Cloud")
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
            
        class ChromaCloudRetriever:
            def __init__(self, vectorstore, k, lambda_mult):
                self.vectorstore = vectorstore
                self.k = k
                self.lambda_mult = lambda_mult
            
            def get_relevant_documents(self, query):
                # Use Cloud-based similarity search
                documents = self.vectorstore.query_collection(query, self.k)
                return [type('Document', (), {'page_content': doc})() for doc in documents]
        
        return ChromaCloudRetriever(self, k, lambda_mult) 