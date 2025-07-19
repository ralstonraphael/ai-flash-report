"""
Vector store module for document embeddings and retrieval using Pinecone.
"""
from pathlib import Path
from typing import List, Optional, Union, Dict, Any
import re
import uuid
import logging
import warnings
import os

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Filter deprecation warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

from langchain.schema import Document as LangChainDocument
from langchain_openai import OpenAIEmbeddings
from langchain.vectorstores import Pinecone as PineconeVectorStore
import pinecone
from pinecone import Pinecone, ServerlessSpec

from src.config import PINECONE_API_KEY, PINECONE_ENVIRONMENT, PINECONE_INDEX

class VectorStore:
    """Manages document embeddings and retrieval using Pinecone."""
    
    def __init__(self, embedding_function: Optional[OpenAIEmbeddings] = None):
        """
        Initialize vector store with Pinecone.
        
        Args:
            embedding_function: OpenAI embeddings instance (will create if None)
        """
        self.embedding_function = embedding_function or OpenAIEmbeddings()
        self.current_namespace = None
        
        # Initialize Pinecone
        logger.info("🌲 Initializing Pinecone client...")
        try:
            # Initialize Pinecone with API key
            self.pc = Pinecone(api_key=PINECONE_API_KEY)
            
            # Connect to existing index
            if not PINECONE_INDEX:
                raise ValueError("PINECONE_INDEX not configured")
            self.index = self.pc.Index(PINECONE_INDEX)
            
            # Test connection
            stats = self.index.describe_index_stats()
            logger.info(f"✅ Connected to Pinecone index '{PINECONE_INDEX}'")
            logger.info(f"📊 Index stats: {stats.total_vector_count} vectors across {len(stats.namespaces)} namespaces")
            
            # Create LangChain vector store wrapper
            self.vectorstore = PineconeVectorStore(
                index=self.index,
                embedding=self.embedding_function,
                text_key="text"
            )
            
        except Exception as e:
            logger.error(f"❌ Failed to initialize Pinecone: {str(e)}")
            raise RuntimeError(f"Pinecone initialization failed: {str(e)}")

    def _clean_namespace_name(self, name: str) -> str:
        """Clean namespace name to be compatible with Pinecone."""
        # Pinecone namespaces can contain alphanumeric characters and hyphens
        return re.sub(r'[^a-zA-Z0-9\-]', '-', name).lower()

    def create_collection(self, documents: List[LangChainDocument],
                        collection_name: str):
        """
        Create a new vector store collection (namespace in Pinecone) from documents.
        
        Args:
            documents: List of LangChain documents to add
            collection_name: Name for the collection (will be used as namespace)
            
        Returns:
            PineconeVectorStore instance
        """
        if not documents:
            raise ValueError("No documents provided")
        
        # Clean namespace name
        namespace = self._clean_namespace_name(collection_name)
        logger.info(f"🔄 Creating namespace '{namespace}' with {len(documents)} documents")
        
        try:
            # Create a new vectorstore instance for this namespace
            namespace_vectorstore = PineconeVectorStore(
                index=self.index,
                embedding=self.embedding_function,
                namespace=namespace,
                text_key="text"
            )
            
            # Add documents to the namespace
            logger.info(f"📤 Adding {len(documents)} documents to Pinecone namespace '{namespace}'...")
            
            # Prepare texts and metadatas
            texts = [doc.page_content for doc in documents]
            metadatas = [doc.metadata for doc in documents]
            
            # Add documents in batches (Pinecone recommends batch sizes of 100)
            batch_size = 100
            for i in range(0, len(texts), batch_size):
                batch_texts = texts[i:i + batch_size]
                batch_metadatas = metadatas[i:i + batch_size]
                
                namespace_vectorstore.add_texts(
                    texts=batch_texts,
                    metadatas=batch_metadatas
                )
                logger.info(f"✅ Added batch {i//batch_size + 1}/{(len(texts)-1)//batch_size + 1}")
            
            self.current_namespace = namespace
            self.vectorstore = namespace_vectorstore
            
            # Verify upload
            stats = self.index.describe_index_stats()
            namespace_count = stats.namespaces.get(namespace, {}).get('vector_count', 0)
            logger.info(f"✅ Successfully created namespace '{namespace}' with {namespace_count} vectors")
            
            return namespace_vectorstore
            
        except Exception as e:
            logger.error(f"❌ Failed to create collection: {str(e)}")
            raise

    def load_collection(self, collection_name: str):
        """
        Load an existing vector store collection (namespace in Pinecone).
        
        Args:
            collection_name: Name of collection to load
            
        Returns:
            PineconeVectorStore instance
        """
        namespace = self._clean_namespace_name(collection_name)
        
        try:
            # Check if namespace exists
            stats = self.index.describe_index_stats()
            if namespace not in stats.namespaces:
                raise ValueError(f"Namespace '{namespace}' not found in index")
            
            vector_count = stats.namespaces[namespace]['vector_count']
            logger.info(f"✅ Loading namespace '{namespace}' with {vector_count} vectors")
            
            # Create vectorstore for this namespace
            self.vectorstore = PineconeVectorStore(
                index=self.index,
                embedding=self.embedding_function,
                namespace=namespace,
                text_key="text"
            )
            self.current_namespace = namespace
            
            return self.vectorstore
            
        except Exception as e:
            logger.error(f"❌ Failed to load collection: {str(e)}")
            raise

    def query_collection(self, query: str, k: int = 5) -> List[str]:
        """
        Query using similarity search in Pinecone.
        
        Args:
            query: Search query string
            k: Number of results to return
            
        Returns:
            List of relevant text chunks
        """
        if not self.vectorstore or not self.current_namespace:
            raise ValueError("No collection loaded. Call create_collection() or load_collection() first.")
            
        logger.info(f"🔍 Querying namespace '{self.current_namespace}' with: {query}")
        
        try:
            # Perform similarity search
            results = self.vectorstore.similarity_search(query, k=k)
            
            # Extract text content from results
            documents = [doc.page_content for doc in results]
            
            logger.info(f"✅ Found {len(documents)} relevant documents")
            return documents
            
        except Exception as e:
            logger.error(f"❌ Query failed: {str(e)}")
            raise

    def get_mmr_retriever(self, k: int = 5, lambda_mult: float = 0.5):
        """
        Get a retriever using Maximal Marginal Relevance search.
        
        Args:
            k: Number of documents to retrieve
            lambda_mult: Trade-off between relevance and diversity (0 to 1)
            
        Returns:
            LangChain retriever object
        """
        if not self.vectorstore or not self.current_namespace:
            raise ValueError("No collection loaded. Call create_collection() or load_collection() first.")
            
        logger.info(f"🔧 Creating MMR retriever for namespace '{self.current_namespace}'")
        
        # Create MMR retriever
        retriever = self.vectorstore.as_retriever(
            search_type="mmr",
            search_kwargs={
                "k": k,
                "lambda_mult": lambda_mult
            }
        )
        
        return retriever
    
    def delete_collection(self, collection_name: str):
        """
        Delete a collection (namespace) from Pinecone.
        
        Args:
            collection_name: Name of collection to delete
        """
        namespace = self._clean_namespace_name(collection_name)
        
        try:
            logger.info(f"🗑️ Deleting namespace '{namespace}'...")
            
            # Delete all vectors in the namespace
            self.index.delete(delete_all=True, namespace=namespace)
            
            logger.info(f"✅ Successfully deleted namespace '{namespace}'")
            
            # Clear current namespace if it was the deleted one
            if self.current_namespace == namespace:
                self.current_namespace = None
                self.vectorstore = None
                
        except Exception as e:
            logger.error(f"❌ Failed to delete collection: {str(e)}")
            raise
    
    def list_collections(self) -> List[str]:
        """
        List all collections (namespaces) in the Pinecone index.
        
        Returns:
            List of collection names
        """
        try:
            stats = self.index.describe_index_stats()
            namespaces = list(stats.namespaces.keys())
            logger.info(f"📋 Found {len(namespaces)} namespaces in index")
            return namespaces
        except Exception as e:
            logger.error(f"❌ Failed to list collections: {str(e)}")
            raise 