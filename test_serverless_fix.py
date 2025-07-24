#!/usr/bin/env python3
"""
Test script to verify the serverless propagation fix works.
"""
import os
import sys
from pathlib import Path

# Add the project root to the Python path
sys.path.insert(0, str(Path(__file__).resolve().parent))

from dotenv import load_dotenv
load_dotenv()

from langchain.schema import Document as LangChainDocument
from src.vectorstore.store import VectorStore

def test_vectorstore_with_serverless():
    """Test the fixed vectorstore with serverless propagation handling."""
    
    print("🧪 Testing VectorStore with Serverless Fix")
    print("=" * 50)
    
    # Create test documents
    test_docs = [
        LangChainDocument(
            page_content="This is a test document about artificial intelligence and machine learning.",
            metadata={"source": "test1.txt", "type": "test"}
        ),
        LangChainDocument(
            page_content="Another test document discussing natural language processing and deep learning.",
            metadata={"source": "test2.txt", "type": "test"}
        )
    ]
    
    try:
        # Initialize vectorstore
        print("🔧 Initializing VectorStore...")
        vs = VectorStore()
        
        # Create collection
        collection_name = "serverless-test-fix"
        print(f"📤 Creating collection '{collection_name}' with {len(test_docs)} documents...")
        
        vectorstore = vs.create_collection(test_docs, collection_name)
        
        print("✅ Collection created successfully!")
        print("🔍 Testing query functionality...")
        
        # Test query
        results = vs.query_collection("artificial intelligence", k=2)
        
        if results:
            print(f"✅ Query successful: Found {len(results)} results")
            print(f"📝 First result preview: {results[0][:100]}...")
            
            # Cleanup
            print(f"🧹 Cleaning up collection '{collection_name}'...")
            vs.delete_collection(collection_name)
            print("✅ Cleanup complete")
            
            print("\n🎉 SUCCESS: Serverless fix is working!")
            return True
        else:
            print("❌ Query returned no results")
            return False
            
    except Exception as e:
        print(f"❌ Test failed: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_vectorstore_with_serverless()
    sys.exit(0 if success else 1) 