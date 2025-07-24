#!/usr/bin/env python3
"""
Debug script to test Pinecone connection and upsert functionality.
Run this to verify Pinecone is working independently of the main app.
"""
import os
import sys
from pathlib import Path
import uuid
import numpy as np

# Add the project root to the Python path
sys.path.insert(0, str(Path(__file__).resolve().parent))

from dotenv import load_dotenv
load_dotenv()

# Import after path setup
from langchain_openai import OpenAIEmbeddings

try:
    from pinecone import Pinecone
    PINECONE_AVAILABLE = True
except ImportError:
    print("❌ Pinecone not available. Install with: pip install pinecone-client==3.0.0")
    sys.exit(1)

def main():
    print("🔍 Pinecone Debug Test")
    print("=" * 50)
    
    # Check environment variables
    api_key = os.getenv("PINECONE_API_KEY")
    index_name = os.getenv("PINECONE_INDEX", "flash-report-index")
    
    if not api_key:
        print("❌ PINECONE_API_KEY not found in environment")
        return
    
    print(f"✅ API Key found: {api_key[:10]}...")
    print(f"✅ Index name: {index_name}")
    
    try:
        # Initialize Pinecone
        print("\n🔧 Initializing Pinecone client...")
        pc = Pinecone(api_key=api_key)
        
        # Connect to index
        print(f"🔧 Connecting to index '{index_name}'...")
        index = pc.Index(index_name)
        
        # Get index stats
        print("🔧 Getting index stats...")
        stats = index.describe_index_stats()
        print(f"📊 Index stats: {stats.total_vector_count} vectors across {len(stats.namespaces)} namespaces")
        print(f"📊 Namespaces: {list(stats.namespaces.keys())}")
        
        # Test embedding generation
        print("\n🔧 Testing embedding generation...")
        embedding_func = OpenAIEmbeddings(
            model="text-embedding-3-small",
            dimensions=512
        )
        
        test_text = "This is a test document for debugging Pinecone integration."
        embedding = embedding_func.embed_query(test_text)
        print(f"✅ Generated embedding with {len(embedding)} dimensions")
        
        # Test namespace creation and upsert
        print("\n🔧 Testing vector upsert...")
        test_namespace = f"debug-test-{uuid.uuid4().hex[:8]}"
        print(f"📝 Using test namespace: {test_namespace}")
        
        # Create test vector
        vector_id = f"test-{uuid.uuid4().hex[:8]}"
        test_vector = {
            "id": vector_id,
            "values": embedding,
            "metadata": {
                "text": test_text,
                "source": "debug_script",
                "namespace": test_namespace
            }
        }  # type: ignore
        
        # Upsert vector
        print("📤 Upserting test vector...")
        upsert_response = index.upsert(
            vectors=[test_vector],  # type: ignore
            namespace=test_namespace
        )
        print(f"✅ Upsert response: {upsert_response}")
        
        # Wait and check stats
        import time
        print("⏳ Waiting 3 seconds for async write...")
        time.sleep(3)
        
        updated_stats = index.describe_index_stats()
        namespace_count = updated_stats.namespaces.get(test_namespace, {}).get('vector_count', 0)
        print(f"📊 Vectors in test namespace: {namespace_count}")
        
        if namespace_count > 0:
            print("✅ SUCCESS: Vector was stored successfully!")
            
            # Test query
            print("\n🔧 Testing vector query...")
            query_response = index.query(  # type: ignore
                vector=embedding,
                top_k=1,
                namespace=test_namespace,
                include_metadata=True
            )
            
            if hasattr(query_response, 'matches') and query_response.matches:  # type: ignore
                match = query_response.matches[0]  # type: ignore
                print(f"✅ Query successful: Found match with score {match.score}")
                print(f"📝 Retrieved text: {match.metadata.get('text', 'No text found')}")
            else:
                print("⚠️ Query returned no matches")
            
            # Cleanup
            print(f"\n🧹 Cleaning up test namespace '{test_namespace}'...")
            index.delete(delete_all=True, namespace=test_namespace)
            print("✅ Cleanup complete")
            
        else:
            print("❌ FAILURE: Vector was not stored!")
            print("🔍 Possible causes:")
            print("  - Index dimension mismatch (should be 512)")
            print("  - API key lacks write permissions")
            print("  - Index is read-only or misconfigured")
            print("  - Quota/limit reached")
        
    except Exception as e:
        print(f"❌ Error: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main() 