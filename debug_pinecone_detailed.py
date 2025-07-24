#!/usr/bin/env python3
"""
Advanced Pinecone debugging for serverless indices.
Tests API key permissions, quotas, and serverless-specific behavior.
"""
import os
import sys
from pathlib import Path
import uuid
import json

# Add the project root to the Python path
sys.path.insert(0, str(Path(__file__).resolve().parent))

from dotenv import load_dotenv
load_dotenv()

try:
    from pinecone import Pinecone
    from langchain_openai import OpenAIEmbeddings
    PINECONE_AVAILABLE = True
except ImportError:
    print("❌ Required packages not available")
    sys.exit(1)

def test_api_key_permissions():
    """Test if API key has proper permissions."""
    print("\n🔧 Testing API Key Permissions...")
    
    api_key = os.getenv("PINECONE_API_KEY")
    if not api_key:
        print("❌ PINECONE_API_KEY not found")
        return False
    
    try:
        pc = Pinecone(api_key=api_key)
        
        # Test: List indexes (requires read permission)
        try:
            indexes = pc.list_indexes()
            print(f"✅ Read permission: Can list {len(indexes)} indexes")
        except Exception as e:
            print(f"❌ Read permission failed: {e}")
            return False
        
        # Test: Get index stats (requires read permission)
        index_name = os.getenv("PINECONE_INDEX", "flash-report-index")
        try:
            index = pc.Index(index_name)
            stats = index.describe_index_stats()
            print(f"✅ Index access: {stats.total_vector_count} total vectors")
        except Exception as e:
            print(f"❌ Index access failed: {e}")
            return False
        
        return True
        
    except Exception as e:
        print(f"❌ API key test failed: {e}")
        return False

def test_quota_limits():
    """Check if we're hitting quota limits."""
    print("\n🔧 Testing Quota Limits...")
    
    try:
        pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
        index = pc.Index(os.getenv("PINECONE_INDEX", "flash-report-index"))
        
        stats = index.describe_index_stats()
        total_vectors = stats.total_vector_count
        
        print(f"📊 Current vector count: {total_vectors}")
        
        # Pinecone free tier typically allows 100k vectors
        if total_vectors > 95000:
            print("⚠️ Approaching free tier limit (100k vectors)")
        elif total_vectors > 100000:
            print("❌ Likely at free tier limit!")
            return False
        else:
            print("✅ Well within quota limits")
        
        return True
        
    except Exception as e:
        print(f"❌ Quota check failed: {e}")
        return False

def test_serverless_behavior():
    """Test serverless-specific behavior."""
    print("\n🔧 Testing Serverless Behavior...")
    
    try:
        pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
        index = pc.Index(os.getenv("PINECONE_INDEX", "flash-report-index"))
        
        # Generate test embedding
        embedding_func = OpenAIEmbeddings(
            model="text-embedding-3-small",
            dimensions=512
        )
        
        test_text = "Serverless test document for debugging."
        embedding = embedding_func.embed_query(test_text)
        
        # Create test namespace
        test_namespace = f"serverless-test-{uuid.uuid4().hex[:8]}"
        print(f"📝 Using test namespace: {test_namespace}")
        
        # Test 1: Single vector upsert
        print("📤 Test 1: Single vector upsert...")
        vector_id = f"test-{uuid.uuid4().hex[:8]}"
        
        upsert_response = index.upsert(  # type: ignore
            vectors=[{
                "id": vector_id,
                "values": embedding,
                "metadata": {"text": test_text, "test": "single"}
            }],
            namespace=test_namespace
        )
        
        print(f"📊 Upsert response: {upsert_response}")
        
        # Test 2: Immediate stats check
        immediate_stats = index.describe_index_stats()
        immediate_count = immediate_stats.namespaces.get(test_namespace, {}).get('vector_count', 0)
        print(f"📊 Immediate count: {immediate_count}")
        
        # Test 3: Wait and recheck (serverless might have different propagation)
        import time
        for wait_time in [1, 3, 5, 10]:
            print(f"⏳ Waiting {wait_time} seconds...")
            time.sleep(wait_time)
            
            stats = index.describe_index_stats()
            count = stats.namespaces.get(test_namespace, {}).get('vector_count', 0)
            print(f"📊 Count after {wait_time}s: {count}")
            
            if count > 0:
                print(f"✅ SUCCESS: Vector appeared after {wait_time} seconds!")
                
                # Test query to make sure it's really there
                query_response = index.query(  # type: ignore
                    vector=embedding,
                    top_k=1,
                    namespace=test_namespace,
                    include_metadata=True
                )
                
                if hasattr(query_response, 'matches') and query_response.matches:  # type: ignore
                    print("✅ Query successful - vector is retrievable")
                    
                    # Cleanup
                    index.delete(delete_all=True, namespace=test_namespace)
                    print("✅ Cleanup complete")
                    return True
                else:
                    print("⚠️ Vector exists but not queryable")
        
        print("❌ Vector never appeared after 19 seconds")
        
        # Test 4: Try different vector format
        print("\n🔧 Test 4: Alternative vector format...")
        alt_namespace = f"alt-test-{uuid.uuid4().hex[:8]}"
        
        # Use tuple format instead of dict
        try:
            alt_response = index.upsert(  # type: ignore
                vectors=[(f"alt-{uuid.uuid4().hex[:8]}", embedding, {"text": test_text})],
                namespace=alt_namespace
            )
            print(f"📊 Alternative format response: {alt_response}")
            
            time.sleep(5)
            alt_stats = index.describe_index_stats()
            alt_count = alt_stats.namespaces.get(alt_namespace, {}).get('vector_count', 0)
            print(f"📊 Alternative format count: {alt_count}")
            
            if alt_count > 0:
                print("✅ Alternative format worked!")
                index.delete(delete_all=True, namespace=alt_namespace)
                return True
            
        except Exception as e:
            print(f"❌ Alternative format failed: {e}")
        
        return False
        
    except Exception as e:
        print(f"❌ Serverless test failed: {e}")
        return False

def main():
    print("🚀 Advanced Pinecone Serverless Debug")
    print("=" * 60)
    
    # Test 1: API Key Permissions
    if not test_api_key_permissions():
        print("\n❌ CRITICAL: API key permission issues detected!")
        print("🔧 Solution: Generate a new API key with read/write permissions")
        return
    
    # Test 2: Quota Limits  
    if not test_quota_limits():
        print("\n❌ CRITICAL: Quota limit issues detected!")
        print("🔧 Solution: Upgrade Pinecone plan or clean up old vectors")
        return
    
    # Test 3: Serverless Behavior
    if not test_serverless_behavior():
        print("\n❌ CRITICAL: Serverless upsert issues detected!")
        print("🔧 Possible solutions:")
        print("  1. Switch to pod-based index")
        print("  2. Use different vector format")
        print("  3. Add longer wait times for propagation")
        print("  4. Contact Pinecone support about serverless issues")
    else:
        print("\n✅ SUCCESS: All tests passed!")
        print("🎉 Your Pinecone setup should be working correctly")

if __name__ == "__main__":
    main() 