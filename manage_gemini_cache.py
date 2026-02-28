"""
Utility script to manage Gemini embeddings cache
"""

import os
import pickle
import json
from datetime import datetime


from pathlib import Path

# Set BASE_DIR relative to this file's location (good for GitHub/Deployment)
BASE_DIR = Path(__file__).parent.absolute()

def get_cache_info():
    """Get information about the embeddings cache"""
    cache_file = os.path.join(BASE_DIR, "gemini_embeddings.pkl")
    
    if not os.path.exists(cache_file):
        return {
            "exists": False,
            "message": "Cache file does not exist"
        }
    
    try:
        # Get file stats
        stats = os.stat(cache_file)
        size_mb = stats.st_size / (1024 * 1024)
        modified = datetime.fromtimestamp(stats.st_mtime)
        
        # Load and check contents
        with open(cache_file, 'rb') as f:
            embeddings = pickle.load(f)
        
        return {
            "exists": True,
            "path": os.path.abspath(cache_file),
            "size_mb": size_mb,
            "modified": modified.strftime("%Y-%m-%d %H:%M:%S"),
            "num_embeddings": len(embeddings),
            "embedding_dim": embeddings.shape[1] if len(embeddings.shape) > 1 else 0
        }
    except Exception as e:
        return {
            "exists": True,
            "error": str(e)
        }


def get_metadata_info():
    """Get information about the chunk metadata"""
    metadata_file = os.path.join(BASE_DIR, "cpc_metadata.json")
    
    if not os.path.exists(metadata_file):
        return {
            "exists": False,
            "message": "Metadata file does not exist"
        }
    
    try:
        with open(metadata_file, 'r', encoding='utf-8') as f:
            chunks = json.load(f)
        
        stats = os.stat(metadata_file)
        size_mb = stats.st_size / (1024 * 1024)
        
        # Calculate some statistics
        chunk_lengths = [len(chunk) for chunk in chunks]
        avg_length = sum(chunk_lengths) / len(chunk_lengths) if chunk_lengths else 0
        
        return {
            "exists": True,
            "num_chunks": len(chunks),
            "size_mb": size_mb,
            "avg_chunk_length": int(avg_length),
            "min_chunk_length": min(chunk_lengths) if chunk_lengths else 0,
            "max_chunk_length": max(chunk_lengths) if chunk_lengths else 0
        }
    except Exception as e:
        return {
            "exists": True,
            "error": str(e)
        }


def delete_cache():
    """Delete the embeddings cache file"""
    cache_file = os.path.join(BASE_DIR, "gemini_embeddings.pkl")
    
    if not os.path.exists(cache_file):
        print("❌ Cache file does not exist")
        return False
    
    try:
        os.remove(cache_file)
        print("✓ Cache file deleted successfully")
        print(f"  Deleted: {cache_file}")
        return True
    except Exception as e:
        print(f"✗ Error deleting cache: {e}")
        return False


def verify_cache_consistency():
    """Verify that cache matches metadata"""
    cache_info = get_cache_info()
    metadata_info = get_metadata_info()
    
    if not cache_info["exists"]:
        return {"status": "no_cache", "message": "Cache does not exist"}
    
    if not metadata_info["exists"]:
        return {"status": "no_metadata", "message": "Metadata does not exist"}
    
    if "error" in cache_info:
        return {"status": "error", "message": f"Cache error: {cache_info['error']}"}
    
    if "error" in metadata_info:
        return {"status": "error", "message": f"Metadata error: {metadata_info['error']}"}
    
    # Check if counts match
    if cache_info["num_embeddings"] != metadata_info["num_chunks"]:
        return {
            "status": "mismatch",
            "message": f"Mismatch: {cache_info['num_embeddings']} embeddings vs {metadata_info['num_chunks']} chunks",
            "action": "Delete cache and regenerate"
        }
    
    return {
        "status": "ok",
        "message": "Cache and metadata are consistent"
    }


def print_status():
    """Print current status of cache and metadata"""
    print("="*80)
    print("GEMINI EMBEDDINGS CACHE STATUS")
    print("="*80)
    
    # Cache info
    print("\n[Cache File]")
    print("-"*80)
    cache_info = get_cache_info()
    
    if not cache_info["exists"]:
        print("Status: ❌ Not Found")
        print("Action: Run test_gemini_rag.py to generate embeddings")
    elif "error" in cache_info:
        print(f"Status: ⚠️  Error")
        print(f"Error: {cache_info['error']}")
    else:
        print("Status: ✓ Found")
        print(f"Path: {cache_info['path']}")
        print(f"Size: {cache_info['size_mb']:.2f} MB")
        print(f"Modified: {cache_info['modified']}")
        print(f"Embeddings: {cache_info['num_embeddings']}")
        print(f"Dimensions: {cache_info['embedding_dim']}")
    
    # Metadata info
    print("\n[Chunk Metadata]")
    print("-"*80)
    metadata_info = get_metadata_info()
    
    if not metadata_info["exists"]:
        print("Status: ❌ Not Found")
        print("Action: Run embedding_comparison.py to generate chunks")
    elif "error" in metadata_info:
        print(f"Status: ⚠️  Error")
        print(f"Error: {metadata_info['error']}")
    else:
        print("Status: ✓ Found")
        print(f"Chunks: {metadata_info['num_chunks']}")
        print(f"Size: {metadata_info['size_mb']:.2f} MB")
        print(f"Avg chunk length: {metadata_info['avg_chunk_length']} chars")
        print(f"Range: {metadata_info['min_chunk_length']} - {metadata_info['max_chunk_length']} chars")
    
    # Consistency check
    print("\n[Consistency Check]")
    print("-"*80)
    consistency = verify_cache_consistency()
    
    if consistency["status"] == "ok":
        print("Status: ✓ Consistent")
        print(consistency["message"])
    elif consistency["status"] == "mismatch":
        print("Status: ⚠️  Mismatch")
        print(consistency["message"])
        print(f"Action: {consistency['action']}")
    else:
        print(f"Status: ⚠️  {consistency['status']}")
        print(consistency["message"])
    
    print("\n" + "="*80)


def main():
    """Main menu"""
    print("\n" + "="*80)
    print("GEMINI CACHE MANAGEMENT UTILITY")
    print("="*80)
    
    while True:
        print("\nOptions:")
        print("  1. Show cache status")
        print("  2. Delete cache (force regeneration)")
        print("  3. Verify cache consistency")
        print("  4. Exit")
        
        choice = input("\nSelect option (1-4): ").strip()
        
        if choice == "1":
            print_status()
        
        elif choice == "2":
            print("\n⚠️  WARNING: This will delete the embeddings cache.")
            print("Embeddings will be regenerated on next run (takes time).")
            confirm = input("\nAre you sure? (yes/no): ").strip().lower()
            
            if confirm == "yes":
                delete_cache()
            else:
                print("Cancelled.")
        
        elif choice == "3":
            print("\nChecking consistency...")
            consistency = verify_cache_consistency()
            print(f"\nStatus: {consistency['status']}")
            print(f"Message: {consistency['message']}")
            if "action" in consistency:
                print(f"Action: {consistency['action']}")
        
        elif choice == "4":
            print("\nExiting...")
            break
        
        else:
            print("\n❌ Invalid option")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nInterrupted by user")
    except Exception as e:
        print(f"\n✗ Error: {e}")
