import sys
import os
import time
import logging

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Configure logging to file and console
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("verify_cache.log"),
        logging.StreamHandler(sys.stdout)
    ]
)

print("🚀 Script started...")

try:
    from src.cache.dragonfly_client import cache
    print("✅ Import successful")
except ImportError as e:
    print(f"❌ Import failed: {e}")
    sys.exit(1)

def verify_cache():
    print("🚀 Starting Cache Verification Function...")
    logging.info("🚀 Starting Cache Verification...")
    
    # Check if enabled
    if not cache.is_enabled():
        msg = "❌ Cache is NOT enabled. Check configuration and connection."
        print(msg)
        logging.error(msg)
        return False
        
    msg = f"✅ Cache is enabled. Connected to {cache.host}:{cache.port}"
    print(msg)
    logging.info(msg)
    
    # Test SET
    test_key = "test:verification:key"
    test_value = {"message": "Hello Dragonfly!", "timestamp": time.time(), "data": [1, 2, 3]}
    
    print(f"📝 Testing SET operation for key '{test_key}'...")
    if cache.set(test_key, test_value, ttl=60):
        print("✅ SET successful.")
    else:
        print("❌ SET failed.")
        return False
        
    # Test GET
    print(f"🔍 Testing GET operation for key '{test_key}'...")
    retrieved_value = cache.get(test_key)
    
    if retrieved_value == test_value:
        print(f"✅ GET successful. Value matches.")
    else:
        print(f"❌ GET failed. Expected {test_value}, got {retrieved_value}")
        return False
        
    # Test DELETE
    print(f"🗑️ Testing DELETE operation for key '{test_key}'...")
    if cache.delete(test_key):
        print("✅ DELETE successful.")
    else:
        print("❌ DELETE failed.")
        return False
        
    # Verify Deletion
    if not cache.exists(test_key):
        print("✅ Verification successful: Key no longer exists.")
    else:
        print("❌ Verification failed: Key still exists after delete.")
        return False
        
    # Get Stats
    stats = cache.get_stats()
    print(f"📊 Cache Stats: {stats}")
    
    print("🎉 All cache verification tests PASSED!")
    return True

if __name__ == "__main__":
    if verify_cache():
        sys.exit(0)
    else:
        sys.exit(1)
