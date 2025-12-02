import sys
import os
import time
import logging
import pickle
import zstandard as zstd
import redis

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def verify_buffering():
    logging.info("🚀 Starting Buffering Verification...")
    
    # Configuration
    CACHE_HOST = os.getenv('DRAGONFLY_HOST', 'localhost')
    CACHE_PORT = int(os.getenv('DRAGONFLY_PORT', '6379'))
    CACHE_STREAM_KEY = "test:simulation:stream"
    
    try:
        client = redis.Redis(host=CACHE_HOST, port=CACHE_PORT, decode_responses=False)
        client.ping()
        logging.info(f"✅ Connected to Dragonfly at {CACHE_HOST}:{CACHE_PORT}")
    except Exception as e:
        logging.error(f"❌ Could not connect to Dragonfly: {e}")
        return False

    # Clear previous test data
    client.delete(CACHE_STREAM_KEY)
    
    # 1. Simulate Producer
    logging.info("🏭 Simulating Producer...")
    compressor = zstd.ZstdCompressor(level=3)
    
    frames_to_produce = 10
    for i in range(frames_to_produce):
        payload = {"step": i, "data": f"frame_{i}"}
        serialized = pickle.dumps(payload)
        compressed = compressor.compress(serialized)
        
        client.rpush(CACHE_STREAM_KEY, compressed)
        logging.info(f"   -> Pushed frame {i}")
        
    # Verify list length
    length = client.llen(CACHE_STREAM_KEY)
    logging.info(f"📊 Buffer length: {length}")
    if length != frames_to_produce:
        logging.error(f"❌ Expected {frames_to_produce} frames, got {length}")
        return False
        
    # 2. Simulate Consumer
    logging.info("CONSUMER: Simulating Consumer (Reading from buffer)...")
    decompressor = zstd.ZstdDecompressor()
    
    frames_consumed = 0
    while True:
        compressed = client.lpop(CACHE_STREAM_KEY)
        if compressed:
            serialized = decompressor.decompress(compressed)
            payload = pickle.loads(serialized)
            logging.info(f"   <- Consumed frame {payload['step']}")
            frames_consumed += 1
        else:
            break
            
    if frames_consumed == frames_to_produce:
        logging.info(f"✅ Successfully consumed all {frames_consumed} frames.")
    else:
        logging.error(f"❌ Consumed {frames_consumed} frames, expected {frames_to_produce}")
        return False
        
    logging.info("🎉 Buffering verification PASSED!")
    return True

if __name__ == "__main__":
    if verify_buffering():
        sys.exit(0)
    else:
        sys.exit(1)
