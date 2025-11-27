
import sys
import os

# Add project root to path
sys.path.append('/home/jonathan.correa/Projects/Atheria')

try:
    from src.pipelines.handlers.inference_handlers import HANDLERS
    
    if "set_config" in HANDLERS:
        print("✅ set_config found in HANDLERS")
        if HANDLERS["set_config"].__name__ == "handle_set_inference_config":
             print("✅ set_config maps to handle_set_inference_config")
        else:
             print(f"❌ set_config maps to {HANDLERS['set_config'].__name__}")
             exit(1)
    else:
        print("❌ set_config NOT found in HANDLERS")
        exit(1)

except ImportError as e:
    print(f"❌ ImportError: {e}")
    exit(1)
except Exception as e:
    print(f"❌ Error: {e}")
    exit(1)
