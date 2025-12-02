#!/usr/bin/env python3
"""
IonQ Connection Tester
======================

This script verifies your connection to the IonQ Quantum Cloud.
It checks:
1. If 'qiskit-ionq' is installed.
2. If 'IONQ_API_KEY' is set.
3. If we can authenticate and list available backends.

Usage:
    export IONQ_API_KEY="your_api_key"
    python3 scripts/test_ionq_connection.py
"""

import os
import sys
import logging

# Configure Logging
logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger()

def print_header(msg):
    print(f"\n{'='*60}")
    print(f" {msg}")
    print(f"{'='*60}")

def check_library():
    print_header("1. Checking Library Installation")
    try:
        import qiskit_ionq
        print(f"✅ qiskit-ionq is installed (Version: {qiskit_ionq.__version__})")
        return True
    except ImportError:
        print("❌ qiskit-ionq is NOT installed.")
        print("   -> Please run: pip install qiskit-ionq")
        return False
    except Exception as e:
        print(f"❌ Error checking library: {e}")
        return False

def check_api_key():
    print_header("2. Checking API Key")
    api_key = os.getenv('IONQ_API_KEY')
    if api_key:
        masked_key = api_key[:4] + "*" * (len(api_key) - 8) + api_key[-4:] if len(api_key) > 8 else "****"
        print(f"✅ IONQ_API_KEY is set: {masked_key}")
        return api_key
    else:
        print("❌ IONQ_API_KEY is NOT set.")
        print("   -> Export it in your terminal: export IONQ_API_KEY='your_key'")
        return None

def check_connection(api_key):
    print_header("3. Testing Connection to IonQ Cloud")
    try:
        from qiskit_ionq import IonQProvider
        provider = IonQProvider(token=api_key)
        
        print("⏳ Connecting to IonQ...")
        backends = provider.backends()
        
        print(f"✅ Connection Successful! Found {len(backends)} backends:")
        for backend in backends:
            try:
                # Handle backend.name (property vs method)
                b_name = backend.name() if callable(getattr(backend, 'name', None)) else backend.name
                
                try:
                    status = backend.status()
                except Exception as e:
                    print(f"   - {b_name} [⚠️ Error getting status: {e}]")
                    continue
                
                # Debugging
                # print(f"   [DEBUG] Backend: {b_name}, Status Type: {type(status)}, Value: {status}")
                
                is_operational = False
                pending = "Unknown"
                
                # Handle Qiskit BackendStatus object
                if hasattr(status, 'operational'):
                    is_operational = status.operational
                    if hasattr(status, 'pending_jobs'):
                        pending = status.pending_jobs
                # Handle simple boolean return (some providers)
                elif isinstance(status, bool):
                     is_operational = status
                     pending = "N/A"
                # Handle string status
                elif isinstance(status, str):
                    is_operational = (status.lower() == 'active' or status.lower() == 'online')
                    pending = "N/A"
                    
                status_str = "🟢 Online" if is_operational else "🔴 Offline"
                print(f"   - {b_name} [{status_str}] (Pending Jobs: {pending})")
            except Exception as e:
                print(f"   - [⚠️ Error processing backend: {e}]")
            
        return True
    except Exception as e:
        import traceback
        traceback.print_exc()
        print(f"❌ Connection Failed: {e}")
        print("   -> Check your API Key and internet connection.")
        return False

def main():
    print_header("🚀 IonQ Connection Tester")
    
    lib_ok = check_library()
    if not lib_ok:
        sys.exit(1)
        
    api_key = check_api_key()
    if not api_key:
        sys.exit(1)
        
    conn_ok = check_connection(api_key)
    if conn_ok:
        print_header("🎉 READY TO QUANTUM")
        print("You are all set! You can now run Atheria experiments on IonQ.")
        print("Try running: python3 scripts/experiment_ionq.py")
    else:
        sys.exit(1)

if __name__ == "__main__":
    main()
