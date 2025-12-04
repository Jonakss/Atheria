#!/bin/bash
set -e

# Cleanup previous
rm -f server.log client.log

# Start server in background WITH AUTOSTART
echo "Starting server with --autostart..."
python run_server.py --no-frontend --autostart > server.log 2>&1 &
SERVER_PID=$!

echo "Server PID: $SERVER_PID"
echo "Waiting 15 seconds for server to initialize..."
sleep 15

# Check if server is running
if ! kill -0 $SERVER_PID 2>/dev/null; then
    echo "Server failed to start or died early. Logs:"
    cat server.log
    exit 1
fi

echo "Server seems alive. Last 10 lines of log:"
tail -n 10 server.log

echo "Starting gRPC client test..."
# Run client for 15 seconds then kill it
timeout 15s python scripts/test_grpc_client.py > client.log 2>&1 || true

echo "Client finished. Checking logs..."
cat client.log

# Check if client received any frames
if grep -q "Frame:" client.log; then
    echo "SUCCESS: gRPC client received frames."
else
    echo "FAILURE: gRPC client did NOT receive frames."
    echo "Server logs (full):"
    cat server.log
    kill $SERVER_PID
    exit 1
fi

# Cleanup
echo "Stopping server..."
kill $SERVER_PID
