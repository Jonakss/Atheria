"""
Benchmark script for gRPC streaming performance.
Connects to a running simulation server and measures throughput.

Usage:
    python -m src.analysis.benchmark_grpc --port 50051 --duration 10
"""
import asyncio
import logging
import time
import argparse
import grpc
from protos import simulation_pb2, simulation_pb2_grpc

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

async def run_benchmark(port, duration):
    """Runs the gRPC benchmark."""
    target = f'localhost:{port}'
    logging.info(f"🔌 Connecting to gRPC server at {target}...")

    # Increase max message size for large grids (e.g. 50MB)
    options = [('grpc.max_receive_message_length', 50 * 1024 * 1024)]

    async with grpc.aio.insecure_channel(target, options=options) as channel:
        stub = simulation_pb2_grpc.SimulationServiceStub(channel)

        logging.info("🚀 Starting stream...")
        start_time = time.time()
        frame_count = 0
        total_bytes = 0

        try:
            # Protocol uses Empty message for request
            request = simulation_pb2.Empty()

            async for state in stub.StreamUniverseState(request):
                frame_count += 1
                payload_size = len(state.data)
                total_bytes += payload_size

                step = state.step

                if frame_count % 10 == 0:
                    elapsed = time.time() - start_time
                    fps = frame_count / elapsed
                    mbps = (total_bytes / 1024 / 1024) / elapsed
                    logging.info(f"📊 Step {step}: {fps:.2f} FPS | {mbps:.2f} MB/s | Payload: {payload_size/1024:.2f} KB")

                if time.time() - start_time > duration:
                    break

        except asyncio.CancelledError:
            logging.info("Stream cancelled.")
        except grpc.RpcError as e:
            logging.error(f"❌ gRPC Error: {e.code()} - {e.details()}")
        except Exception as e:
            logging.error(f"❌ Error: {e}")

        elapsed = time.time() - start_time
        fps = frame_count / elapsed if elapsed > 0 else 0
        mbps = (total_bytes / 1024 / 1024) / elapsed if elapsed > 0 else 0

        print("\n" + "="*40)
        print("BENCHMARK RESULTS")
        print("="*40)
        print(f"Duration: {elapsed:.2f} s")
        print(f"Frames:   {frame_count}")
        print(f"Data:     {total_bytes / 1024 / 1024:.2f} MB")
        print(f"Avg FPS:  {fps:.2f}")
        print(f"Avg Rate: {mbps:.2f} MB/s")
        print("="*40)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Benchmark gRPC Streaming")
    parser.add_argument("--port", type=int, default=50051, help="gRPC server port")
    parser.add_argument("--duration", type=int, default=10, help="Benchmark duration in seconds")

    args = parser.parse_args()

    try:
        asyncio.run(run_benchmark(args.port, args.duration))
    except KeyboardInterrupt:
        pass
