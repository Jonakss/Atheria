
import asyncio
import logging
import grpc
import time
import json
import sys
import os

# Ensure project root is in path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from protos import simulation_pb2, simulation_pb2_grpc

# Configuración de logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

async def run_client():
    server_address = 'localhost:50051'
    logging.info(f"Conectando a {server_address}...")

    async with grpc.aio.insecure_channel(server_address) as channel:
        stub = simulation_pb2_grpc.SimulationServiceStub(channel)

        logging.info("Solicitando stream de datos (UniverseState)...")
        try:
            # Crear iterador asíncrono
            # Establecer un timeout de espera inicial
            stream = stub.StreamUniverseState(simulation_pb2.Empty())

            start_time = time.time()
            count = 0
            bytes_received = 0

            logging.info("Escuchando stream... (Ctrl+C para detener)")

            # Esperar el primer mensaje para confirmar conexión
            try:
                # Iterar sobre el stream
                async for state in stream:
                    count += 1
                    data_len = len(state.data)
                    bytes_received += data_len

                    # Calcular stats cada 30 frames
                    if count % 30 == 0:
                        elapsed = time.time() - start_time
                        if elapsed > 0:
                            fps = 30 / elapsed
                            mbps = (bytes_received / (1024 * 1024)) / elapsed
                        else:
                            fps = 0
                            mbps = 0

                        logging.info(f"Frame: {state.step} | Metrics: {len(state.metrics)} keys | "
                                     f"Data Size: {data_len/1024:.2f} KB | "
                                     f"FPS: {fps:.2f} | Throughput: {mbps:.2f} MB/s")

                        # Reset stats window
                        start_time = time.time()
                        # bytes_received = 0 # Keep cumulative or reset? Resetting is better for "Current Throughput"
                        bytes_received = 0

            except grpc.RpcError as e:
                logging.error(f"Error durante el stream: {e.code()} - {e.details()}")

        except asyncio.CancelledError:
            logging.info("Cliente detenido.")
        except grpc.RpcError as e:
            logging.error(f"Error al conectar gRPC: {e.code()} - {e.details()}")
        except Exception as e:
            logging.error(f"Error inesperado: {e}")

if __name__ == '__main__':
    try:
        asyncio.run(run_client())
    except KeyboardInterrupt:
        pass
