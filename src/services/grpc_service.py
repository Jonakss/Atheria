import asyncio
import logging
import grpc
import time
from typing import Optional
from concurrent import futures

from .base_service import BaseService
from protos import simulation_pb2, simulation_pb2_grpc

class GRPCService(BaseService, simulation_pb2_grpc.SimulationServiceServicer):
    """
    Servicio gRPC para streaming de datos de simulación.
    Implementa el protocolo definido en simulation.proto.
    """

    def __init__(self, data_queue: asyncio.Queue, port: int = 50051):
        # BaseService init
        BaseService.__init__(self, "GRPCService")
        # gRPC Servicer init
        self.data_queue = data_queue
        self.port = port
        self.server = None

    async def _start_impl(self):
        """Inicia el servidor gRPC."""
        # Usamos aio (AsyncIO) de gRPC
        self.server = grpc.aio.server(futures.ThreadPoolExecutor(max_workers=10))
        simulation_pb2_grpc.add_SimulationServiceServicer_to_server(self, self.server)
        self.server.add_insecure_port(f'[::]:{self.port}')

        await self.server.start()
        logging.info(f"🚀 GRPCService: Escuchando en puerto {self.port}")

    async def _stop_impl(self):
        """Detiene el servidor gRPC."""
        if self.server:
            logging.info("🛑 GRPCService: Deteniendo servidor...")
            await self.server.stop(grace=5)

    async def StreamUniverseState(self, request, context):
        """
        Implementación del RPC StreamUniverseState.
        Envía estados del universo a medida que llegan a la cola.
        """
        client_id = id(context)
        logging.info(f"🔌 Cliente gRPC conectado: {client_id}")

        try:
            while self._is_running:
                # Esperar datos de la cola
                # Nota: data_queue debe ser poblada por DataProcessingService
                # Usamos wait_for para poder chequear cancelación regularmente
                try:
                    data = await asyncio.wait_for(self.data_queue.get(), timeout=1.0)
                except asyncio.TimeoutError:
                    if context.done():
                        break
                    continue

                # 'data' es lo que DataProcessingService puso en la cola
                # Esperamos que tenga 'payload' con 'viz_data' y 'simulation_info'
                # O si decidimos enviar raw bytes, lo adaptamos aquí.

                # Por ahora asumimos que DataProcessingService nos manda el dict con 'payload'
                payload = data.get('payload', {})
                sim_info = payload.get('simulation_info', {})
                viz_data = payload.get('viz_data')

                step = sim_info.get('step', 0)

                # Convertir viz_data a bytes si no lo es ya
                # viz_data puede ser un dict (si es 'particles') o bytes (si es imagen)
                # Para prueba de eficiencia, si es buffer, lo mandamos.

                data_bytes = b""
                if isinstance(viz_data, bytes):
                    data_bytes = viz_data
                elif isinstance(viz_data, dict):
                    # Si es JSON-like, lo serializamos simple para prueba
                    import json
                    data_bytes = json.dumps(viz_data).encode('utf-8')

                # Extraer métricas básicas
                metrics = {}
                if 'epoch_metrics' in sim_info and sim_info['epoch_metrics']:
                    # Aplanar métricas para el map<string, float>
                    for k, v in sim_info['epoch_metrics'].items():
                        if isinstance(v, (int, float)):
                            metrics[k] = float(v)

                response = simulation_pb2.UniverseState(
                    step=step,
                    data=data_bytes,
                    metrics=metrics
                )

                yield response

                # No olvidar marcar tarea como hecha si usamos queue
                self.data_queue.task_done()

        except asyncio.CancelledError:
            logging.info(f"🔌 Cliente gRPC desconectado (Cancelado): {client_id}")
        except Exception as e:
            logging.error(f"❌ Error en stream gRPC: {e}")
        finally:
            logging.info(f"🔌 Fin de stream gRPC para: {client_id}")
