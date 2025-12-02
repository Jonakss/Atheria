import asyncio
import logging
import json
from typing import Dict, Any, Set
from aiohttp import web, WSMsgType
from .base_service import BaseService
from ..server.server_state import g_state
from ..pipelines.handlers import HANDLERS
from ..utils import get_experiment_list
from ..server.data_serialization import should_use_binary, serialize_frame_binary

class WebSocketService(BaseService):
    """
    Servicio responsable de la comunicación WebSocket.
    Maneja conexiones, recibe comandos y transmite actualizaciones.
    """
    
    def __init__(self, broadcast_queue: asyncio.Queue):
        super().__init__("WebSocket")
        self.broadcast_queue = broadcast_queue
        self.active_websockets: Set[web.WebSocketResponse] = set()
        
    async def _start_impl(self):
        """Inicia el bucle de transmisión."""
        self._task = asyncio.create_task(self._broadcast_loop())
        # Iniciar loop de streaming si el buffering está habilitado
        from src.config import CACHE_BUFFERING_ENABLED
        if CACHE_BUFFERING_ENABLED:
            self._stream_task = asyncio.create_task(self._stream_loop())
        
    async def _stop_impl(self):
        """Detiene el servicio y cierra conexiones."""
        for ws in list(self.active_websockets):
            await ws.close(code=1001, message=b'Server shutting down')
        self.active_websockets.clear()
        
    async def handle_connection(self, request):
        """Maneja una nueva conexión WebSocket."""
        ws = web.WebSocketResponse()
        await ws.prepare(request)
        
        ws_id = id(ws)
        self.active_websockets.add(ws)
        
        # Registrar en g_state para compatibilidad con handlers existentes
        if 'websockets' not in g_state:
            g_state['websockets'] = {}
        g_state['websockets'][ws_id] = ws
        
        logging.info(f"🔌 Nueva conexión WebSocket: {ws_id}")
        
        try:
            # Enviar estado inicial
            await self._send_initial_state(ws)
            
            async for msg in ws:
                if msg.type == WSMsgType.TEXT:
                    await self._handle_message(ws, msg.data, ws_id)
                elif msg.type == WSMsgType.ERROR:
                    logging.error(f"❌ Error en WebSocket {ws_id}: {ws.exception()}")
        finally:
            self.active_websockets.discard(ws)
            if ws_id in g_state['websockets']:
                del g_state['websockets'][ws_id]
            logging.info(f"🔌 Conexión cerrada: {ws_id}")
            
        return ws
        
    async def _handle_message(self, ws, data, ws_id):
        """Procesa un mensaje entrante."""
        try:
            command_data = json.loads(data)
            command_type = command_data.get('type')
            payload = command_data.get('payload', {})
            
            # Soporte para formato legacy/frontend: { scope, command, args }
            if command_type is None:
                scope = command_data.get('scope')
                command = command_data.get('command')
                args = command_data.get('args', {})
                
                if scope and command:
                    command_type = f"{scope}.{command}"
                    payload = args
            
            # Soporte para formato legacy/frontend: { scope, command, args }
            if command_type is None:
                scope = command_data.get('scope')
                command = command_data.get('command')
                args = command_data.get('args', {})
                
                if scope and command:
                    command_type = f"{scope}.{command}"
                    payload = args
            
            # Agregar ID de WS al payload para que los handlers sepan a quién responder
            payload['ws_id'] = ws_id
            
            # Enrutamiento de comandos (usando el sistema de handlers existente)
            # Formato esperado: "scope.action" (ej: "inference.play")
            if '.' in command_type:
                scope, action = command_type.split('.', 1)
                if scope in HANDLERS and action in HANDLERS[scope]:
                    handler = HANDLERS[scope][action]
                    print(f"DEBUG: Ejecutando handler para {scope}.{action}")
                    # Ejecutar handler (puede ser async)
                    if asyncio.iscoroutinefunction(handler):
                        await handler(payload)
                    else:
                        handler(payload)
                else:
                    logging.warning(f"⚠️ Comando desconocido: {command_type}")
            else:
                logging.warning(f"⚠️ Formato de comando inválido: {command_type}")
                
        except json.JSONDecodeError:
            logging.error("❌ Error decodificando JSON del mensaje WebSocket")
        except Exception as e:
            logging.error(f"❌ Error procesando mensaje: {e}")
            
    async def _send_initial_state(self, ws):
        """Envía el estado inicial al cliente."""
        # Esto replica la lógica de send_initial_state en pipeline_server.py
        # pero simplificada.
        initial_payload = {
            "experiments": get_experiment_list(), # Cargar lista real de experimentos
            "training_status": g_state.get('training_status', 'idle'),
            "inference_status": "paused" if g_state.get('is_paused', True) else "running",
            "compile_status": g_state.get('compile_status', {'is_compiled': False}),
            "active_experiment": g_state.get('active_experiment')
        }
        await ws.send_json({"type": "initial_state", "payload": initial_payload})

    async def _broadcast_loop(self):
        """Bucle que consume de la cola y transmite a todos los clientes."""
        logging.info("📡 WebSocketService: Bucle de transmisión iniciado.")
        
        while self._is_running:
            try:
                # Obtener mensaje de la cola
                message = await self.broadcast_queue.get()
                
                if not self.active_websockets:
                    self.broadcast_queue.task_done()
                    continue
                
                # Determinar si usar binario o JSON
                use_binary = should_use_binary(message.get('type'), message.get('payload'))
                
                if use_binary:
                    # Serializar a binario (o JSON fallback optimizado)
                    binary_data, format_used = serialize_frame_binary(message.get('payload', {}))
                    
                    # Enviar a todos los clientes
                    tasks = []
                    for ws in self.active_websockets:
                        if not ws.closed:
                            # Enviar primero el header/metadata si es necesario, o confiar en que el cliente
                            # sabe manejar el binario.
                            # En este caso, el protocolo espera que los frames binarios sean solo el payload.
                            # Pero necesitamos decirle al cliente qué es.
                            # El cliente actual (frontend) parece esperar un mensaje JSON con "type" y "payload".
                            # Si enviamos binario puro, ¿cómo sabe el cliente qué es?
                            
                            # Revisando server_state.py broadcast_binary:
                            # Envía primero un JSON con metadata y LUEGO el binario.
                            # Pero aquí estamos reemplazando el mensaje completo.
                            
                            # Si usamos el protocolo existente de broadcast_binary:
                            # 1. Enviar header JSON
                            header = {
                                "type": message.get("type"),
                                "format": format_used,
                                "is_binary": True
                            }
                            tasks.append(ws.send_json(header))
                            
                            # 2. Enviar payload binario
                            tasks.append(ws.send_bytes(binary_data))
                else:
                    # Comportamiento normal JSON
                    json_str = json.dumps(message)
                    tasks = []
                    for ws in self.active_websockets:
                        if not ws.closed:
                            tasks.append(ws.send_str(json_str))
                
                if tasks:
                    await asyncio.gather(*tasks, return_exceptions=True)
                
                self.broadcast_queue.task_done()
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logging.error(f"❌ Error en bucle de transmisión: {e}")
                await asyncio.sleep(0.1)

    async def _stream_loop(self):
        """
        Bucle de streaming que consume del buffer (cache) a una tasa constante.
        Esto desacopla la simulación de la visualización.
        """
        from src.config import CACHE_STREAM_KEY, CACHE_BUFFERING_ENABLED
        from src.server.server_state import g_state
        import pickle
        import zstandard as zstd
        import time
        
        logging.info("🌊 WebSocketService: Bucle de streaming iniciado.")
        
        # Compresor para descomprimir
        decompressor = zstd.ZstdDecompressor()
        
        # Rate limiting
        target_fps = g_state.get('target_fps', 30.0)
        frame_time = 1.0 / target_fps
        
        while self._is_running:
            try:
                start_time = time.time()
                
                # Verificar si cache está disponible
                if not g_state.get('cache') or not g_state['cache'].is_enabled():
                    await asyncio.sleep(1.0)
                    continue
                
                cache_client = g_state['cache'].client
                if not cache_client:
                    await asyncio.sleep(1.0)
                    continue
                
                # Intentar obtener frame del buffer (LPOP)
                # LPOP saca el elemento más antiguo (FIFO queue)
                compressed = cache_client.lpop(CACHE_STREAM_KEY)
                
                if compressed:
                    # Descomprimir y deserializar
                    try:
                        serialized = decompressor.decompress(compressed)
                        payload = pickle.loads(serialized)
                        
                        # Construir mensaje
                        message = {
                            'type': 'simulation_frame',
                            'payload': payload
                        }
                        
                        # Enviar a través de la lógica de broadcast existente
                        # pero sin pasar por la cola broadcast_queue para evitar duplicados
                        # si el productor también enviara a la cola (que ya no lo hace si buffering=True)
                        
                        # Reutilizamos la lógica de envío directo
                        await self._broadcast_message(message)
                        
                    except Exception as e:
                        logging.error(f"❌ Error procesando frame de stream: {e}")
                else:
                    # Buffer vacío, esperar un poco
                    await asyncio.sleep(0.01)
                    continue
                
                # Control de FPS (Rate Limiting)
                # Actualizar target_fps dinámicamente
                target_fps = g_state.get('target_fps', 30.0)
                frame_time = 1.0 / max(target_fps, 1.0)
                
                elapsed = time.time() - start_time
                wait_time = max(0, frame_time - elapsed)
                
                if wait_time > 0:
                    await asyncio.sleep(wait_time)
                else:
                    await asyncio.sleep(0)
                    
            except asyncio.CancelledError:
                break
            except Exception as e:
                logging.error(f"❌ Error en bucle de streaming: {e}")
                await asyncio.sleep(1.0)

    async def _broadcast_message(self, message):
        """Helper para enviar un mensaje a todos los clientes (lógica extraída de _broadcast_loop)."""
        if not self.active_websockets:
            return

        use_binary = should_use_binary(message.get('type'), message.get('payload'))
        
        if use_binary:
            binary_data, format_used = serialize_frame_binary(message.get('payload', {}))
            tasks = []
            for ws in self.active_websockets:
                if not ws.closed:
                    header = {
                        "type": message.get("type"),
                        "format": format_used,
                        "is_binary": True
                    }
                    tasks.append(ws.send_json(header))
                    tasks.append(ws.send_bytes(binary_data))
        else:
            json_str = json.dumps(message)
            tasks = []
            for ws in self.active_websockets:
                if not ws.closed:
                    tasks.append(ws.send_str(json_str))
        
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)
