import asyncio
import websockets
import json
import logging
import sys

# Configurar logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

async def test_simulation_flow():
    uri = "ws://localhost:8000/ws"
    
    try:
        async with websockets.connect(uri) as websocket:
            logging.info("✅ Conectado al WebSocket")
            
            # 1. Esperar mensaje de bienvenida/estado inicial
            initial_msg = await websocket.recv()
            logging.info(f"📩 Mensaje inicial recibido: {len(initial_msg)} bytes")
            
            # 2. Cargar experimento (Forzando motor Python para evitar crash nativo)
            load_cmd = {
                "scope": "inference",
                "command": "load_experiment",
                "args": {
                    "experiment_name": "MLP-d4-h16-g16-lr1e-4",
                    "force_engine": "python" 
                }
            }
            
            logging.info(f"📤 Enviando comando de carga: {load_cmd}")
            await websocket.send(json.dumps(load_cmd))
            
            # 3. Esperar confirmación de carga
            # Podríamos recibir varios mensajes antes de la confirmación
            experiment_loaded = False
            for _ in range(20):
                msg = await websocket.recv()
                data = json.loads(msg)
                logging.info(f"📩 Recibido: {data.get('type')}")
                
                if data.get('type') == 'notification' and data.get('payload', {}).get('type') == 'success':
                    if "cargado exitosamente" in data.get('payload', {}).get('message', ''):
                        logging.info("✅ Experimento cargado exitosamente")
                        experiment_loaded = True
                        break
            
            if not experiment_loaded:
                logging.error("❌ No se recibió confirmación de carga del experimento")
                return

            # 4. Iniciar simulación
            play_cmd = {
                "scope": "inference",
                "command": "play",
                "args": {}
            }
            logging.info(f"📤 Enviando comando play: {play_cmd}")
            await websocket.send(json.dumps(play_cmd))
            
            # 5. Verificar recepción de frames con datos
            frames_received = 0
            valid_frames = 0
            
            # Escuchar por 5 segundos
            try:
                while frames_received < 10:
                    msg = await asyncio.wait_for(websocket.recv(), timeout=2.0)
                    data = json.loads(msg)
                    
                    if data.get('type') == 'simulation_frame':
                        frames_received += 1
                        payload = data.get('payload', {})
                        map_data = payload.get('map_data')
                        
                        if map_data and len(map_data) > 0:
                            valid_frames += 1
                            logging.info(f"✅ Frame {frames_received}: map_data válido (len={len(map_data)})")
                        else:
                            logging.warning(f"⚠️ Frame {frames_received}: map_data VACÍO o NULO")
                            
            except asyncio.TimeoutError:
                logging.info("⏱️ Timeout esperando frames")
            
            logging.info(f"📊 Resumen: Frames recibidos: {frames_received}, Frames válidos: {valid_frames}")
            
            if valid_frames > 0:
                logging.info("✅ TEST EXITOSO: Se reciben datos de simulación")
            else:
                logging.error("❌ TEST FALLIDO: No se recibieron datos válidos")

    except Exception as e:
        logging.error(f"❌ Error en el test: {e}")

if __name__ == "__main__":
    asyncio.run(test_simulation_flow())
