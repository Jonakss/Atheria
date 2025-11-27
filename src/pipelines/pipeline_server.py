# src/pipeline_server.py
import asyncio
import logging
import os
import shutil
import zipfile
import json
import io
from datetime import datetime
from aiohttp import web
from pathlib import Path

# Importar servicios
from ..services.simulation_service import SimulationService
from ..services.data_processing_service import DataProcessingService
from ..services.websocket_service import WebSocketService
from .. import config as global_cfg
from ..utils import save_experiment_config
from ..model_loader import MODEL_MAP

# Configuración de logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - [%(module)s] - %(message)s')

class ServiceManager:
    """
    Orquestador de servicios.
    Maneja el ciclo de vida de los servicios y sus dependencias.
    """
    def __init__(self):
        # Colas de comunicación
        self.state_queue = asyncio.Queue(maxsize=2) # Buffer pequeño para evitar latencia
        self.broadcast_queue = asyncio.Queue(maxsize=10)
        
        # Inicializar servicios
        self.simulation_service = SimulationService(self.state_queue)
        self.data_processing_service = DataProcessingService(self.state_queue, self.broadcast_queue)
        self.websocket_service = WebSocketService(self.broadcast_queue)
        
        self.services = [
            self.simulation_service,
            self.data_processing_service,
            self.websocket_service
        ]
        
    async def start_all(self):
        """Inicia todos los servicios."""
        logging.info("🚀 ServiceManager: Iniciando servicios...")
        for service in self.services:
            await service.start()
            
    async def stop_all(self):
        """Detiene todos los servicios."""
        logging.info("🛑 ServiceManager: Deteniendo servicios...")
        for service in reversed(self.services): # Detener en orden inverso
            await service.stop()

# Instancia global del manager
service_manager = ServiceManager()

async def websocket_handler(request):
    """Handler HTTP para endpoint WebSocket."""
    # Delegar al servicio de WebSocket
    return await service_manager.websocket_service.handle_connection(request)

async def on_startup(app):
    """Callback de inicio de aiohttp."""
    await service_manager.start_all()

async def on_cleanup(app):
    """Callback de cierre de aiohttp."""
    await service_manager.stop_all()

# --- Handlers de Upload/Export ---

async def _process_zip_upload(filename, file_data):
    """Procesa un archivo zip subido."""
    # Crear un directorio temporal para extraer
    temp_dir = os.path.join(global_cfg.OUTPUT_DIR, "temp_upload_" + datetime.now().strftime("%Y%m%d%H%M%S"))
    os.makedirs(temp_dir, exist_ok=True)

    try:
        with zipfile.ZipFile(io.BytesIO(file_data)) as zip_ref:
            zip_ref.extractall(temp_dir)

        # Validar estructura
        # Esperamos config.json en la raíz o en una subcarpeta
        config_path = None
        checkpoints_dir = None

        # Buscar config.json recursivamente
        for root, dirs, files in os.walk(temp_dir):
            if 'config.json' in files:
                config_path = os.path.join(root, 'config.json')
                # Asumir que los checkpoints están cerca
                if 'checkpoints' in dirs:
                    checkpoints_dir = os.path.join(root, 'checkpoints')
                elif 'training_checkpoints' in dirs: # Compatibilidad con estructura de repo
                    checkpoints_dir = os.path.join(root, 'training_checkpoints')
                else:
                    # Buscar .pth en el mismo directorio
                    pth_files = [f for f in files if f.endswith('.pth')]
                    if pth_files:
                        checkpoints_dir = root
                break

        if not config_path:
            raise ValueError("No se encontró config.json en el archivo zip")

        # Leer configuración para obtener nombre del experimento
        with open(config_path, 'r') as f:
            config_data = json.load(f)

        exp_name = config_data.get('EXPERIMENT_NAME') or config_data.get('experiment_name')
        if not exp_name:
            # Intentar deducir del nombre del archivo zip
            exp_name = os.path.splitext(filename)[0]
            config_data['EXPERIMENT_NAME'] = exp_name

        # Validar si ya existe
        target_exp_dir = os.path.join(global_cfg.EXPERIMENTS_DIR, exp_name)
        if os.path.exists(target_exp_dir):
            # Agregar sufijo timestamp para no sobrescribir
            timestamp = datetime.now().strftime("_%Y%m%d_%H%M%S")
            exp_name = f"{exp_name}{timestamp}"
            config_data['EXPERIMENT_NAME'] = exp_name
            target_exp_dir = os.path.join(global_cfg.EXPERIMENTS_DIR, exp_name)

        # Crear directorios finales
        target_checkpoints_dir = os.path.join(global_cfg.TRAINING_CHECKPOINTS_DIR, exp_name)
        os.makedirs(target_exp_dir, exist_ok=True)
        os.makedirs(target_checkpoints_dir, exist_ok=True)

        # Guardar configuración actualizada
        save_experiment_config(exp_name, config_data)

        # Mover checkpoints
        count = 0
        if checkpoints_dir and os.path.exists(checkpoints_dir):
            for f in os.listdir(checkpoints_dir):
                if f.endswith('.pth') or f.endswith('.pt'):
                    src = os.path.join(checkpoints_dir, f)
                    dst = os.path.join(target_checkpoints_dir, f)
                    shutil.copy2(src, dst)
                    count += 1

        logging.info(f"Experimento importado: {exp_name} con {count} checkpoints")
        return web.json_response({
            'success': True,
            'message': f"Experimento '{exp_name}' importado correctamente.",
            'experiment_name': exp_name
        })

    finally:
        # Limpiar temporal
        shutil.rmtree(temp_dir, ignore_errors=True)

async def _process_pth_upload(filename, file_data, form_fields):
    """Procesa un archivo .pth subido."""

    exp_name = form_fields.get('experiment_name')
    if not exp_name:
        # Usar nombre del archivo
        exp_name = "Imported_" + os.path.splitext(filename)[0]

    # Security check for experiment name
    if '..' in exp_name or '/' in exp_name or '\\' in exp_name:
        raise ValueError("Nombre de experimento inválido")

    # Validar si existe
    if os.path.exists(os.path.join(global_cfg.EXPERIMENTS_DIR, exp_name)):
        timestamp = datetime.now().strftime("_%Y%m%d_%H%M%S")
        exp_name = f"{exp_name}{timestamp}"

    # Construir configuración
    try:
        d_state = int(form_fields.get('d_state', 8))
        hidden_channels = int(form_fields.get('hidden_channels', 32))
        model_arch = form_fields.get('model_architecture', 'UNET')

        config_data = {
            "EXPERIMENT_NAME": exp_name,
            "MODEL_ARCHITECTURE": model_arch,
            "MODEL_PARAMS": {
                "d_state": d_state,
                "hidden_channels": hidden_channels
            },
            "created_at": datetime.now().isoformat(),
            "updated_at": datetime.now().isoformat(),
            "GRID_SIZE_TRAINING": 64, # Default seguro
            "QCA_STEPS_TRAINING": 16,
            "TOTAL_EPISODES": 0,
            "LR_RATE_M": 1e-4,
            "GAMMA_DECAY": 0.01,
            "INITIAL_STATE_MODE_INFERENCE": "complex_noise",
            "import_source": "upload_pth"
        }

        # Crear directorios
        target_exp_dir = os.path.join(global_cfg.EXPERIMENTS_DIR, exp_name)
        target_checkpoints_dir = os.path.join(global_cfg.TRAINING_CHECKPOINTS_DIR, exp_name)
        os.makedirs(target_exp_dir, exist_ok=True)
        os.makedirs(target_checkpoints_dir, exist_ok=True)

        # Guardar configuración
        save_experiment_config(exp_name, config_data)

        # Guardar archivo .pth
        target_path = os.path.join(target_checkpoints_dir, filename)

        with open(target_path, 'wb') as f:
            f.write(file_data)

        logging.info(f"Modelo importado: {exp_name}")
        return web.json_response({
            'success': True,
            'message': f"Modelo '{exp_name}' importado correctamente.",
            'experiment_name': exp_name
        })

    except Exception as e:
        logging.error(f"Error creando configuración para .pth: {e}")
        return web.json_response({'error': f"Error procesando configuración: {str(e)}"}, status=400)

async def handle_upload_model(request):
    """
    Maneja la carga de modelos (zip o pth).
    """
    reader = await request.multipart()

    # Datos del formulario
    file_data = None
    filename = None
    form_fields = {}

    while True:
        part = await reader.next()
        if part is None:
            break

        if part.name == 'file':
            filename = part.filename
            # Leer el contenido del archivo en memoria
            file_data = await part.read(decode=False)
        else:
            value = await part.text()
            form_fields[part.name] = value

    if not file_data or not filename:
        return web.json_response({'error': 'No se proporcionó ningún archivo'}, status=400)

    try:
        if filename.endswith('.zip'):
            return await _process_zip_upload(filename, file_data)
        elif filename.endswith('.pth') or filename.endswith('.pt'):
            return await _process_pth_upload(filename, file_data, form_fields)
        else:
            return web.json_response({'error': 'Formato de archivo no soportado. Use .zip o .pth'}, status=400)

    except Exception as e:
        logging.error(f"Error procesando carga de archivo: {e}", exc_info=True)
        return web.json_response({'error': str(e)}, status=500)

async def handle_export_experiment(request):
    """
    Exporta un experimento como archivo ZIP.
    """
    exp_name = request.query.get('name')
    if not exp_name:
        return web.json_response({'error': 'Nombre de experimento requerido'}, status=400)

    # Security check
    if '..' in exp_name or '/' in exp_name or '\\' in exp_name:
        return web.json_response({'error': 'Nombre de experimento inválido'}, status=400)

    exp_dir = os.path.join(global_cfg.EXPERIMENTS_DIR, exp_name)
    checkpoints_dir = os.path.join(global_cfg.TRAINING_CHECKPOINTS_DIR, exp_name)

    if not os.path.exists(exp_dir):
        return web.json_response({'error': 'Experimento no encontrado'}, status=404)

    try:
        # Usaremos archivo temporal para ser seguros con memoria
        temp_zip_path = os.path.join(global_cfg.OUTPUT_DIR, f"export_{exp_name}.zip")

        with zipfile.ZipFile(temp_zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
            # Agregar config.json en la raíz
            config_path = os.path.join(exp_dir, 'config.json')
            if os.path.exists(config_path):
                zipf.write(config_path, 'config.json')

            # Agregar checkpoints en carpeta 'checkpoints'
            if os.path.exists(checkpoints_dir):
                for root, dirs, files in os.walk(checkpoints_dir):
                    for file in files:
                        if file.endswith('.pth') or file.endswith('.pt'):
                            file_path = os.path.join(root, file)
                            arcname = os.path.join('checkpoints', file)
                            zipf.write(file_path, arcname)

        # Servir el archivo
        response = web.FileResponse(temp_zip_path)
        response.headers['Content-Disposition'] = f'attachment; filename="experiment_{exp_name}.zip"'

        return response

    except Exception as e:
        logging.error(f"Error exportando experimento: {e}", exc_info=True)
        return web.json_response({'error': str(e)}, status=500)

async def main(shutdown_event=None, serve_frontend=True):
    """Punto de entrada principal."""
    app = web.Application()
    
    # Configurar rutas
    app.router.add_get('/ws', websocket_handler)
    app.router.add_post('/api/upload_model', handle_upload_model)
    app.router.add_get('/api/export_experiment', handle_export_experiment)
    
    # Configurar callbacks
    app.on_startup.append(on_startup)
    app.on_cleanup.append(on_cleanup)
    
    # Servir frontend estático si existe y está habilitado
    if serve_frontend:
        frontend_path = Path(__file__).parent.parent.parent / 'frontend' / 'dist'
        if frontend_path.exists():
            app.router.add_static('/', frontend_path, show_index=True)
            logging.info(f"📂 Sirviendo frontend desde: {frontend_path}")
        else:
            logging.warning(f"⚠️ Frontend no encontrado en: {frontend_path}")
    
    # Iniciar servidor
    port = int(os.environ.get('PORT', 8000))
    logging.info(f"🌍 Servidor escuchando en http://0.0.0.0:{port}")
    
    # Crear runner
    runner = web.AppRunner(app)
    await runner.setup()
    site = web.TCPSite(runner, '0.0.0.0', port)
    await site.start()
    
    # Si hay shutdown_event, esperar a que se active
    if shutdown_event:
        await shutdown_event.wait()
        logging.info("🛑 Shutdown signal recibido, cerrando servidor...")
    else:
        # Si no hay shutdown_event, esperar indefinidamente
        try:
            await asyncio.Event().wait()
        except asyncio.CancelledError:
            logging.info("🛑 Servidor interrumpido...")
    
    await runner.cleanup()

if __name__ == '__main__':
    asyncio.run(main())
