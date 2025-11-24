# src/pipeline_server.py
import asyncio
import logging
import os
from aiohttp import web
from pathlib import Path

# Importar servicios
from ..services.simulation_service import SimulationService
from ..services.data_processing_service import DataProcessingService
from ..services.websocket_service import WebSocketService

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

async def main(shutdown_event=None, serve_frontend=True):
    """Punto de entrada principal."""
    app = web.Application()
    
    # Configurar rutas
    app.router.add_get('/ws', websocket_handler)
    
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
