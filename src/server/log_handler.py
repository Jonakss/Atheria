import logging
import asyncio
from .server_state import broadcast

class WebSocketLogHandler(logging.Handler):
    """
    Handler de logging que envía los logs a través de WebSockets.
    """
    def __init__(self):
        super().__init__()
        self.enabled = True
        # Formato por defecto
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        self.setFormatter(formatter)

    def emit(self, record):
        """
        Envía el registro de log a los clientes WebSocket.
        """
        if not self.enabled:
            return

        try:
            msg = self.format(record)
            
            # Determinar el tipo de log para el frontend
            log_type = "simulation_log"  # Default
            if record.levelno >= logging.ERROR:
                log_type = "error_log"
            elif "Entrenamiento" in msg:
                log_type = "training_log"
            
            # Preparar payload
            payload = {
                "type": log_type,
                "payload": msg
            }

            # Enviar de forma asíncrona (fire and forget)
            # Necesitamos obtener el loop actual o crear una tarea si hay uno corriendo
            try:
                loop = asyncio.get_running_loop()
                if loop.is_running():
                    loop.create_task(broadcast(payload))
            except RuntimeError:
                # No hay loop corriendo (ej: durante inicio/cierre), ignorar
                pass
                
        except Exception:
            self.handleError(record)
