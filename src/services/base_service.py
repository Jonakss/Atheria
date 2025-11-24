import asyncio
import logging
from abc import ABC, abstractmethod
from typing import Optional

class BaseService(ABC):
    """
    Clase base abstracta para todos los servicios de Atheria.
    Define la interfaz común para inicio, parada y estado.
    """
    
    def __init__(self, name: str):
        self.name = name
        self.logger = logging.getLogger(f"Service.{name}")
        self._is_running = False
        self._task: Optional[asyncio.Task] = None
    
    async def start(self):
        """Inicia el servicio."""
        if self._is_running:
            self.logger.warning(f"Servicio {self.name} ya está corriendo.")
            return
            
        self._is_running = True
        self.logger.info(f"🚀 Iniciando servicio {self.name}...")
        try:
            await self._start_impl()
            self.logger.info(f"✅ Servicio {self.name} iniciado correctamente.")
        except Exception as e:
            self.logger.error(f"❌ Error iniciando servicio {self.name}: {e}")
            self._is_running = False
            raise e

    async def stop(self):
        """Detiene el servicio."""
        if not self._is_running:
            return
            
        self.logger.info(f"🛑 Deteniendo servicio {self.name}...")
        self._is_running = False
        try:
            await self._stop_impl()
            if self._task and not self._task.done():
                self._task.cancel()
                try:
                    await self._task
                except asyncio.CancelledError:
                    pass
            self.logger.info(f"✅ Servicio {self.name} detenido.")
        except Exception as e:
            self.logger.error(f"❌ Error deteniendo servicio {self.name}: {e}")

    @abstractmethod
    async def _start_impl(self):
        """Implementación específica del inicio del servicio."""
        pass

    @abstractmethod
    async def _stop_impl(self):
        """Implementación específica de la parada del servicio."""
        pass

    @property
    def is_running(self) -> bool:
        return self._is_running
