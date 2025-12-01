import logging
import os
from typing import Dict, List, Optional
from ..engines.compute_backend import ComputeBackend, LocalBackend, MockQuantumBackend

class ConnectionManager:
    """
    Manages connections to various compute backends (Local, Cloud QPU, Simulators).
    Acts as a registry and factory for backends, similar to Lightning AI's accelerator registry.
    """
    
    _instance = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(ConnectionManager, cls).__new__(cls)
            cls._instance._initialized = False
        return cls._instance
    
    def __init__(self):
        if self._initialized:
            return
            
        self.backends: Dict[str, ComputeBackend] = {}
        self.active_backend_id: str = "local"
        
        # Register default backends
        self.register_backend("local", LocalBackend())
        self.register_backend("quantum_mock", MockQuantumBackend())
        
        # Load API keys from env (placeholder)
        self.ionq_api_key = os.getenv("IONQ_API_KEY")
        
        self._initialized = True
        logging.info("🔌 ConnectionManager initialized")

    def register_backend(self, name: str, backend: ComputeBackend):
        """Registers a new backend instance."""
        self.backends[name] = backend
        logging.info(f"🔌 Backend registered: {name} ({backend.__class__.__name__})")

    def get_backend(self, name: str) -> Optional[ComputeBackend]:
        """Retrieves a registered backend by name."""
        return self.backends.get(name)

    def get_active_backend(self) -> ComputeBackend:
        """Returns the currently active backend."""
        return self.backends[self.active_backend_id]

    def set_active_backend(self, name: str):
        """Sets the active backend."""
        if name not in self.backends:
            raise ValueError(f"Backend '{name}' not found. Available: {list(self.backends.keys())}")
        self.active_backend_id = name
        logging.info(f"🔌 Active backend switched to: {name}")

    def list_backends(self) -> List[Dict[str, str]]:
        """Lists available backends and their status."""
        return [
            {
                "id": name,
                "type": backend.__class__.__name__,
                "status": backend.get_status().get("status", "unknown")
            }
            for name, backend in self.backends.items()
        ]
        
    def check_connection(self, name: str) -> bool:
        """Checks if a specific backend is reachable."""
        backend = self.get_backend(name)
        if not backend:
            return False
        status = backend.get_status()
        return status.get("status") in ["ready", "online"]

# Global singleton instance
connection_manager = ConnectionManager()
