import abc
import logging
import torch
import numpy as np
from typing import Any, Dict, Optional, Union

class ComputeBackend(abc.ABC):
    """
    Abstract base class for compute backends.
    A backend abstracts the execution hardware (CPU, GPU, QPU, Simulator).
    """
    
    @abc.abstractmethod
    def execute(self, operation: str, *args, **kwargs) -> Any:
        """
        Executes a specific operation on the backend.
        
        Args:
            operation: Name of the operation to execute (e.g., 'forward', 'step', 'measure').
            *args: Positional arguments for the operation.
            **kwargs: Keyword arguments for the operation.
            
        Returns:
            The result of the operation.
        """
        pass

    @abc.abstractmethod
    def get_device(self) -> Any:
        """Returns the underlying device object (e.g., torch.device)."""
        pass
    
    @abc.abstractmethod
    def get_status(self) -> Dict[str, Any]:
        """Returns the status of the backend (e.g., connected, queue depth)."""
        pass

class LocalBackend(ComputeBackend):
    """
    Backend for local execution using PyTorch (CPU or GPU).
    This is the default backend for Aetheria.
    """
    
    def __init__(self, device: Union[str, torch.device] = "cuda" if torch.cuda.is_available() else "cpu"):
        self.device = torch.device(device)
        logging.info(f"🖥️ Initialized LocalBackend on {self.device}")

    def execute(self, operation: str, *args, **kwargs) -> Any:
        # For local backend, we might just return the args/kwargs to be used by the engine directly,
        # or execute specific tensor operations if we move logic here.
        # Currently, the Engine holds the logic, so this might be a pass-through or
        # used for specific offloaded tasks.
        
        # In the future, this could dispatch to specific kernels.
        pass

    def get_device(self) -> torch.device:
        return self.device
    
    def get_status(self) -> Dict[str, Any]:
        return {
            "type": "local",
            "device": str(self.device),
            "status": "ready",
            "memory_allocated": torch.cuda.memory_allocated(self.device) if self.device.type == 'cuda' else 0
        }

class MockQuantumBackend(ComputeBackend):
    """
    A mock backend that simulates a Quantum Processing Unit (QPU) connection.
    Used for testing UI integration and fallback logic without real hardware.
    """
    
    def __init__(self, num_qubits: int = 25):
        self.num_qubits = num_qubits
        self.device = torch.device("cpu") # Use CPU for classical simulation of the mock
        logging.info(f"🔮 Initialized MockQuantumBackend with {num_qubits} qubits")

    def execute(self, operation: str, *args, **kwargs) -> Any:
        if operation == 'run_circuit':
            # Simulate a delay and return random results
            import time
            time.sleep(0.1) 
            return {"00": 0.5, "11": 0.5} # Bell state-ish
        return None

    def get_device(self) -> torch.device:
        return self.device
    
    def get_status(self) -> Dict[str, Any]:
        return {
            "type": "quantum_mock",
            "device": "MockQPU",
            "status": "online",
            "qubits": self.num_qubits,
            "queue_depth": 0
        }

class IonQBackend(ComputeBackend):
    """
    Backend for execution on IonQ quantum computers via Qiskit or API.
    """
    
    def __init__(self, api_key: Optional[str] = None, backend_name: str = "ionq_simulator"):
        self.api_key = api_key
        self.backend_name = backend_name
        self.device = torch.device("cpu") # Classical interface
        self.provider = None
        self.backend = None
        
        # Try to initialize Qiskit IonQ Provider
        try:
            from qiskit_ionq import IonQProvider
            if self.api_key:
                self.provider = IonQProvider(token=self.api_key)
                self.backend = self.provider.get_backend(self.backend_name)
                logging.info(f"⚛️ Initialized IonQBackend connected to {self.backend_name}")
            else:
                logging.warning("⚠️ IonQBackend initialized without API Key. Execution will fail.")
        except ImportError:
            logging.error("❌ qiskit-ionq not installed. Please run: pip install qiskit-ionq")
        except Exception as e:
            logging.error(f"❌ Error initializing IonQBackend: {e}")

    def execute(self, operation: str, *args, **kwargs) -> Any:
        """
        Executes a quantum circuit on IonQ.
        
        Args:
            operation: Must be 'run_circuit'
            args[0]: QuantumCircuit object
            kwargs: 'shots' (default 1024)
        """
        if operation != 'run_circuit':
            raise ValueError(f"IonQBackend does not support operation: {operation}")
            
        if not self.backend:
            raise RuntimeError("IonQ backend not initialized (missing key or library)")
            
        circuit = args[0]
        shots = kwargs.get('shots', 1024)
        
        try:
            job = self.backend.run(circuit, shots=shots)
            logging.info(f"🚀 Job submitted to IonQ: {job.job_id()}")
            
            # Wait for result (blocking for now, can be made async)
            result = job.result()
            counts = result.get_counts()
            return counts
        except Exception as e:
            logging.error(f"❌ IonQ execution failed: {e}")
            raise

    def get_device(self) -> torch.device:
        return self.device
    
    def get_status(self) -> Dict[str, Any]:
        status = "offline"
        queue = 0
        if self.backend:
            try:
                # This might vary based on backend status availability
                b_status = self.backend.status()
                status = "online" if b_status.operational else "maintenance"
                queue = b_status.pending_jobs
            except:
                status = "unknown"
                
        return {
            "type": "quantum_ionq",
            "device": self.backend_name,
            "status": status,
            "queue_depth": queue
        }
