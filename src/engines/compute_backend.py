import abc
import logging
import torch
import numpy as np
from typing import Any, Dict, Optional, Union
import os

# Cargar variables de .env si existe
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass  # dotenv no instalado, usar variables de entorno del sistema

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
            import time
            time.sleep(0.1) 
            return {"00": 0.5, "11": 0.5}
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
    Backend for execution on IonQ quantum computers via Qiskit.
    """
    
    def __init__(self, api_key: Optional[str] = None, backend_name: str = "ionq_simulator"):
        self.api_key = api_key or os.getenv("IONQ_API_KEY")
        self.backend_name = backend_name
        self.device = torch.device("cpu") # Classical interface
        self.provider = None
        self.backend = None
        
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
        if operation != 'run_circuit':
            raise ValueError(f"IonQBackend does not support operation: {operation}")
            
        if not self.backend:
            raise RuntimeError("IonQ backend not initialized (missing key or library)")
            
        circuit = args[0]
        shots = kwargs.get('shots', 1024)
        
        try:
            job = self.backend.run(circuit, shots=shots)
            logging.info(f"🚀 Job submitted to IonQ: {job.job_id()}")
            
            # Wait for result (blocking)
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

class IBMBackend(ComputeBackend):
    """
    Backend for execution on IBM Quantum via Qiskit Runtime.
    """

    def __init__(self, api_key: Optional[str] = None, backend_name: str = "ibm_brisbane"):
        self.api_key = api_key or os.getenv("QISKIT_IBM_TOKEN")
        self.backend_name = backend_name
        self.device = torch.device("cpu")
        self.service = None
        self.backend = None

        try:
            from qiskit_ibm_runtime import QiskitRuntimeService
            if self.api_key:
                # Initialize service (save_account is typically done once, but we can pass token directly)
                # If channel is 'ibm_quantum', token is required.
                try:
                    self.service = QiskitRuntimeService(channel="ibm_quantum", token=self.api_key)
                except Exception as e:
                    logging.warning(f"⚠️ Failed to initialize QiskitRuntimeService with token: {e}. Attempting fallback.")
                    # Fallback if already saved on disk
                    self.service = QiskitRuntimeService(channel="ibm_quantum")

                # Get backend
                # If specifically requested generic 'ibm_qpu', pick least busy
                if backend_name == 'ibm_qpu':
                     self.backend = self.service.least_busy(operational=True, simulator=False)
                else:
                     self.backend = self.service.backend(backend_name)

                logging.info(f"🟦 Initialized IBMBackend connected to {self.backend.name}")
            else:
                 logging.warning("⚠️ IBMBackend initialized without API Key. Execution will fail.")
        except ImportError:
            logging.error("❌ qiskit-ibm-runtime not installed. Please run: pip install qiskit-ibm-runtime")
        except Exception as e:
            logging.error(f"❌ Error initializing IBMBackend: {e}")

    def execute(self, operation: str, *args, **kwargs) -> Any:
        if operation != 'run_circuit':
             raise ValueError(f"IBMBackend does not support operation: {operation}")

        if not self.backend:
             raise RuntimeError("IBM backend not initialized")

        circuit = args[0]
        shots = kwargs.get('shots', 1024)

        try:
            # Use SamplerV2 for execution
            from qiskit_ibm_runtime import SamplerV2
            from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager

            # Transpile
            pm = generate_preset_pass_manager(backend=self.backend, optimization_level=1)
            isa_circuit = pm.run(circuit)

            sampler = SamplerV2(mode=self.backend)
            job = sampler.run([isa_circuit], shots=shots)
            logging.info(f"🚀 Job submitted to IBM Quantum: {job.job_id()}")

            result = job.result()
            # Result structure for V2 is different (PubResult)
            # We assume 1 pub, 1 circuit
            pub_result = result[0]
            # Access bitstrings from the first register (usually 'meas' or 'c')
            # This depends on circuit construction.
            # Fallback to checking available data
            data = pub_result.data
            # Assuming standard measurement register
            if hasattr(data, 'meas'):
                counts = data.meas.get_counts()
            elif hasattr(data, 'c'): # Classical register often named 'c'
                counts = data.c.get_counts()
            else:
                # Try to get first available attribute that has counts
                counts = {}
                for attr in dir(data):
                    if not attr.startswith('_'):
                        val = getattr(data, attr)
                        if hasattr(val, 'get_counts'):
                            counts = val.get_counts()
                            break

            return counts

        except Exception as e:
             logging.error(f"❌ IBM execution failed: {e}")
             raise

    def get_device(self) -> torch.device:
        return self.device

    def get_status(self) -> Dict[str, Any]:
         status = "offline"
         queue = 0
         if self.backend:
             try:
                 status_val = self.backend.status()
                 status = "online" if status_val.operational else "maintenance"
                 queue = status_val.pending_jobs
             except:
                 status = "unknown"

         return {
             "type": "quantum_ibm",
             "device": self.backend.name if self.backend else self.backend_name,
             "status": status,
             "queue_depth": queue
         }

class LocalQuantumBackend(ComputeBackend):
    """
    Backend for local quantum simulation using Qiskit Aer.
    """
    def __init__(self, backend_name: str = "aer_simulator"):
        self.backend_name = backend_name
        self.device = torch.device("cpu")
        self.backend = None

        try:
            from qiskit_aer import AerSimulator
            self.backend = AerSimulator()
            logging.info(f"💻 Initialized LocalQuantumBackend ({self.backend_name})")
        except ImportError:
            logging.error("❌ qiskit-aer not installed. Please run: pip install qiskit-aer")

    def execute(self, operation: str, *args, **kwargs) -> Any:
        if operation != 'run_circuit':
            raise ValueError(f"LocalQuantumBackend does not support: {operation}")

        if not self.backend:
             raise RuntimeError("Aer backend not initialized")

        circuit = args[0]
        shots = kwargs.get('shots', 1024)

        try:
            # Transpile for Aer
            from qiskit import transpile
            t_circuit = transpile(circuit, self.backend)
            job = self.backend.run(t_circuit, shots=shots)
            result = job.result()
            counts = result.get_counts()
            return counts
        except Exception as e:
            logging.error(f"❌ Local Aer execution failed: {e}")
            raise

    def get_device(self) -> torch.device:
        return self.device

    def get_status(self) -> Dict[str, Any]:
        return {
            "type": "quantum_local",
            "device": "AerSimulator",
            "status": "online",
            "queue_depth": 0
        }
