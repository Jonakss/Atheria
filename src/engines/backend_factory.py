import logging
from typing import Optional
from .compute_backend import ComputeBackend, IonQBackend, IBMBackend, LocalQuantumBackend, MockQuantumBackend

class BackendFactory:
    """
    Factory class to create and manage compute backends.
    """

    @staticmethod
    def get_backend(backend_name: str) -> ComputeBackend:
        """
        Returns an instance of the requested backend.

        Args:
            backend_name: Identifier for the backend.
                Supported values:
                - 'ionq_simulator': IonQ Simulator
                - 'ionq_aria': IonQ Aria QPU
                - 'ionq_qpu': Generic IonQ QPU
                - 'ibm_brisbane': IBM Brisbane QPU
                - 'ibm_kyoto': IBM Kyoto QPU
                - 'ibm_qpu': Generic IBM QPU (Least Busy)
                - 'local_aer': Local Qiskit Aer Simulator
                - 'local_mock': Local Mock Backend (Debug)

        Returns:
            ComputeBackend instance.
        """
        logging.info(f"🏭 Requesting backend: {backend_name}")

        if backend_name.startswith('ionq'):
            # IonQ Backends
            # Ensure we pass the specific target to the backend class
            target = backend_name if backend_name != 'ionq' else 'ionq_simulator'
            # If specifically 'ionq_qpu', we might want to let IonQBackend pick a default QPU or pass it through
            if backend_name == 'ionq_qpu':
                target = 'ionq_qpu'
            return IonQBackend(backend_name=target)

        elif backend_name.startswith('ibm'):
            # IBM Backends
            return IBMBackend(backend_name=backend_name)

        elif backend_name == 'local_aer':
            return LocalQuantumBackend()

        elif backend_name == 'local_mock':
            return MockQuantumBackend()

        else:
            logging.warning(f"⚠️ Unknown backend '{backend_name}'. Defaulting to Local Mock.")
            return MockQuantumBackend()
