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
                - 'ionq_simulator', 'ionq_aria', 'ionq_qpu': IonQBackend
                - 'ibm_brisbane', 'ibm_kyoto', 'ibm_qpu': IBMBackend
                - 'local_aer': LocalQuantumBackend
                - 'local_mock': MockQuantumBackend

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
