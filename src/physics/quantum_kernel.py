import torch
import logging
import numpy as np
from .. import config as cfg

class QuantumMicroscope:
    """
    Módulo de 'Quantum Microscope' (Deep Quantum Kernel).
    
    Utiliza un circuito cuántico profundo (ZZ Feature Map + Entanglement) para analizar
    correlaciones no locales y estructuras ocultas en el grid de simulación.
    
    Concepto:
    Mapea datos clásicos (parche del grid) a un espacio de Hilbert de alta dimensión
    donde patrones complejos (vórtices, entrelazamiento) son linealmente separables o
    detectables vía medidas de complejidad.
    """
    def __init__(self, device):
        self.device = device
        self.backend = None
        self._setup_backend()
        
    def _setup_backend(self):
        try:
            from ..engines.compute_backend import IonQBackend
            if cfg.IONQ_API_KEY:
                self.backend = IonQBackend(api_key=cfg.IONQ_API_KEY, backend_name=cfg.IONQ_BACKEND_NAME)
            else:
                logging.warning("⚠️ IonQ API Key missing. QuantumMicroscope will run in MOCK mode.")
        except Exception as e:
            logging.error(f"❌ Failed to init IonQBackend for Microscope: {e}")

    def analyze_patch(self, state_patch):
        """
        Analiza un parche del estado usando el Kernel Cuántico.
        
        Args:
            state_patch: Tensor [C, H, W] o [H, W, C].
                         Se espera un parche pequeño (ej: 4x4 o 2x2).
                         
        Returns:
            Dict con métricas: {'complexity': float, 'coherence': float, 'features': list}
        """
        # 1. Preprocesamiento: Downsample a N qubits
        # IonQ Basic soporta 11 qubits. Usemos 4 qubits para un análisis rápido y robusto (2x2 grid).
        n_qubits = 4
        
        # Aplanar y reducir dimensionalidad
        # Tomamos la magnitud promedio de los canales (dim 0)
        # Asumimos input (C, H, W)
        if state_patch.is_complex():
            magnitude = state_patch.abs().mean(dim=0) 
        else:
            magnitude = state_patch.mean(dim=0)
            
        if len(magnitude.shape) == 3:
            magnitude = magnitude[0]
            
        print(f"DEBUG Magnitude Shape: {magnitude.shape}")
        print(f"DEBUG Magnitude Content:\n{magnitude}")
            
        # Reducir a 4 qubits (2x2) si es más grande
        # El backend IonQ básico suele ser de 11 qubits, pero para este kernel usamos 4
        # Interpolamos a 2x2 (4 valores)
        # Asumimos input torch tensor
        if len(magnitude.shape) == 2:
            H, W = magnitude.shape
            # Simple pooling to 2x2
            h_step, w_step = max(1, H//2), max(1, W//2)
            inputs = []
            inputs.append(magnitude[0:h_step, 0:w_step].mean().item())
            inputs.append(magnitude[0:h_step, w_step:].mean().item())
            inputs.append(magnitude[h_step:, 0:w_step].mean().item())
            inputs.append(magnitude[h_step:, w_step:].mean().item())
        else:
            # Fallback
            inputs = [0.0] * n_qubits
            
        # Normalizar a [0, 2pi] para rotaciones
        inputs = np.array(inputs)
        print(f"DEBUG Pre-Norm Inputs: {inputs}")
        inputs = (inputs - inputs.min()) / (inputs.max() - inputs.min() + 1e-6) # [0, 1]
        inputs = inputs * np.pi # [0, pi] (ZZ map usa esto mejor)
        
        print(f"DEBUG Microscope Inputs: {inputs}")
        
        if self.backend is None:
            return self._mock_analysis(inputs)
            
        try:
            from qiskit import QuantumCircuit
            from qiskit.circuit.library import ZZFeatureMap, RealAmplitudes
            
            # --- 1. PREPARACIÓN & CODIFICACIÓN (ZZ Feature Map) ---
            # Mapea datos a correlaciones cuánticas (difícil de simular clásicamente)
            # reps=1 para velocidad, entanglement='linear' o 'circular'
            feature_map = ZZFeatureMap(feature_dimension=n_qubits, reps=1, entanglement='circular')
            qc_encode = feature_map.assign_parameters(inputs)
            
            # --- 2. PROCESAMIENTO PROFUNDO (Ansatz) ---
            # "Strongly Entangling Layers" simplificado para Qiskit
            # RealAmplitudes añade capas de Ry y CNOTs
            ansatz = RealAmplitudes(num_qubits=n_qubits, reps=1, entanglement='full')
            # Fijamos parámetros del ansatz a valores aleatorios (o entrenados) para que actúe como un "filtro" fijo
            # Por ahora, usamos parámetros fijos para consistencia
            fixed_params = [0.5] * ansatz.num_parameters
            qc_process = ansatz.assign_parameters(fixed_params)
            
            # Combinar
            qc = QuantumCircuit(n_qubits)
            qc.compose(qc_encode, inplace=True)
            qc.compose(qc_process, inplace=True)
            qc.measure_all()
            
            # Transpilar para el backend (necesario para IonQ)
            # Usamos basis_gates genéricos de IonQ para evitar errores de atributo en el backend object
            from qiskit import transpile
            qc = transpile(qc, basis_gates=['rx', 'ry', 'rz', 'rxx', 'cx'], optimization_level=1)
            
            # --- 3. EJECUCIÓN ---
            # Execute the transpiled circuit
            # Our IonQBackend wrapper uses 'execute'
            if hasattr(self.backend, 'run'):
                job = self.backend.run(qc, shots=1024)
                counts = job.result().get_counts(qc)
            else:
                # Fallback for our custom wrapper
                counts = self.backend.execute('run_circuit', qc, shots=1024)
            
            # --- 4. ANÁLISIS DE RESULTADOS ---
            # Calcular Expectation Values <Z_i>
            # P(0) - P(1) para cada qubit
            exp_vals = []
            total_shots = sum(counts.values())
            
            for i in range(n_qubits):
                p0 = 0
                p1 = 0
                for bitstring, count in counts.items():
                    # bitstring es little-endian en Qiskit usualmente, pero IonQ backend puede variar
                    # Asumimos orden estándar string index
                    # bitstring "0010" -> qubit 0 es el último char? Qiskit es qubit 0 = rightmost.
                    # Vamos a iterar simple.
                    bit = bitstring[n_qubits - 1 - i] # Qiskit style
                    if bit == '0': p0 += count
                    else: p1 += count
                
                exp_val = (p0 - p1) / total_shots
                exp_vals.append(exp_val)
                
            # Métricas Derivadas
            # Complexity: Varianza de las expectaciones (si todo es igual, es simple. Si varía, hay estructura)
            # Coherence: Proporción de estados "puros" vs mezcla (aproximado por max count probability)
            
            max_prob = max(counts.values()) / total_shots
            complexity = np.std(exp_vals) * 10.0 # Escalar
            
            # Detectar "Quantum Advantage" (simulado):
            # Si hay correlaciones fuertes (ZZ), los valores de Z deberían estar correlacionados
            # Calculamos correlación par a par <Zi Zj> - <Zi><Zj> (Covarianza)
            # Simplificado: Suma de valores absolutos de expectación
            activity = np.mean(np.abs(exp_vals))
            
            return {
                'complexity': float(complexity),
                'activity': float(activity),
                'coherence': float(max_prob),
                'features': [float(x) for x in exp_vals]
            }
            
        except Exception as e:
            logging.error(f"❌ Quantum Microscope Failed: {e}")
            return self._mock_analysis(inputs)

    def _mock_analysis(self, inputs):
        """Simulación local del análisis."""
        # Generar métricas basadas en la varianza del input
        variance = np.var(inputs)
        return {
            'complexity': float(variance * 10),
            'activity': float(np.mean(inputs)),
            'coherence': 0.5 + float(np.random.rand() * 0.1),
            'features': [float(x) for x in inputs] # Pass-through
        }
