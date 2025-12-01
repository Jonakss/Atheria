# Quantum Compute Services & Integration Strategy

## Overview
This document summarizes research on major Quantum Processing Unit (QPU) cloud services and outlines a strategy for integrating them into the Aetheria project. The goal is to move from local simulations to hybrid quantum-classical execution.

## Cloud Quantum Providers

### 1. IonQ (via Cloud & Partners)
*   **Hardware**: Trapped-ion quantum computers (Aria, Forte). High fidelity, long coherence.
*   **Access**:
    *   **Direct API**: REST API for job submission.
    *   **Cloud Partners**: Available on AWS Braket, Azure Quantum, Google Cloud.
*   **Pricing**:
    *   Per-shot pricing (e.g., ~$0.01/shot) + per-task fees.
    *   Subscription models (e.g., Aria plan ~$25k/mo).
    *   Free tier simulators often available via partners.
*   **Integration**: Python SDK (`ionq-python`), Qiskit provider, or direct REST.

### 2. AWS Braket
*   **Hardware**: Aggregator for IonQ, Rigetti, OQC, QuEra, D-Wave.
*   **Features**:
    *   Unified API (Braket SDK) for multiple backends.
    *   Hybrid Jobs (Classical + Quantum co-processing).
    *   Simulators: SV1 (State Vector), DM1 (Density Matrix), TN1 (Tensor Network).
*   **Pricing**:
    *   Simulators: ~$4.50/hour or per-minute.
    *   QPU: Per-task ($0.30) + Per-shot (varies, e.g., $0.00035 - $0.01).
*   **Integration**: `amazon-braket-sdk` (Python).

### 3. IBM Quantum (Qiskit Runtime)
*   **Hardware**: Superconducting qubits (Eagle, Osprey, Heron processors).
*   **Features**:
    *   **Qiskit Runtime**: Primitives (Sampler, Estimator) for optimized execution.
    *   Error suppression and mitigation built-in.
*   **Pricing**:
    *   **Open Plan**: Free access (limited time/backends).
    *   **Pay-As-You-Go**: ~$1.60/sec of runtime.
    *   **Premium**: Dedicated access.
*   **Integration**: `qiskit-ibm-runtime` (Python).

### 4. Azure Quantum
*   **Hardware**: Aggregator for IonQ, Quantinuum, Rigetti, PASQAL.
*   **Features**:
    *   Integration with Q# and QDK.
    *   Resource Estimator tool.
*   **Pricing**: Provider-specific (similar to AWS Braket).

## Integration Strategy: "Compute Backend" Abstraction

To support these diverse services alongside our local PyTorch (CPU/GPU) simulation, we propose a **Compute Backend** abstraction layer.

### Architecture
We will introduce a `ComputeBackend` interface that abstracts the execution of quantum/classical circuits.

```python
class ComputeBackend(ABC):
    @abstractmethod
    def execute(self, circuit_or_state, **kwargs):
        """Executes a simulation step or quantum circuit."""
        pass

    @abstractmethod
    def get_status(self):
        """Returns backend status (connected, queue depth, etc.)."""
        pass
```

### Backend Types
1.  **`LocalBackend`**:
    *   **Engine**: PyTorch (Native/Legacy).
    *   **Hardware**: Local CPU or CUDA GPU.
    *   **Use Case**: Real-time simulation, development, training.

2.  **`SimulatorBackend` (Cloud/Local)**:
    *   **Engine**: Qiskit Aer, Braket SV1, IonQ Simulator.
    *   **Hardware**: Local CPU/GPU or Cloud CPU/GPU instances.
    *   **Use Case**: Verifying quantum logic before spending on QPU.

3.  **`QPUBackend` (Cloud)**:
    *   **Engine**: IonQ, Rigetti, IBMQ via API.
    *   **Hardware**: Real Quantum Processor.
    *   **Use Case**: "Edge of Chaos" experiments, finding true quantum emergence.
    *   **Connection**: Managed via API Keys and async job submission.

### Implementation Phases

1.  **Phase 1: Abstraction (Current)**
    *   Refactor `MotorFactory` to use `ComputeBackend`.
    *   Implement `LocalBackend` (wrapping current logic).
    *   Create `MockQuantumBackend` for UI testing.

2.  **Phase 2: Simulation Connectors**
    *   Integrate `qiskit` or `pennylane` to run on local simulators via the backend interface.

3.  **Phase 3: Cloud Integration**
    *   Implement `IonQBackend` / `BraketBackend`.
    *   Add "Connection Manager" in UI to input API keys and select providers.

## Connection System (Lightning AI Style)
We will implement a "Connection Manager" service that:
*   Stores API credentials securely (env vars or encrypted local storage).
*   Checks connectivity/quota for cloud services.
*   Allows "One-Click" switching between Local (GPU) and Cloud (QPU).

This aligns with the user's vision of a system similar to Lightning AI or Grid, where compute resources are abstract and pluggable.
