# PLAN: Integración Frontend Quantum Fast Forward
**Para:** Agente Frontend (Jules)
**De:** Agente Backend (Antigravity)

## Objetivo
Integrar la funcionalidad de "Salto Temporal Cuántico" (Quantum Fast Forward) en la interfaz de Atheria, permitiendo al usuario ejecutar la evolución de 1 millón de pasos en IonQ (o simulador) con un solo clic.

## 1. Nuevos Componentes UI

### A. Panel de Control Cuántico (`QuantumControls.tsx`)
Agregar una sección o modal dedicado a "Time Warp / Fast Forward":
- **Botón:** "Quantum Jump (1M Steps)".
- **Selector de Backend:** [IonQ Simulator | IonQ Aria | Local Mock].
- **Visualizador de Estado:**
    - Mostrar el circuito cuántico (QASM) que se va a ejecutar.
    - Mostrar el estado de la cola de IonQ (Status: Queued, Running, Completed).

### B. Visualizador de Circuito (`QuantumCircuitViewer.tsx`)
- Implementar un visualizador simple de texto o gráfico para el código QASM.
- El QASM está disponible en `models/quantum_fastforward.qasm`.
- Puedes cargarlo vía API o tenerlo estático si no cambia.

## 2. Integración WebSocket

### A. Nuevo Evento: `QUANTUM_FAST_FORWARD`
El frontend debe emitir este evento para iniciar el proceso.

**Payload Request:**
```json
{
  "action": "quantum_fast_forward",
  "params": {
    "steps": 1000000,
    "backend": "ionq_simulator"
  }
}
```

**Payload Response (Progress):**
```json
{
  "type": "quantum_status",
  "status": "submitted",
  "job_id": "ionq-12345"
}
```

**Payload Response (Complete):**
```json
{
  "type": "state_update",
  "data": { ... }, // Nuevo estado físico
  "metadata": {
    "quantum_execution_time": "120ms",
    "fidelity": 0.9999
  }
}
```

## 3. Flujo de Usuario (UX)
1.  Usuario hace clic en "Quantum Jump".
2.  UI muestra "Compiling Circuit..." -> "Sending to IonQ...".
3.  UI muestra spinner/loader mientras espera la respuesta del QPU.
4.  Al recibir el resultado, la simulación visual se actualiza instantáneamente al estado futuro.
5.  Mostrar un "Toast" o notificación: "Time Jump Successful (Fidelity: 99.99%)".

## 4. Recursos
- **Modelo:** El backend usará `models/quantum_fastforward_final.pt`.
- **QASM:** Puedes mostrar el contenido de `models/quantum_fastforward.qasm` como "Source Code" del universo.
