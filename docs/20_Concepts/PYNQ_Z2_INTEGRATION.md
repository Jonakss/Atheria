# Integración de Aceleración por Hardware con PYNQ-Z2

## 1. Visión General de la Arquitectura
La placa **PYNQ-Z2** es un Sistema en Chip (SoC) Zynq-7000 que integra:
*   **PS (Processing System):** Un procesador ARM Cortex-A9 dual-core que ejecuta Linux (Ubuntu) y Python.
*   **PL (Programmable Logic):** El tejido FPGA (Artix-7) donde implementaremos la "Ley M" física.

El objetivo es descargar la evolución de las células (`evolve_step`) al FPGA para lograr una aceleración masiva y paralela, mientras el servidor `aiohttp` y la lógica de control permanecen en Python.

### Diagrama de Flujo de Datos
```mermaid
graph LR
    subgraph "Host (PC/Servidor)"
        A[Frontend React] <-->|WebSocket| B[Backend Python]
    end

    subgraph "PYNQ-Z2 (Edge Device)"
        B <-->|Ethernet/API| C[PYNQ Python API]
        C <-->|DMA (AXI bus)| D[FPGA Core (Aetheria Mesh)]
    end
```

## 2. Diseño del Hardware (Overlay)
En el ecosistema PYNQ, el diseño de hardware compilado (`.bit`) se llama **Overlay**.

### Componentes del Overlay "AetheriaOne"
1.  **Aetheria Mesh IP:** Bloque personalizado (HLS/Verilog) que contiene la cuadrícula de células.
    *   **Input:** Stream de estado actual ($128 \times 128 \times 8$ bits).
    *   **Lógica:** Implementa la regla de transición (promedio, fase, no-linealidad) en paralelo.
    *   **Output:** Stream de estado siguiente.
2.  **AXI DMA:** Permite transferir grandes bloques de datos memoria-FPGA sin cargar al procesador.
3.  **AXI Interconnect:** Gestiona comunicaciones.

### Especificación de la Célula (Verilog/HLS)
Versión optimizada para pipeline sistólico.
```verilog
// Ejemplo conceptual de la lógica en el PL
always @(posedge clk) begin
    if (valid_in) begin
        // Operación de mezcla óptica en hardware (1 ciclo)
        t_mix <= (in_left + in_right) >> 1;
        t_phase <= t_mix + phase_constant;
        out_state <= t_phase;
    end
end
```

## 3. Integración de Software (Python en PYNQ)
### Driver de Aetheria (`src/drivers/pynq_driver.py`)
Código que corre en el ARM de la PYNQ.

```python
from pynq import Overlay, allocate
import numpy as np

class AetheriaFPGA:
    def __init__(self, bitstream_path="aetheria.bit"):
        # 1. Cargar el Hardware
        self.overlay = Overlay(bitstream_path)
        self.dma = self.overlay.axi_dma_0

        # 2. Reservar memoria contigua para DMA
        self.buffer_size = 128 * 128
        self.input_buffer = allocate(shape=(self.buffer_size,), dtype=np.uint8)
        self.output_buffer = allocate(shape=(self.buffer_size,), dtype=np.uint8)

    def evolve(self, current_state_array):
        # A. Copiar datos al Buffer
        np.copyto(self.input_buffer, current_state_array.flatten())

        # B. Iniciar transferencia
        self.dma.sendchannel.transfer(self.input_buffer)
        self.dma.recvchannel.transfer(self.output_buffer)

        # C. Esperar
        self.dma.sendchannel.wait()
        self.dma.recvchannel.wait()

        # D. Devolver resultado
        return self.output_buffer.reshape((128, 128))
```

## 4. Conexión con el Servidor Principal
Estrategia: **"Remote Worker"**. La PYNQ-Z2 ejecuta un servidor ligero (FastAPI) que expone el hardware.

### En la PYNQ (`worker.py`)
```python
from fastapi import FastAPI
from pynq_driver import AetheriaFPGA
import numpy as np

app = FastAPI()
fpga = AetheriaFPGA()

@app.post("/evolve")
async def evolve_step(data: dict):
    state = np.array(data['state'], dtype=np.uint8)
    new_state = fpga.evolve(state)
    return {"state": new_state.tolist()}
```

### En el PC Host (`src/qca_engine_remote.py`)
```python
import requests

class RemoteFPGAEngine:
    def __init__(self, url="http://pynq:8000"):
        self.url = url

    def evolve_step(self, current_psi):
        response = requests.post(f"{self.url}/evolve", json={"state": current_psi.tolist()})
        return np.array(response.json()['state'])
```

## 5. Ventajas y Consideraciones
| Ventaja | Descripción |
| :--- | :--- |
| **Paralelismo Real** | Actualización simultánea de 16k celdas. |
| **Baja Latencia** | Microsegundos una vez los datos están en chip. |
| **Eficiencia** | Fracción de energía comparado con GPU. |

**Limitaciones:**
*   **Ancho de Banda de Red:** Enviar $128 \times 128$ por Ethernet a 60fps es el cuello de botella.
*   **Capacidad Artix-7:** Limitado para grids masivos ($1024^2$).

## 6. Próximos Pasos
1.  **Adquisición:** Conseguir PYNQ-Z2.
2.  **HLS:** Escribir núcleo en C++ (Vivado HLS).
3.  **Integración:** Script Python con `pynq`.
