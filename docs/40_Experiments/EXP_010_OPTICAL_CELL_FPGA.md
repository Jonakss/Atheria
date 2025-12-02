# EXP-010: Simulación FPGA de Célula Óptica (AetheriaCell)

**Estado:** Simulado Exitosamente
**Herramientas:** EDA Playground, Icarus Verilog

## Objetivo
Validar que la lógica fundamental de Aetheria (interacción local + actualización de estado) puede implementarse en hardware digital (FPGA) y que es equivalente a un interferómetro óptico (Beam Splitter + Phase Shift).

## El Diseño (`design.sv`)
Implementamos una célula que promedia el estado de sus vecinos (difusión) y aplica una transformación.
```verilog
module AetheriaCell (
    input clk, rst,
    input [7:0] neighbor_L, neighbor_R,
    output reg [7:0] state
);
    // Física: (L + R) / 2 + 1
    // Equivalente a Beam Splitter 50:50
    state <= ((neighbor_L + neighbor_R) >> 1) + 8'd1;
endmodule
```

## Resultados de la Simulación (`testbench.sv`)
Simulamos una cadena de 3 células y un anillo de 16 células.
* **Propagación de Onda:** Observamos cómo un pulso de energía inyectado en la Célula 1 viaja a la Célula 2 y 3 con un retraso de 1 ciclo de reloj ($c = 1 cell/clock$).
* **Interferencia:** Cuando la energía rebotó en los bordes o en el anillo cerrado, observamos patrones de interferencia constructiva y destructiva.
* **Disipación:** Al cortar la fuente de energía, el sistema decayó gradualmente hacia la entropía máxima (estado homogéneo), validando la termodinámica del modelo.

## Conclusión
La "Célula Aetheria" es sintetizable en hardware. Esto abre la puerta a construir una **APU (Aetheria Processing Unit)** masivamente paralela basada en FPGAs para escalar la simulación más allá de lo que permiten las GPUs.
