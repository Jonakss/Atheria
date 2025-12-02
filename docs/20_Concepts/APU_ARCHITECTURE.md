# Unidad de Procesamiento Aetheria (APU): Arquitectura de Hardware

## Resumen Ejecutivo
La APU es un acelerador de hardware diseñado para la simulación eficiente de Lattice QFT. Se aleja de las arquitecturas Von Neumann tradicionales para adoptar un enfoque de **Dataflow** y **Compute-in-Memory**, optimizado para la aritmética polar y la preservación de la unitariedad.

## 1. Núcleo de Cómputo: Matriz CORDIC
A diferencia de las TPUs/GPUs que usan multiplicadores MAC (Multiply-Accumulate), la APU utiliza motores **CORDIC** (Coordinate Rotation Digital Computer).
* **Operación Nativa:** Rotación de fase ($R_\theta$).
* **Ventaja:** Calcula senos, cosenos y rotaciones vectoriales usando solo desplazamientos y sumas (shift-and-add), eliminando multiplicadores costosos y garantizando la conservación de la norma ($x^2 + y^2 = cte$).

## 2. Aritmética: Posit32 + Quire
Para evitar la degradación numérica en simulaciones largas:
* **Posit32:** Formato numérico que ofrece mayor precisión en el rango $[-1, 1]$ (donde residen las amplitudes cuánticas) que el IEEE Float32.
* **Quire:** Acumulador de alta precisión (512-bit) para realizar sumas exactas sin errores de redondeo intermedios.

## 3. Estrategia de Implementación
### Prototipo FPGA
* **Plataforma:** Xilinx Alveo o Intel Agilex.
* **Lógica:** Implementación de celdas "OpticalCell" (Beam Splitter + Phase Shifter) en Verilog.
* **Simulación:** Verificada en EDA Playground usando Icarus Verilog.

### ASIC Analógico (Futuro)
* **Tecnología:** ReRAM (Resistive RAM) o Flash Analógica.
* **Concepto:** Utilizar la Ley de Ohm ($I = V \cdot G$) para realizar multiplicaciones matriciales instantáneas en el dominio analógico.
* **Desafío:** Ruido térmico y variabilidad de dispositivos (mitigado por el entrenamiento "Hardware-Aware").

## 4. Diagrama de Bloques (Conceptual)
```mermaid
graph TD
    A[Memoria Tensor (HBM/CXL)] -->|Tensores Polares| B(Matriz Sistólica CORDIC)
    B -->|Rotaciones| C{ALU Posit/Quire}
    C -->|Acumulación Exacta| D[Activación No-Lineal]
    D -->|Nuevo Estado| A
```

## Referencias
* **Mythic AI:** Arquitectura CIM basada en Flash.
* **Gustafson:** Posit Arithmetic.
* **KLM Protocol:** Computación cuántica óptica lineal.
