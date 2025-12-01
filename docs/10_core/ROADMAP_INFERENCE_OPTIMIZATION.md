# 🚀 Roadmap: Inference Optimization & Serving

> **Objetivo:** Transformar Atheria de un prototipo de investigación a una solución de producción escalable mediante optimización de inferencia, servicio asíncrono y cuantización.

**Estado:** 📅 Planificación  
**Documento Base:** [[INFERENCE_OPTIMIZATION_STRATEGIES]]

---

## 📋 Resumen de Fases

| Fase | Nombre | Objetivo | Estado |
|------|--------|----------|--------|
| **Fase 1** | **Infraestructura Asíncrona** | Desacoplar servicio de inferencia | 🔴 Pendiente |
| **Fase 2** | **Compresión de Modelo** | Reducir VRAM y coste | 🔴 Pendiente |
| **Fase 3** | **Aceleración de Grafo** | Maximizar throughput puro | 🔴 Pendiente |
| **Fase 4** | **Despliegue Productivo** | Escalar y monitorizar | 🔴 Pendiente |

---

## 🛠️ Fase 1: Infraestructura Asíncrona (LitServe)

**Meta:** Eliminar bloqueos en el bucle de simulación y permitir concurrencia.

- [ ] **Migración a LitServe**
  - [ ] Crear clase `AtheriaInferenceAPI` heredando de `LitAPI`
  - [ ] Implementar método `setup()` para carga de modelo Ley M
  - [ ] Implementar método `predict()` para inferencia de un paso
- [ ] **Gestión de Concurrencia**
  - [ ] Configurar `max_batch_size` y `batch_timeout` (e.g., 4 requests / 50ms)
  - [ ] Implementar manejo de colas para múltiples clientes WebSocket
- [ ] **Integración con WebSocket Existente**
  - [ ] Adaptar `SimulationService` para usar el endpoint de LitServe (o integrarlo in-process)
  - [ ] Asegurar que el streaming de frames no se bloquee por la inferencia

## 📉 Fase 2: Compresión de Modelo (Quantization)

**Meta:** Reducir requisitos de hardware (A100 -> L4) sin perder emergencia.

- [ ] **Implementación de NF4 (4-bit)**
  - [ ] Integrar `bitsandbytes` en el pipeline de carga de Ley M
  - [ ] Configurar `QuantizationConfig` para backbone del modelo
- [ ] **Validación de Calidad**
  - [ ] Crear script de comparación de fidelidad (FP16 vs NF4)
  - [ ] Verificar métricas de emergencia (entropía, complejidad) en simulación larga
- [ ] **Optimización de Memoria**
  - [ ] Medir reducción de VRAM
  - [ ] Ajustar batch size máximo permitido con la nueva memoria disponible

## ⚡ Fase 3: Aceleración de Grafo (torch.compile)

**Meta:** Reducir overhead de Python y optimizar kernels de GPU.

- [ ] **Compilación JIT**
  - [ ] Envolver modelo Ley M con `torch.compile(mode="reduce-overhead")`
  - [ ] Identificar y eliminar "graph breaks" en el código del modelo
- [ ] **Optimización de Tensores**
  - [ ] Asegurar formas estáticas (static shapes) en los tensores de entrada
  - [ ] Pre-asignar memoria para buffers recurrentes
- [ ] **Benchmarking**
  - [ ] Medir latencia de inferencia (ms/step) antes y después
  - [ ] Comparar throughput (steps/sec)

## 🚀 Fase 4: Despliegue y Escalado

**Meta:** Infraestructura robusta para múltiples usuarios o simulaciones masivas.

- [ ] **Contenerización**
  - [ ] Actualizar Dockerfile para incluir dependencias de optimización (LitServe, bitsandbytes)
  - [ ] Crear imagen optimizada para inferencia (separada de entrenamiento)
- [ ] **Orquestación**
  - [ ] Configurar despliegue en Lightning AI o Kubernetes
  - [ ] Implementar auto-escalado basado en profundidad de cola
- [ ] **Monitorización**
  - [ ] Exponer métricas de inferencia (latencia, throughput, VRAM)
  - [ ] Integrar alertas de degradación de rendimiento

---

## 🔗 Referencias

- [[INFERENCE_OPTIMIZATION_STRATEGIES]] - Estrategias detalladas
- [[EXTERNAL_RESEARCH_GEMINI_OPTIMIZATION_ANALYSIS]] - Investigación original
- [[ROADMAP_PHASE_2]] - Relacionado: Motor Nativo C++ (otra vía de optimización)
