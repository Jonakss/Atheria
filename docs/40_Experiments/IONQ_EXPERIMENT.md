# Experimento: Integración con IonQ

**Fecha:** 2025-12-02
**Estado:** Implementado (Requiere API Key)

## Objetivo
Habilitar la ejecución de circuitos cuánticos desde Atheria utilizando el hardware de IonQ, permitiendo experimentos híbridos donde parte del cómputo se delega a una QPU real.

## Implementación
Se ha creado una abstracción `ComputeBackend` en `src/engines/compute_backend.py`.
- **IonQBackend:** Implementación específica que usa `qiskit-ionq` para comunicarse con IonQ.
- **Configuración:** Se utilizan variables de entorno `IONQ_API_KEY` y `IONQ_BACKEND_NAME`.

## Script de Prueba
El script `scripts/experiment_ionq.py` inicializa el backend y envía un circuito Bell State simple.

### Resultados de Verificación
El script se ejecutó exitosamente en el simulador de IonQ (`ionq_simulator`).

**Salida del Experimento:**
```
✅ Backend Initialized: {'type': 'quantum_ionq', 'device': 'ionq_simulator', 'status': 'unknown', 'queue_depth': 0}

🧪 Submitting Circuit to IonQ...
🏆 Results:
{'00': 47, '11': 53}
✅ Bell state correlation observed!
```

Esto confirma que la integración es funcional y capaz de enviar circuitos, esperar la ejecución y recuperar resultados.

## Quantum Genesis (Inicialización Cuántica)

Atheria soporta **"Quantum Genesis"**: inicializar el estado del universo ($t=0$) usando datos cuánticos reales en lugar de pseudo-aleatorios.

### Cómo funciona
1. Se ejecuta un circuito cuántico en IonQ (Superposición + Entrelazamiento).
2. Se miden los resultados (bitstrings).
3. Se usan estos bits para llenar el grid inicial de la simulación.
4. Esto crea un patrón de ruido inicial con correlaciones cuánticas reales.

### Uso
Para usar Quantum Genesis, el motor debe inicializarse con `initial_mode='ionq'`.
Actualmente esto se puede probar con:
```bash
python3 scripts/test_ionq_init.py
```

## Pasos para Ejecución Real
1. Instalar dependencia: `pip install qiskit-ionq`
2. Obtener API Key de IonQ.
3. Ejecutar:
   ```bash
   export IONQ_API_KEY="tu_api_key"
   python3 scripts/experiment_ionq.py
   ```
