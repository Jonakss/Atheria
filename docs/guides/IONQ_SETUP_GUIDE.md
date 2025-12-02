# Guía de Configuración IonQ para Atheria

Esta guía te ayudará a configurar y probar la integración de Atheria con los ordenadores cuánticos de IonQ.

## 1. Requisitos Previos

### Obtener una API Key
1. Ve a [IonQ Quantum Cloud Console](https://cloud.ionq.com/).
2. Regístrate o inicia sesión.
3. Navega a "API Keys" y genera una nueva clave.

### Instalar Librería
Atheria utiliza `qiskit-ionq` para comunicarse con el hardware.

```bash
pip install qiskit-ionq
```

## 2. Configuración del Entorno

Para que Atheria pueda usar tu cuenta, necesitas exportar tu API Key como variable de entorno.

**En tu terminal:**
```bash
export IONQ_API_KEY="tu_api_key_aqui"
```

*(Opcional) Para hacerlo permanente, añade esa línea a tu `~/.bashrc` o `~/.zshrc`.*

## 3. Verificación de Conexión

Hemos creado un script simple para verificar que todo esté correcto antes de lanzar experimentos complejos.

Ejecuta:
```bash
python3 scripts/test_ionq_connection.py
```

Este script verificará:
1. Que la librería esté instalada.
2. Que la API Key sea visible.
3. Que pueda conectarse a IonQ y listar los backends disponibles (Simulador y QPU).

## 4. Ejecutar un Experimento en Atheria

Una vez verificado, puedes correr el experimento de prueba de Atheria:

```bash
python3 scripts/experiment_ionq.py
```

### ¿Qué hace este experimento?
1. Inicializa el `IonQBackend` de Atheria.
2. Crea un circuito cuántico simple (Estado de Bell).
3. Lo envía a la cola de IonQ (por defecto al simulador `ionq_simulator`).
4. Espera los resultados y los muestra.

## Solución de Problemas常见

- **Error: `qiskit-ionq not installed`**: Ejecuta `pip install qiskit-ionq`.
- **Error: `Backend is offline`**: Verifica que tu API Key sea correcta y que tengas créditos o acceso al simulador.
- **Resultados inesperados**: El simulador puede tener ruido si se configura así, o si usas la QPU real (`ionq_qpu`), los resultados tendrán ruido cuántico real.
