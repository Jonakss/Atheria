# AETHERIA: Laboratorio de Emergencia Cuántica

AETHERIA es un entorno de simulación e investigación para estudiar la emergencia de complejidad en autómatas celulares cuánticos (QCA). El sistema utiliza un modelo de Deep Learning, denominado "Ley M", para gobernar la evolución del estado cuántico, con el objetivo de descubrir operadores que generen patrones estructurales complejos en el "borde del caos".

Este proyecto es una aplicación web interactiva con un backend de `aiohttp` y un frontend de `React`, permitiendo el entrenamiento de modelos en GPU y la visualización de simulaciones en tiempo real.

## Estado del Proyecto

El proyecto ha sido refactorizado para asegurar su estabilidad y mantenibilidad. Todo el código obsoleto y los puntos de entrada conflictivos han sido eliminados.

## Cómo Empezar

### Requisitos

- Python 3.10+
- Conda o venv para la gestión de entornos
- Node.js y npm para el frontend

### Instalación

1.  **Clonar el repositorio:**
    ```bash
    git clone <repository-url>
    cd Atheria
    ```

2.  **Configurar el Backend:**
    ```bash
    # Crear y activar un entorno virtual
    python3 -m venv ath_venv
    source ath_venv/bin/activate

    # Instalar dependencias de Python
    pip install -r requirements.txt
    ```

3.  **Configurar el Frontend:**
    ```bash
    cd frontend
    npm install
    npm run build
    cd ..
    ```

### **Ejecutar la Aplicación**

El único punto de entrada para la aplicación es `run_server.py`. Este script sirve tanto el backend de la API como el frontend de React.

**Desde la raíz del proyecto, ejecuta:**
```bash
# Asegúrate de que tu entorno virtual esté activado
source ath_venv/bin/activate

# Establece la variable de entorno para desarrollo
export AETHERIA_ENV=development

# Inicia el servidor
python3 run_server.py
```

La aplicación estará disponible en `http://localhost:8000`.