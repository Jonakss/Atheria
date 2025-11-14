# AETHERIA: Laboratorio de Emergencia Cuántica

AETHERIA es un entorno de simulación e investigación para estudiar la emergencia de complejidad en autómatas celulares cuánticos (QCA). El sistema utiliza un modelo de Deep Learning, denominado "Ley M", para gobernar la evolución del estado cuántico.

Este proyecto es una aplicación web interactiva con un backend de `aiohttp` y un frontend de `React`, permitiendo el entrenamiento de modelos en GPU y la visualización de simulaciones en tiempo real.

## Cómo Empezar

### Requisitos
- Python 3.10+
- Node.js y npm

### Instalación
1.  **Backend:**
    ```bash
    python3 -m venv ath_venv
    source ath_venv/bin/activate
    pip install -r requirements.txt
    ```
2.  **Frontend:**
    ```bash
    cd frontend
    npm install
    npm run build
    cd ..
    ```

### Ejecutar la Aplicación
El único punto de entrada es `run_server.py`.

```bash
source ath_venv/bin/activate
export AETHERIA_ENV=development
python3 run_server.py
```
La aplicación estará disponible en `http://localhost:8000`.