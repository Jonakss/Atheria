# app.py
import lightning.app as la  # <--- CAMBIO AQUÍ
from lightning.app.compute import CloudCompute  # <--- CAMBIO AQUÍ
import os
import sys
import asyncio

# --- Configuración del Path (Importante) ---
# Añade 'src' al path para que Lightning pueda encontrar tus módulos
script_dir = os.path.dirname(os.path.abspath(__file__))
src_dir = os.path.join(script_dir, "src")
if src_dir not in sys.path:
    sys.path.append(src_dir)

# Importa la lógica de tu servidor desde 'src'
try:
    from src.pipeline_server import run_server_pipeline
    from src.config import WEBSOCKET_PORT
except ImportError as e:
    print(f"Error: No se pudieron importar los módulos desde 'src'. Verifica tu 'src/__init__.py'.")
    print(f"Error: {e}")
    sys.exit(1)


# 1. El Backend (Computación)
# Esto corre en su propia máquina (¡con GPU!)
class SimulationServer(la.LightningWork):  # <--- CAMBIO AQUÍ
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # El servidor debe escuchar en 0.0.0.0 para aceptar conexiones externas
        self.host = "0.0.0.0" 
        self.port = WEBSOCKET_PORT # Usa el puerto de tu config (ej. 8765)

    def run(self):
        """Este método se ejecuta en la máquina remota."""
        print(f"🚀 Iniciando servidor de simulación en {self.host}:{self.port}")
        
        # NOTA: M_FILENAME se pasa como None.
        # Tu 'run_server_pipeline' (en src/pipeline_server.py)
        # ya tiene la lógica para encontrar el último modelo si M_FILENAME es None.
        asyncio.run(run_server_pipeline(M_FILENAME=None))


# 2. El Orquestador (Flow) y Frontend
# Esto corre en una máquina pequeña, sirve la UI y maneja el estado.
class AetheriaApp(la.LightningFlow):  # <--- CAMBIO AQUÍ
    def __init__(self):
        super().__init__()
        # Inicia el backend, pidiendo una GPU (ej. A10G)
        self.backend = SimulationServer(
            cloud_compute=CloudCompute("gpu-T4")  # <--- CAMBIO AQUÍ
        )
        
    def run(self):
        # Lanza el trabajo del backend.
        # Lightning se encarga de iniciarlo en la nube.
        self.backend.run()

    def configure_layout(self):
        # Define el frontend que se mostrará en el navegador.
        # Le decimos que use el script 'ui.py'
        return la.frontend.StreamlitFrontend(render_fn=self_render_fn)  # <--- CAMBIO AQUÍ

def self_render_fn(app_state: AetheriaApp):
    """
    Esta función es llamada por Streamlit para renderizar la UI.
    Actúa como un puente para pasar el estado del Flow (app_state) a la UI.
    """
    # Importa la función de renderizado de tu UI
    # (El import se hace aquí para que se recargue en caliente)
    from ui import render_app 

    # --- ¡Magia de Lightning! ---
    # Obtenemos la URL interna donde el backend está corriendo
    # y la pasamos a nuestra función de renderizado de la UI.
    
    # El backend.url será algo como 'http://10.X.X.X'
    if not app_state.backend.url:
        import streamlit as st
        st.set_page_config(page_title="Visor AETHERIA", layout="wide")
        st.info("🚀 Iniciando el backend en la GPU... por favor espera (puede tardar ~1-2 min la primera vez).")
        st.spinner("Esperando que el servidor de simulación esté listo...")
        return

    # Convertimos la URL HTTP a una URL de WebSocket
    ws_url = app_state.backend.url.replace("http", "ws")
    
    # Renderiza la app, pasándole la URL dinámica
    render_app(ws_url=f"{ws_url}:{app_state.backend.port}")


# Punto de entrada para ejecutar la app con: lightning run app app.py
app = la.LightningApp(AetheriaApp())  # <--- CAMBIO AQUÍ