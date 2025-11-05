# app.py
import lightning.app as la
from lightning.app.compute import CloudCompute
import os
import sys
import asyncio

# --- Configuración del Path (Importante) ---
script_dir = os.path.dirname(os.path.abspath(__file__))
src_dir = os.path.join(script_dir, "src")
if src_dir not in sys.path:
    sys.path.append(src_dir)

# Importa la lógica de tu servidor desde 'src'
try:
    # Ahora importamos la función 'run_server_pipeline'
    from src.pipeline_server import run_server_pipeline
    from src.config import WEBSOCKET_PORT
except ImportError as e:
    print(f"Error: No se pudieron importar los módulos desde 'src'. Verifica tu 'src/__init__.py'.")
    print(f"Error: {e}")
    sys.exit(1)


# 1. El Backend (Computación)
class SimulationServer(la.LightningWork):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.host = "0.0.0.0" 
        self.port = WEBSOCKET_PORT
        
        # --- NUEVO: Estado del Backend ---
        # El Flow controlará esta variable
        self.viz_type = "change" # Valor inicial por defecto

    def run(self):
        """Este método se ejecuta en la máquina remota."""
        print(f"🚀 Iniciando servidor de simulación en {self.host}:{self.port}")
        
        # --- MODIFICADO: Pasamos 'self' (el Work) al pipeline ---
        # Esto permite que el pipeline lea self.viz_type en tiempo real
        asyncio.run(run_server_pipeline(M_FILENAME=None, work=self))


# 2. El Orquestador (Flow) y Frontend
class AetheriaApp(la.LightningFlow):
    def __init__(self):
        super().__init__()
        
        # --- NUEVO: Estado del Flow ---
        # Este es el "estado maestro" que la UI modificará
        self.viz_type = "change" # Valor inicial por defecto
        
        self.backend = SimulationServer(
            cloud_compute=CloudCompute("gpu-a10g") 
        )
        
    def run(self):
        # --- NUEVO: Sincronización de Estado (Flow -> Work) ---
        # Copia el estado maestro del Flow al estado del Work
        # Si el usuario cambia self.viz_type en la UI,
        # esto lo enviará al backend en el próximo ciclo.
        self.backend.viz_type = self.viz_type
        
        # Inicia el backend
        self.backend.run()

    def configure_layout(self):
        # Define el frontend
        return la.frontend.StreamlitFrontend(render_fn=self_render_fn)

def self_render_fn(app_state: AetheriaApp):
    """
    Esta función se ejecuta para renderizar la UI de Streamlit.
    Pasa el 'app_state' (el Flow) a la función de renderizado.
    """
    # Importa la función de renderizado de tu UI
    from ui import render_app 

    if not app_state.backend.url:
        import streamlit as st
        st.set_page_config(page_title="Visor AETHERIA", layout="wide")
        st.info("🚀 Iniciando el backend en la GPU... por favor espera...")
        st.spinner("Esperando que el servidor de simulación esté listo...")
        return

    # Convertimos la URL HTTP a una URL de WebSocket
    ws_url = app_state.backend.url.replace("http", "ws")
    
    # --- MODIFICADO: Pasamos el 'app_state' completo a la UI ---
    # Esto permite a la UI leer Y escribir en el estado del Flow
    render_app(
        app_state=app_state,
        ws_url=f"{ws_url}:{app_state.backend.port}"
    )


# Punto de entrada para ejecutar la app
app = la.LightningApp(AetheriaApp())