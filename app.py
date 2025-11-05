# app.py
import lightning.app as la
from lightning.app.compute import CloudCompute
from lightning.app.structures import List
import os
import sys
import glob
import time
import asyncio # Necesario para el run() del SimulationServer

# --- Configuración del Path (Importante) ---
script_dir = os.path.dirname(os.path.abspath(__file__))
src_dir = os.path.join(script_dir, "src")
if src_dir not in sys.path:
    sys.path.append(src_dir)

# Importa la lógica de tus pipelines
try:
    from src.pipeline_server import run_server_pipeline
    from src.pipeline_train import run_training_pipeline
    from src import config as cfg # Para rutas de carpetas
except ImportError as e:
    print(f"Error: No se pudieron importar los módulos desde 'src'. Verifica tu 'src/__init__.py'.")
    print(f"Error: {e}")
    sys.exit(1)


class TrainingWork(la.LightningWork):
    """Este Work ejecuta el pipeline de entrenamiento (Fase 5) UNA VEZ."""
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.model_path = "" # Expondrá la ruta al modelo final
        
    def run(self):
        print("\n" + "="*30)
        print("🚀 INICIANDO TRABAJO DE ENTRENAMIENTO...")
        print("="*30)
        # run_training_pipeline es síncrono, se ejecuta y termina.
        # Pasa None para que el pipeline busque el último checkpoint si es necesario
        _, self.model_path = run_training_pipeline() 
        print(f"✅ ENTRENAMIENTO FINALIZADO. Modelo en: {self.model_path}")


class SimulationServer(la.LightningWork):
    """Este Work ejecuta el pipeline del servidor (Fase 7) INDEFINIDAMENTE."""
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.host = "0.0.0.0" 
        self.port = cfg.WEBSOCKET_PORT
        # El Flow (gerente) controlará estas variables
        self.viz_type = "change"
        self.pause_event_set = True # True = Resumido, False = Pausado
        self.reset_event_set = False

    def run(self, model_path: str, start_step: int):
        print(f"🚀 Iniciando servidor de simulación en {self.host}:{self.port}")
        # Pasa el 'self' (el Work) al pipeline para control interactivo
        asyncio.run(run_server_pipeline(M_FILENAME=model_path, work=self, initial_step=start_step))


class FileLister(la.LightningWork):
    """Este Work revisa el disco cada 30s y publica la lista de checkpoints."""
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.training_models = List()
        self.sim_states = List()

    def run(self):
        while True:
            # 1. Listar modelos de entrenamiento (.pth)
            train_files = glob.glob(os.path.join(cfg.CHECKPOINT_DIR, "*.pth"))
            train_files.sort(key=os.path.getmtime, reverse=True)
            self.training_models = [os.path.basename(f) for f in train_files]
            
            # 2. Listar estados de simulación (.pth)
            sim_files = glob.glob(os.path.join(cfg.LARGE_SIM_CHECKPOINT_DIR, "*.pth"))
            sim_files.sort(key=os.path.getmtime, reverse=True)
            self.sim_states = [os.path.basename(f) for f in sim_files]
            
            time.sleep(30) # Esperar 30 segundos


# 2. El Orquestador (Flow) y Frontend
class AetheriaLab(la.LightningFlow):
    def __init__(self):
        super().__init__()
        
        # --- Estado del Laboratorio (El "cerebro" del gerente) ---
        self.mode = "idle" # 'idle', 'training', 'simulating'
        self.selected_model = "" # El .pth de entrenamiento a usar
        self.selected_state = "" # El .pth de simulación a cargar
        self.start_step = 0
        self.viz_type = "change"
        
        # --- Componentes (Workers) ---
        self.file_lister = FileLister(parallel=True)
        self.training_work = TrainingWork(cloud_compute=la.CloudCompute("gpu-t4"))
        self.simulation_server = SimulationServer(cloud_compute=la.CloudCompute("gpu-t4"))
        
    def run(self):
        # 1. Siempre ejecutar el listador de archivos
        self.file_lister.run()

        # 2. Lógica de cambio de modo
        if self.mode == "training":
            self.training_work.run()
            # Cuando termina, vuelve a 'idle' y actualiza el modelo
            if self.training_work.status.is_succeeded:
                print(f"Flow: Entrenamiento terminado. Modelo: {self.training_work.model_path}")
                self.selected_model = self.training_work.model_path
                self.mode = "idle" # Vuelve al modo 'idle'
        
        elif self.mode == "simulating":
            # Pasa el modelo y el estado seleccionados al servidor
            self.simulation_server.run(
                model_path=self.selected_model,
                start_step=self.start_step
            )
            
            # Sincroniza la UI con el backend
            self.simulation_server.viz_type = self.viz_type
            
    def configure_layout(self):
        # Le dice a Lightning que use 'ui.py' como el frontend
        return la.frontend.StreamlitFrontend(render_fn=render_ui_entrypoint)

def render_ui_entrypoint(app_state: AetheriaLab):
    """
    Punto de entrada que Streamlit usa para renderizar la UI.
    """
    # Importa la función de renderizado de tu UI
    # (El import se hace aquí para que se recargue en caliente)
    from ui import render_lab_ui

    # Renderiza la app, pasándole el app_state completo
    render_lab_ui(app_state)


# Punto de entrada para ejecutar la app
app = la.LightningApp(AetheriaLab())