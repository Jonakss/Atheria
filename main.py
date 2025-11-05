# main.py
import asyncio
import os
import sys

# --- Configuración del Path ---
script_dir = os.path.dirname(os.path.abspath(__file__))
src_dir = os.path.join(script_dir, "src")
if src_dir not in sys.path:
    sys.path.append(src_dir)

# Ahora podemos importar de forma segura desde 'src'
try:
    import src.config as cfg
    from src.pipeline_train import run_training_pipeline
    from src.pipeline_viz import run_visualization_pipeline
    from src.pipeline_server import run_server_pipeline
    from src.qca_engine import Aetheria_Motor # Para type hinting
except ImportError as e:
    print(f"Error: No se pudieron importar los módulos desde 'src'.")
    print(f"Asegúrate que la carpeta 'src' existe y contiene un archivo '__init__.py'.")
    print(f"Directorio 'src' buscado: {src_dir}")
    print(f"Detalle del error: {e}")
    sys.exit(1)


async def main_pipeline():
    """
    Orquesta el pipeline completo de AETHERIA.
    """
    print("--- INICIANDO EJECUCIÓN DEL PIPELINE AETHERIA ---")
    
    Aetheria_Motor_Train: Aetheria_Motor | None = None
    M_FILENAME: str | None = None

    # --------------------------------------------------------------------------
    # FASE 5: LÓGICA PRINCIPAL DE ENTRENAMIENTO (Síncrono)
    # --------------------------------------------------------------------------
    if cfg.RUN_TRAINING:
        Aetheria_Motor_Train, M_FILENAME = run_training_pipeline()
    else:
        print("\n>>> FASE DE ENTRENAMIENTO (FASE 5) OMITIDA <<<")

    # --------------------------------------------------------------------------
    # FASE 6: VISUALIZACIÓN POST-ENTRENAMIENTO (Síncrono)
    # --------------------------------------------------------------------------
    if cfg.RUN_POST_TRAINING_VIZ:
        run_visualization_pipeline(Aetheria_Motor_Train, M_FILENAME)
    else:
        print("\n>>> FASE DE VISUALIZACIÓN POST-ENTRENAMIENTO (FASE 6) OMITIDA <<<")


    # --------------------------------------------------------------------------
    # FASE 7: LÓGICA DE SIMULACIÓN GRANDE (Asíncrono/Servidor)
    # --------------------------------------------------------------------------
    if cfg.RUN_LARGE_SIM:
        
        # --- ¡¡MODIFICACIÓN IMPORTANTE!! ---
        # Pasamos 'work=None' explícitamente para ejecutar en modo local/agnóstico
        await run_server_pipeline(M_FILENAME, work=None)
        
    else:
        print("\n>>> FASE DE SIMULACIÓN GRANDE (FASE 7) OMITIDA <<<")

    print("\n--- EJECUCIÓN DEL PIPELINE AETHERIA FINALIZADA ---")


# ==============================================================================
# PUNTO DE ENTRADA DEL SCRIPT
# ==============================================================================
if __name__ == "__main__":
    try:
        asyncio.run(main_pipeline())
    except KeyboardInterrupt:
        print("\n🛑 Proceso principal interrumpido por el usuario.")