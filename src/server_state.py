# src/server_state.py

# Estado global de la aplicación
g_state = {
    "websockets": {},
    "training_process": None,
    "motor": None,
    "is_paused": True,
    "simulation_step": 0,
    "viz_type": "density",
    "inference_running": False,
}