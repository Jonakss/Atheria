# src/server_state.py
import asyncio
from . import config as cfg

class GlobalState:
    """Una clase para mantener el estado global del servidor de forma organizada."""
    def __init__(self):
        # --- Estado del Servidor de Laboratorio ---
        self.training_process = None
        self.clients = set()  # Clientes conectados al WebSocket principal

        # --- Estado de la Simulación ---
        self.simulation_tasks = []
        self.sim_is_paused = False
        self.sim_viz_type = 'density'
        self.sim_motor = None
        self.sim_model = None
        self.sim_step_count = 0
        self.sim_connected_clients = {}  # {ws_aiohttp: {"viewport": ...}}

g_state = GlobalState()
