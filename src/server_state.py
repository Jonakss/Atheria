# src/server_state.py

# --- Estado Global del Servidor ---
# Este diccionario contiene el estado en tiempo real de la aplicación.
# Es accesible desde todos los módulos que lo importan.

g_state = {
    # --- ¡¡CORRECCIÓN CLAVE!! Usar un diccionario para los websockets ---
    # Esto permite asociar cada websocket con un ID único.
    'websockets': {},
    
    'training_process': None,
    'simulation_running': False,
    'motor': None,
    'viz_type': 'density_map', # Valor por defecto
}