# ui.py
import streamlit as st
import streamlit.components.v1 as components

# --- NUEVO: Lista de todas las visualizaciones posibles ---
VIZ_OPTIONS = [
    "change",
    "density",
    "phase",
    "magnitude",
    "channels"
]

def render_app(app_state, ws_url: str):
    """
    Renderiza el visor de AETHERIA usando Streamlit components.
    
    Args:
        app_state: El objeto LightningFlow (AetheriaApp).
        ws_url: La URL del WebSocket del backend.
    """
    
    st.set_page_config(page_title="Visor AETHERIA", layout="wide")
    
    # --- NUEVO: Sidebar para controles ---
    with st.sidebar:
        st.image("https://storage.googleapis.com/lightning-ai-docs/app/components/logos/A-L-A-logo-color-ondark.png", width=200)
        st.header("Controles de AETHERIA")
        
        # El valor actual se lee desde el estado del Flow
        current_index = VIZ_OPTIONS.index(app_state.viz_type)
        
        # --- El Menú Desplegable ---
        selected_viz_type = st.selectbox(
            "Tipo de Visualización",
            options=VIZ_OPTIONS,
            index=current_index
        )
        
        # --- Aquí está la magia (UI -> Flow) ---
        # Si el usuario cambia la selección, actualizamos el estado del Flow.
        if selected_viz_type != app_state.viz_type:
            app_state.viz_type = selected_viz_type
            # No es necesario hacer nada más, el Flow lo detectará
            # y se lo comunicará al backend automáticamente.
        
        st.info("La simulación se ejecuta en un backend de GPU. El visor solo recibe los frames.")

    # El HTML/JS del visor (el mismo de antes)
    html_content = f"""
    <!DOCTYPE html>
    <html lang="es">
    <head>
        <meta charset="UTF-8">
        <title>Visor AETHERIA</title>
        <style>
            body {{ font-family: system-ui, sans-serif; background: #111; color: #eee; margin: 0; padding: 0; }}
            #container {{ text-align: center; padding-top: 1rem; }}
            /* Quita el h1 del HTML, lo manejamos con Streamlit */
            #simCanvas {{
                width: 90vmin;
                height: 90vmin;
                max-width: 800px;
                max-height: 800px;
                background: #000;
                border: 1px solid #555;
                image-rendering: pixelated; 
            }}
            #status {{ font-size: 1.2rem; margin-top: 1rem; color: #eee; }}
            #status.connected {{ color: #4ade80; }}
            #status.error {{ color: #f87171; }}
            #stepCounter {{ color: #aaa; }}
        </style>
    </head>
    <body>
        <div id="container">
            <canvas id="simCanvas" width="100" height="100"></canvas>
            <div id="status">Conectando a {ws_url}...</div>
            <div id="stepCounter">Paso: 0</div>
        </div>

        <script>
            const canvas = document.getElementById('simCanvas');
            const ctx = canvas.getContext('2d');
            const statusEl = document.getElementById('status');
            const stepCounterEl = document.getElementById('stepCounter');
            const WEBSOCKET_URL = "{ws_url}"; 
            
            console.log("Conectando a:", WEBSOCKET_URL);

            const ws = new WebSocket(WEBSOCKET_URL);
            const img = new Image();
            
            img.onload = () => {{
                if (canvas.width !== img.width || canvas.height !== img.height) {{
                    canvas.width = img.width;
                    canvas.height = img.height;
                }}
                ctx.drawImage(img, 0, 0, canvas.width, canvas.height);
            }};

            ws.onopen = () => {{
                console.log("Conectado al servidor AETHERIA.");
                statusEl.textContent = "Conectado";
                statusEl.className = "connected";
            }};
            ws.onclose = () => {{
                console.log("Desconectado del servidor.");
                statusEl.textContent = "Desconectado.";
                statusEl.className = "error";
            }};
            ws.onerror = (err) => {{
                console.error("Error de WebSocket:", err);
                statusEl.textContent = "Error de conexión.";
                statusEl.className = "error";
            }};

            ws.onmessage = (event) => {{
                try {{
                    const data = JSON.parse(event.data);
                    stepCounterEl.textContent = `Paso: ${{data.step}} (Tipo: ${{data.frame_type}})`;
                    img.src = data.image_data;
                }} catch (e) {{
                    console.error("Error procesando mensaje:", e, event.data);
                }}
            }};
        </script>
    </body>
    </html>
    """
    
    # Título principal de la app Streamlit
    st.title("Visor de Simulación AETHERIA")
    
    # Renderiza el componente HTML
    components.html(html_content, height=900, scrolling=False)

# Esto permite probar el UI localmente (sin backend)
if __name__ == "__main__":
    # Simula un 'app_state' falso para la prueba
    class FakeAppState:
        viz_type = "change"
        backend = None
    
    render_app(app_state=FakeAppState(), ws_url="ws://127.0.0.1:8765")