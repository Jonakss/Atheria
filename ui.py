# ui.py
import streamlit as st
import streamlit.components.v1 as components

def render_app(ws_url: str):
    """
    Renderiza el visor de AETHERIA usando Streamlit components.
    Recibe la URL del WebSocket (ws_url) desde el Lightning Flow.
    """
    
    # Inyecta la URL del backend directamente en tu código HTML/JavaScript
    html_content = f"""
    <!DOCTYPE html>
    <html lang="es">
    <head>
        <meta charset="UTF-8">
        <title>Visor AETHERIA</title>
        <style>
            /* Estilos para que se vea bien dentro de Streamlit */
            body {{ font-family: system-ui, sans-serif; background: #111; color: #eee; margin: 0; padding: 0; }}
            #container {{ text-align: center; padding-top: 1rem; }}
            h1 {{ font-weight: 300; color: #eee; }}
            #simCanvas {{
                width: 90vmin; /* Escala la imagen */
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
            <h1>Visor de Simulación AETHERIA</h1>
            <canvas id="simCanvas" width="100" height="100"></canvas>
            <div id="status">Conectando a {ws_url}...</div>
            <div id="stepCounter">Paso: 0</div>
        </div>

        <script>
            // Tu código JS original, sin cambios,
            // excepto por la variable WEBSOCKET_URL
            
            const canvas = document.getElementById('simCanvas');
            const ctx = canvas.getContext('2d');
            const statusEl = document.getElementById('status');
            const stepCounterEl = document.getElementById('stepCounter');

            // --- ¡AQUÍ ESTÁ LA MAGIA! ---
            // Usamos la URL que Streamlit/Lightning inyectó
            const WEBSOCKET_URL = "wss://8765-01k95p45n43bdtt9r4rzw4a5z1.cloudspaces.litng.ai"; 
            
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
                statusEl.textContent = "Error de conexión. ¿Está el backend corriendo?";
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
    
    # Configura la página de Streamlit
    st.set_page_config(page_title="Visor AETHERIA", layout="wide")
    
    # Renderiza el componente HTML
    components.html(html_content, height=900, scrolling=True)

# Esto permite probar el UI localmente si se desea, 
# aunque fallará al conectar sin un backend.
if __name__ == "__main__":
    render_app(ws_url="ws://127.0.0.1:8765")