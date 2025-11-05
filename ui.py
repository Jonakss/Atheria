# ui.py
import streamlit as st
import streamlit.components.v1 as components
import os
import re

# Esta función de plantilla se usa en la pestaña "Visor"
def load_html_template(file_name: str) -> str:
    """Carga un archivo HTML (ej. 'viewer.html') y lo devuelve como un string."""
    try:
        template_path = os.path.join(os.path.dirname(__file__), file_name)
        with open(template_path, "r", encoding="utf-8") as f:
            return f.read()
    except Exception:
        return "<h2>Error: No se encontró 'viewer.html'.</h2>"

def render_lab_ui(app_state):
    """
    Renderiza la UI completa del Laboratorio con pestañas.
    'app_state' es el LightningFlow (AetheriaLab).
    """
    st.set_page_config(page_title="AETHERIA Lab", layout="wide")
    st.title("Laboratorio AETHERIA")

    tab_visor, tab_lab = st.tabs(["Visor en Vivo", "Laboratorio (Control)"])

    # -----------------------------------------------------------------
    # Pestaña 1: El Visor en Vivo
    # -----------------------------------------------------------------
    with tab_visor:
        st.header("Visor de Simulación en Tiempo Real")
        
        # --- Sidebar de Controles del Visor ---
        with st.sidebar:
            st.header("Controles del Visor")
            
            # 1. Selector de Visualización
            viz_options = ["change", "density", "phase", "magnitude", "channels"]
            current_viz_index = viz_options.index(app_state.viz_type) if app_state.viz_type in viz_options else 0
            
            selected_viz_type = st.selectbox(
                "Tipo de Visualización:",
                options=viz_options,
                index=current_viz_index
            )
            
            # Si el usuario cambia, actualiza el estado del Flow
            if selected_viz_type != app_state.viz_type:
                app_state.viz_type = selected_viz_type

            # 2. Controles de Pausa/Reset
            # (Estos comandos los envía el 'viewer.html', no el Flow)
            st.info("Los controles de Pausa/Reset están en el visor.")

        # --- Cargar el Visor HTML ---
        html_template = load_html_template("viewer.html")
        
        ws_url = ""
        if app_state.mode == "simulating" and app_state.simulation_server.url:
            ws_url = app_state.simulation_server.url.replace("http", "ws")
            ws_url = f"{ws_url}:{app_state.simulation_server.port}"
        
        if "{ws_url}" in html_template:
            html_content = html_template.replace("{ws_url}", ws_url)
        else:
            st.error("Error: 'viewer.html' no tiene el placeholder {ws_url}.")
            html_content = ""

        if app_state.mode != "simulating":
             st.info("El servidor de simulación no está activo. Inícialo desde la pestaña 'Laboratorio'.")
        
        components.html(html_content, height=950, scrolling=False)


    # -----------------------------------------------------------------
    # Pestaña 2: El Laboratorio
    # -----------------------------------------------------------------
    with tab_lab:
        st.header("Panel de Control Principal")
        
        # --- Sección de Estado Actual ---
        st.subheader("Estado del Sistema")
        
        current_status = app_state.mode.capitalize()
        if app_state.mode == "training" and not app_state.training_work.status.is_succeeded:
            current_status = "Entrenando..."
            st.info("El trabajo de entrenamiento se está ejecutando en la GPU. El servidor de simulación está detenido.")
            st.spinner("Entrenando...")
        
        st.metric("Estado Actual", current_status)
        
        col1, col2 = st.columns(2)
        
        # --- Sección de Entrenamiento ---
        with col1:
            st.subheader("Entrenamiento (Fase 5)")
            st.write("Inicia un nuevo trabajo de entrenamiento. Esto detendrá la simulación si se está ejecutando.")
            
            # Botón de Entrenamiento
            if st.button("🚀 Iniciar Entrenamiento", disabled=(app_state.mode == "training")):
                app_state.mode = "training" # ¡Le dice al Flow que entrene!
                if app_state.simulation_server.status.is_running:
                    app_state.simulation_server.stop()
            
            st.write("**Modelos Entrenados (.pth):**")
            st.dataframe(app_state.file_lister.training_models, use_container_width=True)

        # --- Sección de Simulación ---
        with col2:
            st.subheader("Simulación (Fase 7)")
            st.write("Inicia el servidor de simulación en vivo.")
            
            # 1. Selector de Modelo
            model_options = list(app_state.file_lister.training_models)
            if not model_options:
                st.warning("No se han encontrado modelos. Entrena un modelo primero.")
            
            selected_model = st.selectbox(
                "Elige un Modelo (.pth) para cargar:",
                options=model_options,
                index=0 if model_options else -1
            )
            
            # 2. Selector de Estado
            state_options = ["(Empezar desde cero)"] + list(app_state.file_lister.sim_states)
            selected_state_file = st.selectbox(
                "Elige un Estado (.pth) para reanudar (opcional):",
                options=state_options,
                index=0
            )
            
            # Botón de Simulación
            if st.button("🛰️ Iniciar Servidor de Simulación", disabled=(app_state.mode == "simulating" or not selected_model)):
                app_state.mode = "simulating" # ¡Le dice al Flow que simule!
                app_state.selected_model = os.path.join("output/training_checkpoints", selected_model)
                
                # Extraer el número de paso del nombre del archivo
                if selected_state_file != "(Empezar desde cero)":
                    try:
                        match = re.search(r"step_(\d+)\.pth", selected_state_file)
                        app_state.start_step = int(match.group(1)) if match else 0
                        app_state.selected_state = os.path.join("output/simulation_checkpoints", selected_state_file)
                    except Exception:
                        app_state.start_step = 0
                else:
                    app_state.start_step = 0
                
                if app_state.training_work.status.is_running:
                    app_state.training_work.stop()