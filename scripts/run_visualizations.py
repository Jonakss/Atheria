# /home/jonathan.correa/Projects/Atheria/run_visualizations.py
#
# Este script carga el modelo U-Net, genera datos de ejemplo y utiliza
# las herramientas de visualización para crear gráficos.

import torch
import os
import sys

# --- Configuración del Path ---
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
src_dir = os.path.join(PROJECT_ROOT, "src")
if src_dir not in sys.path:
    sys.path.append(src_dir)

from src.visualization_tools import visualize_poincare_disk, visualize_3d_slices
from src.qca_operator_unet import QCA_Operator_UNet
from src import config as cfg # <-- ¡NUEVO!

def run_visualization_pipeline():
    """
    Función principal para ejecutar el pipeline de visualización.
    """
    print("--- INICIANDO PIPELINE DE VISUALIZACIÓN ---")

    # 1. Configurar y cargar el modelo U-Net
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Usando dispositivo: {device}")

    try:
        # Corregido: Usar parámetros desde src/config.py
        model = QCA_Operator_UNet(
            d_state=cfg.D_STATE, 
            hidden_channels=cfg.HIDDEN_CHANNELS
        ).to(device)
        print("Modelo U-Net cargado exitosamente con la configuración de 'src/config.py'.")
    except Exception as e:
        print(f"Error al cargar QCA_Operator_UNet: {e}")
        print("Asegúrate de que 'src/qca_operator_unet.py' existe y es correcto.")
        return

    # Hook para capturar la salida del bottleneck
    latent_space = None
    def hook(module, input, output):
        nonlocal latent_space
        latent_space = output

    # Adjuntar el hook a la capa bottleneck del modelo U-Net
    try:
        # Corregido: La capa se llama 'bot', no 'bottleneck'
        model.bot.register_forward_hook(hook)
        print("Hook registrado en la capa 'bot' de la U-Net.")
    except AttributeError:
        print("ERROR: El modelo U-Net no tiene un atributo 'bot'.")
        print("No se podrá generar la visualización de Poincaré.")
        return

    # 2. Generar un tensor de datos de ejemplo
    # El modelo espera una entrada con shape [B, 2 * d_state, H, W]
    batch_size = 64
    # Usar tamaño de config para inferencia
    height, width = cfg.GRID_SIZE_INFERENCE, cfg.GRID_SIZE_INFERENCE
    input_tensor = torch.randn(batch_size, 2 * cfg.D_STATE, height, width).to(device)
    print(f"Tensor de entrada de ejemplo generado con shape: {input_tensor.shape}")

    # 3. Pasar los datos por el modelo para capturar el espacio latente y la salida
    with torch.no_grad():
        # El modelo devuelve dos tensores: delta_real y delta_imag
        delta_real, delta_imag = model(input_tensor)
    
    # Concatenamos la salida para visualización 3D
    output_tensor_for_viz = torch.stack([delta_real.squeeze(0), delta_imag.squeeze(0)], dim=0)
    print(f"Tensor de salida (real, imag) generado con shapes: {delta_real.shape}, {delta_imag.shape}")

    # 4. Generar visualizaciones
    if latent_space is not None:
        energy_values = torch.mean(torch.abs(input_tensor), dim=(1, 2, 3)).cpu().numpy()
        visualize_poincare_disk(
            latent_space,
            energy_values=energy_values, 
            title="Espacio Latente de la U-Net (Disco de Poincaré)"
        )
    
    # Visualizar los canales de salida (real, imag) del primer elemento del batch
    visualize_3d_slices(
        output_tensor_for_viz.unsqueeze(0), # Añadimos batch dim para que coincida
        title="Canales de Salida de la U-Net (real/imag)"
    )

    print("--- PIPELINE DE VISUALIZACIÓN COMPLETADO ---")

if __name__ == "__main__":
    run_visualization_pipeline()
