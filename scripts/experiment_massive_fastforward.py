import os
import sys
import torch
import torch.nn as nn
import numpy as np
import logging
import matplotlib.pyplot as plt

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.models.unet_unitary import UNetUnitary
from scripts.experiment_holographic_layer import HolographicConv2d

# Configure Logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def main():
    print("🚀 Iniciando EXP-007: Massive Fast Forward (1 Millón de Pasos)")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 1. Cargar Checkpoint Entrenado
    checkpoint_path = "output/UNET_UNITARIA-D8-H32-G64-LR1e-4/output/checkpoints/UNetUnitary_G64_Eps130_1762992979_FINAL.pth"
    
    if not os.path.exists(checkpoint_path):
        logging.error(f"❌ Checkpoint no encontrado: {checkpoint_path}")
        return

    logging.info(f"📂 Cargando modelo desde: {checkpoint_path}")
    
    # Instanciar modelo (parámetros deben coincidir con el nombre del checkpoint D8-H32-G64)
    # D8 -> d_state=8
    # H32 -> hidden_channels=32
    d_state = 8
    hidden_channels = 32
    model = UNetUnitary(d_state=d_state, hidden_channels=hidden_channels).to(device)
    
    try:
        checkpoint = torch.load(checkpoint_path, map_location=device)
        # Manejar si el checkpoint es el state_dict directo o un dict completo
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        else:
            model.load_state_dict(checkpoint)
        logging.info("✅ Modelo cargado exitosamente.")
    except Exception as e:
        logging.error(f"❌ Error cargando pesos: {e}")
        return

    model.eval()

    # 2. Extraer Operador Efectivo (Linearización Holográfica)
    logging.info("🔮 Extrayendo Operador Holográfico Efectivo...")
    
    grid_size = 64 # Según el checkpoint G64
    
    # Usamos un estado de prueba (onda plana + ruido para excitar todos los modos)
    # Input shape: [B, 2*d_state, H, W] (Real + Imag concatenados)
    # Para simplificar la extracción de un operador escalar efectivo por canal espacial,
    # usaremos un solo canal activo o promediaremos.
    # Vamos a excitar el primer canal (Real de psi_0) y ver respuesta.
    probe_input = torch.randn(1, 2 * d_state, grid_size, grid_size, device=device)
    probe_input = probe_input / torch.norm(probe_input) # Normalizar
    
    with torch.no_grad():
        # UNetUnitary retorna delta_psi, así que psi_next = psi + delta_psi
        delta_psi = model(probe_input)
        probe_output = probe_input + delta_psi
    
    # Pasar al dominio de la frecuencia (usando FFT clásica para extracción rápida)
    # Tomamos el primer par (Real, Imag) correspondiente al primer estado del campo
    # Asumimos que d_state=8 son 8 campos independientes o acoplados.
    # Para la demo holográfica 1-channel, colapsamos a 1 canal complejo representativo.
    input_complex = torch.complex(probe_input[:, 0], probe_input[:, 1])
    output_complex = torch.complex(probe_output[:, 0], probe_output[:, 1])
    
    input_freq = torch.fft.fft2(input_complex)
    output_freq = torch.fft.fft2(output_complex)
    
    # Calcular función de transferencia H(k) = Y(k) / X(k)
    # Evitar división por cero
    epsilon = 1e-8
    transfer_function = output_freq / (input_freq + epsilon)
    
    # 3. Massive Fast Forward (Potenciación)
    # U(t)^N -> H(k)^N
    N_STEPS = 1_000_000
    logging.info(f"⏩ Calculando Fast Forward de {N_STEPS} pasos...")
    
    # Potenciación eficiente: H^N = exp(N * log(H)) o simplemente potencia elemento a elemento
    # Como es complejo, esto rota la fase N veces y escala la magnitud N veces.
    # Si el modelo es estable/unitario, la magnitud de H debería ser ~1.
    
    # Forzamos unitariedad para estabilidad a largo plazo (opcional, pero recomendado para física)
    # H_unitary = H / |H|
    transfer_function_unitary = transfer_function / (transfer_function.abs() + epsilon)
    
    # H_final = H_unitary ^ N
    # Usamos coordenadas polares para precisión: r^N * exp(i * N * theta)
    r = transfer_function_unitary.abs()
    theta = transfer_function_unitary.angle()
    
    r_final = r.pow(N_STEPS)
    theta_final = theta * N_STEPS
    
    transfer_function_final = torch.polar(r_final, theta_final)
    
    # 4. Crear Capa Holográfica con el Operador Final
    # Usamos nuestra clase HolographicConv2d pero inyectamos los pesos calculados
    holo_layer = HolographicConv2d(in_channels=1, out_channels=1, grid_size=grid_size, device=device)
    
    # Adaptar shape: [Out, In, H, W]. Aquí es 1-to-1 mapping del campo complejo.
    # HolographicConv2d espera pesos complejos.
    # Nuestra transfer_function es [1, H, W] (Batch=1).
    # Asignamos a weights_freq [1, 1, H, W]
    with torch.no_grad():
        holo_layer.weights_freq.data = transfer_function_final.unsqueeze(0).unsqueeze(0)
        
    # 5. Ejecutar Simulación (Input Real -> Fast Forward -> Output)
    logging.info("⚡ Ejecutando Simulación Holográfica...")
    
    # Estado inicial: Pulso Gaussiano en el centro
    x = torch.linspace(-1, 1, grid_size, device=device)
    y = torch.linspace(-1, 1, grid_size, device=device)
    Y, X = torch.meshgrid(y, x, indexing='ij')
    R = torch.sqrt(X**2 + Y**2)
    initial_state_spatial = torch.exp(-R**2 / 0.1).unsqueeze(0).unsqueeze(0) # [1, 1, H, W]
    
    # Forward Pass (usa QFT/IQFT internamente)
    final_state_spatial = holo_layer(initial_state_spatial)
    
    # 6. Guardar Checkpoint
    save_path = "output/checkpoints/fastforward_1M.pt"
    os.makedirs("checkpoints", exist_ok=True)
    
    torch.save({
        'steps': N_STEPS,
        'initial_state': initial_state_spatial,
        'final_state': final_state_spatial,
        'transfer_function': transfer_function_final,
        'description': 'Massive Fast Forward (1M steps) using Holographic Layer'
    }, save_path)
    
    logging.info(f"💾 Resultado guardado en: {save_path}")
    
    # Métricas
    input_energy = initial_state_spatial.abs().pow(2).sum()
    output_energy = final_state_spatial.abs().pow(2).sum()
    logging.info(f"📊 Energía Inicial: {input_energy:.2f}")
    logging.info(f"📊 Energía Final (1M pasos): {output_energy:.2f}")
    
    if output_energy > input_energy * 1.1:
        logging.warning("⚠️ Advertencia: La energía creció significativamente (posible inestabilidad).")
    elif output_energy < input_energy * 0.9:
        logging.warning("⚠️ Advertencia: La energía disipó significativamente.")
    else:
        logging.info("✅ Conservación de energía aceptable.")

if __name__ == "__main__":
    main()
