import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import logging
import matplotlib.pyplot as plt
from qiskit import QuantumCircuit

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src import config

# Configure Logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class QuantumNativeConv2d(nn.Module):
    """
    Capa de Convolución Cuántica Nativa (Hardware Efficient).
    En lugar de una matriz diagonal arbitraria (cara), usa un circuito parametrizado (PQC)
    con compuertas Rz (locales) y Rzz (entrelazamiento) que son nativas o baratas en IonQ.
    
    Ansatz:
    1. QFT (Fixed)
    2. PQC Layer:
       - Rz(theta_i) en cada qubit i
       - Rzz(phi_j) en qubits vecinos (i, i+1)
    3. IQFT (Fixed)
    """
    def __init__(self, grid_size=32, n_layers=1, device='cpu'):
        super().__init__()
        self.grid_size = grid_size
        self.n_qubits = int(np.log2(grid_size * grid_size))
        self.n_layers = n_layers
        self.device = device
        
        # Parámetros entrenables (Ángulos reales)
        # Rz por qubit por capa
        self.theta = nn.Parameter(torch.randn(n_layers, self.n_qubits) * 0.1)
        # Rzz por par vecino por capa (Topología lineal 1D para simplificar)
        # N-1 pares
        self.phi = nn.Parameter(torch.randn(n_layers, self.n_qubits - 1) * 0.1)
        
        self.to(device)

    def forward_pqc(self, x_freq):
        """
        Aplica el PQC en el dominio de la frecuencia.
        Simulación matemática eficiente (sin Qiskit loop en entrenamiento).
        
        Rz(theta) -> Phase shift e^(-i * theta / 2)
        Rzz(phi) -> Entrelazamiento e^(-i * phi / 2 * Z_i Z_j)
        
        Nota: Rzz es diagonal en la base Z, así que también es solo fase!
        Esto significa que nuestro ansatz (QFT -> Diagonal PQC -> IQFT) 
        sigue siendo una convolución, pero con un kernel restringido por la topología del circuito.
        """
        B, C, H, W = x_freq.shape
        # Aplanar espacialmente [B, C, N_PIXELS]
        x_flat = x_freq.view(B, C, -1)
        
        # Aplicar capas
        for l in range(self.n_layers):
            # 1. Rz (Local Phase)
            # theta: [N_QUBITS]. Pero aquí estamos en la base computacional (2^N estados).
            # Un Rz en el qubit k añade una fase dependiendo de si el bit k es 0 o 1.
            # Esto es equivalente a multiplicar por una diagonal construida a partir de los ángulos.
            
            # Construcción eficiente de la diagonal del PQC:
            # Estado base |k> = |b_n ... b_0>
            # Fase total = Sum_i (theta_i * (-1)^b_i) ... (aprox, Rz es e^-i*theta/2 * Z)
            # Z|0> = |0>, Z|1> = -|1>.
            
            # Vamos a construir el vector de fases completo (2^N) a partir de theta (N).
            # Esto es costoso (2^N) pero necesario para simular el forward pass clásico.
            # En hardware cuántico, esto es O(1) profundidad.
            
            # Generar todos los bitstrings (0 a 2^N - 1)
            indices = torch.arange(2**self.n_qubits, device=self.device)
            
            total_phase = torch.zeros(2**self.n_qubits, device=self.device)
            
            # Rz contributions
            for q in range(self.n_qubits):
                # Bit value (0 or 1) at position q
                bit_val = (indices >> q) & 1
                # Z eigenvalue: 0->+1, 1->-1. 
                # Rz(theta) = exp(-i * theta/2 * Z).
                # Phase: -theta/2 if 0, +theta/2 if 1. (Wait, Z|0>=|0>, Z|1>=-|1>. So exp(-i th/2 Z)|0> = e^-i th/2 |0>)
                z_val = 1.0 - 2.0 * bit_val.float() # 0->1, 1->-1
                total_phase += -0.5 * self.theta[l, q] * z_val
                
            # Rzz contributions (Nearest Neighbor)
            for q in range(self.n_qubits - 1):
                bit_val_1 = (indices >> q) & 1
                bit_val_2 = (indices >> (q+1)) & 1
                # ZZ eigenvalue: same->+1, diff->-1
                z1 = 1.0 - 2.0 * bit_val_1.float()
                z2 = 1.0 - 2.0 * bit_val_2.float()
                zz_val = z1 * z2
                total_phase += -0.5 * self.phi[l, q] * zz_val
            
            # Aplicar fase
            phase_factor = torch.exp(1j * total_phase)
            # Reshape para broadcast [1, 1, 2^N]
            phase_factor = phase_factor.view(1, 1, -1)
            
            x_flat = x_flat * phase_factor
            
        return x_flat.view(B, C, H, W)

    def forward(self, x_spatial):
        # 1. QFT (FFT Clásica para simulación)
        x_complex = torch.complex(x_spatial, torch.zeros_like(x_spatial)) if not x_spatial.is_complex() else x_spatial
        x_freq = torch.fft.fft2(x_complex)
        
        # 2. PQC (Holographic Mask parametrizada)
        x_freq_processed = self.forward_pqc(x_freq)
        
        # 3. IQFT
        x_spatial_out = torch.fft.ifft2(x_freq_processed)
        return x_spatial_out.real # Asumimos output real por ahora

    def export_circuit(self):
        """Genera el circuito Qiskit equivalente."""
        qc = QuantumCircuit(self.n_qubits)
        
        for l in range(self.n_layers):
            # Rz
            for q in range(self.n_qubits):
                theta_val = self.theta[l, q].item()
                qc.rz(theta_val, q)
            
            # Rzz
            for q in range(self.n_qubits - 1):
                phi_val = self.phi[l, q].item()
                qc.rzz(phi_val, q, q+1)
                
        return qc

def main():
    print("🚀 Iniciando EXP-008: Entrenamiento Cuántico Nativo")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 1. Cargar Target (Operador Fast Forward de EXP-007)
    # Si no existe, lo generamos sintéticamente para la demo.
    checkpoint_path = "checkpoints/fastforward_1M.pt"
    if os.path.exists(checkpoint_path):
        logging.info(f"📂 Cargando target desde {checkpoint_path}")
        cp = torch.load(checkpoint_path, map_location=device)
        # El transfer_function es [1, 1, H, W] complejo
        raw_transfer = cp['transfer_function'].to(device)
        # Asegurar que sea unitario (solo fase) para el target
        target_phases = raw_transfer / (raw_transfer.abs() + 1e-8)
        grid_size = target_phases.shape[-1]
    else:
        logging.warning("⚠️ No se encontró checkpoint de EXP-007. Usando target sintético.")
        grid_size = 32
        target_phases = torch.ones(1, 1, grid_size, grid_size, dtype=torch.cfloat, device=device)
    
    logging.info(f"   Target Grid: {grid_size}x{grid_size}")
    
    # 2. Inicializar Modelo Cuántico Nativo
    # Usamos 3 capas para dar suficiente expresividad
    model = QuantumNativeConv2d(grid_size=grid_size, n_layers=3, device=device)
    optimizer = optim.Adam(model.parameters(), lr=0.01)
    
    # 3. Loop de Entrenamiento
    # Queremos que PQC(ones) ~ Target_Phases
    # O más simple: Minimizar distancia entre fases generadas y target.
    
    logging.info("🧠 Entrenando PQC para aproximar operador holográfico...")
    losses = []
    
    # Input constante (1s en frecuencia) para ver solo la máscara
    dummy_input_freq = torch.ones(1, 1, grid_size, grid_size, dtype=torch.cfloat, device=device)
    
    for epoch in range(100):
        optimizer.zero_grad()
        
        # Forward del PQC
        generated_freq = model.forward_pqc(dummy_input_freq)
        
        # Loss: Distancia en el plano complejo (Fase + Magnitud, aunque magnitud es 1)
        # Loss = |Generated - Target|^2
        loss = nn.MSELoss()(generated_freq.real, target_phases.real) + \
               nn.MSELoss()(generated_freq.imag, target_phases.imag)
        
        loss.backward()
        optimizer.step()
        
        losses.append(loss.item())
        if epoch % 10 == 0:
            logging.info(f"   Epoch {epoch}: Loss = {loss.item():.6f}")
            
    # 4. Exportar Circuito
    logging.info("⚛️ Exportando circuito optimizado...")
    qc = model.export_circuit()
    print("\nCircuito Nativo Generado (Primeras capas):")
    print(qc.draw(fold=80))
    
    depth = qc.depth()
    n_gates = sum(qc.count_ops().values())
    logging.info(f"📊 Estadísticas del Circuito: Profundidad={depth}, Compuertas={n_gates}")
    
    # Comparación con Diagonal
    n_qubits = model.n_qubits
    diagonal_cost = 2**n_qubits
    logging.info(f"💡 Ahorro vs Diagonal: {n_gates} vs ~{diagonal_cost} CNOTs")
    
    # Guardar
    try:
        import qiskit.qasm2
        qasm_str = qiskit.qasm2.export(qc)
    except:
        # Fallback for older qiskit
        try:
            qasm_str = qc.qasm()
        except:
            qasm_str = "QASM export failed"

    torch.save({
        'model_state_dict': model.state_dict(),
        'circuit_qasm': qasm_str,
        'losses': losses
    }, "checkpoints/quantum_native_model.pt")
    logging.info("💾 Modelo guardado en checkpoints/quantum_native_model.pt")

if __name__ == "__main__":
    main()
