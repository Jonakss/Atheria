import torch
import torch.nn as nn
import logging

class PolarEngine(nn.Module):
    def __init__(self, model, grid_size, d_state=1, device='cpu'):
        super().__init__()
        self.model = model
        self.grid_size = grid_size
        self.d_state = d_state
        self.device = device
        self.model.to(device)
        
        # Initialize state
        self.state = PolarStateContainer(grid_size, d_state, device)
        
        # Visualization artifacts
        self.last_delta_psi = None
        self.is_compiled = False
        
    def forward(self, x):
        # Placeholder forward pass
        return x

    @property
    def operator(self):
        """Alias for model, required by QC_Trainer_v4 for checkpointing."""
        return self.model
        
    def evolve_step(self, current_psi):
        # Evolución física para entrenamiento
        # current_psi: [B, C, H, W] (Complex)
        
        # 1. Preparar input para el modelo
        # El modelo espera [B, 2*C, H, W] (Real)
        if hasattr(current_psi, 'to_cartesian'):
            real, imag = current_psi.to_cartesian()
        else:
            real = current_psi.real
            imag = current_psi.imag
            
        model_input = torch.cat([real, imag], dim=1)
        
        # 2. Inferencia
        # Output: [B, 2*C, H, W] (Real)
        model_output = self.model(model_input)
        
        # 3. Convertir output a complejo
        out_real, out_imag = torch.chunk(model_output, 2, dim=1)
        delta_psi = torch.complex(out_real, out_imag)
        
        # 4. Actualizar estado (Euler integration o similar)
        # Por ahora asumimos que el modelo predice el nuevo estado directamente o un delta
        # Si es UNetUnitary, suele ser una transformación.
        # Asumimos que el modelo retorna el NUEVO estado o un delta aditivo.
        # Para consistencia con CartesianEngine, asumimos delta si es aditivo, o estado si es directo.
        # Dado que es "Polar", quizás deberíamos trabajar en magnitud/fase, pero el modelo es convolucional estándar.
        
        # Simplemente retornamos el output como el nuevo estado por ahora.
        return delta_psi

    def evolve_internal_state(self, step=None):
        # Evolución interna para simulación en vivo (sin gradientes)
        with torch.no_grad():
            # 1. Obtener estado actual como tensor complejo [1, 1, H, W]
            current_psi = self.get_initial_state(batch_size=1) # Hack: usar get_initial_state para obtener tensor desde contenedor
            # Mejor: construir tensor desde self.state
            mag = self.state.psi.magnitude
            phase = self.state.psi.phase
            real = mag * torch.cos(phase)
            imag = mag * torch.sin(phase)
            current_psi = torch.complex(real, imag) # [1, 1, H, W]
            
            # 2. Evolucionar
            new_psi = self.evolve_step(current_psi)
            
            # 3. Actualizar estado interno
            self.state.psi.magnitude = new_psi.abs()
            self.state.psi.phase = new_psi.angle()
        
    def get_dense_state(self, roi=None, check_pause_callback=None):
        """
        Retorna el estado denso para visualización.
        """
        # Construir tensor complejo desde estado interno
        mag = self.state.psi.magnitude
        phase = self.state.psi.phase
        real = mag * torch.cos(phase)
        imag = mag * torch.sin(phase)
        # Permute to [B, H, W, C] for consistency with other engines
        return torch.complex(real, imag).permute(0, 2, 3, 1)

    def get_visualization_data(self, viz_type: str = "density"):
        """
        Retorna datos de visualización para shaders.
        
        Args:
            viz_type: Tipo de visualización ("density", "phase", "energy", "magnitude")
            
        Returns:
            dict con 'data' (array 2D) y 'metadata'
        """
        import numpy as np
        
        try:
            mag = self.state.psi.magnitude  # [1, C, H, W]
            phase = self.state.psi.phase    # [1, C, H, W]
            
            # Reducir canales si múltiples (promediar)
            if mag.shape[1] > 1:
                mag = mag.mean(dim=1, keepdim=True)
                phase = phase.mean(dim=1, keepdim=True)
            
            if viz_type == "density" or viz_type == "magnitude":
                data = (mag ** 2).squeeze()  # [H, W]
                
            elif viz_type == "phase":
                phase_norm = (phase / (2 * 3.14159) + 0.5) % 1.0
                data = phase_norm.squeeze()  # [H, W]
                
            elif viz_type == "energy":
                data = (mag ** 2).squeeze()
                
            elif viz_type == "gradient":
                density = mag.squeeze(0).squeeze(0)  # [H, W]
                gy = torch.diff(density, dim=0, prepend=density[:1, :])
                gx = torch.diff(density, dim=1, prepend=density[:, :1])
                data = torch.sqrt(gx**2 + gy**2)
                
            elif viz_type == "real":
                real = mag * torch.cos(phase)
                data = real.squeeze()
                
            elif viz_type == "imag":
                imag = mag * torch.sin(phase)
                data = imag.squeeze()
                
            else:
                logging.warning(f"⚠️ Tipo de visualización '{viz_type}' no soportado. Usando density.")
                data = (mag ** 2).squeeze()
            
            # Convertir a numpy
            data_np = data.cpu().numpy().astype(np.float32)
            
            return {
                "data": data_np,
                "type": viz_type,
                "shape": list(data_np.shape),
                "min": float(data_np.min()),
                "max": float(data_np.max()),
                "engine": "PolarEngine"
            }
            
        except Exception as e:
            logging.error(f"Error en get_visualization_data: {e}")
            return {"data": None, "type": viz_type, "error": str(e)}

    def apply_tool(self, action, params):
        """
        Aplica una herramienta cuántica al estado polar.
        """
        logging.info(f"🛠️ PolarEngine aplicando herramienta: {action} | Params: {params}")
        
        try:
            # 1. Obtener estado denso actual
            current_dense = self.get_dense_state()
            
            # 2. Aplicar herramienta (usando lógica compartida)
            from ..physics import IonQCollapse, QuantumSteering
            device = self.device
            new_psi = None
            
            if action == 'collapse':
                intensity = float(params.get('intensity', 0.5))
                center = None
                if 'x' in params and 'y' in params:
                    center = (int(params['y']), int(params['x']))
                collapser = IonQCollapse(device)
                new_psi = collapser.collapse(current_dense, region_center=center, intensity=intensity)
                
            elif action == 'vortex':
                x = int(params.get('x', self.grid_size // 2))
                y = int(params.get('y', self.grid_size // 2))
                radius = int(params.get('radius', 5))
                strength = float(params.get('strength', 1.0))
                steering = QuantumSteering(device)
                new_psi = steering.inject(current_dense, 'vortex', x=x, y=y, radius=radius, strength=strength)
                
            elif action == 'wave':
                k_x = float(params.get('k_x', 1.0))
                k_y = float(params.get('k_y', 1.0))
                cx, cy = self.grid_size // 2, self.grid_size // 2
                radius = self.grid_size
                steering = QuantumSteering(device)
                new_psi = steering.inject(current_dense, 'plane_wave', x=cx, y=cy, radius=radius, k_x=k_x, k_y=k_y)
            
            if new_psi is not None:
                # 3. Actualizar estado interno (Polar)
                self.state.psi.magnitude = new_psi.abs()
                self.state.psi.phase = new_psi.angle()
                return True
                
            return False
            
        except Exception as e:
            logging.error(f"❌ Error aplicando herramienta en PolarEngine: {e}", exc_info=True)
            return False

    def get_model_for_params(self):
        return self.model

    def compile_model(self):
        """
        Compila el modelo para optimización.
        """
        if not self.is_compiled and self.model is not None:
            try:
                import torch._dynamo
                self.model = torch.compile(self.model)
                self.is_compiled = True
                logging.info("✅ PolarEngine: Modelo compilado con torch.compile()")
            except Exception as e:
                logging.warning(f"⚠️ PolarEngine: No se pudo compilar el modelo: {e}")

    def get_initial_state(self, batch_size=1):
        # Retorna estado inicial aleatorio [B, C, H, W]
        # PolarStateContainer maneja el estado interno, pero para entrenamiento
        # necesitamos retornar el tensor inicial.
        # Por ahora, generamos uno nuevo.
        mag = torch.rand(batch_size, self.d_state, self.grid_size, self.grid_size, device=self.device)
        phase = torch.rand(batch_size, self.d_state, self.grid_size, self.grid_size, device=self.device) * 2 * 3.14159
        # Retornar complejo [B, C, H, W]
        real = mag * torch.cos(phase)
        imag = mag * torch.sin(phase)
        return torch.complex(real, imag)

class PolarStateContainer:
    def __init__(self, grid_size, d_state, device):
        self.psi = QuantumStatePolar(grid_size, d_state, device)
        
    def _reset_state_random(self):
        # Reiniciar estado a aleatorio
        self.psi.magnitude = torch.rand(1, self.psi.d_state, self.psi.grid_size, self.psi.grid_size, device=self.psi.device)
        self.psi.phase = torch.rand(1, self.psi.d_state, self.psi.grid_size, self.psi.grid_size, device=self.psi.device) * 2 * 3.14159

class QuantumStatePolar:
    def __init__(self, grid_size, d_state, device):
        self.grid_size = grid_size
        self.d_state = d_state
        self.device = device
        # Initialize with some random data for visualization
        self.magnitude = torch.rand(1, d_state, grid_size, grid_size, device=device)
        self.phase = torch.rand(1, d_state, grid_size, grid_size, device=device) * 2 * 3.14159
        
    def to_cartesian(self):
        real = self.magnitude * torch.cos(self.phase)
        imag = self.magnitude * torch.sin(self.phase)
        return real, imag
    
    @property
    def real(self):
        return self.magnitude * torch.cos(self.phase)
    
    @property
    def imag(self):
        return self.magnitude * torch.sin(self.phase)
        
    def abs(self):
        return self.magnitude
        
    @property
    def shape(self):
        return self.magnitude.shape

    def squeeze(self, dim=None):
        """
        Returns a squeezed complex tensor representation of the state.
        Required by VisualizationPipeline which expects a tensor.
        """
        real = self.magnitude * torch.cos(self.phase)
        imag = self.magnitude * torch.sin(self.phase)
        complex_tensor = torch.complex(real, imag)
        
        if dim is None:
            return complex_tensor.squeeze()
        return complex_tensor.squeeze(dim)

    def clone(self):
        """
        Returns a deep copy of the QuantumStatePolar object.
        Required by QC_Trainer_v4 for state preservation.
        """
        new_state = QuantumStatePolar(self.grid_size, self.d_state, self.device)
        new_state.magnitude = self.magnitude.clone()
        new_state.phase = self.phase.clone()
        return new_state
