import torch
import torch.nn as nn
import torch.nn.functional as F
import logging
import numpy as np
from .qca_engine import CartesianEngine

class HolographicEngine(CartesianEngine):
    """
    Motor Holográfico (AdS/CFT).
    
    Extiende el motor Cartesiano (QCA 2D) para implementar el Principio Holográfico.
    El estado fundamental vive en la frontera 2D (Boundary), pero el motor puede
    proyectar este estado hacia un "Bulk" 3D emergente.
    
    La dimensión extra (Z) emerge de la escala (renormalización).
    """
    
    def __init__(self, model_operator, grid_size, d_state, device, cfg=None, bulk_depth=16):
        super().__init__(model_operator, grid_size, d_state, device, cfg)
        self.bulk_depth = bulk_depth
        logging.info(f"🌌 HolographicEngine inicializado: Boundary={grid_size}x{grid_size}, Bulk Depth={bulk_depth}")
        
    def get_bulk_state(self):
        """
        Proyecta el estado de frontera 2D hacia el bulk 3D.
        
        Usa una transformación de escala (Scale-Space) donde la profundidad Z
        corresponde al nivel de "coarse-graining" (borrosidad/baja frecuencia).
        
        Returns:
            Tensor [1, D, H, W, C] (o similar, dependiendo de lo que necesite el viz)
            Aquí retornamos [1, D, H, W] de energía/magnitud para visualización volumétrica.
        """
        if self.state.psi is None:
            return None
            
        # 1. Obtener estado base (Boundary)
        # [1, H, W, C]
        psi = self.state.psi
        
        # Calcular magnitud/energía para simplificar visualización 3D
        # [1, 1, H, W]
        magnitude = torch.sqrt(torch.sum(psi.abs().pow(2), dim=-1, keepdim=True)).permute(0, 3, 1, 2)
        
        bulk_layers = []
        
        # Layer 0: Frontera (Original)
        bulk_layers.append(magnitude)
        
        # Proyectar hacia adentro (aumentando Z = disminuyendo resolución/aumentando escala)
        current_layer = magnitude
        
        for z in range(1, self.bulk_depth):
            # Aplicar Average Pooling para simular renormalización
            # Esto suaviza y reduce la información de alta frecuencia
            # Padding para mantener el tamaño espacial (H, W) constante para visualización volumétrica fácil
            # O podríamos reducir el tamaño espacial para simular geometría hiperbólica real
            
            # Opción A: Mantener tamaño (Cilindro) - Más fácil para Texture3D
            # Usamos AvgPool con stride 1 y kernel creciente o Gaussian Blur
            
            sigma = z * 0.5 + 0.5
            kernel_size = int(sigma * 4) + 1
            if kernel_size % 2 == 0: kernel_size += 1
            
            # Gaussian Blur simple
            blurred = self._gaussian_blur(magnitude, kernel_size, sigma)
            bulk_layers.append(blurred)
            
        # Stack en dimensión de profundidad (D)
        # [1, D, H, W]
        bulk_volume = torch.cat(bulk_layers, dim=1)
        
        return bulk_volume

    def _gaussian_blur(self, img, kernel_size, sigma):
        """Aplica Gaussian Blur 2D a un tensor [B, C, H, W]."""
        k = self._get_gaussian_kernel(kernel_size, sigma, img.device)
        # Separable convolution for efficiency
        k_x = k.unsqueeze(0).unsqueeze(0) # [1, 1, K, 1]
        k_y = k.unsqueeze(0).unsqueeze(0).transpose(2, 3) # [1, 1, 1, K]
        
        # Apply to each channel
        channels = img.shape[1]
        padded = F.pad(img, (kernel_size//2, kernel_size//2, kernel_size//2, kernel_size//2), mode='reflect')
        
        # Convolve X then Y
        # We assume independent channels, so groups=channels if we expanded kernel
        # But here we iterate or use groups
        
        # Expand kernel for groups
        k_x = k_x.expand(channels, 1, kernel_size, 1)
        k_y = k_y.expand(channels, 1, 1, kernel_size)
        
        blurred = F.conv2d(padded, k_x, groups=channels)
        blurred = F.conv2d(blurred, k_y, groups=channels)
        
        return blurred

    def _get_gaussian_kernel(self, kernel_size, sigma, device):
        x = torch.arange(kernel_size, device=device).float() - kernel_size // 2
        k = torch.exp(-x**2 / (2 * sigma**2))
        k = k / k.sum()
        return k.view(kernel_size, 1)

    def get_holographic_entropy(self):
        """
        Calcula la entropía de entrelazamiento holográfica (Ryu-Takayanagi).
        En este modelo simplificado, es proporcional al área de la superficie mínima en el bulk.
        """
        # TODO: Implementar cálculo de entropía basado en la geometría del bulk
        pass

    def get_visualization_data(self, viz_type: str = "density"):
        """
        Retorna datos para visualización frontend.
        
        Extiende CartesianEngine para incluir visualización del bulk 3D.
        
        Args:
            viz_type: Tipo de visualización
                - Todos los tipos de CartesianEngine (density, phase, energy, etc.)
                - 'bulk': Volumen 3D proyectado
                
        Returns:
            dict con 'data' y 'metadata'
        """
        # Para 'bulk', usar proyección holográfica
        if viz_type == "bulk":
            bulk_volume = self.get_bulk_state()
            if bulk_volume is None:
                return {"data": None, "type": viz_type, "error": "No bulk state"}
                
            # Convertir a numpy
            data_np = bulk_volume.cpu().numpy().astype(np.float32)
            
            # NORMALIZACIÓN: Los shaders esperan datos en [0, 1]
            min_val = float(data_np.min())
            max_val = float(data_np.max())
            if max_val > min_val and abs(max_val - min_val) > 1e-10:
                data_np = (data_np - min_val) / (max_val - min_val)
            else:
                data_np = np.full_like(data_np, 0.5)
            
            return {
                "data": data_np,
                "type": viz_type,
                "shape": list(data_np.shape),
                "min": 0.0,  # Ya normalizado
                "max": 1.0,  # Ya normalizado
                "engine": "HolographicEngine",
                "bulk_depth": self.bulk_depth
            }
        
        # Para otros tipos, delegar a CartesianEngine
        result = super().get_visualization_data(viz_type)
        result["engine"] = "HolographicEngine"
        return result
