import torch
import torch.nn.functional as F
import numpy as np
import logging

class HolographicMixin:
    """
    Mixin universal para dotar a cualquier motor de capacidades holográficas.
    Implementa la proyección del estado de frontera (2D) hacia el bulk (3D)
    usando principios de Renormalización (Scale-Space).
    """

    def _generate_bulk_state(self, base_field: torch.Tensor, depth: int):
        """
        Genera el volumen del bulk aplicando desenfoque progresivo (coarse-graining)
        al campo base.

        Args:
            base_field (torch.Tensor): Tensor 2D [B, C, H, W] o [B, H, W].
                                     Debe ser energía/magnitud normalizada.
            depth (int): Número de capas en la dimensión Z (bulk).

        Returns:
            torch.Tensor: Volumen 3D [B, D, H, W] (donde D=depth).
        """
        if base_field is None:
            return None

        # Asegurar formato [B, C, H, W]
        if base_field.dim() == 3:
            base_field = base_field.unsqueeze(1) # [B, 1, H, W]

        # Tomar magnitud si hay múltiples canales (o usar el primer canal si es escalar)
        # Para visualización volumétrica simple, preferimos escalar.
        # Si C > 1, calculamos magnitud L2
        if base_field.shape[1] > 1:
            magnitude = torch.sqrt(torch.sum(base_field.abs().pow(2), dim=1, keepdim=True)) # [B, 1, H, W]
        else:
            magnitude = base_field.abs() # [B, 1, H, W]

        bulk_layers = []

        # Layer 0: Frontera (Original)
        # Normalizamos la frontera para asegurar visibilidad base
        bulk_layers.append(magnitude)

        # Proyectar hacia el bulk (aumentando Z)
        for z in range(1, depth):
            # La profundidad Z corresponde a la escala sigma
            # sigma = 0.5 * z + 0.5 (ajustable)
            sigma = z * 0.5 + 0.5
            kernel_size = int(sigma * 4) + 1
            if kernel_size % 2 == 0: kernel_size += 1

            blurred = self._gaussian_blur(magnitude, kernel_size, sigma)
            bulk_layers.append(blurred)

        # Concatenar en dimensión de profundidad (D) (que será el canal 1)
        # Resultado: [B, D, H, W]
        # Nota: torch.cat usa la dimensión especificada. Queremos que la dimensión 1 sea Depth.
        bulk_volume = torch.cat(bulk_layers, dim=1)

        return bulk_volume

    def _gaussian_blur(self, img, kernel_size, sigma):
        """Aplica Gaussian Blur 2D a un tensor [B, C, H, W]."""
        # Generar kernel
        k = self._get_gaussian_kernel(kernel_size, sigma, img.device)
        
        # Separable convolution for efficiency
        k_x = k.unsqueeze(0).unsqueeze(0) # [1, 1, K, 1]
        k_y = k.unsqueeze(0).unsqueeze(0).transpose(2, 3) # [1, 1, 1, K]

        # Apply to each channel
        channels = img.shape[1]
        # Padding 'reflect' para evitar bordes oscuros artificiales
        padded = F.pad(img, (kernel_size//2, kernel_size//2, kernel_size//2, kernel_size//2), mode='reflect')

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

    def get_bulk_visualization_data(self, viz_type: str, base_field_getter, bulk_depth: int = 8):
        """
        Helper para get_visualization_data de los motores.
        
        Args:
            viz_type: Tipo de visualización. Checks for 'holographic_bulk' or 'bulk'.
            base_field_getter: Callable que retorna el campo 2D base (tensor).
            bulk_depth: Profundidad del bulk.
        """
        if viz_type not in ["holographic_bulk", "bulk"]:
            return None

        # Obtener campo base
        base_field = base_field_getter()
        if base_field is None:
            return {"data": [], "error": "No base field state"}

        # Generar bulk
        bulk_volume = self._generate_bulk_state(base_field, bulk_depth)
        
        if bulk_volume is None:
             return {"data": [], "error": "Failed to generate bulk"}

        # Convertir a numpy flat array para transmisión eficiente
        # Formato esperado por frontend: 
        # Si es volumétrico, [D, H, W] o similar.
        # Aquí normalizamos y mandamos como array plano.
        
        # CPU
        data_np = bulk_volume.detach().cpu().numpy().astype(np.float32) # [1, D, H, W]
        data_np = data_np.squeeze(0) # [D, H, W]

        # Normalización Global del Volumen
        min_val = float(data_np.min())
        max_val = float(data_np.max())
        if max_val > min_val and abs(max_val - min_val) > 1e-10:
            data_np = (data_np - min_val) / (max_val - min_val)
        else:
            data_np = np.zeros_like(data_np)

        # Flatten explícito: [Layer0_Row0...RowN, Layer1_Row0...]
        # Esto es equivalente a .flatten() por defecto (C-order)
        # El frontend esperaría un array largo.
        # Es vital que el frontend sepa que es 3D.
        
        return {
            "data": data_np.flatten(), # Flattened array
            "type": viz_type,
            "shape": list(data_np.shape), # [D, H, W] - Frontend usa esto para reconstruir
            "min": 0.0,
            "max": 1.0,
            "is_volumetric": True
        }
