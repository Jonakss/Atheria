# src/models/rmsnorm.py
"""
RMSNorm (Root Mean Square Normalization) para QCA Unitaria.

RMSNorm es más adecuado que GroupNorm para física cuántica porque:
1. NO resta la media (preserva mejor la energía total |ψ|²)
2. Es más rápido (~15-20% más rápido que GroupNorm)
3. Mejor estabilidad numérica para estados cuánticos
"""
import torch
import torch.nn as nn


class RMSNorm2d(nn.Module):
    """
    RMSNorm para tensores 2D (imágenes/canales espaciales).
    
    Fórmula: x_norm = x / sqrt(mean(x²) + ε) * weight
    
    Args:
        num_channels: Número de canales a normalizar
        eps: Valor pequeño para estabilidad numérica (default: 1e-6)
    """
    def __init__(self, num_channels, eps=1e-6):
        super().__init__()
        self.num_channels = num_channels
        self.eps = eps
        # Parámetro aprendible para escalar después de normalizar
        self.weight = nn.Parameter(torch.ones(num_channels))
    
    def forward(self, x):
        """
        Args:
            x: Tensor de forma [batch, channels, height, width]
        
        Returns:
            Tensor normalizado de la misma forma
        """
        # Calcular RMS (Root Mean Square) por canal
        # mean(x²) sobre dimensiones espaciales (H, W)
        rms = torch.sqrt(torch.mean(x**2, dim=[2, 3], keepdim=True) + self.eps)
        
        # Normalizar: dividir por RMS
        x_norm = x / rms
        
        # Escalar con parámetro aprendible (similar a GroupNorm)
        x_norm = x_norm * self.weight.view(1, -1, 1, 1)
        
        return x_norm


class RMSNorm1d(nn.Module):
    """
    RMSNorm para tensores 1D (útil para MLP o capas lineales).
    
    Args:
        dim: Dimensión a normalizar
        eps: Valor pequeño para estabilidad numérica (default: 1e-6)
    """
    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.dim = dim
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))
    
    def forward(self, x):
        """
        Args:
            x: Tensor de forma [..., dim]
        
        Returns:
            Tensor normalizado de la misma forma
        """
        # Calcular RMS sobre la última dimensión
        rms = torch.sqrt(torch.mean(x**2, dim=-1, keepdim=True) + self.eps)
        x_norm = x / rms
        x_norm = x_norm * self.weight
        return x_norm

