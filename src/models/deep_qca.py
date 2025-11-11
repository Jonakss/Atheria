# src/models/deep_qca.py
import torch
import torch.nn as nn
import torch.nn.functional as F

from . import register_model

# ------------------------------------------------------------------------------
# 1.2: QCA_Operator_Deep Class (The "Deep" M-Law)
# ------------------------------------------------------------------------------
@register_model('DEEP_QCA')
class QCA_Operator_Deep(nn.Module):
    def __init__(self, d_state, hidden_channels):
        super().__init__()
        self.d_state = d_state
        # 3x3 Neighborhood convolution (non-trainable, center-masked)
        self.conv_neighbors = nn.Conv2d(2*d_state, 2*d_state*8, kernel_size=3,
                                        padding=1, groups=2*d_state, bias=False)
        weights = torch.ones(2*d_state*8, 1, 3, 3)
        weights[:, 0, 1, 1] = 0.0 # Zero out the center weight.
        self.conv_neighbors.weight.data = weights
        self.conv_neighbors.weight.requires_grad = False

        # Trainable 1x1 Convolutional MLP
        self.processing_net = nn.Sequential(
            nn.Conv2d(2 * d_state * 8, hidden_channels, kernel_size=1, bias=False),
            nn.ELU(inplace=True),
            nn.Conv2d(hidden_channels, hidden_channels, kernel_size=1, bias=False),
            nn.ELU(inplace=True),
            nn.Conv2d(hidden_channels, hidden_channels, kernel_size=1, bias=False),
            nn.ELU(inplace=True),
            nn.Conv2d(hidden_channels, hidden_channels, kernel_size=1, bias=False),
            nn.ELU(inplace=True),
            nn.Conv2d(hidden_channels, 8 * d_state, kernel_size=1, bias=False)
        )

        # Trainable bias parameters
        self.M_bias_real = nn.Parameter(torch.zeros(d_state))
        self.M_bias_imag = nn.Parameter(torch.zeros(d_state))

    def forward(self, x_cat):
        """Applies the evolution operator."""
        x_neighbors = self.conv_neighbors(x_cat.to(self.conv_neighbors.weight.device))
        F_int = self.processing_net(x_neighbors)

        # Reformatear a la salida esperada
        # Permutar para que los canales estén al final para la división
        F_int_permuted = F_int.permute(0, 2, 3, 1) # [B, H, W, 8*d_state]
        B, H, W, C_total = F_int_permuted.shape
        D4 = 4 * self.d_state

        F_int_real_raw = F_int_permuted[..., :D4]
        F_int_imag_raw = F_int_permuted[..., D4:]
        
        F_int_real = F_int_real_raw.reshape(B, H, W, 4, self.d_state).mean(dim=3) * 0.1
        F_int_imag = F_int_imag_raw.reshape(B, H, W, 4, self.d_state).mean(dim=3) * 0.1

        # Permutar de nuevo a [B, C, H, W] para concatenar
        delta_real = F_int_real.permute(0, 3, 1, 2) # [B, d_state, H, W]
        delta_imag = F_int_imag.permute(0, 3, 1, 2) # [B, d_state, H, W]

        delta_psi = torch.cat([delta_real, delta_imag], dim=1) # [B, 2*d_state, H, W]
        
        return delta_psi
