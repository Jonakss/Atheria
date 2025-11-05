# src/qca_operator_unet.py
import torch
import torch.nn as nn
import torch.nn.functional as F

# ------------------------------------------------------------------------------
# 1.2: QCA_Operator_Deep Class (The "Deep" M-Law)
# ------------------------------------------------------------------------------
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

        F_int = F_int.squeeze(0).permute(1, 2, 0) # (H, W, Channels)
        H, W, C = F_int.shape
        D4 = 4 * self.d_state

        F_int_real_raw = F_int[:, :, :D4]
        F_int_imag_raw = F_int[:, :, D4:]
        
        F_int_real = F_int_real_raw.reshape(H, W, 4, self.d_state).mean(dim=2) * 0.1
        F_int_imag = F_int_imag_raw.reshape(H, W, 4, self.d_state).mean(dim=2) * 0.1

        return F_int_real, F_int_imag
