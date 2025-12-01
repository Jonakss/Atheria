import torch
import torch.nn as nn
import numpy as np

class QuantumStatePolar:
    def __init__(self, magnitude, phase):
        """
        magnitude: Tensor (B, 1, H, W) -> r
        phase: Tensor (B, 1, H, W) -> theta
        """
        self.magnitude = magnitude
        self.phase = phase
        self.device = magnitude.device

    @classmethod
    def from_cartesian(cls, real, imag):
        magnitude = torch.sqrt(real**2 + imag**2 + 1e-8)
        phase = torch.atan2(imag, real)
        return cls(magnitude, phase)

    def to_cartesian(self):
        real = self.magnitude * torch.cos(self.phase)
        imag = self.magnitude * torch.sin(self.phase)
        return real, imag

    def get_tensor(self):
        """Returns concatenated [magnitude, phase]"""
        return torch.cat([self.magnitude, self.phase], dim=1)

class PolarEngine(nn.Module):
    def __init__(self, model, grid_size=128):
        super().__init__()
        self.model = model
        self.grid_size = grid_size

        # Spatial Mixing Kernel (Laplacian for diffusion)
        # We apply this in Cartesian space
        self.register_buffer('laplacian', torch.tensor([
            [0.05, 0.2, 0.05],
            [0.2, -1.0, 0.2],
            [0.05, 0.2, 0.05]
        ]).unsqueeze(0).unsqueeze(0)) # (1, 1, 3, 3)

    def spatial_mixing(self, state: QuantumStatePolar):
        """
        Applies diffusion/convolution in Cartesian space.
        """
        real, imag = state.to_cartesian()

        # Apply Laplacian to Real and Imag parts independently
        # Padding=1, circular (periodic boundary)
        # Note: PyTorch F.conv2d expects (B, C, H, W). We treat Real/Imag as channels.

        # 1. Pad for periodic boundaries
        real_pad = torch.nn.functional.pad(real, (1, 1, 1, 1), mode='circular')
        imag_pad = torch.nn.functional.pad(imag, (1, 1, 1, 1), mode='circular')

        # 2. Convolve
        diff_real = torch.nn.functional.conv2d(real_pad, self.laplacian)
        diff_imag = torch.nn.functional.conv2d(imag_pad, self.laplacian)

        # 3. Add diffusion to original state (Euler step)
        # psi_new = psi + alpha * Laplacian(psi)
        alpha = 0.1 # Diffusion rate
        real_new = real + alpha * diff_real
        imag_new = imag + alpha * diff_imag

        return QuantumStatePolar.from_cartesian(real_new, imag_new)

    def forward(self, state: QuantumStatePolar):
        """
        Single Evolution Step
        1. Spatial Mixing (Cartesian)
        2. Non-linear Update (Polar / Rotational UNet)
        """
        # 1. Spatial Mixing
        mixed_state = self.spatial_mixing(state)

        # 2. Prepare Input for Model
        # Input: Magnitude, Sin(Phase), Cos(Phase) -> 3 Channels
        # This helps the model understand circularity
        mag = mixed_state.magnitude
        phase = mixed_state.phase

        model_input = torch.cat([
            mag,
            torch.sin(phase),
            torch.cos(phase)
        ], dim=1) # (B, 3, H, W)

        # 3. Model Prediction
        # Output: Delta Magnitude, Delta Phase
        output = self.model(model_input)
        delta_mag = output[:, 0:1, :, :]
        delta_phase = output[:, 1:2, :, :]

        # 4. Apply Update
        # Phase accumulates
        new_phase = phase + delta_phase
        # Wrap phase to [-pi, pi] for cleanliness (optional but good)
        new_phase = torch.atan2(torch.sin(new_phase), torch.cos(new_phase))

        # Magnitude is squashed (Sigmoid or Tanh) or additive
        # Let's assume model predicts additive change, but we clamp or sigmoid
        new_mag = torch.sigmoid(torch.log(mag + 1e-8) + delta_mag) # Multiplicative-ish update

        return QuantumStatePolar(new_mag, new_phase)
