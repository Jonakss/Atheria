# src/visualization.py
import torch
import numpy as np
from PIL import Image

# ¡Importaciones relativas!
from .config import DEVICE
from .qca_engine import QCA_State # Para type hinting

# ------------------------------------------------------------------------------
# 4.1: Visualization Helper Functions
# ------------------------------------------------------------------------------

def downscale_frame(frame, downscale_factor):
    """Downscales an image frame (numpy array) using PIL."""
    if downscale_factor <= 1:
        return frame
    height, width = frame.shape[:2]
    new_height, new_width = height // downscale_factor, width // downscale_factor
    img = Image.fromarray(frame)
    img_resized = img.resize((new_width, new_height), Image.Resampling.LANCZOS)
    return np.array(img_resized)


def get_density_frame_gpu(state: QCA_State):
    """Generates a frame visualizing total probability density."""
    prob_sq = state.x_real.pow(2) + state.x_imag.pow(2)
    density_map = prob_sq.squeeze(0).sum(dim=2).detach()
    d_min, d_max = density_map.min(), density_map.max()
    norm_factor = d_max - d_min
    if norm_factor < 1e-8:
        normalized_density = torch.zeros_like(density_map).to(state.x_real.device)
    else:
        normalized_density = (density_map - d_min) / norm_factor
    normalized_density_clamped = normalized_density.clamp(0.0, 1.0)
    R = normalized_density_clamped
    G = torch.zeros_like(normalized_density_clamped)
    B = 1.0 - normalized_density_clamped
    img_rgb = torch.stack([R, G, B], dim=2).clamp(0.0, 1.0)
    final_image = (img_rgb * 255).byte().cpu().numpy()
    return final_image

def get_channel_frame_gpu(state: QCA_State, num_channels=3):
    """Generates a frame visualizing the first N channels as RGB."""
    prob_sq = state.x_real.pow(2) + state.x_imag.pow(2)
    combined_image = torch.zeros(state.size, state.size, 3, device=state.x_real.device)
    
    num_channels_to_viz = min(num_channels, state.d_state)
    if num_channels_to_viz == 0:
        return (combined_image * 255).byte().cpu().numpy()

    for i in range(num_channels_to_viz):
        channel_data = prob_sq[0, :, :, i].detach()
        ch_min, ch_max = channel_data.min(), channel_data.max()
        if (ch_max - ch_min) < 1e-8:
            channel_scaled = torch.zeros_like(channel_data)
        else:
            channel_scaled = (channel_data - ch_min) / (ch_max - ch_min)
        color_index = i % 3
        combined_image[:, :, color_index] += channel_scaled * (1.0 / num_channels_to_viz)

    final_image = (combined_image.clamp(0, 1) * 255).byte().cpu().numpy()
    return final_image

def get_state_magnitude_frame_gpu(state: QCA_State):
    """Generates a frame visualizing the state vector magnitude (grayscale)."""
    prob_sq = state.x_real.pow(2) + state.x_imag.pow(2)
    magnitude_map = torch.sqrt(prob_sq.squeeze(0).sum(dim=2) + 1e-8).detach()
    m_min, m_max = magnitude_map.min(), magnitude_map.max()
    norm_factor = m_max - m_min
    if norm_factor < 1e-8:
        normalized_magnitude = torch.zeros_like(magnitude_map).to(state.x_real.device)
    else:
        normalized_magnitude = (magnitude_map - m_min) / norm_factor
    normalized_magnitude_clamped = normalized_magnitude.clamp(0.0, 1.0)
    img_gray = normalized_magnitude_clamped
    img_rgb = torch.stack([img_gray, img_gray, img_gray], dim=2)
    final_image = (img_rgb * 255).byte().cpu().numpy()
    return final_image

def get_state_phase_frame_gpu(state: QCA_State):
    """Generates a frame visualizing the state vector phase (mapped to Hue)."""
    sum_real = state.x_real.squeeze(0).sum(dim=2)
    sum_imag = state.x_imag.squeeze(0).sum(dim=2)
    phase_map = torch.atan2(sum_imag, sum_real).detach()
    normalized_phase = (phase_map + torch.pi) / (2 * torch.pi)
    normalized_phase_clamped = normalized_phase.clamp(0.0, 1.0)
    hue = normalized_phase_clamped
    R = torch.sin(2 * torch.pi * hue + torch.pi/2) * 0.5 + 0.5
    G = torch.sin(2 * torch.pi * hue + torch.pi*3/2) * 0.5 + 0.5
    B = torch.sin(2 * torch.pi * hue + torch.pi*5/2) * 0.5 + 0.5
    img_rgb = torch.stack([R, G, B], dim=2).clamp(0.0, 1.0)
    final_image = (img_rgb * 255).byte().cpu().numpy()
    return final_image

def get_state_change_magnitude_frame_gpu(state: QCA_State, prev_state: QCA_State):
    """Generates a frame visualizing the magnitude of state change (activity)."""
    state_real = state.x_real.detach()
    state_imag = state.x_imag.detach()
    prev_state_real = prev_state.x_real.detach().to(DEVICE)
    prev_state_imag = prev_state.x_imag.detach().to(DEVICE)
    diff_real = state_real - prev_state_real
    diff_imag = state_imag - prev_state_imag
    change_magnitude_sq = diff_real.pow(2) + diff_imag.pow(2)
    change_magnitude_map = torch.sqrt(change_magnitude_sq.squeeze(0).sum(dim=2) + 1e-8)
    m_min, m_max = change_magnitude_map.min(), change_magnitude_map.max()
    norm_factor = m_max - m_min
    if norm_factor < 1e-12:
        normalized_change = torch.zeros_like(change_magnitude_map).to(DEVICE)
    else:
        normalized_change = (change_magnitude_map - m_min) / norm_factor # Normalized
    normalized_change_clamped = normalized_change.clamp(0.0, 1.0)
    img_gray = normalized_change_clamped
    img_rgb = torch.stack([img_gray, img_gray, img_gray], dim=2)
    final_image = (img_rgb * 255).byte().cpu().numpy()
    return final_image