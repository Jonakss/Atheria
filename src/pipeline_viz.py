# src/pipeline_viz.py
import torch
import numpy as np
from sklearn.decomposition import PCA

pca = PCA(n_components=2)

def get_visualization_data(psi: torch.Tensor, viz_type: str):
    if psi.dim() == 4 and psi.shape[0] == 1:
        psi = psi.squeeze(0)

    density = torch.sum(psi.abs()**2, dim=-1).cpu().numpy()
    phase = torch.angle(psi).cpu().numpy()
    real_part = psi.real.cpu().numpy()
    imag_part = psi.imag.cpu().numpy()

    # --- Lógica de selección de datos del mapa ---
    map_data = None
    if viz_type == 'density':
        map_data = density
    elif viz_type == 'phase':
        map_data = (phase + np.pi) / (2 * np.pi)
    else:
        map_data = density

    min_val, max_val = np.min(map_data), np.max(map_data)
    if max_val > min_val:
        map_data = (map_data - min_val) / (max_val - min_val)

    # --- Cálculo de datos para Poincaré ---
    psi_flat_real = psi.real.reshape(-1, psi.shape[-1]).cpu().numpy()
    psi_flat_imag = psi.imag.reshape(-1, psi.shape[-1]).cpu().numpy()
    psi_flat_for_pca = np.concatenate([psi_flat_real, psi_flat_imag], axis=1)
    poincare_coords = pca.fit_transform(psi_flat_for_pca)
    max_abs_val = np.max(np.abs(poincare_coords))
    if max_abs_val > 0:
        poincare_coords = poincare_coords / max_abs_val

    # --- ¡¡CORRECCIÓN!! Lógica de histogramas restaurada ---
    density_flat = density.flatten()
    phase_flat = phase.flatten()
    real_flat = real_part.flatten()
    imag_flat = imag_part.flatten()

    density_hist, density_bins = np.histogram(density_flat, bins=30, range=(0, np.max(density_flat) if np.max(density_flat) > 0 else 1))
    phase_hist, phase_bins = np.histogram(phase_flat, bins=30, range=(-np.pi, np.pi))
    real_hist, real_bins = np.histogram(real_flat, bins=30, range=(-1, 1))
    imag_hist, imag_bins = np.histogram(imag_flat, bins=30, range=(-1, 1))

    hist_data = {
        'density': [{"bin": f"{density_bins[i]:.2f}", "count": int(density_hist[i])} for i in range(len(density_hist))],
        'phase': [{"bin": f"{phase_bins[i]:.2f}", "count": int(phase_hist[i])} for i in range(len(phase_hist))],
        'real': [{"bin": f"{real_bins[i]:.2f}", "count": int(real_hist[i])} for i in range(len(real_hist))],
        'imag': [{"bin": f"{imag_bins[i]:.2f}", "count": int(imag_hist[i])} for i in range(len(imag_hist))],
    }

    return {
        "map_data": map_data.tolist(),
        "hist_data": hist_data,
        "poincare_coords": poincare_coords.tolist()
    }