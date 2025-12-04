import torch
from typing import List, Tuple, Dict, Any, Callable

# Type alias for Loss Function
# Args: psi_history (List[Tensor]), psi_initial (Tensor), **kwargs (Configurable params)
# Returns: (Total Loss Tensor, Metrics Dict)
LossFunction = Callable[..., Tuple[torch.Tensor, Dict[str, float]]]

def evolutionary_loss(psi_history: List[torch.Tensor], psi_initial: torch.Tensor, **kwargs) -> Tuple[torch.Tensor, Dict[str, float]]:
    """
    Standard Evolutionary Loss for Aetheria.
    Calculates how "alive" the simulation is based on:
    1. Survival (Energy Retention)
    2. Symmetry (Local Structure)
    3. Complexity (Spatial Entropy)

    Configurable kwargs:
    - target_energy_retention: Target energy relative to initial (default: 0.8)
    - w_survival: Weight for survival loss (default: 10.0)
    - w_symmetry: Weight for symmetry loss (default: 5.0)
    - w_complexity: Weight for complexity loss (default: 1.0)
    """
    psi_final = psi_history[-1]

    # Config parameters
    target_retention = kwargs.get('target_energy_retention', 0.8)
    w_survival = kwargs.get('w_survival', 10.0)
    w_symmetry = kwargs.get('w_symmetry', 5.0)
    w_complexity = kwargs.get('w_complexity', 1.0)

    # A. Supervivencia (Energy Retention)
    # Queremos que la energía se mantenga a pesar del Gamma Decay y el Ruido.
    # No forzamos a que sea idéntica (permitimos metabolismo), pero penalizamos la muerte (0) o explosión (inf).
    final_energy = torch.sum(psi_final.abs().pow(2))
    initial_energy = torch.sum(psi_initial.abs().pow(2))
    target_energy = initial_energy * target_retention # Permitimos disipación natural configurable
    loss_survival = torch.abs(final_energy - target_energy) / (initial_energy + 1e-6)

    # B. Simetría Local (IonQ Hypothesis)
    # Rotamos el estado 90 grados. Si es una partícula estable, debería parecerse a sí misma.
    # Esto fomenta la creación de "átomos" geométricos.
    psi_rot = torch.rot90(psi_final, 1, [1, 2]) # Rotar en ejes H, W
    loss_symmetry = torch.mean((psi_final.abs() - psi_rot.abs())**2)

    # C. Complejidad (Entropía Espacial)
    # Evitar el truco fácil de "llenar todo de ceros" o "llenar todo de unos".
    # Queremos islas de materia. Usamos la desviación estándar espacial.
    # Queremos MAXIMIZAR la complejidad -> MINIMIZAR el negativo.
    spatial_variance = torch.std(psi_final.abs().sum(dim=-1))
    loss_complexity = -torch.log(spatial_variance + 1e-6)

    # Ponderación de la Evolución (Ajustar según fase)
    total_loss = (w_survival * loss_survival) + (w_symmetry * loss_symmetry) + (w_complexity * loss_complexity)

    metrics = {
        "survival": loss_survival.item(),
        "symmetry": loss_symmetry.item(),
        "complexity": loss_complexity.item()
    }
    return total_loss, metrics

def mse_energy_loss(psi_history: List[torch.Tensor], psi_initial: torch.Tensor, **kwargs) -> Tuple[torch.Tensor, Dict[str, float]]:
    """
    Simple MSE Loss on Total Energy.
    Forces the system to strictly conserve energy (unitary evolution proxy).
    """
    psi_final = psi_history[-1]

    final_energy = torch.sum(psi_final.abs().pow(2))
    initial_energy = torch.sum(psi_initial.abs().pow(2))

    # Strict conservation (Target = 100% of initial)
    loss = (final_energy - initial_energy).pow(2)

    metrics = {
        "energy_error": loss.item(),
        "final_energy": final_energy.item()
    }

    return loss, metrics

def structure_loss(psi_history: List[torch.Tensor], psi_initial: torch.Tensor, **kwargs) -> Tuple[torch.Tensor, Dict[str, float]]:
    """
    Structure Loss.
    Encourages the formation of distinct structures by penalizing diffuse states.
    Uses L1 norm sparsity (assuming sparse particles) combined with total variation.

    Configurable kwargs:
    - tv_weight: Weight for Total Variation loss (default: 0.1)
    """
    psi_final = psi_history[-1]
    mag = psi_final.abs()

    # Config parameters
    tv_weight = kwargs.get('tv_weight', 0.1)

    # 1. Sparsity (L1 norm) - we want most space to be empty
    loss_sparsity = torch.mean(mag)

    # 2. Total Variation (Smoothness/Compactness)
    # Penalize high frequency noise, encourage blobs
    diff_h = torch.abs(mag[:, :, 1:] - mag[:, :, :-1])
    diff_w = torch.abs(mag[:, 1:, :] - mag[:, :-1, :])
    loss_tv = torch.mean(diff_h) + torch.mean(diff_w)

    total_loss = loss_sparsity + (tv_weight * loss_tv)

    metrics = {
        "sparsity": loss_sparsity.item(),
        "total_variation": loss_tv.item()
    }

    return total_loss, metrics

# Registry of available loss functions
LOSS_REGISTRY: Dict[str, LossFunction] = {
    "evolutionary": evolutionary_loss,
    "mse_energy": mse_energy_loss,
    "structure": structure_loss
}

def get_loss_function(name: str) -> LossFunction:
    """Retrieve a loss function by name."""
    if name not in LOSS_REGISTRY:
        raise ValueError(f"Loss function '{name}' not found. Available: {list(LOSS_REGISTRY.keys())}")
    return LOSS_REGISTRY[name]
