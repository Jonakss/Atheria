import torch

class QuantumNoiseInjector:
    """
    Simula el entorno hostil que fuerza la evolución de la simetría y la robustez.
    Basado en principios de corrección de errores cuánticos (IonQ / Bacon-Shor).
    """
    def __init__(self, device):
        self.device = device

    def apply_phase_flip(self, psi, rate=0.01):
        """
        Ruido Cuántico (Z-Error).
        Rota aleatoriamente la fase sin cambiar la magnitud (energía).
        
        Efecto Evolutivo: Destruye la coherencia necesaria para el movimiento.
        Solución que busca la IA: Redundancia de fase y simetría paridad.
        """
        # Máscara booleana de dónde ocurre el error
        mask = torch.rand(psi.shape[:3], device=self.device) < rate
        mask = mask.unsqueeze(-1).expand_as(psi)
        
        # Rotación aleatoria (0 a 2pi)
        random_angle = torch.rand(psi.shape, device=self.device) * 2 * torch.pi
        phase_noise = torch.complex(torch.cos(random_angle), torch.sin(random_angle))
        
        psi_out = psi.clone()
        psi_out[mask] = psi[mask] * phase_noise[mask]
        return psi_out

    def apply_bit_flip(self, psi, rate=0.005):
        """
        Ruido Clásico (X-Error).
        Invierte la amplitud o satura la energía. Simula 'rayos cósmicos'.
        
        Efecto Evolutivo: Destruye la estructura local.
        Solución que busca la IA: Robustez estructural (partículas densas).
        """
        mask = torch.rand(psi.shape[:3], device=self.device) < rate
        mask = mask.unsqueeze(-1).expand_as(psi)
        
        psi_out = psi.clone()
        # Inversión de signo drástica (Flip de 180 grados)
        psi_out[mask] = -psi[mask]
        return psi_out
    
    def apply_thermal_jitter(self, psi, temperature=0.01):
        """
        Ruido Térmico (Aditivo).
        Pequeñas fluctuaciones constantes en todo el campo.
        Evita que el sistema se congele en mínimos locales.
        """
        noise_real = torch.randn_like(psi.real) * temperature
        noise_imag = torch.randn_like(psi.imag) * temperature
        noise = torch.complex(noise_real, noise_imag)
        return psi + noise
