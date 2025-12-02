import torch
import numpy as np

class EpochDetector:
    """
    Analiza el estado del universo (psi) y determina en qué 'Era Cosmológica' estamos.
    Se usa para ajustar el Curriculum Learning automáticamente (ej. activar gravedad cuando hay planetas).
    """
    def __init__(self):
        self.history = []
        # Definición de las Eras de Atheria
        self.epochs = {
            0: "0. Vacío Inestable (Big Bang)",
            1: "I. Era Cuántica (Sopa de Probabilidad)",
            2: "II. Era de Partículas (Cristalización Simétrica)",
            3: "III. Era Química (Polímeros y Movimiento)",
            4: "IV. Era Gravitacional (Acreción de Materia)",
            5: "V. Era Biológica (A-Life / Homeostasis)"
        }

    def analyze_state(self, psi_tensor):
        """
        Calcula métricas clave del estado actual para diagnóstico.
        
        Args:
            psi_tensor: Tensor de estado [Batch, Height, Width, Channels] o [H, W, C]
            
        Returns:
            dict: Métricas calculadas (energía, simetría, clustering, entropía)
        """
        # Asegurar dimensiones correctas (remover batch si es 1)
        if len(psi_tensor.shape) == 4:
            psi = psi_tensor.squeeze(0)
        else:
            psi = psi_tensor

        # Detectar formato (C, H, W) y convertir a (H, W, C)
        # Esto pasa con PolarEngine/HarmonicEngine que usan convención PyTorch
        if psi.ndim == 3 and psi.shape[0] < psi.shape[1] and psi.shape[1] == psi.shape[2]:
            psi = psi.permute(1, 2, 0)

        metrics = {}
        
        # 1. Energía Total (Estabilidad del Vacío)
        # Suma de la magnitud al cuadrado de todos los canales
        # Ayuda a saber si el universo explotó o se congeló
        energy_map = psi.abs().pow(2).sum(dim=-1) # [H, W]
        total_energy = torch.sum(energy_map).item()
        metrics['energy'] = total_energy
        
        # 2. Entropía Espacial / Clustering (¿Es ruido o estructura?)
        # Varianza alta = Estructuras densas separadas por vacío (Planetas/Partículas)
        # Varianza baja = Ruido uniforme (Sopa Cuántica)
        spatial_variance = torch.var(energy_map).item()
        metrics['clustering'] = spatial_variance
        
        # 3. Simetría Local (Firma de Partículas IonQ)
        # Comparamos el estado con su versión rotada 90 grados
        # Si es idéntico, es un cristal perfecto (Fermión estable)
        psi_rot = torch.rot90(psi, 1, [0, 1])
        diff = torch.mean((psi.abs() - psi_rot.abs())**2).item()
        # Score: 1.0 = Simetría Perfecta, 0.0 = Caos Asimétrico
        symmetry_score = 1.0 / (1.0 + diff * 100)
        metrics['symmetry'] = symmetry_score
        
        return metrics

    def determine_epoch(self, metrics):
        """
        Clasifica el estado actual basándose en las métricas físicas.
        """
        e = metrics.get('energy', 0)
        c = metrics.get('clustering', 0)
        s = metrics.get('symmetry', 0)
        
        # Lógica de clasificación difusa para las Eras
        
        # Caso 0: Universo muerto o no nacido
        if e < 1.0:
            return 0 
            
        # Caso 1: Sopa Cuántica
        # Hay energía, pero está dispersa (bajo clustering) y es caótica (baja simetría)
        if s < 0.2 and c < 0.1:
            return 1 
            
        # Caso 2: Era de Partículas
        # La simetría emerge fuerte (defensa contra el ruido IonQ)
        if s > 0.6:
            return 2 
            
        # Caso 3: Era Química
        # La simetría baja un poco (se rompe para permitir movimiento) 
        # y el clustering sube (se forman estructuras más grandes)
        if s > 0.3 and c > 0.5:
            return 3 
            
        # Caso 4: Era Gravitacional
        # Clustering extremo. Grandes islas de materia separadas por vacío absoluto.
        if c > 2.0:
            return 4 
            
        # Por defecto, asumimos transición o caos cuántico
        return 1

# Bloque de prueba rápida si ejecutas el archivo directamente
if __name__ == "__main__":
    detector = EpochDetector()
    
    # Simular un estado de "Sopa Cuántica" (Ruido)
    dummy_soup = torch.randn(1, 64, 64, 16)
    metrics_soup = detector.analyze_state(dummy_soup)
    epoch_soup = detector.determine_epoch(metrics_soup)
    
    print(f"Estado Sopa: {metrics_soup}")
    print(f"Época Detectada: {detector.epochs[epoch_soup]}")
    
    # Simular un estado de "Partícula" (Centro denso y simétrico)
    dummy_particle = torch.zeros(1, 64, 64, 16)
    dummy_particle[:, 30:34, 30:34, :] = 1.0 # Cuadrado central
    metrics_part = detector.analyze_state(dummy_particle)
    epoch_part = detector.determine_epoch(metrics_part)
    
    print(f"\nEstado Partícula: {metrics_part}")
    print(f"Época Detectada: {detector.epochs[epoch_part]}")