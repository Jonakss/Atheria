import torch
import numpy as np
import random

class GenesisModule:
    """
    El Arquitecto del Big Bang.
    Contiene recetas para inyectar energía inicial en el Motor Disperso.
    """
    def __init__(self, engine):
        self.engine = engine
        self.d_state = engine.d_state
        self.device = engine.device

    def _get_random_state(self, intensity=1.0):
        """Genera un vector de estado aleatorio normalizado."""
        state = torch.randn(self.d_state, device=self.device)
        # Normalizar para que la energía total sea 'intensity'
        norm = torch.sqrt(torch.sum(state.abs().pow(2)))
        return (state / norm) * np.sqrt(intensity)

    def inject_primordial_soup(self, center_coords, radius=10, density=0.3):
        """
        Crea una nebulosa de gas aleatorio alrededor de un punto.
        """
        cx, cy, cz = center_coords
        count = 0
        
        for x in range(cx - radius, cx + radius):
            for y in range(cy - radius, cy + radius):
                for z in range(cz - radius, cz + radius):
                    # Probabilidad de existencia basada en distancia al centro (Gaussian blob)
                    dist = np.sqrt((x-cx)**2 + (y-cy)**2 + (z-cz)**2)
                    prob = density * np.exp(-dist / (radius/2))
                    
                    if random.random() < prob:
                        state = self._get_random_state(intensity=random.uniform(0.1, 0.5))
                        self.engine.add_particle((x, y, z), state)
                        count += 1
        
        print(f"🧪 Sopa Primordial inyectada: {count} partículas.")

    def inject_monolith(self, center_coords, size=5, intensity=2.0):
        """
        Crea un cubo denso y uniforme de energía. (Prueba de erosión).
        """
        cx, cy, cz = center_coords
        # Estado base sólido (mismo vector para todo el bloque)
        base_state = self._get_random_state(intensity)
        
        count = 0
        for x in range(cx - size, cx + size):
            for y in range(cy - size, cy + size):
                for z in range(cz - size, cz + size):
                    # Pequeña variación para que no sea matemáticamente perfecto (aburrido)
                    noise = torch.randn_like(base_state) * 0.05
                    self.engine.add_particle((x, y, z), base_state + noise)
                    count += 1
                    
        print(f"⬛ Monolito inyectado: {count} celdas densas.")

    def inject_collider(self, coord_a, coord_b, axis='x'):
        """
        Crea dos haces de partículas en trayectoria de colisión.
        Nota: Esto requiere que la IA ya sepa mover cosas, o que iniciemos con 'momento'.
        Como la U-Net aprende movimiento de los canales, inyectamos patrones de fase.
        """
        # Haz A (Fase positiva)
        state_a = self._get_random_state(1.0)
        # Haz B (Fase opuesta/negativa)
        state_b = -state_a 
        
        # Dibujar líneas
        length = 10
        for i in range(length):
            if axis == 'x':
                self.engine.add_particle((coord_a[0]+i, coord_a[1], coord_a[2]), state_a)
                self.engine.add_particle((coord_b[0]-i, coord_b[1], coord_b[2]), state_b)
        
        print(f"⚡ Colisionador configurado. Impacto inminente.")

    def inject_symmetric_seed(self, center_coords, size=4):
        """
        Crea un patrón con simetría de espejo forzada.
        Ideal para probar la hipótesis de IonQ.
        """
        cx, cy, cz = center_coords
        
        # Generar solo un octante
        octant_data = {}
        for x in range(size):
            for y in range(size):
                for z in range(size):
                    if random.random() > 0.5:
                        octant_data[(x,y,z)] = self._get_random_state(1.0)
        
        # Reflejar a los 8 octantes
        count = 0
        for (ox, oy, oz), state in octant_data.items():
            # Reflejos: (+x,+y,+z), (-x,+y,+z), etc.
            for sx in [1, -1]:
                for sy in [1, -1]:
                    for sz in [1, -1]:
                        px, py, pz = cx + ox*sx, cy + oy*sy, cz + oz*sz
                        # Importante: Para simetría paridad, el estado también podría reflejarse
                        # Aquí usamos simetría simple (mismo estado en espejo)
                        self.engine.add_particle((px, py, pz), state)
                        count += 1
                        
        print(f"❄️ Semilla Simétrica inyectada: {count} partículas (Simetría 8x).")