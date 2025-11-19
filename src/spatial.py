"""
Optimización Espacial usando Códigos Morton (Z-order curve).

Este módulo implementa índices espaciales para mejorar la localidad
de acceso a memoria y la eficiencia de búsqueda en el SparseEngine.
"""
import torch
import numpy as np
from typing import Union, List, Tuple
import logging


class SpatialIndexer:
    """
    Índice espacial usando Códigos Morton para coordenadas 3D.
    
    Los códigos Morton intercalan los bits de X, Y, Z para crear
    un único entero que preserva la localidad espacial.
    
    Ejemplo:
        >>> indexer = SpatialIndexer()
        >>> coords = torch.tensor([[10, 10, 10], [11, 10, 10]])
        >>> codes = indexer.coords_to_morton(coords)
        >>> coords_recovered = indexer.morton_to_coords(codes)
        >>> assert torch.allclose(coords, coords_recovered)
    """
    
    def __init__(self, max_coord_bits: int = 21):
        """
        Inicializa el indexador espacial.
        
        Args:
            max_coord_bits: Número máximo de bits por coordenada (default: 21)
                          Permite coordenadas hasta 2^21 ≈ 2 millones
                          Total: 63 bits (signed int64) = 3 * 21
        """
        self.max_coord_bits = max_coord_bits
        self.max_coord = (1 << max_coord_bits) - 1  # 2^21 - 1
        
        # Máscaras para separar bits en el código Morton
        # Para 21 bits, necesitamos máscaras para bits 0, 1, 2, ..., 20
        self._init_bit_masks()
    
    def _init_bit_masks(self):
        """Inicializa máscaras para intercalación de bits."""
        # Máscaras para expandir bits (separación)
        # x: bits en posiciones 0, 3, 6, 9, ...
        # y: bits en posiciones 1, 4, 7, 10, ...
        # z: bits en posiciones 2, 5, 8, 11, ...
        
        # Máscara para obtener bits de X (cada 3er bit desde posición 0)
        self.x_mask = 0x09249249  # 0b00001001001001001001001001001001 (32 bits)
        self.y_mask = 0x12492492  # 0b00010010010010010010010010010010
        self.z_mask = 0x24924924  # 0b00100100100100100100100100100100
        
        # Para 64 bits, necesitamos extender estas máscaras
        # Pero usaremos operaciones bitwise más directas para 3D
    
    def _expand_bits_3d(self, v: torch.Tensor) -> torch.Tensor:
        """
        Expande los bits de una coordenada para intercalación 3D.
        
        Para coordenada X: coloca bits en posiciones 0, 3, 6, 9, ...
        Para coordenada Y: coloca bits en posiciones 1, 4, 7, 10, ...
        Para coordenada Z: coloca bits en posiciones 2, 5, 8, 11, ...
        
        Args:
            v: Tensor de coordenadas (int32 o int64)
        
        Returns:
            Tensor con bits expandidos
        """
        # Convertir a int64 para asegurar suficientes bits
        v = v.to(torch.int64)
        
        # Máscara inicial para primeros 10 bits
        v = v & 0x3FF  # Solo primeros 10 bits
        
        # Expandir bits usando operaciones bitwise
        # Patrón: para 3D, expandimos cada bit a 3 posiciones
        v = (v | (v << 32)) & 0x1F00000000FFFF
        v = (v | (v << 16)) & 0x1F0000FF0000FF
        v = (v | (v << 8)) & 0x100F00F00F00F00F
        v = (v | (v << 4)) & 0x10C30C30C30C30C3
        v = (v | (v << 2)) & 0x1249249249249249
        
        return v
    
    def _compact_bits_3d(self, m: torch.Tensor) -> torch.Tensor:
        """
        Compacta bits intercalados de vuelta a coordenada normal.
        
        Operación inversa de _expand_bits_3d.
        
        Args:
            m: Tensor con código Morton (bits intercalados)
        
        Returns:
            Tensor con coordenada compacta
        """
        # Máscara para extraer bits de una coordenada
        mask = 0x0924924924924924  # Máscara para bits X (cada 3er bit)
        
        # Aplicar máscara y revertir expansión
        x = m & mask
        x = (x | (x >> 2)) & 0x30C30C30C30C30C3
        x = (x | (x >> 4)) & 0xF00F00F00F00F00F
        x = (x | (x >> 8)) & 0x00FF0000FF0000FF
        x = (x | (x >> 16)) & 0x00FF00000000FFFF
        x = (x | (x >> 32)) & 0x3FF
        
        return x.to(torch.int32)
    
    def coords_to_morton(self, coords: Union[torch.Tensor, np.ndarray, List]) -> torch.Tensor:
        """
        Convierte coordenadas 3D a códigos Morton.
        
        Args:
            coords: Tensor/array de forma [N, 3] con coordenadas [x, y, z]
                   o lista de tuplas [(x, y, z), ...]
        
        Returns:
            Tensor de forma [N] con códigos Morton (int64)
        """
        # Convertir a tensor
        if isinstance(coords, (list, tuple)):
            coords = torch.tensor(coords, dtype=torch.int32)
        elif isinstance(coords, np.ndarray):
            coords = torch.from_numpy(coords).to(torch.int32)
        
        if coords.dim() == 1:
            coords = coords.unsqueeze(0)  # [3] -> [1, 3]
        
        if coords.shape[1] != 3:
            raise ValueError(f"coords debe tener forma [N, 3], pero tiene {coords.shape}")
        
        # Clampear coordenadas al rango válido
        coords = torch.clamp(coords, 0, self.max_coord).to(torch.int32)
        
        # Separar coordenadas
        x = coords[:, 0]
        y = coords[:, 1]
        z = coords[:, 2]
        
        # Expandir bits para cada coordenada
        # Para 3D: intercalamos x, y, z como: z1 y1 x1 z0 y0 x0 ...
        x_expanded = self._expand_bits_3d_for_coord(x, 0)  # Posiciones 0, 3, 6, ...
        y_expanded = self._expand_bits_3d_for_coord(y, 1)  # Posiciones 1, 4, 7, ...
        z_expanded = self._expand_bits_3d_for_coord(z, 2)  # Posiciones 2, 5, 8, ...
        
        # Combinar bits intercalados
        morton = x_expanded | y_expanded | z_expanded
        
        return morton.to(torch.int64)
    
    def _expand_bits_3d_for_coord(self, v: torch.Tensor, offset: int) -> torch.Tensor:
        """
        Expande bits para una coordenada específica en orden 3D.
        
        Args:
            v: Tensor de coordenadas (int32)
            offset: Offset de posición (0=X, 1=Y, 2=Z)
        
        Returns:
            Tensor con bits expandidos en posiciones offset, offset+3, offset+6, ...
        """
        v = v.to(torch.int64)
        
        # Expandir bits usando operaciones bitwise para 3D
        # Cada bit de v se coloca en posición offset + 3*i
        result = torch.zeros_like(v, dtype=torch.int64)
        
        # Máscaras para cada bit
        for bit_pos in range(self.max_coord_bits):
            # Obtener el bit en posición bit_pos
            bit = (v >> bit_pos) & 1
            
            # Posición en código Morton: offset + 3 * bit_pos
            morton_pos = offset + 3 * bit_pos
            
            if morton_pos < 64:  # Asegurar que no exceda int64
                result |= (bit << morton_pos)
        
        return result
    
    def morton_to_coords(self, codes: Union[torch.Tensor, np.ndarray, List]) -> torch.Tensor:
        """
        Recupera coordenadas 3D desde códigos Morton.
        
        Args:
            codes: Tensor/array de códigos Morton (int64)
        
        Returns:
            Tensor de forma [N, 3] con coordenadas [x, y, z]
        """
        # Convertir a tensor
        if isinstance(codes, (list, tuple, np.ndarray)):
            codes = torch.tensor(codes, dtype=torch.int64)
        
        if codes.dim() == 0:
            codes = codes.unsqueeze(0)  # Escalar -> [1]
        
        codes = codes.to(torch.int64)
        
        # Extraer bits para cada coordenada
        x = self._extract_bits_3d_for_coord(codes, 0)  # Bits en posiciones 0, 3, 6, ...
        y = self._extract_bits_3d_for_coord(codes, 1)  # Bits en posiciones 1, 4, 7, ...
        z = self._extract_bits_3d_for_coord(codes, 2)  # Bits en posiciones 2, 5, 8, ...
        
        # Combinar en tensor [N, 3]
        coords = torch.stack([x, y, z], dim=1)
        
        return coords.to(torch.int32)
    
    def _extract_bits_3d_for_coord(self, codes: torch.Tensor, offset: int) -> torch.Tensor:
        """
        Extrae bits para una coordenada específica desde código Morton.
        
        Args:
            codes: Tensor de códigos Morton (int64)
            offset: Offset de posición (0=X, 1=Y, 2=Z)
        
        Returns:
            Tensor con coordenada extraída (int32)
        """
        result = torch.zeros_like(codes, dtype=torch.int64)
        
        # Extraer bits en posiciones offset, offset+3, offset+6, ...
        for bit_pos in range(self.max_coord_bits):
            morton_pos = offset + 3 * bit_pos
            
            if morton_pos < 64:
                # Extraer bit en posición morton_pos
                bit = (codes >> morton_pos) & 1
                # Colocar en posición bit_pos del resultado
                result |= (bit << bit_pos)
        
        return result.to(torch.int32)
    
    def get_active_chunks(self, coords: Union[torch.Tensor, np.ndarray, List]) -> List[int]:
        """
        Identifica bloques de espacio activos para simulación.
        
        Args:
            coords: Tensor/array de forma [N, 3] con coordenadas
        
        Returns:
            Lista de códigos Morton únicos representando chunks activos
        """
        morton_codes = self.coords_to_morton(coords)
        unique_chunks = torch.unique(morton_codes).tolist()
        return unique_chunks


# Instancia global para uso rápido
_default_indexer = None

def get_spatial_indexer() -> SpatialIndexer:
    """Retorna una instancia singleton de SpatialIndexer."""
    global _default_indexer
    if _default_indexer is None:
        _default_indexer = SpatialIndexer()
    return _default_indexer

