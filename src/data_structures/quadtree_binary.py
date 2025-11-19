# src/quadtree_binary.py
"""
Quadtree con representación binaria directa (bytes) para datos 2D.

En lugar de usar objetos Python, representamos el quadtree directamente en bytes:
- Cada nodo = 5 bytes (1 byte flags + 4 bytes hijos)
- Estructura plana en memoria
- Acceso O(1) por índice calculado
- Perfecto para grids 2D de simulación
"""
import numpy as np
import struct
from typing import Optional, Tuple, List
from enum import IntFlag
import logging


class NodeFlags(IntFlag):
    """Flags para cada nodo del quadtree."""
    EMPTY = 0
    LEAF = 1      # Nodo hoja (contiene datos)
    BRANCH = 2    # Nodo rama (tiene hijos)
    FULL = 4      # Nodo completamente ocupado
    PARTIAL = 8   # Nodo parcialmente ocupado


class BinaryQuadtree:
    """
    Quadtree representado directamente en bytes para máxima eficiencia en 2D.
    
    Estructura de memoria:
    - Header: 24 bytes (metadatos)
    - Nodos: Array plano de bytes
    - Datos: Array plano de valores (solo hojas)
    
    Cada nodo = 5 bytes:
    - Byte 0: Flags (NodeFlags)
    - Bytes 1-4: Índices de hijos (uint32 cada uno, o 0 si no existe)
    
    Cuadrantes (2D):
    0: (-, -)  2: (-, +)
    1: (+, -)  3: (+, +)
    """
    
    # Tamaños en bytes
    HEADER_SIZE = 24
    NODE_SIZE = 5  # 1 byte flags + 4 bytes (índice hijo)
    CHILD_INDEX_SIZE = 4  # uint32
    
    def __init__(self, max_depth: int = 8, initial_capacity: int = 1024):
        """
        Args:
            max_depth: Profundidad máxima del quadtree
            initial_capacity: Capacidad inicial de nodos
        """
        self.max_depth = max_depth
        self.max_size = 2 ** max_depth  # Tamaño máximo en cada dimensión
        
        # Arrays de bytes para nodos y datos
        self.node_buffer = bytearray(initial_capacity * self.NODE_SIZE)
        self.data_buffer = []  # Lista de valores (se puede convertir a numpy array)
        
        # Contadores
        self.node_count = 0
        self.data_count = 0
        self.next_node_index = 1  # 0 es el nodo raíz
        
        # Índices libres (para reutilización)
        self.free_indices = []
        
        # Inicializar nodo raíz (índice 0)
        self._init_root()
    
    def _init_root(self):
        """Inicializa el nodo raíz."""
        self._set_node_flags(0, NodeFlags.EMPTY)
        for i in range(4):  # 4 cuadrantes en 2D
            self._set_child_index(0, i, 0)  # 0 = no hay hijo
        self.node_count = 1
    
    def _get_node_offset(self, node_index: int) -> int:
        """Calcula el offset en bytes para un nodo."""
        return node_index * self.NODE_SIZE
    
    def _ensure_capacity(self, min_capacity: int):
        """Asegura que el buffer tenga capacidad suficiente."""
        current_capacity = len(self.node_buffer) // self.NODE_SIZE
        if min_capacity > current_capacity:
            # Expandir buffer
            new_capacity = max(min_capacity, current_capacity * 2)
            new_size = new_capacity * self.NODE_SIZE
            self.node_buffer.extend(bytearray(new_size - len(self.node_buffer)))
            logging.debug(f"Quadtree buffer expandido a {new_capacity} nodos")
    
    def _allocate_node(self) -> int:
        """Asigna un nuevo nodo y retorna su índice."""
        if self.free_indices:
            node_index = self.free_indices.pop()
        else:
            node_index = self.next_node_index
            self.next_node_index += 1
            self._ensure_capacity(self.next_node_index)
        
        self.node_count += 1
        return node_index
    
    def _free_node(self, node_index: int):
        """Libera un nodo para reutilización."""
        if node_index > 0:  # No liberar raíz
            self._set_node_flags(node_index, NodeFlags.EMPTY)
            for i in range(4):
                self._set_child_index(node_index, i, 0)
            self.free_indices.append(node_index)
            self.node_count -= 1
    
    def _get_node_flags(self, node_index: int) -> NodeFlags:
        """Obtiene los flags de un nodo."""
        offset = self._get_node_offset(node_index)
        return NodeFlags(self.node_buffer[offset])
    
    def _set_node_flags(self, node_index: int, flags: NodeFlags):
        """Establece los flags de un nodo."""
        offset = self._get_node_offset(node_index)
        self.node_buffer[offset] = int(flags)
    
    def _get_child_index(self, node_index: int, quadrant: int) -> int:
        """Obtiene el índice de un hijo (0-3 para quadtree 2D)."""
        offset = self._get_node_offset(node_index) + 1 + (quadrant * self.CHILD_INDEX_SIZE)
        return struct.unpack('<I', self.node_buffer[offset:offset + self.CHILD_INDEX_SIZE])[0]
    
    def _set_child_index(self, node_index: int, quadrant: int, child_index: int):
        """Establece el índice de un hijo."""
        offset = self._get_node_offset(node_index) + 1 + (quadrant * self.CHILD_INDEX_SIZE)
        struct.pack_into('<I', self.node_buffer, offset, child_index)
    
    def _get_quadrant(self, x: int, y: int, center_x: int, center_y: int) -> int:
        """
        Calcula el cuadrante (0-3) para un punto dado.
        
        Cuadrantes 2D:
        0: (-, -)  2: (-, +)
        1: (+, -)  3: (+, +)
        """
        quadrant = 0
        if x >= center_x:
            quadrant |= 1
        if y >= center_y:
            quadrant |= 2
        return quadrant
    
    def _get_quadrant_bounds(self, min_x: int, min_y: int,
                            max_x: int, max_y: int, quadrant: int) -> Tuple[int, int, int, int]:
        """Calcula los bounds de un cuadrante dentro de un área 2D."""
        center_x = (min_x + max_x) // 2
        center_y = (min_y + max_y) // 2
        
        if quadrant & 1:
            new_min_x, new_max_x = center_x, max_x
        else:
            new_min_x, new_max_x = min_x, center_x
        
        if quadrant & 2:
            new_min_y, new_max_y = center_y, max_y
        else:
            new_min_y, new_max_y = min_y, center_y
        
        return new_min_x, new_min_y, new_max_x, new_max_y
    
    def insert(self, x: int, y: int, value: float, depth: int = 0, node_index: int = 0,
               min_x: int = 0, min_y: int = 0,
               max_x: Optional[int] = None, max_y: Optional[int] = None) -> bool:
        """
        Inserta un valor en el quadtree.
        
        Args:
            x, y: Coordenadas del punto (2D)
            value: Valor a insertar
            depth: Profundidad actual (recursión)
            node_index: Índice del nodo actual
            min_x, min_y: Bounds mínimos del área actual
            max_x, max_y: Bounds máximos del área actual
        
        Returns:
            True si se insertó exitosamente
        """
        if max_x is None:
            max_x = self.max_size
            max_y = self.max_size
        
        # Validar coordenadas
        if not (min_x <= x < max_x and min_y <= y < max_y):
            return False
        
        flags = self._get_node_flags(node_index)
        
        # Si es hoja y estamos en la profundidad máxima, insertar valor
        if depth >= self.max_depth or (flags & NodeFlags.LEAF):
            # Convertir a hoja si no lo es
            if not (flags & NodeFlags.LEAF):
                self._set_node_flags(node_index, NodeFlags.LEAF)
                # Guardar índice de datos en el primer hijo (reutilizamos el campo)
                data_index = len(self.data_buffer)
                self.data_buffer.append(value)
                self._set_child_index(node_index, 0, data_index)
                self.data_count += 1
            else:
                # Actualizar valor existente
                data_index = self._get_child_index(node_index, 0)
                if data_index < len(self.data_buffer):
                    self.data_buffer[data_index] = value
            return True
        
        # Si es vacío, convertir a hoja si estamos cerca de la profundidad máxima
        if flags == NodeFlags.EMPTY:
            if depth >= self.max_depth - 1:
                self._set_node_flags(node_index, NodeFlags.LEAF)
                data_index = len(self.data_buffer)
                self.data_buffer.append(value)
                self._set_child_index(node_index, 0, data_index)
                self.data_count += 1
                return True
            else:
                # Convertir a rama
                self._set_node_flags(node_index, NodeFlags.BRANCH | NodeFlags.PARTIAL)
        
        # Calcular cuadrante
        center_x = (min_x + max_x) // 2
        center_y = (min_y + max_y) // 2
        quadrant = self._get_quadrant(x, y, center_x, center_y)
        
        # Obtener o crear hijo
        child_index = self._get_child_index(node_index, quadrant)
        if child_index == 0:
            # Crear nuevo hijo
            child_index = self._allocate_node()
            self._set_child_index(node_index, quadrant, child_index)
        
        # Calcular bounds del cuadrante
        new_min_x, new_min_y, new_max_x, new_max_y = \
            self._get_quadrant_bounds(min_x, min_y, max_x, max_y, quadrant)
        
        # Insertar recursivamente
        return self.insert(x, y, value, depth + 1, child_index,
                          new_min_x, new_min_y, new_max_x, new_max_y)
    
    def query(self, x: int, y: int, node_index: int = 0,
              min_x: int = 0, min_y: int = 0,
              max_x: Optional[int] = None, max_y: Optional[int] = None) -> Optional[float]:
        """
        Consulta un valor en el quadtree.
        
        Returns:
            Valor encontrado o None si no existe
        """
        if max_x is None:
            max_x = self.max_size
            max_y = self.max_size
        
        if not (min_x <= x < max_x and min_y <= y < max_y):
            return None
        
        flags = self._get_node_flags(node_index)
        
        # Si es hoja, retornar valor
        if flags & NodeFlags.LEAF:
            data_index = self._get_child_index(node_index, 0)
            if data_index < len(self.data_buffer):
                return self.data_buffer[data_index]
            return None
        
        # Si es vacío, no hay datos
        if flags == NodeFlags.EMPTY:
            return None
        
        # Calcular cuadrante y buscar en hijo
        center_x = (min_x + max_x) // 2
        center_y = (min_y + max_y) // 2
        quadrant = self._get_quadrant(x, y, center_x, center_y)
        
        child_index = self._get_child_index(node_index, quadrant)
        if child_index == 0:
            return None
        
        new_min_x, new_min_y, new_max_x, new_max_y = \
            self._get_quadrant_bounds(min_x, min_y, max_x, max_y, quadrant)
        
        return self.query(x, y, child_index, new_min_x, new_min_y, new_max_x, new_max_y)
    
    def get_statistics(self) -> dict:
        """Retorna estadísticas del quadtree."""
        total_nodes = self.node_count
        total_data = self.data_count
        memory_nodes = len(self.node_buffer)
        memory_data = len(self.data_buffer) * 8  # Asumiendo float64
        
        return {
            "total_nodes": total_nodes,
            "total_data_points": total_data,
            "memory_nodes_bytes": memory_nodes,
            "memory_data_bytes": memory_data,
            "total_memory_bytes": memory_nodes + memory_data,
            "compression_ratio": (self.max_size ** 2 * 8) / (memory_nodes + memory_data) if (memory_nodes + memory_data) > 0 else 0,
            "max_depth": self.max_depth,
            "max_size": self.max_size
        }
    
    def to_numpy(self) -> np.ndarray:
        """
        Convierte el quadtree a un array numpy denso 2D (para visualización).
        
        WARNING: Esto puede usar mucha memoria si el quadtree es grande.
        """
        result = np.zeros((self.max_size, self.max_size), dtype=np.float32)
        
        for x in range(self.max_size):
            for y in range(self.max_size):
                value = self.query(x, y)
                if value is not None:
                    result[x, y] = value
        
        return result
    
    def from_dense_array(self, array: np.ndarray, threshold: float = 0.01):
        """
        Construye el quadtree desde un array numpy denso 2D.
        
        Solo inserta valores que superan el threshold (para compresión).
        
        Args:
            array: Array 2D numpy (H, W)
            threshold: Valor mínimo para considerar no-vacío
        """
        shape = array.shape
        for x in range(shape[0]):
            for y in range(shape[1]):
                value = array[x, y]
                if abs(value) > threshold:
                    self.insert(x, y, float(value))
    
    def from_map_data(self, map_data: List[List[float]], threshold: float = 0.01):
        """
        Construye el quadtree desde map_data (formato de la simulación).
        
        Args:
            map_data: Lista de listas 2D (formato del pipeline)
            threshold: Valor mínimo para considerar no-vacío
        """
        if not map_data or not map_data[0]:
            return
        
        height = len(map_data)
        width = len(map_data[0])
        
        for x in range(height):
            for y in range(width):
                value = map_data[x][y]
                if abs(value) > threshold:
                    self.insert(x, y, float(value))
    
    def save_to_file(self, filepath: str):
        """Guarda el quadtree a un archivo binario."""
        with open(filepath, 'wb') as f:
            # Header
            f.write(struct.pack('<I', self.max_depth))  # 4 bytes
            f.write(struct.pack('<I', self.node_count))  # 4 bytes
            f.write(struct.pack('<I', self.data_count))  # 4 bytes
            f.write(struct.pack('<I', len(self.node_buffer)))  # 4 bytes
            f.write(struct.pack('<I', len(self.data_buffer)))  # 4 bytes
            f.write(struct.pack('<I', 0))  # Padding a 24 bytes
            
            # Node buffer
            f.write(self.node_buffer)
            
            # Data buffer
            if self.data_buffer:
                data_array = np.array(self.data_buffer, dtype=np.float64)
                f.write(data_array.tobytes())
    
    @classmethod
    def load_from_file(cls, filepath: str) -> 'BinaryQuadtree':
        """Carga un quadtree desde un archivo binario."""
        with open(filepath, 'rb') as f:
            # Header
            max_depth = struct.unpack('<I', f.read(4))[0]
            node_count = struct.unpack('<I', f.read(4))[0]
            data_count = struct.unpack('<I', f.read(4))[0]
            buffer_size = struct.unpack('<I', f.read(4))[0]
            _ = struct.unpack('<I', f.read(4))[0]  # data_buffer_size
            _ = struct.unpack('<I', f.read(4))[0]  # Padding
            
            # Crear instancia
            quadtree = cls(max_depth=max_depth)
            quadtree.node_count = node_count
            quadtree.data_count = data_count
            
            # Node buffer
            quadtree.node_buffer = bytearray(f.read(buffer_size))
            
            # Data buffer
            if data_count > 0:
                data_bytes = f.read(data_count * 8)  # float64 = 8 bytes
                data_array = np.frombuffer(data_bytes, dtype=np.float64)
                quadtree.data_buffer = data_array.tolist()
            
            return quadtree

