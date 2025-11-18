# src/gpu_optimizer.py
"""
Módulo de optimización de GPU y CPU para simulaciones.
Gestiona el uso eficiente de recursos y evita transferencias innecesarias.
"""
import torch
import logging
from typing import Optional

class GPUOptimizer:
    """
    Optimizador de recursos GPU/CPU para simulaciones.
    """
    
    def __init__(self, device: torch.device):
        self.device = device
        self.is_cuda = device.type == 'cuda'
        self.empty_cache_interval = 50  # Limpiar cache cada N pasos (reducido para mejor gestión de memoria)
        self.step_count = 0
        
        # Configurar allocator de CUDA para evitar fragmentación
        if self.is_cuda:
            import os
            # Configurar expandable_segments para mejor gestión de memoria
            # Usar PYTORCH_ALLOC_CONF (PYTORCH_CUDA_ALLOC_CONF está deprecado)
            os.environ.setdefault('PYTORCH_ALLOC_CONF', 'expandable_segments:True')
        
        # Configurar optimizaciones iniciales
        if self.is_cuda:
            # Habilitar optimizaciones de memoria CUDA
            torch.backends.cudnn.benchmark = True  # Optimizar para tamaños fijos
            torch.backends.cudnn.deterministic = False  # Permitir optimizaciones no deterministas (más rápido)
            logging.info("Optimizaciones CUDA habilitadas")
        else:
            logging.info("Modo CPU: optimizaciones limitadas")
    
    def optimize_model(self, model: torch.nn.Module):
        """
        Optimiza un modelo para inferencia.
        
        Args:
            model: Modelo a optimizar
        
        Nota: No compila el modelo aquí, eso se hace después con compile_model()
        para mantener una referencia al modelo original.
        """
        # Poner en modo evaluación (desactiva dropout, batch norm en modo train, etc.)
        model.eval()
        
        # NO compilar aquí - la compilación se hace después en compile_model()
        # para mantener una referencia al modelo original
        
        return model
    
    def empty_cache_if_needed(self):
        """
        Limpia la caché de GPU si es necesario.
        Debe llamarse periódicamente durante la simulación.
        """
        self.step_count += 1
        
        if self.is_cuda and self.step_count % self.empty_cache_interval == 0:
            torch.cuda.empty_cache()
            # Sincronizar para asegurar que la limpieza se complete
            torch.cuda.synchronize()
            logging.debug(f"Cache de GPU limpiado (paso {self.step_count})")
    
    def should_keep_on_gpu(self, tensor: torch.Tensor, size_threshold_mb: float = 10.0) -> bool:
        """
        Decide si un tensor debería mantenerse en GPU o moverse a CPU.
        
        Args:
            tensor: Tensor a evaluar
            size_threshold_mb: Tamaño en MB por encima del cual considerar mover a CPU
        
        Returns:
            True si debe mantenerse en GPU, False si debe moverse a CPU
        """
        if not self.is_cuda:
            return False
        
        # Calcular tamaño aproximado en MB
        element_size = tensor.element_size()  # bytes por elemento
        num_elements = tensor.numel()
        size_mb = (element_size * num_elements) / (1024 * 1024)
        
        # Si es pequeño, mantener en GPU
        # Si es grande y no se usa frecuentemente, considerar CPU
        return size_mb < size_threshold_mb
    
    @staticmethod
    def move_to_cpu_batch(tensors: list, keep_on_gpu: Optional[list] = None):
        """
        Mueve una lista de tensores a CPU de forma eficiente.
        
        Args:
            tensors: Lista de tensores a mover
            keep_on_gpu: Lista opcional de booleanos indicando cuáles mantener en GPU
        """
        if keep_on_gpu is None:
            keep_on_gpu = [False] * len(tensors)
        
        moved = []
        for tensor, keep in zip(tensors, keep_on_gpu):
            if keep or tensor.device.type == 'cpu':
                moved.append(tensor)
            else:
                moved.append(tensor.cpu())
        
        return moved
    
    @staticmethod
    def enable_inference_mode():
        """
        Configura PyTorch para modo de inferencia optimizado.
        Desactiva gradientes y otras optimizaciones.
        """
        # Usar torch.inference_mode() si está disponible (PyTorch 1.9+)
        # Es más rápido que torch.no_grad()
        return torch.inference_mode()
    
    def get_memory_stats(self) -> dict:
        """
        Obtiene estadísticas de memoria GPU/CPU.
        
        Returns:
            Dict con estadísticas de memoria
        """
        stats = {
            'device': str(self.device),
            'step_count': self.step_count
        }
        
        if self.is_cuda:
            stats['gpu_allocated_mb'] = torch.cuda.memory_allocated(self.device) / (1024 ** 2)
            stats['gpu_reserved_mb'] = torch.cuda.memory_reserved(self.device) / (1024 ** 2)
            stats['gpu_max_allocated_mb'] = torch.cuda.max_memory_allocated(self.device) / (1024 ** 2)
        else:
            stats['cpu_memory_available'] = True
        
        return stats

# Instancia global del optimizador (se inicializa cuando se carga el motor)
_global_optimizer: Optional[GPUOptimizer] = None

def get_optimizer(device: torch.device) -> GPUOptimizer:
    """
    Obtiene o crea la instancia global del optimizador.
    
    Args:
        device: Dispositivo a usar
    
    Returns:
        Instancia de GPUOptimizer
    """
    global _global_optimizer
    if _global_optimizer is None or _global_optimizer.device != device:
        _global_optimizer = GPUOptimizer(device)
    return _global_optimizer

