"""
ExperimentLogger: Sistema de documentación automática para experimentos de Atheria 4.

Este módulo proporciona una clase para generar y mantener archivos Markdown
con el historial de resultados de entrenamiento, especialmente para los hitos
importantes (is_best checkpoints).
"""

import os
import json
from datetime import datetime
from typing import Dict, Any, Optional, List
import logging

# Configurar logging
logging.basicConfig(level=logging.INFO)


class ExperimentLogger:
    """
    Logger para documentar experimentos de Atheria en archivos Markdown.
    
    Mantiene un archivo Markdown por experimento en `docs/40_Experiments/`
    con una tabla histórica de resultados y metadatos del experimento.
    """
    
    def __init__(self, experiment_name: str, docs_base_dir: str = "docs/40_Experiments"):
        """
        Inicializa el logger para un experimento.
        
        Args:
            experiment_name: Nombre del experimento (ej: "UNET_32ch_D5_LR2e-5")
            docs_base_dir: Directorio base para documentación (por defecto docs/40_Experiments)
        """
        # Resolver ruta absoluta relativa al root del proyecto si es relativa
        if not os.path.isabs(docs_base_dir):
            # Asumiendo estructura: root/src/utils/experiment_logger.py
            current_file = os.path.abspath(__file__)
            project_root = os.path.dirname(os.path.dirname(os.path.dirname(current_file)))
            docs_base_dir = os.path.join(project_root, docs_base_dir)

        self.experiment_name = experiment_name
        self.docs_base_dir = docs_base_dir
        self.log_file = os.path.join(docs_base_dir, f"{experiment_name}.md")
        
        # Asegurar que el directorio existe (incluyendo subdirectorios del experimento)
        log_dir = os.path.dirname(self.log_file)
        os.makedirs(log_dir, exist_ok=True)
        
        # Historial de resultados (se carga desde el archivo si existe)
        self.results_history: List[Dict[str, Any]] = []
        
        # Configuración del experimento (se inicializa con initialize_or_load)
        self.config: Optional[Dict[str, Any]] = None
        
        logging.info(f"ExperimentLogger inicializado para: {experiment_name}")
    
    def initialize_or_load(self, config: Dict[str, Any]) -> None:
        """
        Inicializa el archivo Markdown o carga uno existente.
        
        Args:
            config: Diccionario con la configuración del experimento
                   (ej: {'model_architecture': 'UNET', 'lr': 0.0001, ...})
        """
        self.config = config
        
        # Si el archivo ya existe, cargar el historial existente
        if os.path.exists(self.log_file):
            try:
                self._load_existing_log()
                logging.info(f"Log existente cargado desde: {self.log_file}")
            except Exception as e:
                logging.warning(f"Error al cargar log existente: {e}. Creando nuevo log.")
                self.results_history = []
        else:
            # Crear nuevo archivo con encabezado
            self._create_new_log(config)
            logging.info(f"Nuevo log creado en: {self.log_file}")
    
    def log_result(self, 
                   episodes: int,
                   metrics: Dict[str, float],
                   loss: float,
                   is_best: bool = False,
                   checkpoint_path: Optional[str] = None) -> None:
        """
        Registra un resultado de entrenamiento en el log Markdown.
        
        Args:
            episodes: Número de episodios de entrenamiento
            metrics: Diccionario con métricas (ej: {'survival': 0.5, 'symmetry': 0.3, ...})
            loss: Pérdida total del episodio
            is_best: Si este checkpoint es el mejor hasta ahora
            checkpoint_path: Ruta al archivo de checkpoint (opcional)
        """
        timestamp = datetime.now().isoformat()
        
        result_entry = {
            'timestamp': timestamp,
            'episodes': episodes,
            'loss': loss,
            'metrics': metrics.copy(),
            'is_best': is_best,
            'checkpoint_path': checkpoint_path
        }
        
        # Añadir a historial
        self.results_history.append(result_entry)
        
        # Actualizar el archivo Markdown
        self._update_log_file(result_entry, is_best)
        
        if is_best:
            logging.info(f"✅ Hito registrado: Episodio {episodes} - Loss: {loss:.6f} - Mejor checkpoint hasta ahora")
    
    def _create_new_log(self, config: Dict[str, Any]) -> None:
        """Crea un nuevo archivo Markdown con la configuración inicial."""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        content = f"""# Experimento: {self.experiment_name}

**Fecha de Creación:** {timestamp}

## Configuración del Experiment

```json
{json.dumps(config, indent=2, ensure_ascii=False)}
```

## Historial de Resultados

### Tabla de Hitos (Mejores Checkpoints)

| Episodio | Fecha | Loss Total | Survival | Symmetry | Complexity | Métrica Combinada | Checkpoint |
|----------|-------|------------|----------|----------|------------|-------------------|------------|
"""
        
        with open(self.log_file, 'w', encoding='utf-8') as f:
            f.write(content)
    
    def _load_existing_log(self) -> None:
        """
        Carga el historial desde un archivo Markdown existente.
        
        Nota: Esta implementación es básica. Para una carga completa,
        se necesitaría parsear el Markdown completamente. Por ahora,
        solo verificamos que el archivo existe.
        """
        # Por simplicidad, no parseamos el Markdown existente.
        # Solo mantenemos el historial en memoria durante la sesión.
        # Si necesitamos persistir entre sesiones, se podría usar un JSON auxiliar.
        pass
    
    def _update_log_file(self, result_entry: Dict[str, Any], is_best: bool) -> None:
        """
        Actualiza el archivo Markdown con un nuevo resultado.
        
        Solo actualiza si es un hito importante (is_best=True), para evitar
        que el archivo crezca demasiado rápido.
        """
        if not is_best:
            # No actualizamos el archivo para checkpoints normales
            # Solo mantenemos en memoria
            return
        
        # Leer el contenido actual
        if os.path.exists(self.log_file):
            with open(self.log_file, 'r', encoding='utf-8') as f:
                content = f.read()
        else:
            # Si no existe, crear uno nuevo
            self._create_new_log(self.config or {})
            with open(self.log_file, 'r', encoding='utf-8') as f:
                content = f.read()
        
        # Calcular métrica combinada (para V4: survival + symmetry ponderados)
        metrics = result_entry['metrics']
        survival = metrics.get('survival', 0.0)
        symmetry = metrics.get('symmetry', 0.0)
        complexity = metrics.get('complexity', 0.0)
        
        # Métrica combinada: menor es mejor (pérdida normalizada)
        # Para ordenar checkpoints, queremos maximizar supervivencia y simetría
        # Entonces usamos el negativo de survival + symmetry (o su inverso ponderado)
        # Como survival y symmetry son pérdidas, menor es mejor
        # Para ordenar "mejores", queremos el menor valor de la combinación
        combined_metric = (10.0 * survival) + (5.0 * symmetry)  # Misma ponderación que en loss_function_evolutionary
        
        # Formatear fecha
        timestamp = result_entry['timestamp']
        try:
            dt = datetime.fromisoformat(timestamp)
            date_str = dt.strftime("%Y-%m-%d %H:%M:%S")
        except:
            date_str = timestamp
        
        # Ruta del checkpoint (relativa o absoluta)
        checkpoint_str = result_entry.get('checkpoint_path', 'N/A')
        if checkpoint_str and checkpoint_str != 'N/A':
            # Hacer ruta relativa si es absoluta
            if os.path.isabs(checkpoint_str):
                # Intentar hacerla relativa al proyecto
                try:
                    import sys
                    project_root = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
                    checkpoint_str = os.path.relpath(checkpoint_str, project_root)
                except:
                    pass
        
        # Nueva fila de tabla
        new_row = (
            f"| {result_entry['episodes']} | {date_str} | "
            f"{result_entry['loss']:.6f} | "
            f"{survival:.6f} | {symmetry:.6f} | {complexity:.6f} | "
            f"{combined_metric:.6f} | `{checkpoint_str}` |\n"
        )
        
        # Insertar la nueva fila después del encabezado de la tabla
        # Buscar el final de la tabla (antes de cualquier sección adicional)
        lines = content.split('\n')
        table_end_idx = len(lines)
        
        # Buscar si hay una sección adicional después de la tabla
        for i, line in enumerate(lines):
            if line.strip().startswith('##') and i > 10:  # Después de la tabla inicial
                table_end_idx = i
                break
        
        # Insertar la nueva fila después del encabezado de la tabla
        # La tabla comienza después de "| Episodio | ..."
        for i, line in enumerate(lines):
            if line.strip().startswith('| Episodio'):
                # Insertar después del separador de la tabla (siguiente línea que sea "|----------|")
                for j in range(i + 1, min(i + 3, len(lines))):
                    if '---' in lines[j] or '|' in lines[j] and '---' in lines[j]:
                        # Insertar después del separador
                        lines.insert(j + 1, new_row.strip())
                        break
                else:
                    # Si no hay separador, insertar después del encabezado
                    lines.insert(i + 2, new_row.strip())
                break
        
        # Escribir el contenido actualizado
        with open(self.log_file, 'w', encoding='utf-8') as f:
            f.write('\n'.join(lines))
        
        logging.debug(f"Log actualizado con hito en episodio {result_entry['episodes']}")
    
    def get_best_results(self, n: int = 5) -> List[Dict[str, Any]]:
        """
        Retorna los N mejores resultados según la métrica combinada.
        
        Args:
            n: Número de mejores resultados a retornar
            
        Returns:
            Lista de diccionarios con los mejores resultados ordenados
        """
        if not self.results_history:
            return []
        
        # Filtrar solo los que son "best"
        best_results = [r for r in self.results_history if r.get('is_best', False)]
        
        # Ordenar por métrica combinada (menor es mejor)
        def get_combined_metric(result):
            metrics = result['metrics']
            survival = metrics.get('survival', 0.0)
            symmetry = metrics.get('symmetry', 0.0)
            return (10.0 * survival) + (5.0 * symmetry)
        
        best_results.sort(key=get_combined_metric)
        
        return best_results[:n]

