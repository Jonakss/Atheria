# src/managers/snapshot_manager.py
"""
Gestor de Snapshots de Inferencia.

Permite guardar y cargar el estado completo de una simulación para reanudarla.
"""
import logging
import torch
import json
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Tuple

SNAPSHOT_DIR = Path("output/inference_snapshots")
SNAPSHOT_DIR.mkdir(parents=True, exist_ok=True)

logger = logging.getLogger(__name__)

def save_snapshot(experiment_name: str, step: int, psi: torch.Tensor, metadata: Optional[Dict] = None) -> Optional[Path]:
    """
    Guarda el estado de la simulación (psi) en un archivo.

    Args:
        experiment_name (str): Nombre del experimento activo.
        step (int): Paso actual de la simulación.
        psi (torch.Tensor): El tensor de estado a guardar.
        metadata (Optional[Dict]): Metadatos adicionales para guardar.

    Returns:
        Optional[Path]: La ruta al archivo de snapshot guardado, o None si falla.
    """
    if not experiment_name:
        logger.error("Error al guardar snapshot: El nombre del experimento es inválido.")
        return None

    try:
        # Crear subdirectorio para el experimento si no existe
        experiment_dir = SNAPSHOT_DIR / experiment_name
        experiment_dir.mkdir(parents=True, exist_ok=True)

        # Usar timestamp para asegurar nombres únicos y facilitar ordenamiento
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename_base = f"snapshot_{timestamp}_step_{step}"
        filepath_pt = experiment_dir / f"{filename_base}.pt"
        filepath_json = experiment_dir / f"{filename_base}.json"

        # Guardar el tensor
        torch.save(psi, filepath_pt)

        # Guardar metadatos
        snapshot_metadata = {
            "experiment_name": experiment_name,
            "step": step,
            "timestamp": datetime.now().isoformat(),
            "tensor_shape": list(psi.shape),
            "tensor_dtype": str(psi.dtype),
            "filepath_pt": str(filepath_pt),
        }
        if metadata:
            snapshot_metadata.update(metadata)

        with open(filepath_json, 'w') as f:
            json.dump(snapshot_metadata, f, indent=4)

        logger.info(f"✅ Snapshot guardado exitosamente en: {filepath_pt}")
        return filepath_pt

    except Exception as e:
        logger.error(f"❌ Error catastrófico al guardar el snapshot: {e}", exc_info=True)
        return None


def list_snapshots(experiment_name: str) -> List[Dict]:
    """
    Lista todos los snapshots disponibles para un experimento.

    Args:
        experiment_name (str): Nombre del experimento.

    Returns:
        List[Dict]: Una lista de metadatos de los snapshots, ordenada del más reciente al más antiguo.
    """
    snapshots = []
    experiment_dir = SNAPSHOT_DIR / experiment_name
    if not experiment_dir.exists():
        return []

    # Buscar archivos JSON, que contienen los metadatos
    for filepath_json in experiment_dir.glob("*.json"):
        try:
            with open(filepath_json, 'r') as f:
                metadata = json.load(f)

            # Verificar que el archivo de tensor correspondiente exista
            filepath_pt = Path(metadata.get("filepath_pt", ""))
            if filepath_pt.exists():
                snapshots.append(metadata)
            else:
                logger.warning(f"Se encontró el metadata '{filepath_json}' pero falta el archivo de tensor '{filepath_pt}'.")

        except Exception as e:
            logger.error(f"Error al leer el metadata del snapshot '{filepath_json}': {e}")

    # Ordenar por timestamp, del más reciente al más antiguo
    snapshots.sort(key=lambda x: x.get("timestamp", ""), reverse=True)

    return snapshots


def load_snapshot(snapshot_path: str) -> Optional[Tuple[torch.Tensor, Dict]]:
    """
    Carga el estado de la simulación desde un archivo de snapshot.

    Args:
        snapshot_path (str): La ruta al archivo .pt del snapshot.

    Returns:
        Optional[Tuple[torch.Tensor, Dict]]: Una tupla conteniendo el tensor de estado (psi)
        y sus metadatos, o None si falla la carga.
    """
    filepath_pt = Path(snapshot_path)
    filepath_json = filepath_pt.with_suffix(".json")

    if not filepath_pt.exists() or not filepath_json.exists():
        logger.error(f"Error al cargar snapshot: No se encontró el archivo de tensor '{filepath_pt}' o de metadata '{filepath_json}'.")
        return None

    try:
        # Cargar el tensor
        # Mapear a la CPU para evitar problemas si se guardó en GPU y ahora no hay
        psi = torch.load(filepath_pt, map_location=torch.device('cpu'))

        # Cargar metadatos
        with open(filepath_json, 'r') as f:
            metadata = json.load(f)

        logger.info(f"✅ Snapshot cargado exitosamente desde: {filepath_pt}")
        return psi, metadata

    except Exception as e:
        logger.error(f"❌ Error catastrófico al cargar el snapshot '{filepath_pt}': {e}", exc_info=True)
        return None
