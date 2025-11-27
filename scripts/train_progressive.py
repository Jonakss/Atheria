
import os
import sys
import time
import json
import shutil
import psutil
import torch
import subprocess
from pathlib import Path
from datetime import datetime, timedelta
from types import SimpleNamespace
import matplotlib.pyplot as plt

# ============================================================================
# 1. CONFIGURACIÓN DE ENTORNO
# ============================================================================

# Detectar entorno
IN_KAGGLE = os.path.exists("/kaggle/input") or os.path.exists("/kaggle/working")
IN_COLAB = False
if not IN_KAGGLE:
    try:
        import google.colab
        IN_COLAB = os.path.exists("/content")
    except:
        IN_COLAB = False

ENV_NAME = "Kaggle" if IN_KAGGLE else "Colab" if IN_COLAB else "Local"
print(f"🌍 Entorno detectado: {ENV_NAME}")

# Configurar directorios
if IN_COLAB:
    from google.colab import drive
    print("📁 Montando Google Drive...")
    drive.mount('/content/drive')
    DRIVE_ROOT = Path("/content/drive/MyDrive/Atheria")
elif IN_KAGGLE:
    DRIVE_ROOT = Path("/kaggle/working/atheria_checkpoints")
else:
    # Local: usar carpeta en home o en el proyecto
    DRIVE_ROOT = Path.home() / "atheria_checkpoints"

print(f"📁 Directorio de Checkpoints/Logs: {DRIVE_ROOT}")

# Crear estructura de carpetas
DRIVE_CHECKPOINT_DIR = DRIVE_ROOT / "checkpoints"
DRIVE_LOGS_DIR = DRIVE_ROOT / "logs"
DRIVE_EXPORTS_DIR = DRIVE_ROOT / "exports"

for directory in [DRIVE_CHECKPOINT_DIR, DRIVE_LOGS_DIR, DRIVE_EXPORTS_DIR]:
    directory.mkdir(parents=True, exist_ok=True)

# Configurar PROJECT_ROOT
if IN_KAGGLE:
    PROJECT_ROOT = Path("/kaggle/working/Atheria")
    if not PROJECT_ROOT.exists():
        subprocess.run(["git", "clone", "https://github.com/Jonakss/Atheria.git", "/kaggle/working/Atheria"])
elif IN_COLAB:
    PROJECT_ROOT = Path("/content/Atheria")
    if not PROJECT_ROOT.exists():
        subprocess.run(["git", "clone", "https://github.com/Jonakss/Atheria.git", "/content/Atheria"])
else:
    # Local: Asumimos que el script está en scripts/ o root
    # Si __file__ es scripts/train_progressive.py, parent es scripts, parent.parent es root
    current_file = Path(__file__).resolve()
    if current_file.parent.name == "scripts":
        PROJECT_ROOT = current_file.parent.parent
    else:
        PROJECT_ROOT = Path.cwd()

print(f"📁 Proyecto configurado en: {PROJECT_ROOT}")

# Agregar src al path
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Imports del proyecto
try:
    from src.trainers.qc_trainer_v4 import QC_Trainer_v4
    from src.model_loader import instantiate_model, load_weights, load_checkpoint_data, load_model
except ImportError as e:
    print(f"❌ Error importando módulos del proyecto: {e}")
    print("Asegúrate de estar en la raíz del proyecto o que PYTHONPATH esté configurado.")
    sys.exit(1)

# ============================================================================
# 2. CLASES Y UTILIDADES
# ============================================================================

class ResourceMonitor:
    """Monitor de recursos GPU/RAM/Tiempo"""
    
    def __init__(self, max_training_hours=10):
        self.start_time = time.time()
        self.max_training_seconds = max_training_hours * 3600
        self.max_training_hours = max_training_hours
        
    def get_gpu_usage(self):
        if not torch.cuda.is_available():
            return 0.0
        try:
            return torch.cuda.utilization()
        except:
            return 0.0
    
    def get_gpu_memory(self):
        if not torch.cuda.is_available():
            return 0.0, 0.0
        allocated = torch.cuda.memory_allocated() / 1e9
        reserved = torch.cuda.memory_reserved() / 1e9
        return allocated, reserved
    
    def get_ram_usage(self):
        mem = psutil.virtual_memory()
        return mem.used / 1e9, mem.total / 1e9
    
    def get_elapsed_time(self):
        elapsed = time.time() - self.start_time
        remaining = max(0, self.max_training_seconds - elapsed)
        return elapsed, remaining
    
    def should_stop(self):
        elapsed, remaining = self.get_elapsed_time()
        return remaining < (self.max_training_seconds * 0.1)
    
    def get_status_str(self):
        gpu_usage = self.get_gpu_usage()
        gpu_mem_used, gpu_mem_reserved = self.get_gpu_memory()
        ram_used, ram_total = self.get_ram_usage()
        elapsed, remaining = self.get_elapsed_time()
        
        elapsed_str = str(timedelta(seconds=int(elapsed)))
        remaining_str = str(timedelta(seconds=int(remaining)))
        
        return (
            f"📊 RECURSOS: GPU: {gpu_usage:.1f}% | Mem: {gpu_mem_used:.2f}/{gpu_mem_reserved:.2f}GB | "
            f"RAM: {ram_used:.2f}/{ram_total:.2f}GB | "
            f"Tiempo: {elapsed_str} (Restante: {remaining_str})"
        )

def find_latest_checkpoint(checkpoint_dir):
    if not checkpoint_dir.exists():
        return None
    checkpoints = list(checkpoint_dir.glob("*.pth"))
    if not checkpoints:
        return None
    
    last_model = checkpoint_dir / "last_model.pth"
    if last_model.exists():
        return str(last_model)
    
    latest = max(checkpoints, key=lambda p: p.stat().st_mtime)
    return str(latest)

def sync_checkpoint_to_drive(local_path, drive_dir, filename=None):
    try:
        if filename is None:
            filename = Path(local_path).name
        drive_path = drive_dir / filename
        shutil.copy2(local_path, drive_path)
        return True
    except Exception as e:
        print(f"❌ Error sincronizando a Drive: {e}")
        return False

# ============================================================================
# 3. CONFIGURACIÓN DEL CURRICULUM
# ============================================================================

TRAINING_PHASES = [
    {
        "PHASE_NAME": "Fase1_Vacuum_Stability",
        "LOAD_FROM_PHASE": None,
        "MODEL_ARCHITECTURE": "UNET",
        "MODEL_PARAMS": {"d_state": 4, "hidden_channels": 32},
        "GRID_SIZE_TRAINING": 32,
        "QCA_STEPS_TRAINING": 50,
        "LR_RATE_M": 1e-4,
        "GAMMA_DECAY": 0.001,
        "TOTAL_EPISODES": 500,
        "SAVE_EVERY_EPISODES": 50,
    },
    {
        "PHASE_NAME": "Fase2_Matter_Emergence",
        "LOAD_FROM_PHASE": "Fase1_Vacuum_Stability",
        "MODEL_ARCHITECTURE": "UNET",
        "MODEL_PARAMS": {"d_state": 4, "hidden_channels": 64},
        "GRID_SIZE_TRAINING": 64,
        "QCA_STEPS_TRAINING": 100,
        "LR_RATE_M": 5e-5,
        "GAMMA_DECAY": 0.01,
        "TOTAL_EPISODES": 1000,
        "SAVE_EVERY_EPISODES": 20,
    },
]

GLOBAL_CONFIG = {
    "DRIVE_SYNC_EVERY": 50,
    "MAX_TRAINING_HOURS": 10,
    "AUTO_RESUME": True,
    "MAX_CHECKPOINTS_TO_KEEP": 3,
}

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
EXPERIMENT_ROOT_NAME = "MultiPhase_Experiment_v1"
BASE_CHECKPOINT_DIR = DRIVE_CHECKPOINT_DIR / EXPERIMENT_ROOT_NAME
BASE_LOG_DIR = DRIVE_LOGS_DIR / EXPERIMENT_ROOT_NAME
BASE_CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
BASE_LOG_DIR.mkdir(parents=True, exist_ok=True)

monitor = ResourceMonitor(max_training_hours=GLOBAL_CONFIG["MAX_TRAINING_HOURS"])

# ============================================================================
# 4. LOOP PRINCIPAL DE ENTRENAMIENTO
# ============================================================================

for phase_idx, phase_cfg in enumerate(TRAINING_PHASES):
    PHASE_NAME = phase_cfg["PHASE_NAME"]
    
    print("\n" + "#" * 70)
    print(f"🚀 INICIANDO FASE {phase_idx+1}/{len(TRAINING_PHASES)}: {PHASE_NAME}")
    print("#" * 70)
    
    # Configuración de Directorios
    PHASE_CHECKPOINT_DIR = BASE_CHECKPOINT_DIR / PHASE_NAME
    PHASE_LOG_DIR = BASE_LOG_DIR / PHASE_NAME
    PHASE_CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    PHASE_LOG_DIR.mkdir(parents=True, exist_ok=True)
    
    LOCAL_PHASE_DIR = PROJECT_ROOT / "output" / "checkpoints" / EXPERIMENT_ROOT_NAME / PHASE_NAME
    LOCAL_PHASE_DIR.mkdir(parents=True, exist_ok=True)
    
    # Verificar si la fase ya está completada
    final_marker = PHASE_CHECKPOINT_DIR / "PHASE_COMPLETED.marker"
    if final_marker.exists() and GLOBAL_CONFIG["AUTO_RESUME"]:
        print(f"✅ Fase {PHASE_NAME} ya completada. Saltando...")
        continue

    # Configuración de Fase
    current_exp_cfg = SimpleNamespace(**phase_cfg)
    current_exp_cfg.MODEL_PARAMS = SimpleNamespace(**phase_cfg["MODEL_PARAMS"])
    current_exp_cfg.DEVICE = device
    
    # Instanciar Modelo
    print(f"🛠️ Instanciando modelo {phase_cfg['MODEL_ARCHITECTURE']}...")
    model = instantiate_model(current_exp_cfg)
    
    # Cargar Pesos
    resume_from_episode = 0
    weights_loaded = False
    
    # A) Resume fase actual
    if GLOBAL_CONFIG["AUTO_RESUME"]:
        latest_ckpt = find_latest_checkpoint(PHASE_CHECKPOINT_DIR)
        if not latest_ckpt:
            latest_ckpt = find_latest_checkpoint(LOCAL_PHASE_DIR)
            
        if latest_ckpt:
            print(f"🔄 Resumiendo fase actual desde: {Path(latest_ckpt).name}")
            ckpt_data = load_checkpoint_data(latest_ckpt)
            if ckpt_data:
                load_weights(model, ckpt_data)
                resume_from_episode = ckpt_data.get('episode', 0)
                weights_loaded = True
    
    # B) Transfer Learning
    if not weights_loaded and phase_cfg["LOAD_FROM_PHASE"]:
        prev_phase_name = phase_cfg["LOAD_FROM_PHASE"]
        print(f"📥 Buscando pesos de fase anterior: {prev_phase_name}")
        
        prev_dir = BASE_CHECKPOINT_DIR / prev_phase_name
        best_prev = prev_dir / "best_model.pth"
        if not best_prev.exists():
             best_prev = find_latest_checkpoint(prev_dir)
             
        if best_prev and Path(best_prev).exists():
            print(f"✅ Cargando pesos previos de: {Path(best_prev).name}")
            ckpt_data = load_checkpoint_data(str(best_prev))
            
            model_state = model.state_dict()
            pretrained_state = ckpt_data['model_state_dict']
            filtered_state = {k: v for k, v in pretrained_state.items() 
                            if k in model_state and v.shape == model_state[k].shape}
            
            missing, unexpected = model.load_state_dict(filtered_state, strict=False)
            print(f"ℹ️ Transfer Learning: {len(missing)} capas nuevas, {len(unexpected)} descartadas.")
            weights_loaded = True
        else:
            print(f"⚠️ No se encontraron pesos previos. Iniciando desde cero.")
            
    if not weights_loaded:
        print("🆕 Iniciando fase con pesos aleatorios.")

    # Inicializar Trainer
    trainer = QC_Trainer_v4(
        experiment_name=f"{EXPERIMENT_ROOT_NAME}/{PHASE_NAME}",
        model=model,
        model_params=phase_cfg['MODEL_PARAMS'],
        device=device,
        lr=phase_cfg["LR_RATE_M"],
        grid_size=phase_cfg["GRID_SIZE_TRAINING"],
        qca_steps=phase_cfg["QCA_STEPS_TRAINING"],
        gamma_decay=phase_cfg["GAMMA_DECAY"],
        max_checkpoints_to_keep=GLOBAL_CONFIG["MAX_CHECKPOINTS_TO_KEEP"]
    )
    trainer.checkpoint_dir = str(LOCAL_PHASE_DIR)
    
    training_log = {"episodes": [], "losses": []}
    
    print(f"\n▶️ Entrenando {phase_cfg['TOTAL_EPISODES'] - resume_from_episode} episodios...")
    
    try:
        for episode in range(resume_from_episode, phase_cfg["TOTAL_EPISODES"]):
            
            if monitor.should_stop():
                print(f"\n⏰ TIEMPO GLOBAL AGOTADO")
                trainer.save_checkpoint(episode, is_best=False)
                sync_checkpoint_to_drive(str(LOCAL_PHASE_DIR / f"checkpoint_ep{episode}.pth"), PHASE_CHECKPOINT_DIR)
                raise KeyboardInterrupt("Global Timeout")
                
            loss, metrics = trainer.train_episode(episode)
            
            training_log["episodes"].append(episode)
            training_log["losses"].append(loss)
            
            if (episode + 1) % phase_cfg["SAVE_EVERY_EPISODES"] == 0:
                is_best = (episode + 1) % (phase_cfg["SAVE_EVERY_EPISODES"] * 2) == 0
                trainer.save_checkpoint(episode, is_best=is_best)
                
            if (episode + 1) % GLOBAL_CONFIG["DRIVE_SYNC_EVERY"] == 0:
                sync_checkpoint_to_drive(str(LOCAL_PHASE_DIR / "last_checkpoint.pth"), PHASE_CHECKPOINT_DIR, "last_model.pth")
                if (LOCAL_PHASE_DIR / "best_model.pth").exists():
                    sync_checkpoint_to_drive(str(LOCAL_PHASE_DIR / "best_model.pth"), PHASE_CHECKPOINT_DIR)
                print(f"☁️ Sync Drive Ep {episode}")

            if (episode + 1) % 10 == 0:
                print(f"Ep {episode+1}/{phase_cfg['TOTAL_EPISODES']} | Loss: {loss:.4f} | {monitor.get_status_str()}")
        
        print(f"\n✅ FASE {PHASE_NAME} COMPLETADA")
        
        with open(final_marker, 'w') as f:
            f.write(f"Completed at {datetime.now()}")
            
        sync_checkpoint_to_drive(str(LOCAL_PHASE_DIR / "best_model.pth"), PHASE_CHECKPOINT_DIR)
        
    except KeyboardInterrupt:
        print("\n🛑 Entrenamiento interrumpido.")
        break
    except Exception as e:
        print(f"\n❌ Error en fase {PHASE_NAME}: {e}")
        raise e

# ============================================================================
# 5. EXPORTACIÓN FINAL
# ============================================================================

print("\n" + "="*70)
print("🏁 ENTRENAMIENTO FINALIZADO. EXPORTANDO...")
print("="*70)

last_phase = TRAINING_PHASES[-1]
last_phase_name = last_phase["PHASE_NAME"]
LAST_PHASE_CHECKPOINT_DIR = BASE_CHECKPOINT_DIR / last_phase_name

best_checkpoint = LAST_PHASE_CHECKPOINT_DIR / "best_model.pth"
if not best_checkpoint.exists():
    best_checkpoint = find_latest_checkpoint(LAST_PHASE_CHECKPOINT_DIR)

if best_checkpoint:
    print(f"📥 Cargando modelo para exportación: {Path(best_checkpoint).name}")
    
    export_cfg = SimpleNamespace(**last_phase)
    export_cfg.MODEL_PARAMS = SimpleNamespace(**last_phase["MODEL_PARAMS"])
    export_cfg.DEVICE = device
    export_cfg.EXPERIMENT_NAME = f"{EXPERIMENT_ROOT_NAME}_{last_phase_name}"
    
    model = load_model(export_cfg, str(best_checkpoint))
    model.eval()
    model.to(device)
    
    d_state = last_phase["MODEL_PARAMS"]["d_state"]
    grid_size = last_phase["GRID_SIZE_TRAINING"]
    example_input = torch.randn(1, 2 * d_state, grid_size, grid_size, device=device)
    
    torchscript_path = DRIVE_EXPORTS_DIR / f"{EXPERIMENT_ROOT_NAME}_{last_phase_name}_model.pt"
    
    try:
        traced_model = torch.jit.trace(model, example_input, strict=False)
        traced_model.save(str(torchscript_path))
        print(f"✅ Modelo TorchScript exportado: {torchscript_path.name}")
    except Exception as e:
        print(f"⚠️ Error exportando TorchScript: {e}")
    
    export_checkpoint_path = DRIVE_EXPORTS_DIR / f"{EXPERIMENT_ROOT_NAME}_{last_phase_name}_best.pth"
    shutil.copy2(best_checkpoint, export_checkpoint_path)
    print(f"✅ Mejor checkpoint exportado: {export_checkpoint_path.name}")
    
    report_path = DRIVE_EXPORTS_DIR / f"{EXPERIMENT_ROOT_NAME}_REPORT.md"
    with open(report_path, 'w') as f:
        f.write(f"# Reporte: {EXPERIMENT_ROOT_NAME}\n\n")
        f.write(f"**Fecha:** {datetime.now()}\n\n")
        f.write(f"## Fases\n")
        for i, phase in enumerate(TRAINING_PHASES):
             f.write(f"- {phase['PHASE_NAME']}: {phase['TOTAL_EPISODES']} eps\n")
    
    print(f"✅ Reporte generado: {report_path.name}")
else:
    print("⚠️ No se encontró checkpoint final para exportar.")
