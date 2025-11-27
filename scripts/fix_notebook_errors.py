import json
import nbformat

NOTEBOOK_PATH = "notebooks/Atheria_Progressive_Training.ipynb"

def fix_notebook():
    print(f"Reading {NOTEBOOK_PATH}...")
    with open(NOTEBOOK_PATH, 'r', encoding='utf-8') as f:
        nb = nbformat.read(f, as_version=4)

    new_cells = []
    
    # 1. Remove Section 6 (Auto-Resume Check)
    skip_next = False
    for cell in nb.cells:
        # Check for Section 6 markdown
        if cell.cell_type == 'markdown' and "Sección 6" in cell.source:
            print("Removing Section 6 Markdown cell")
            skip_next = True 
            continue
        
        if skip_next:
            if cell.cell_type == 'code':
                print("Removing Section 6 Code cell")
                skip_next = False
                continue
            else:
                skip_next = False
        
        # 2. Update Section 9 (Export)
        if cell.cell_type == 'code' and "Exportando modelo final" in cell.source:
            print("Updating Section 9 Code cell")
            cell.source = update_export_code()
            
        new_cells.append(cell)

    nb.cells = new_cells
    
    print(f"Writing updated notebook to {NOTEBOOK_PATH}...")
    with open(NOTEBOOK_PATH, 'w', encoding='utf-8') as f:
        nbformat.write(nb, f)
    print("Done!")

def update_export_code():
    return """from src.model_loader import load_model
import shutil
from types import SimpleNamespace

print("📤 Exportando modelo final de la ÚLTIMA FASE...\\n")

# Obtener configuración de la última fase
last_phase = TRAINING_PHASES[-1]
last_phase_name = last_phase["PHASE_NAME"]
print(f"🔹 Última fase: {last_phase_name}")

# Directorios de la última fase
LAST_PHASE_CHECKPOINT_DIR = BASE_CHECKPOINT_DIR / last_phase_name
LAST_PHASE_LOG_DIR = BASE_LOG_DIR / last_phase_name

# Encontrar mejor checkpoint de la última fase
best_checkpoint = LAST_PHASE_CHECKPOINT_DIR / "best_model.pth"
if not best_checkpoint.exists():
    best_checkpoint = find_latest_checkpoint(LAST_PHASE_CHECKPOINT_DIR)

if best_checkpoint:
    print(f"📥 Cargando modelo desde: {Path(best_checkpoint).name}")
    
    # Preparar config para cargar modelo
    # Necesitamos reconstruir el objeto config esperado por load_model
    export_cfg = SimpleNamespace(**last_phase)
    export_cfg.MODEL_PARAMS = SimpleNamespace(**last_phase["MODEL_PARAMS"])
    export_cfg.DEVICE = device
    export_cfg.EXPERIMENT_NAME = f"{EXPERIMENT_ROOT_NAME}_{last_phase_name}"
    
    # Cargar modelo
    model = load_model(export_cfg, str(best_checkpoint))
    model.eval()
    model.to(device)
    
    # Exportar a TorchScript
    # Usamos dimensiones de la última fase
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
    
    # Copiar mejor checkpoint a Drive (Exports)
    export_checkpoint_path = DRIVE_EXPORTS_DIR / f"{EXPERIMENT_ROOT_NAME}_{last_phase_name}_best.pth"
    shutil.copy2(best_checkpoint, export_checkpoint_path)
    print(f"✅ Mejor checkpoint exportado: {export_checkpoint_path.name}")
    
    # Generar reporte de entrenamiento
    report_path = DRIVE_EXPORTS_DIR / f"{EXPERIMENT_ROOT_NAME}_REPORT.md"
    
    with open(report_path, 'w') as f:
        f.write(f"# Reporte de Entrenamiento: {EXPERIMENT_ROOT_NAME}\\n\\n")
        f.write(f"**Fecha:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\\n\\n")
        f.write(f"## Fases Completadas\\n\\n")
        
        for i, phase in enumerate(TRAINING_PHASES):
             f.write(f"### Fase {i+1}: {phase['PHASE_NAME']}\\n")
             f.write(f"- Grid: {phase['GRID_SIZE_TRAINING']}x{phase['GRID_SIZE_TRAINING']}\\n")
             f.write(f"- Episodes: {phase['TOTAL_EPISODES']}\\n\\n")
        
        f.write(f"## Archivos Exportados\\n\\n")
        f.write(f"- TorchScript: `{torchscript_path.name}`\\n")
        f.write(f"- Checkpoint: `{export_checkpoint_path.name}`\\n")
    
    print(f"✅ Reporte generado: {report_path.name}")
    
else:
    print("⚠️ No se encontró checkpoint para exportar en la última fase")

print("\\n" + "=" * 70)
print("🎉 EXPORTACIÓN COMPLETADA")
print("=" * 70)
print(f"\\n📂 Todos los archivos están en Drive: {DRIVE_EXPORTS_DIR}")
"""

if __name__ == "__main__":
    fix_notebook()
