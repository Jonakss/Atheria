# AGENTS.md - AI Agent Development Log

This file tracks significant changes and improvements made to the AETHERIA project by AI agents and automated systems.

---

## 2025-11-10: Major Project Refactoring and Dynamic Model System

**Commit:** `45328c6` - feat: Implement dynamic model loading and reorganize project  
**Author:** Jonathan Correa Paiva  
**Type:** Feature Enhancement + Bug Fixes

### Overview
Major architectural overhaul introducing dynamic model registration, improved project organization, and several critical bug fixes. This update establishes a more maintainable and extensible codebase.

### Key Features

#### 1. Dynamic Model Registration System
- **Created `src/models/` package** to centralize all model architectures
- **Implemented decorator-based registration**: Models auto-register themselves, eliminating manual `if/elif` chains
- **Refactored model loading pipeline**: 
  - Updated `model_loader.py` to use `models.get_model_class()`
  - Modified `pipeline_train.py` to leverage the new dynamic system
- **Models now include**:
  - `mlp.py` - MLP 1x1 operator (local processing)
  - `unet.py` - Standard U-Net with regional vision
  - `unet_unitary.py` - Unitary-constrained U-Net variant
  - `deep_qca.py` - Deep QCA architecture

#### 2. Project Structure Reorganization
Established cleaner separation of concerns:

```
aetheria/
├── web/                # Web server and UI
│   ├── app.py          # Unified Flask/aiohttp server
│   └── index.html      # Browser interface
├── scripts/            # Utility scripts
│   ├── train.py        # Training entry point
│   └── run_visualizations.py
├── notebooks/          # Jupyter analysis notebooks
├── src/                # Core engine and models
│   ├── models/         # NEW: Model architectures
│   └── ...
└── checkpoints/        # Trained model storage
```

**Benefits:**
- Root directory decluttered
- Clear functional separation
- Easier navigation and maintenance
- Absolute path resolution via `PROJECT_ROOT`

#### 3. Model Type Differentiation
- UI now explicitly distinguishes between:
  - `UNET` (Standard U-Net)
  - `UNET_UNITARIA` (Unitary U-Net)
  - `MLP` (Multi-Layer Perceptron)
- Allows users to select specific architectures during training and simulation

### Critical Bug Fixes

#### 1. Checkpoint Loading (`TypeError` on model instantiation)
**Problem:** Models failed to load from checkpoints due to missing architecture information.

**Solution:**
- Checkpoints now save model metadata (architecture type, hyperparameters)
- `model_loader.py` reads metadata to correctly instantiate model class before loading weights
- Prevents runtime errors during simulation initialization

#### 2. Model Compilation Error
**Problem:** Training pipeline passed an integer to `torch.compile()` instead of the model instance.

**Solution:**
- Fixed argument order in compilation call
- Proper model instance now passed to PyTorch's compilation system
- Improves training performance with optimized graph execution

#### 3. Import Path Corrections
**Problem:** File reorganization broke existing import statements.

**Solution:**
- Updated all imports to reflect new directory structure
- Implemented absolute path resolution using `PROJECT_ROOT`
- All modules now correctly reference relocated files

### Technical Impact

**For Developers:**
- Adding new models: Simply create a new file in `src/models/` with the `@register_model` decorator
- No need to modify loading logic or training pipeline
- Reduces boilerplate and potential for bugs

**For Users:**
- More reliable model loading
- Clear model selection in UI
- Better error messages when models fail to load

**For Training:**
- `torch.compile()` now works correctly, improving training speed
- Metadata in checkpoints enables better experiment tracking
- Path resolution more robust across different environments

### Files Changed
- **45 files modified** with 6,371 insertions
- New directories: `web/`, `scripts/`, `notebooks/`, `src/models/`
- Refactored: `model_loader.py`, `pipeline_train.py`, `web/app.py`
- Documentation: Updated `README.md` with new structure

---

## Future Agent Sessions

*This section will be populated with subsequent AI-assisted development sessions.*

### Guidelines for Future Entries
- **Date and commit reference** at the top
- **Clear categorization**: Feature, Bug Fix, Refactor, Documentation
- **Impact summary**: What changed and why it matters
- **Technical details**: For significant architectural changes
- **Focus on substance**: Skip trivial changes like formatting

---

**Last Updated:** 2025-11-13  
**Tracking Since:** Project initialization (2025-11-10)
