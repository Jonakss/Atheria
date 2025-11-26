# 2025-11-26: Fix Crítico de Build y Mejoras CLI

**Fecha:** 2025-11-26
**Autor:** Antigravity (Google Deepmind)
**Tipo:** `fix`, `feat`, `chore`
**Componentes:** `setup.py`, `pyproject.toml`, `src/cli.py`, `src/model_loader.py`

---

## 📝 Resumen Ejecutivo

Se resolvieron problemas críticos que impedían la instalación y ejecución del proyecto en entornos nuevos, específicamente relacionados con el aislamiento de build de `pip` y dependencias faltantes. Además, se mejoró el CLI para permitir instalaciones rápidas.

## 🚀 Cambios Principales

### 1. Fix de Aislamiento de Build (`pyproject.toml`)
- **Problema:** `pip install -e .` fallaba porque el entorno de build aislado no tenía `torch` instalado, necesario para que CMake detectara LibTorch.
- **Solución:** Se creó `pyproject.toml` declarando explícitamente las dependencias de build (`torch`, `pybind11`, `numpy`, `wheel`).

### 2. Modo Rápido en CLI (`--fast`)
- **Problema:** La instalación con aislamiento de build es lenta porque descarga `torch` cada vez.
- **Solución:** Se agregó el flag `--fast` al comando `ath dev` y `ath install`.
- **Comando:** `ath dev --fast` ejecuta `pip install -e . --no-build-isolation`, usando las librerías del sistema (mucho más rápido).

### 3. Fix Runtime Import y Signature (`model_loader.py`)
- **Problema:** Error `ImportError: cannot import name 'load_model'` y posteriormente `TypeError: cannot unpack non-iterable` en `inference_handlers.py`.
- **Solución:** 
    - Se implementó la función `load_model` unificando `instantiate_model` y `load_weights`.
    - Se corrigió el retorno de `load_model` para devolver una tupla `(model, checkpoint_data)` como esperan los handlers.

### 4. Estructura de Paquetes (`__init__.py`)
- **Problema:** Error `package init file ... not found` durante la creación del wheel.
- **Solución:** Se crearon los archivos `__init__.py` faltantes en `src/engines/`, `src/physics/` y `src/physics/analysis/`.

## 🔗 Archivos Afectados
- `pyproject.toml` (Nuevo)
- `src/cli.py`
- `src/model_loader.py`
- `src/engines/__init__.py`
- `src/physics/__init__.py`
- `src/physics/analysis/__init__.py`
- `docs/50_Guides/HOW_TO_RUN.md`
