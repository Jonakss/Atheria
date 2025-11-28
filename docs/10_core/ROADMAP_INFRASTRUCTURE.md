# 🏗️ Roadmap Infraestructura: DevOps & Tooling

**Objetivo:** Establecer una base sólida de CI/CD, automatización y herramientas de desarrollo que permita escalar el proyecto y facilitar la colaboración.

---

## 1. CI/CD & Automatización (GitHub Actions)

**Referencia:** [[CI_CD_SETUP|Configuración CI/CD]]

### A. Pipeline de Construcción y Test
- **Multi-plataforma:** Asegurar builds correctos en Linux, macOS y Windows.
- **C++ Compilation:** Automatizar la compilación del motor nativo (`setup.py build_ext`).
- **Tests Automatizados:** Ejecutar `pytest` y tests de frontend en cada PR.
- **Linting & Formatting:** Enforce de estilo (Black, Isort, ESLint, Prettier).

### B. Sistema de Release
- **Versionado Semántico:** Automatizar el bump de versiones basado en commits (Conventional Commits).
- **Artifacts:** Generar y subir binarios pre-compilados (Wheels) para diferentes plataformas.
- **Docker:** Construir y publicar imágenes Docker optimizadas (`atheria-server`, `atheria-training`).

---

## 2. Infraestructura de Entrenamiento

**Referencia:** [[PROGRESSIVE_TRAINING|Entrenamiento Progresivo]]

### A. Notebooks de Entrenamiento (Colab/Kaggle)
- **Persistencia:** Guardado automático de checkpoints en Google Drive / Kaggle Datasets.
- **Recuperación:** Auto-resume tras desconexiones o timeouts.
- **Monitorización:** Visualización en tiempo real del progreso (WandB o custom dashboard).

### B. Gestión de Datos
- **Datasets:** Pipeline para generar y versionar datasets de entrenamiento.
- **Model Registry:** Sistema para trackear versiones de modelos ("Ley M") y sus métricas.

---

## 3. Herramientas de Desarrollo (DX)

### A. CLI (Command Line Interface)
Mejorar la herramienta `ath` para facilitar tareas comunes.
- `ath dev`: Iniciar entorno de desarrollo completo (backend + frontend).
- `ath train`: Lanzar entrenamientos locales.
- `ath doctor`: Diagnosticar problemas de configuración (CUDA, dependencias).

### B. Documentación & Knowledge Base
- **RAG Pipeline:** Mantener la documentación optimizada para consumo por agentes de IA.
- **Obsidian Vault:** Estructura limpia y enlazada para navegación humana.
- **Auto-docs:** Generación automática de referencia de API.

---

## 4. Testing & Benchmarking

### A. Suite de Benchmarks
- **Comparativas:** Python vs C++ vs CUDA.
- **Métricas:** FPS, Steps/Second, Memoria, Latencia WebSocket.
- **Regresión:** Detectar degradación de rendimiento en PRs.

### B. Tests de Integración
- **End-to-End:** Tests que verifiquen el flujo completo (Frontend -> WebSocket -> Engine -> Model -> Response).

---

**Estado:** En Progreso
**Prioridad:** Alta (Soporte transversal a todas las fases)
