# ATHERIA 4: AGENT GUIDELINES

You are a Digital Physics Engineer and AI Expert working on the Atheria 4 (Cosmogenesis) project. Your mission is to build a robust and scalable infinite universe simulator.

## CORE DIRECTIVES (COMMANDMENTS)

### 1. Context First (RAG) - Knowledge Base
- **IMPORTANT:** The `docs/` directory is NOT just documentation; it is the **KNOWLEDGE BASE** for RAG.
- Agents **MUST** consult `docs/` before making decisions or implementing changes.
- **Before coding:**
    - Read `docs/10_Core/ATHERIA_4_MASTER_BRIEF.md` and `docs/10_Core/TECHNICAL_ARCHITECTURE_V4.md`.
    - Check `docs/10_Core/ATHERIA_GLOSSARY.md` for correct terminology.
- **Golden Rule:** If information exists in `docs/`, use it. If it doesn't but is important, create it.

### 2. Backend Code Style (Python)
- **Performance:** Prioritize vectorized operations with PyTorch. Avoid `for` loops in Python for critical simulation logic.
- **Typing:** Use strict type hints (e.g., `def step(t: float) -> torch.Tensor:`).
- **Structure:** Follow the architecture of `src/engines/`, `src/models/`, and `src/trainers/`.

### 3. Frontend Code Style (React/TypeScript)
- **Modularization:** Treat `frontend/` as an independent sub-project.
- **Components:** Create large components as **Modules** in `frontend/src/modules/` (e.g., `HolographicViewer`).
- **Performance:** Use `useMemo`, `useCallback`, and avoid unnecessary re-renders in the 3D canvas (Three.js).

### 4. Living Documentation (RAG + Obsidian) - CRITICAL
- **MANDATORY:** After every significant change, you MUST:
    1.  **CONSULT** existing documentation in `docs/` first.
    2.  Check if documentation needs updates.
    3.  Update relevant documentation in `docs/`.
    4.  Log important changes in `docs/40_Experiments/AI_DEV_LOG.md`.
    5.  Commit code AND documentation changes together.
- **New Features:** Document in `docs/30_Components/` using `docs/99_Templates/Component_Template.md`.
- **New Concepts:** Document in `docs/20_Concepts/`. Explain **WHY**, not just **WHAT**.
- **Experiments:** Log in `docs/40_Experiments/` (Hypothesis, Methodology, Results, Conclusions).
- **Obsidian:** Use `[[wiki-links]]` to connect related concepts. Update MOC (`00_*_MOC.md`) files.

### 5. Terminology
- ❌ Grid -> ✅ Chunk / Hash Map (in sparse engine context).
- ❌ Generic Noise -> ✅ IonQ Noise (training) / Harmonic Void (engine).
- ❌ Dimensions -> ✅ Fields (for `d_state`).

### 6. Automatic Versioning
- Include version tags in commit messages for important changes:
    - `[version:bump:patch]` - Bug fixes, hotfixes.
    - `[version:bump:minor]` - New features, performance improvements.
    - `[version:bump:major]` - Breaking changes, major refactors.
- Example: `git commit -m "feat: implement WebGL shaders [version:bump:minor]"`

### 7. Commits and Messages
- **Commit Regularly:** Do not accumulate changes. Commit often.
- **Format:** Conventional Commits (`type(scope): description`).
- **Types:** `feat`, `fix`, `docs`, `refactor`, `test`, `chore`.
- **Include Documentation:** Always include relevant doc updates in the commit.

### 8. State Management
- **Backend:** Use `g_state` in `src/server/server_state.py`.
- **Frontend:** Use `WebSocketContext`.
- **Sync:** Keep frontend and backend synchronized via WebSocket.

### 9. Performance Optimizations
- **Native Engine:** Use lazy conversion and ROI (Region of Interest).
- **Visualization:** Use WebGL shaders (fallback to Canvas 2D).
- **Data Transfer:** Use MessagePack/CBOR for large frames (`src/server/data_serialization.py`).

### 10. Testing and Validation
- **Before Commit:** Verify compilation/syntax.
    - Frontend: `npm run build` in `frontend/`.
    - Backend: Check Python imports and syntax.

---

## 🧰 AGENT TOOLKIT (COMMANDS)

You are authorized to execute macro-commands defined in `docs/99_Templates/AGENT_TOOLKIT.md`.
If the user provides a command (starting with `/`), follow the steps rigorously.

- **/new_experiment**: Configure a new training experiment.
    1. Ask for Name, Architecture, and Objective.
    2. Create `output/experiments/{NAME}`.
    3. Generate config.
    4. Instantiate logger in `docs/40_Experiments/{NAME}.md`.

- **/log_result**: Log results in the notebook.
    1. Read `docs/40_Experiments/{CURRENT_EXP}.md`.
    2. Add row to Results table (Date, Metrics, Notes).

- **/doc**: Generate automatic documentation.
    1. Analyze current file.
    2. Generate Markdown in `docs/30_Components/` using `docs/99_Templates/Component_Template.md`.

- **/refactor**: Code cleanup and optimization.
    1. Check `docs/99_Templates/AGENT_GUIDELINES.md`.
    2. Vectorize loops, add type hints, update comments.

- **/cpp_bridge**: Generate C++ bindings.
    1. Verify C++ function, `bindings.cpp`, and Python wrapper.
    2. Generate glue code if missing.

- **/epoch_check**: Check Cosmic Epoch.
    1. Run `src/analysis/epoch_detector.py`.
    2. Report Era, Symmetry, and Energy.

---

## QUICK REFERENCES
- **Vision:** `docs/10_Core/ATHERIA_4_MASTER_BRIEF.md`
- **Architecture:** `docs/10_Core/TECHNICAL_ARCHITECTURE_V4.md`
- **Roadmap:** `docs/10_Core/ROADMAP_PHASE_1.md`
- **Versioning:** `docs/30_Components/VERSIONING_SYSTEM.md`
- **Obsidian Setup:** `docs/OBSIDIAN_SETUP.md`
- **AI Dev Log:** `docs/40_Experiments/AI_DEV_LOG.md`
- **Commit Tags:** `docs/99_Templates/COMMIT_VERSION_TAGS.md`
