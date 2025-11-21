# AETHERIA: Laboratorio de Emergencia Cuántica

AETHERIA es un entorno de simulación e investigación para estudiar la emergencia de complejidad en autómatas celulares cuánticos (QCA). El sistema utiliza un modelo de Deep Learning, denominado "Ley M", para gobernar la evolución del estado cuántico mediante la Ecuación Maestra de Lindblad.

Este proyecto es una aplicación web interactiva con un backend de `aiohttp` y un frontend de `React`, permitiendo el entrenamiento de modelos en GPU y la visualización de simulaciones en tiempo real.

## 🚀 Características Principales

### Simulación Cuántica
- **Autómatas Celulares Cuánticos (QCA)**: Simulación 2D de sistemas cuánticos en cuadrículas
- **Ecuación Maestra de Lindblad**: Implementación completa con evolución unitaria y términos disipativos
- **Múltiples Arquitecturas de Modelos**: U-Net, U-Net Unitaria, MLP, DEEP_QCA, SNN_UNET
- **Motores de Simulación**: Motor Python (PyTorch) y Motor Nativo C++ (LibTorch) para máximo rendimiento
- **Optimizaciones de Rendimiento**: Live feed opcional, frame skipping, control de FPS, lazy conversion, ROI-based rendering

### Visualizaciones Avanzadas

#### Visualizaciones 2D
- **Básicas**: Densidad (`|ψ|²`), Fase, Energía, Parte Real/Imaginaria
- **Análisis**: Entropía, Coherencia, Actividad de Canales, Mapa de Física
- **Flujo**: Campo vectorial `delta_psi` con visualización y estadísticas
- **Atractores**: Atractor de Fase, Visualización Poincaré (2D y 3D)

#### Visualizaciones 3D
- **Evolución Temporal 3D**: Stacking de frames 2D a lo largo del eje temporal
- **Espacio Complejo 3D**: Visualización Real vs Imaginario vs Tiempo
- **Poincaré 3D**: Proyección esférica con renderizado de alta calidad

#### Herramientas de Análisis
- **t-SNE**: Atlas del Universo y Química Celular
- **Histogramas**: Distribución estadística de valores
- **Overlays**: Grid, coordenadas, Quadtree, estadísticas en tiempo real

### Gestión de Experimentos
- **Checkpointing**: Guardado automático de pesos y estados del optimizador
- **Transfer Learning**: Continuar entrenamiento desde checkpoints
- **Historia de Simulación**: Guardar y cargar frames completos para análisis posterior
- **Notas y Metadatos**: Anotaciones asociadas a checkpoints

### Optimizaciones de Rendimiento

#### Motor Nativo C++ (Atheria Core)
- **Rendimiento**: Hasta ~10,000 steps/segundo con motor nativo C++
- **Lazy Conversion**: Conversión sparse→dense solo cuando se necesita
- **ROI Support**: Renderizado basado en Region of Interest para eficiencia
- **Compilado**: Motor C++ con PyBind11 para integración perfecta con Python

#### Control de Live Feed
- **Live Feed Activo**: Calcula y envía visualizaciones en tiempo real
- **Live Feed Desactivado**: Solo evoluciona la física sin calcular visualizaciones
- **Modo Manual**: `steps_interval = 0` permite control manual de actualizaciones
- **Beneficio**: Permite simulaciones más rápidas para experimentos largos sin visualización

#### Protocolo WebSocket Optimizado
- **MessagePack Binario**: Frames de visualización en formato binario eficiente (3-5x más compacto que JSON)
- **JSON para Comandos**: Solo comandos y metadatos usan JSON (pequeños y rápidos)
- **Separación Clara**: Datos grandes en binario, comandos en JSON

#### Sistema de Overlays
- **Grid**: Líneas de cuadrícula configurables
- **Coordenadas**: Referencias espaciales
- **Quadtree**: Visualización de estructura de compresión
- **Estadísticas**: Min/Max/Promedio, tamaño de grilla, zoom

#### Zoom Inteligente
- **Zoom Limitado**: Mantiene la grilla siempre visible
- **Reset de Vista**: Botón para recuperar la vista inicial
- **Pan Constreñido**: Previene perder la vista de la simulación

## 📖 Documentación

La documentación completa está disponible en el directorio [`docs/`](docs/README.md). Este es un **vault de Obsidian** para navegación avanzada y RAG (Retrieval Augmented Generation).

### Guías Principales
- **[Guía de Aprendizaje Progresivo](docs/PROGRESSIVE_LEARNING.md)**: Aprende desde lo básico hasta experimentos avanzados
- **[Guía de Experimentación](docs/EXPERIMENTATION_GUIDE.md)**: Estrategias y mejores prácticas
- **[Pruebas por Visualización](docs/VISUALIZATION_TESTING.md)**: Cómo probar e interpretar cada visualización

### Componentes Técnicos
- **[CLI Tool](docs/30_Components/CLI_TOOL.md)**: Uso del CLI `atheria` para desarrollo
- **[Motor Nativo C++](docs/30_Components/Native_Engine_Core.md)**: Arquitectura y optimizaciones del motor nativo
- **[Protocolo WebSocket](docs/30_Components/WEB_SOCKET_PROTOCOL.md)**: Protocolo binario (MessagePack) vs JSON
- **[Sistema de Historia](docs/HISTORY_SYSTEM.md)**: Guardar y analizar simulaciones completas

### Estructuras de Datos
- **[Quadtree Binario](docs/QUADTREE_BINARY.md)**: Estructuras de datos eficientes para 2D
- **[Octree Binario](docs/OCTREE_BINARY.md)**: Estructuras de datos para 3D
- **[Sistema TimeTree](docs/TIME_TREE_SYSTEM.md)**: Almacenamiento eficiente de historia temporal

## 🛠️ Cómo Empezar

### Requisitos
- Python 3.10+
- Node.js 18+ y npm
- CUDA-capable GPU (opcional pero recomendado)

### Instalación

1. **Backend:**
   ```bash
   python3 -m venv ath_venv
   source ath_venv/bin/activate  # En Windows: ath_venv\Scripts\activate
   pip install -r requirements.txt
   python3 setup.py build_ext --inplace  # Compilar extensiones C++
   pip install -e .  # Instalar en modo desarrollo (habilita CLI)
   ```

2. **Frontend:**
   ```bash
   cd frontend
   npm install
   npm run build
   cd ..
   ```

### Ejecutar la Aplicación

#### Opción 1: Usando el CLI (Recomendado)

El CLI `atheria` (alias `ath`) simplifica el flujo de desarrollo:

```bash
source ath_venv/bin/activate  # En Windows: ath_venv\Scripts\activate

# Modo desarrollo completo (Build + Install + Run)
atheria dev                  # Sin frontend (solo WebSocket API)
atheria dev --frontend       # Con frontend estático
atheria dev --port 8080      # Puerto personalizado

# Comandos individuales
atheria build                # Solo compilar extensiones C++
atheria install              # Solo instalar paquete
atheria run --frontend       # Solo ejecutar servidor
atheria clean                # Limpiar archivos de build
```

**Nota:** El CLI está disponible después de `pip install -e .` (ver instalación).

#### Opción 2: Comando Directo

```bash
source ath_venv/bin/activate  # En Windows: ath_venv\Scripts\activate
export AETHERIA_ENV=development  # En Windows: set AETHERIA_ENV=development
python3 run_server.py  # En Windows: python run_server.py

# Solo WebSocket API (sin frontend)
ATHERIA_NO_FRONTEND=1 python3 run_server.py

# Puerto personalizado
python3 run_server.py --port 8080
```

La aplicación estará disponible en `http://localhost:8000` (o el puerto especificado).

#### Ver ayuda del CLI

```bash
atheria --help
atheria dev --help
```

## 🎯 Uso Rápido

### 1. Crear un Experimento
1. En la pestaña **Laboratorio**, haz clic en **Nuevo Experimento**
2. Configura los parámetros del modelo (d_state, hidden_channels, etc.)
3. Selecciona la arquitectura (U-Net, U-Net Unitaria, etc.)
4. Haz clic en **Crear y Entrenar**

### 2. Cargar un Modelo para Simulación
1. En la pestaña **Gestión de Experimentos**, selecciona un experimento
2. Haz clic en **Cargar Modelo para Inferencia**
3. La simulación comenzará automáticamente (en pausa)

### 3. Visualizar
1. Usa el dropdown **Mapa de Visualización** para cambiar entre visualizaciones
2. Activa **Modo Live Feed** para visualización en tiempo real
3. Usa los controles de zoom y pan en el canvas
4. Activa overlays desde el botón de configuración (⚙️)

### 4. Guardar y Analizar Historia
1. En **Controles Avanzados**, activa **Habilitar Historia**
2. Ejecuta la simulación
3. Cuando termines, haz clic en **Guardar Historia**
4. Ve a la pestaña **Historia** para cargar y reproducir la simulación guardada

## 📊 Visualizaciones Disponibles

### Básicas
- `density`: Densidad cuántica `|ψ|²`
- `phase`: Fase del estado cuántico
- `energy`: Energía local
- `real`: Parte real
- `imag`: Parte imaginaria

### Avanzadas
- `entropy`: Entropía local (complejidad/información)
- `coherence`: Coherencia de fase entre vecinos
- `channel_activity`: Actividad por canal cuántico
- `physics`: Mapa de la matriz de física aprendida

### Análisis Temporal
- `flow`: Campo vectorial de cambio (`delta_psi`)
- `phase_attractor`: Evolución del estado en el espacio de fases
- `poincare`: Visualización Poincaré 2D
- `poincare_3d`: Visualización Poincaré 3D (proyección esférica)

### Evolución Temporal
- `history_3d`: Stacking de frames 2D en el eje temporal
- `complex_3d`: Real vs Imaginario vs Tiempo

### Análisis Estadístico
- `spectral`: Análisis espectral
- `gradient`: Gradientes espaciales
- `universe_atlas`: t-SNE de snapshots temporales
- `cell_chemistry`: t-SNE del estado actual

## ⚙️ Optimizaciones y Configuración

### Motor de Simulación
- **Selección de Motor**: Cambia entre Motor Python y Motor Nativo C++ desde el header
- **Motor Nativo**: Mayor rendimiento (~10,000 steps/segundo), ideal para simulaciones largas
- **Motor Python**: Más flexible, mejor para experimentación y debugging
- **Cambio Dinámico**: Puedes cambiar de motor sin reiniciar (requiere recargar modelo)

### Control de Live Feed
- **Activar/Desactivar**: Switch en la pestaña de Visualización
- **Modo Manual**: Configura `steps_interval = 0` para control manual de actualizaciones
- **Intervalos Grandes**: Soporta hasta 1,000,000 pasos entre frames
- **Efecto**: Cuando está desactivado, la simulación corre más rápido sin calcular visualizaciones
- **Uso**: Ideal para experimentos largos donde no necesitas ver cada frame

### Velocidad y FPS
- **Velocidad de Simulación**: Multiplicador (0.1x - 100x)
- **FPS Objetivo**: Frames por segundo objetivo (0.1 - 120 FPS)
- **Frame Skip**: Saltar frames para acelerar (0 = todos, 1 = cada otro, etc.)
- **FPS Real**: Muestra steps/segundo de simulación y frames/segundo de visualización

### Overlays
- **Grid**: Tamaño de cuadrícula configurable
- **Quadtree**: Threshold para visualización de estructura
- **Estadísticas**: Min/Max/Promedio en tiempo real
- **Coordenadas**: Referencias espaciales

## 🔬 Arquitecturas de Modelos

### U-Net
- Arquitectura estándar convolucional
- Flexible, no garantiza conservación de energía
- Ideal para experimentación general

### U-Net Unitaria
- Conserva energía por diseño
- Usa matrices antisimétricas
- Más estable para simulaciones largas

### MLP
- Red densa simple
- Menor capacidad pero más rápida
- Útil para experimentos rápidos

### DEEP_QCA
- Arquitectura específica para QCA
- Optimizada para patrones espaciales

### SNN_UNET
- Red neuronal espiking
- Dinámicas temporales más complejas

## 📝 Notas de Desarrollo

### Estructura del Proyecto
```
Atheria/
├── src/                    # Backend Python
│   ├── cli.py              # CLI tool (atheria/ath)
│   ├── pipelines/
│   │   ├── pipeline_server.py  # Servidor principal
│   │   └── pipeline_viz.py     # Generación de visualizaciones
│   ├── engines/
│   │   ├── qca_engine.py       # Motor Python (PyTorch)
│   │   └── native_engine_wrapper.py  # Wrapper para motor C++
│   ├── cpp_core/           # Motor nativo C++ (LibTorch)
│   ├── models/             # Arquitecturas de modelos
│   └── ...
├── frontend/               # Frontend React
│   ├── src/
│   │   ├── components/     # Componentes React
│   │   ├── modules/        # Módulos grandes (Dashboard, etc.)
│   │   ├── hooks/          # Hooks personalizados
│   │   └── context/        # Context API
│   └── ...
└── docs/                   # Documentación completa
```

### CLI Tool (`atheria` / `ath`)

El CLI simplifica el flujo de desarrollo:

**Comandos disponibles:**
- `atheria dev [--frontend]` - Build + Install + Run (ciclo completo)
- `atheria build` - Compilar extensiones C++
- `atheria install` - Instalar paquete en modo desarrollo
- `atheria run [--frontend]` - Ejecutar servidor
- `atheria clean` - Limpiar archivos de build y caché

**Ejemplos:**
```bash
# Desarrollo rápido (sin frontend)
atheria dev

# Desarrollo con frontend
atheria dev --frontend

# Solo compilar
atheria build

# Limpiar
atheria clean
```

Ver [`docs/30_Components/CLI_TOOL.md`](docs/30_Components/CLI_TOOL.md) para más detalles.

### Comandos WebSocket Útiles

```javascript
// Habilitar historia
simulation.enable_history({enabled: true})

// Guardar historia
simulation.save_history({filename: "experimento.json"})

// Capturar snapshot para t-SNE
simulation.capture_snapshot({})

// Configurar FPS
simulation.set_fps({fps: 30})

// Configurar velocidad
simulation.set_speed({speed: 2.0})

// Controlar live feed
simulation.set_live_feed({enabled: true})

// Configurar intervalo de pasos (0 = manual)
simulation.set_steps_interval({steps_interval: 1000})

// Actualización manual de visualización
simulation.update_visualization({})

// Cambiar motor
simulation.switch_engine({engine: "native"})  // o "python"
```

## 🎓 Objetivos de Aprendizaje

### Nivel 1: Fundamentos
- Entender física básica (QCA, unitariedad, Lindblad)
- Dominar visualizaciones básicas
- Comparar arquitecturas simples

### Nivel 2: Herramientas
- Dominar todas las visualizaciones
- Usar t-SNE para análisis
- Guardar y analizar historia

### Nivel 3: Optimización
- Encontrar mejores parámetros
- Optimizar para tu hardware
- Documentar configuraciones exitosas

### Nivel 4: A-Life (Artificial Life)
- Buscar gliders (estructuras móviles)
- Buscar osciladores
- Buscar replicadores
- Caracterizar estructuras encontradas

## 🤝 Contribuir

Si encuentras errores o tienes sugerencias:
1. Documenta tus hallazgos
2. Comparte configuraciones exitosas
3. Contribuye con mejoras al código

## 📚 Referencias

- **Ecuación Maestra de Lindblad**: Para sistemas cuánticos abiertos
- **Autómatas Celulares Cuánticos**: Modelado de sistemas cuánticos discretos
- **Artificial Life**: Búsqueda de emergencia de complejidad

## 🔧 Desarrollo

### Versionado
El proyecto usa SemVer (Semantic Versioning) para todas las componentes:
- **Aplicación Principal**: `src/__version__.py` → `4.1.0`
- **Motor Python/Wrapper**: `src/engines/__version__.py` → `4.1.0`
- **Motor Nativo C++**: `src/cpp_core/include/version.h` → `4.1.0`
- **Frontend**: `frontend/package.json` → `4.0.2`

Las versiones se exponen en la UI del dashboard.

### Workflow de Desarrollo
```bash
# 1. Activar entorno
source ath_venv/bin/activate

# 2. Desarrollo completo (build + install + run)
atheria dev --frontend

# 3. Solo recompilar después de cambios C++
atheria build

# 4. Limpiar cuando sea necesario
atheria clean
```

### Compilar Motor Nativo C++
El motor nativo C++ requiere compilación separada:
```bash
# Automático con CLI
atheria build

# Manual
python3 setup.py build_ext --inplace
pip install -e .
```

## 🎯 Próximos Pasos

1. **Instala el proyecto** y activa el CLI: `pip install -e .`
2. **Lee la [Guía de Aprendizaje Progresivo](docs/PROGRESSIVE_LEARNING.md)**
3. **Prueba las visualizaciones** según [VISUALIZATION_TESTING.md](docs/VISUALIZATION_TESTING.md)
4. **Experimenta** siguiendo [EXPERIMENTATION_GUIDE.md](docs/EXPERIMENTATION_GUIDE.md)
5. **Explora el motor nativo** para simulaciones de alto rendimiento
6. **Busca A-Life** usando todas las herramientas disponibles

---

**AETHERIA** - Explorando la emergencia de complejidad en sistemas cuánticos 🚀
