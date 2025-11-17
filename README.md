# AETHERIA: Laboratorio de Emergencia Cuántica

AETHERIA es un entorno de simulación e investigación para estudiar la emergencia de complejidad en autómatas celulares cuánticos (QCA). El sistema utiliza un modelo de Deep Learning, denominado "Ley M", para gobernar la evolución del estado cuántico mediante la Ecuación Maestra de Lindblad.

Este proyecto es una aplicación web interactiva con un backend de `aiohttp` y un frontend de `React`, permitiendo el entrenamiento de modelos en GPU y la visualización de simulaciones en tiempo real.

## 🚀 Características Principales

### Simulación Cuántica
- **Autómatas Celulares Cuánticos (QCA)**: Simulación 2D de sistemas cuánticos en cuadrículas
- **Ecuación Maestra de Lindblad**: Implementación completa con evolución unitaria y términos disipativos
- **Múltiples Arquitecturas de Modelos**: U-Net, U-Net Unitaria, MLP, DEEP_QCA, SNN_UNET
- **Optimizaciones de Rendimiento**: Live feed opcional, frame skipping, control de FPS

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

#### Control de Live Feed
- **Live Feed Activo**: Calcula y envía visualizaciones en tiempo real
- **Live Feed Desactivado**: Solo evoluciona la física sin calcular visualizaciones
- **Beneficio**: Permite simulaciones más rápidas para experimentos largos sin visualización

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

La documentación completa está disponible en el directorio [`docs/`](docs/README.md):

- **[Guía de Aprendizaje Progresivo](docs/PROGRESSIVE_LEARNING.md)**: Aprende desde lo básico hasta experimentos avanzados
- **[Guía de Experimentación](docs/EXPERIMENTATION_GUIDE.md)**: Estrategias y mejores prácticas
- **[Pruebas por Visualización](docs/VISUALIZATION_TESTING.md)**: Cómo probar e interpretar cada visualización
- **[Sistema de Historia](docs/HISTORY_SYSTEM.md)**: Guardar y analizar simulaciones completas
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
   ```

2. **Frontend:**
   ```bash
   cd frontend
   npm install
   npm run build
   cd ..
   ```

### Ejecutar la Aplicación

El único punto de entrada es `run_server.py`.

```bash
source ath_venv/bin/activate  # En Windows: ath_venv\Scripts\activate
export AETHERIA_ENV=development  # En Windows: set AETHERIA_ENV=development
python3 run_server.py  # En Windows: python run_server.py
```

La aplicación estará disponible en `http://localhost:8000`.

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

### Control de Live Feed
- **Activar/Desactivar**: Switch en la pestaña de Visualización
- **Efecto**: Cuando está desactivado, la simulación corre más rápido sin calcular visualizaciones
- **Uso**: Ideal para experimentos largos donde no necesitas ver cada frame

### Velocidad y FPS
- **Velocidad de Simulación**: Multiplicador (0.1x - 100x)
- **FPS Objetivo**: Frames por segundo objetivo (0.1 - 120 FPS)
- **Frame Skip**: Saltar frames para acelerar (0 = todos, 1 = cada otro, etc.)

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
│   ├── pipeline_server.py  # Servidor principal
│   ├── qca_engine.py       # Motor de física
│   ├── models/             # Arquitecturas de modelos
│   ├── pipeline_viz.py     # Generación de visualizaciones
│   └── ...
├── frontend/               # Frontend React
│   ├── src/
│   │   ├── components/     # Componentes React
│   │   ├── hooks/          # Hooks personalizados
│   │   └── context/        # Context API
│   └── ...
└── docs/                   # Documentación completa
```

### Comandos Útiles

```python
# Habilitar historia
simulation.enable_history({enabled: true})

# Guardar historia
simulation.save_history({filename: "experimento.json"})

# Capturar snapshot para t-SNE
simulation.capture_snapshot({})

# Configurar FPS
simulation.set_fps({fps: 30})

# Configurar velocidad
simulation.set_speed({speed: 2.0})

# Controlar live feed
simulation.set_live_feed({enabled: true})
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

## 🎯 Próximos Pasos

1. **Lee la [Guía de Aprendizaje Progresivo](docs/PROGRESSIVE_LEARNING.md)**
2. **Prueba las visualizaciones** según [VISUALIZATION_TESTING.md](docs/VISUALIZATION_TESTING.md)
3. **Experimenta** siguiendo [EXPERIMENTATION_GUIDE.md](docs/EXPERIMENTATION_GUIDE.md)
4. **Busca A-Life** usando todas las herramientas disponibles

---

**AETHERIA** - Explorando la emergencia de complejidad en sistemas cuánticos 🚀
