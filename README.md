# AETHERIA: Laboratorio de Complejidad Emergente

AETHERIA es un laboratorio de software para simular la emergencia de estructuras complejas a partir de reglas físicas fundamentales.

Este proyecto modela un universo discreto como una cuadrícula de Autómatas Celulares Cuánticos (QCA). La evolución de este universo no está pre-programada, sino que es gobernada por una **"Ley M" (Ley Fundamental)**: un modelo de Deep Learning (U-Net) que se entrena desde cero para descubrir las "leyes de la física" de este universo.

El objetivo es descubrir, mediante Aprendizaje por Refuerzo, una Ley M que opere en el **"Borde del Caos"**: el régimen crítico donde la información puede propagarse, la estabilidad se mantiene y la complejidad estructural emerge espontáneamente.

## 🚀 Arquitectura Simplificada

El proyecto ha sido refactorizado en una arquitectura unificada y fácil de usar:

- **`app.py`**: Un único servidor que maneja tanto el backend de simulación como el frontend web.
- **`index.html`**: Una única interfaz de usuario (UI) web para controlar todo: entrenamiento, simulación y visualización.
- **`train.py`**: El script de entrenamiento, que ahora es llamado como un subproceso por el servidor principal.
- **`src/`**: Contiene toda la lógica del núcleo (motor QCA, modelos, configuración).

```
aetheria/
├── app.py              <-- 🚀 El SERVIDOR UNIFICADO (ejecutar este archivo)
├── index.html          <-- 🖥️ La INTERFAZ DE USUARIO web
├── train.py            <-- 🏋️ El script de entrenamiento (llamado por app.py)
├── requirements.txt    <-- 📋 Dependencias del proyecto
│
├── src/                <-- 🧠 Todo el código fuente del núcleo
│   ├── config.py         <-- ⚙️ Parámetros globales (tamaño de grilla, etc.)
│   ├── qca_engine.py     <-- 🌌 Motor de simulación QCA
│   ├── qca_operator_*.py <-- 🧬 Las "Leyes M" (modelos U-Net)
│   └── model_loader.py   <-- 📦 Utilidad para cargar modelos
│
└── checkpoints/        <-- 💾 Los modelos entrenados (.pth) se guardan aquí
```

## ⚙️ Cómo Empezar

### 1. Instalación

Asegúrate de tener Python 3.8+ y `pip`. Clona el repositorio y navega al directorio del proyecto. Luego, instala las dependencias:

```bash
pip install -r requirements.txt
```

### 2. Ejecutar la Aplicación

Para iniciar el laboratorio, simplemente ejecuta el servidor `app.py`:

```bash
python3 app.py
```

El servidor se iniciará y te mostrará la URL para acceder a la interfaz web (normalmente `http://localhost:8000`).

### 3. Usar la Interfaz Web

Abre tu navegador en `http://localhost:8000`. Desde esta única interfaz, puedes:

- **Entrenar un Nuevo Modelo**:
  - En el panel "Controles de Entrenamiento", ajusta los parámetros como el nombre del experimento, la tasa de aprendizaje y los episodios.
  - Haz clic en "🚀 Iniciar Entrenamiento".
  - Verás los logs del entrenamiento en tiempo real en la sección "Log de Entrenamiento".
  - Los modelos (`.pth`) se guardarán en el directorio `checkpoints/`.

- **Ejecutar una Simulación**:
  - Una vez que un modelo ha sido entrenado, haz clic en "🔄 Refrescar Modelos" para que aparezca en la lista desplegable.
  - Selecciona el modelo que deseas cargar en el panel "Cargar Modelo para Simulación".
  - Haz clic en "▶️ Iniciar Simulación".

- **Visualizar y Analizar**:
  - La simulación se mostrará en el visor central.
  - Usa el menú "Tipo de Visualización" para cambiar entre diferentes modos de análisis (densidad, fase, FFT, etc.).
  - **Haz clic y arrastra** para moverte por la simulación (pan).
  - **Usa la rueda del ratón** para acercar y alejar (zoom).
  - Las métricas globales como la entropía y la densidad se actualizan en tiempo real.
  - La configuración de la simulación actual (modelo cargado, tamaño de la grilla) se muestra en el panel "Configuración de Simulación".

## 🔬 Visualizaciones Disponibles

- **Análisis de Grid**:
  - `Densidad`: Mapa de calor de la "materia" o "energía".
  - `Magnitud del Cambio`: Resalta las áreas de mayor actividad entre pasos.
  - `Canales RGB`: Mapea los primeros 3 canales complejos a colores para ver la dinámica interna.
  - `Fase Agregada`: Muestra la coherencia de fase, útil para detectar comportamiento de onda.
  - `Transformada de Fourier 2D`: Analiza las frecuencias espaciales de la estructura.

- **Análisis Temporal y Estadístico**:
  - `Diagrama Espacio-Tiempo`: Muestra la evolución de una fila de píxeles a lo largo del tiempo.
  - `Gráfico de Poincaré`: Ayuda a identificar atractores y caos en la dinámica de la densidad.
  - `Histograma de Densidad`: Muestra la distribución de los valores de densidad en la grilla.
