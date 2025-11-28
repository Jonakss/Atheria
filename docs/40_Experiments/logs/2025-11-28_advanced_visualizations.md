# 2025-11-28: Visualizaciones Avanzadas de Campos (WebGL)

## Objetivo
Implementar visualizaciones de alto rendimiento basadas en shaders WebGL para analizar propiedades avanzadas del campo cuántico en tiempo real.

## Nuevas Visualizaciones Implementadas

### 1. Entropía Local (Shannon)
- **Shader:** `FRAGMENT_SHADER_ENTROPY`
- **Método:** Calcula la varianza local en una ventana de 3x3 píxeles como proxy de la entropía/complejidad local.
- **Uso:** Identificar regiones con alta complejidad estructural o "borde del caos".
- **Visualización:** Colormap "Plasma" (Azul -> Rojo -> Amarillo).

### 2. Gradiente ($|\nabla \rho|$)
- **Shader:** `FRAGMENT_SHADER_GRADIENT`
- **Método:** Operador Sobel para calcular la magnitud del cambio del campo en X e Y.
- **Uso:** Detectar bordes, frentes de onda y límites de estructuras.
- **Visualización:** Colormap "Inferno" (Negro -> Rojo -> Amarillo).

### 3. Flujo Denso (GPU)
- **Shader:** `FRAGMENT_SHADER_FLOW`
- **Método:** Mapea la dirección del gradiente al espacio de color HSV.
    - **Hue (Matiz):** Dirección del flujo (ángulo).
    - **Value (Brillo):** Magnitud del flujo.
- **Uso:** Visualizar la dinámica de fluidos del campo de manera densa y continua.
- **Diferencia con CPU:** A diferencia del "Quiver Plot" (flechas dispersas), este shader visualiza cada píxel, permitiendo ver micro-corrientes.

## Cambios Técnicos
- **Archivo:** `frontend/src/utils/shaderVisualization.ts`
    - Se agregaron las constantes de los shaders GLSL.
    - Se actualizó la interfaz `ShaderConfig`.
- **Archivo:** `frontend/src/components/ui/ShaderCanvas.tsx`
    - Se integraron los nuevos shaders en el selector de visualización.
    - Se corrigieron imports para incluir los nuevos tipos.

## Verificación
- **Build:** `npm run build` completado exitosamente.
- **Integración:** Las opciones aparecen en el selector de visualización del frontend y utilizan la aceleración GPU cuando está disponible.
