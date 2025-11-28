precision mediump float;

varying vec2 vUv;
uniform sampler2D uTexture;
uniform float uTime;
uniform float uZoom;
uniform vec2 uPan;
uniform vec2 uResolution;
uniform vec3 uColorMap[5]; // Mapa de colores personalizado

// Constantes para el modelo de Poincaré
const float DISK_RADIUS = 0.95;

// Función para mapear coordenadas del disco de Poincaré al plano euclidiano (grid)
// Inversa de la proyección estereográfica o mapeo conforme
vec2 poincareToEuclidean(vec2 diskCoord) {
    // En el modelo del disco de Poincaré, la distancia desde el centro representa
    // la profundidad en el espacio hiperbólico (coordenada Z en AdS).
    // El borde del disco (r=1) es el "infinito" o la frontera (Boundary).
    
    // Para visualizar el grid 2D como la frontera, necesitamos un mapeo que:
    // 1. Tome coordenadas del disco (u,v)
    // 2. Las transforme a coordenadas de textura (s,t)
    
    // Mapeo simple: Coordenadas polares
    float r = length(diskCoord);
    float theta = atan(diskCoord.y, diskCoord.x);
    
    // Mapeo hiperbólico: r_euclidiano = 2 * tanh(r_hiperbolico / 2)
    // Invertimos para obtener r_hiperbolico desde r_euclidiano del disco
    // r_hyp = 2 * atanh(r)
    
    // Pero para nuestra visualización "Holográfica", queremos que:
    // - El borde del disco muestre el detalle fino (alta frecuencia)
    // - El centro muestre estructuras grandes (baja frecuencia)
    
    // Usamos una distorsión radial simple para simular la métrica AdS
    // r' = r^alpha (donde alpha < 1 expande el centro, alpha > 1 expande el borde)
    float r_distorted = pow(r, 0.5); // Expandir el centro para ver mejor el "bulk"
    
    // Convertir de vuelta a cartesianas para sampling de textura
    // Esto es una aproximación artística de la proyección holográfica
    vec2 uv = vec2(
        0.5 + 0.5 * r_distorted * cos(theta),
        0.5 + 0.5 * r_distorted * sin(theta)
    );
    
    return uv;
}

// Función de color (Magma/Inferno style)
vec3 getHeatmapColor(float t) {
    // Clamp t to [0, 1]
    t = clamp(t, 0.0, 1.0);
    
    // Interpolación suave entre puntos de control del colormap
    if (t < 0.25) return mix(uColorMap[0], uColorMap[1], t * 4.0);
    if (t < 0.50) return mix(uColorMap[1], uColorMap[2], (t - 0.25) * 4.0);
    if (t < 0.75) return mix(uColorMap[2], uColorMap[3], (t - 0.50) * 4.0);
    return mix(uColorMap[3], uColorMap[4], (t - 0.75) * 4.0);
}

void main() {
    // Coordenadas normalizadas [-1, 1] corregidas por aspecto
    vec2 uv = vUv * 2.0 - 1.0;
    float aspect = uResolution.x / uResolution.y;
    uv.x *= aspect;
    
    // Aplicar Pan y Zoom (en el espacio del disco)
    uv = (uv - uPan) / uZoom;
    
    // Distancia al centro
    float r = length(uv);
    
    // Fondo negro fuera del disco
    if (r > 1.0) {
        gl_FragColor = vec4(0.0, 0.0, 0.0, 1.0);
        return;
    }
    
    // Borde brillante del disco (Boundary)
    if (r > DISK_RADIUS) {
        float edgeIntensity = smoothstep(1.0, DISK_RADIUS, r);
        gl_FragColor = vec4(0.2, 0.4, 1.0, 1.0) * edgeIntensity * 0.5;
        return;
    }
    
    // Mapeo holográfico: Disco -> Grid
    // Transformamos la coordenada del disco (hiperbólico) a coordenada de textura (euclidiano)
    // Usamos una transformación Möbius para navegar el espacio hiperbólico si quisiéramos
    // simular movimiento en el bulk, pero por ahora es estático.
    
    // Efecto de "Lente Gravitacional"
    // Distorsionamos las coordenadas UV basándonos en la distancia al centro
    // para simular la curvatura del espacio AdS.
    float distortion = 1.0 / (1.0 - r*r + 0.1); // Curvatura
    vec2 texCoord = uv * distortion * 0.5 + 0.5;
    
    // Sampling de la textura (el estado del universo 2D)
    // Usamos clamp para evitar repetir la textura fuera de [0,1]
    vec4 texColor = texture2D(uTexture, clamp(texCoord, 0.0, 1.0));
    
    // Extraer valor (asumiendo textura en escala de grises o R=val)
    float value = texColor.r;
    
    // Aplicar colormap
    vec3 color = getHeatmapColor(value);
    
    // Efecto de "Atmósfera" o "Glow" basado en la intensidad
    // Los valores altos brillan más
    color += value * value * 0.2;
    
    // Viñeteado suave hacia el borde del disco para dar sensación de profundidad
    float vignette = smoothstep(DISK_RADIUS, 0.0, r);
    color *= (0.5 + 0.5 * vignette);
    
    gl_FragColor = vec4(color, 1.0);
}
