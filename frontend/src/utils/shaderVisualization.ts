/**
 * Utilidades para visualización con Shaders WebGL/GPU
 *
 * Este módulo implementa visualizaciones procesadas en GPU del navegador
 * para reducir el overhead del backend y mejorar el rendimiento.
 */

export interface ShaderConfig {
  type:
    | "density"
    | "phase"
    | "energy"
    | "complex"
    | "real"
    | "imag"
    | "entropy"
    | "gradient"
    | "flow";
  colormap?: "viridis" | "plasma" | "inferno" | "magma" | "turbo";
  minValue?: number;
  maxValue?: number;
  gamma?: number; // Corrección gamma para colormap
  channelMode?: number; // 0=Composite, 1=Ch0, 2=Ch1, 3=Ch2
}

// ... (existing code) ...

/**
 * Shader de fragment para visualización de Entropía Local (Shannon)
 * Calcula la entropía en una ventana de 3x3 alrededor de cada píxel
 */
export const FRAGMENT_SHADER_ENTROPY = `
    precision mediump float;
    
    uniform sampler2D u_texture;
    uniform vec2 u_resolution;
    uniform float u_gamma;
    uniform int u_colormap;
    
    varying vec2 v_texCoord;
    
    // Colormap Plasma
    vec3 plasma(float t) {
        t = clamp(t, 0.0, 1.0);
        vec3 c0 = vec3(0.050383, 0.029803, 0.527975);
        vec3 c1 = vec3(0.546989, 0.127025, 0.513125);
        vec3 c2 = vec3(0.998156, 0.401833, 0.255082);
        vec3 c3 = vec3(0.988362, 0.998364, 0.644924);
        
        if (t < 0.33) {
            return mix(c0, c1, t * 3.0);
        } else if (t < 0.66) {
            return mix(c1, c2, (t - 0.33) * 3.0);
        } else {
            return mix(c2, c3, (t - 0.66) * 3.0);
        }
    }

    void main() {
        vec2 onePixel = vec2(1.0, 1.0) / u_resolution;
        
        // Calcular histograma local en ventana 3x3
        // Simplificación: Usamos la varianza local como proxy de entropía para rendimiento
        // La entropía real requiere binning que es costoso en fragment shader
        
        float mean = 0.0;
        float values[9];
        
        int idx = 0;
        for(int i = -1; i <= 1; i++) {
            for(int j = -1; j <= 1; j++) {
                float val = texture2D(u_texture, v_texCoord + vec2(float(i), float(j)) * onePixel).r;
                values[idx] = val;
                mean += val;
                idx++;
            }
        }
        mean /= 9.0;
        
        float variance = 0.0;
        for(int i = 0; i < 9; i++) {
            float diff = values[i] - mean;
            variance += diff * diff;
        }
        variance /= 9.0;
        
        // Visualizar desviación estándar normalizada
        float stdDev = sqrt(variance);
        
        // Amplificar para visibilidad (la entropía local suele ser baja en campos suaves)
        float entropyProxy = clamp(stdDev * 10.0, 0.0, 1.0);
        
        vec3 color = plasma(entropyProxy);
        gl_FragColor = vec4(color, 1.0);
    }
`;

/**
 * Shader de fragment para visualización de Gradiente (|∇ρ|)
 */
export const FRAGMENT_SHADER_GRADIENT = `
    precision mediump float;
    
    uniform sampler2D u_texture;
    uniform vec2 u_resolution;
    uniform float u_gamma;
    
    varying vec2 v_texCoord;
    
    // Colormap Inferno (bueno para intensidad)
    vec3 inferno(float t) {
        t = clamp(t, 0.0, 1.0);
        return vec3(t, t*0.5, t*0.2); // Simplificado para demo, idealmente usar tabla completa
    }

    void main() {
        vec2 onePixel = vec2(1.0, 1.0) / u_resolution;
        
        // Operador Sobel para gradiente
        float gx = 0.0;
        float gy = 0.0;
        
        // Kernel X
        // -1 0 1
        // -2 0 2
        // -1 0 1
        gx += -1.0 * texture2D(u_texture, v_texCoord + vec2(-1.0, -1.0) * onePixel).r;
        gx += -2.0 * texture2D(u_texture, v_texCoord + vec2(-1.0,  0.0) * onePixel).r;
        gx += -1.0 * texture2D(u_texture, v_texCoord + vec2(-1.0,  1.0) * onePixel).r;
        gx +=  1.0 * texture2D(u_texture, v_texCoord + vec2( 1.0, -1.0) * onePixel).r;
        gx +=  2.0 * texture2D(u_texture, v_texCoord + vec2( 1.0,  0.0) * onePixel).r;
        gx +=  1.0 * texture2D(u_texture, v_texCoord + vec2( 1.0,  1.0) * onePixel).r;
        
        // Kernel Y
        // -1 -2 -1
        //  0  0  0
        //  1  2  1
        gy += -1.0 * texture2D(u_texture, v_texCoord + vec2(-1.0, -1.0) * onePixel).r;
        gy += -2.0 * texture2D(u_texture, v_texCoord + vec2( 0.0, -1.0) * onePixel).r;
        gy += -1.0 * texture2D(u_texture, v_texCoord + vec2( 1.0, -1.0) * onePixel).r;
        gy +=  1.0 * texture2D(u_texture, v_texCoord + vec2(-1.0,  1.0) * onePixel).r;
        gy +=  2.0 * texture2D(u_texture, v_texCoord + vec2( 0.0,  1.0) * onePixel).r;
        gy +=  1.0 * texture2D(u_texture, v_texCoord + vec2( 1.0,  1.0) * onePixel).r;
        
        float gradientMag = sqrt(gx*gx + gy*gy);
        
        // Amplificar para visibilidad
        float normalized = clamp(gradientMag * 2.0, 0.0, 1.0);
        
        // Color fuego/inferno
        vec3 color = vec3(normalized, pow(normalized, 3.0), pow(normalized, 10.0));
        
        gl_FragColor = vec4(color, 1.0);
    }
`;

/**
 * Shader de fragment para visualización de Flujo Denso
 * Mapea la dirección del gradiente a Hue y la magnitud a Value
 */
export const FRAGMENT_SHADER_FLOW = `
    precision mediump float;
    
    uniform sampler2D u_texture;
    uniform vec2 u_resolution;
    
    varying vec2 v_texCoord;
    
    vec3 hsvToRgb(float h, float s, float v) {
        h = mod(h, 1.0) * 6.0;
        float c = v * s;
        float x = c * (1.0 - abs(mod(h, 2.0) - 1.0));
        float m = v - c;
        
        vec3 rgb;
        if (h < 1.0) rgb = vec3(c, x, 0.0);
        else if (h < 2.0) rgb = vec3(x, c, 0.0);
        else if (h < 3.0) rgb = vec3(0.0, c, x);
        else if (h < 4.0) rgb = vec3(0.0, x, c);
        else if (h < 5.0) rgb = vec3(x, 0.0, c);
        else rgb = vec3(c, 0.0, x);
        
        return rgb + m;
    }

    void main() {
        vec2 onePixel = vec2(1.0, 1.0) / u_resolution;
        
        // Calcular gradiente (dirección del flujo)
        float center = texture2D(u_texture, v_texCoord).r;
        float left = texture2D(u_texture, v_texCoord + vec2(-1.0, 0.0) * onePixel).r;
        float right = texture2D(u_texture, v_texCoord + vec2(1.0, 0.0) * onePixel).r;
        float up = texture2D(u_texture, v_texCoord + vec2(0.0, -1.0) * onePixel).r;
        float down = texture2D(u_texture, v_texCoord + vec2(0.0, 1.0) * onePixel).r;
        
        float dx = right - left;
        float dy = down - up; // Y invertido en texturas a veces, ajustar según coord system
        
        float angle = atan(dy, dx);
        float magnitude = sqrt(dx*dx + dy*dy);
        
        // Normalizar ángulo a Hue [0, 1]
        float hue = (angle + 3.14159) / (2.0 * 3.14159);
        
        // Magnitud a Value (amplificada)
        float value = clamp(magnitude * 5.0, 0.0, 1.0);
        
        // Saturación constante para colores vivos
        float saturation = 1.0;
        
        vec3 color = hsvToRgb(hue, saturation, value);
        gl_FragColor = vec4(color, 1.0);
    }
`;

/**
 * Detecta si WebGL está disponible en el navegador
 */
export function isWebGLAvailable(): boolean {
  try {
    const canvas = document.createElement("canvas");
    const gl =
      canvas.getContext("webgl") || canvas.getContext("experimental-webgl");
    return gl !== null;
  } catch (e) {
    return false;
  }
}

/**
 * Detecta si WebGL2 está disponible (mejor que WebGL1)
 */
export function isWebGL2Available(): boolean {
  try {
    const canvas = document.createElement("canvas");
    const gl = canvas.getContext("webgl2");
    return gl !== null;
  } catch (e) {
    return false;
  }
}

/**
 * Shader de vertex básico para renderizado 2D
 */
export const VERTEX_SHADER_2D = `
    attribute vec2 a_position;
    attribute vec2 a_texCoord;
    
    varying vec2 v_texCoord;
    
    void main() {
        // Posición directa en clip space [-1, 1]
        gl_Position = vec4(a_position, 0.0, 1.0);
        v_texCoord = a_texCoord;
    }
`;

/**
 * Shader de fragment para visualización de densidad (density = |ψ|²)
 */
export const FRAGMENT_SHADER_DENSITY = `
    precision mediump float;
    
    uniform sampler2D u_texture;
    uniform vec2 u_textureSize;
    uniform float u_minValue;
    uniform float u_maxValue;
    uniform float u_gamma;
    uniform int u_colormap;
    
    varying vec2 v_texCoord;
    
    // Colormap Viridis (coloración científica estándar)
    vec3 viridis(float t) {
        // Aproximación de viridis colormap
        t = clamp(t, 0.0, 1.0);
        vec3 c0 = vec3(0.267004, 0.004874, 0.329415);
        vec3 c1 = vec3(0.127568, 0.566949, 0.550556);
        vec3 c2 = vec3(0.369214, 0.788888, 0.382914);
        vec3 c3 = vec3(0.993248, 0.906157, 0.143936);
        
        if (t < 0.33) {
            return mix(c0, c1, t * 3.0);
        } else if (t < 0.66) {
            return mix(c1, c2, (t - 0.33) * 3.0);
        } else {
            return mix(c2, c3, (t - 0.66) * 3.0);
        }
    }
    
    // Colormap Plasma (alternativa vibrante)
    vec3 plasma(float t) {
        t = clamp(t, 0.0, 1.0);
        vec3 c0 = vec3(0.050383, 0.029803, 0.527975);
        vec3 c1 = vec3(0.546989, 0.127025, 0.513125);
        vec3 c2 = vec3(0.998156, 0.401833, 0.255082);
        vec3 c3 = vec3(0.988362, 0.998364, 0.644924);
        
        if (t < 0.33) {
            return mix(c0, c1, t * 3.0);
        } else if (t < 0.66) {
            return mix(c1, c2, (t - 0.33) * 3.0);
        } else {
            return mix(c2, c3, (t - 0.66) * 3.0);
        }
    }
    
    void main() {
        // Obtener valor de densidad desde textura
        // La textura almacena datos normalizados a [0, 255] (UNSIGNED_BYTE)
        // WebGL lee esto automáticamente como [0, 1] cuando usamos LUMINANCE + UNSIGNED_BYTE
        // Por lo tanto, texel.r ya está en [0, 1] y NO necesitamos normalizar de nuevo
        vec4 texel = texture2D(u_texture, v_texCoord);
        float normalized = texel.r; // Ya está normalizado a [0, 1] desde la textura
        normalized = clamp(normalized, 0.0, 1.0);
        
        // Aplicar corrección gamma
        normalized = pow(normalized, u_gamma);
        
        // Aplicar colormap
        vec3 color;
        if (u_colormap == 0) {
            color = viridis(normalized);
        } else if (u_colormap == 1) {
            color = plasma(normalized);
        } else {
            // Default: grayscale
            color = vec3(normalized);
        }
        
        gl_FragColor = vec4(color, 1.0);
    }
`;

/**
 * Shader de fragment para visualización de fase (phase = angle(ψ))
 */
export const FRAGMENT_SHADER_PHASE = `
    precision mediump float;
    
    uniform sampler2D u_texture;
    uniform vec2 u_textureSize;
    
    varying vec2 v_texCoord;
    
    // Convertir fase a HSL (Hue basado en fase)
    vec3 phaseToHSL(float phase) {
        // Normalizar fase [-π, π] a [0, 1]
        float hue = (phase + 3.14159) / (2.0 * 3.14159);
        
        // HSL: H = hue, S = 1.0, L = 0.5
        float h = mod(hue, 1.0);
        float s = 1.0;
        float l = 0.5;
        
        // Convertir HSL a RGB
        float c = (1.0 - abs(2.0 * l - 1.0)) * s;
        float x = c * (1.0 - abs(mod(h * 6.0, 2.0) - 1.0));
        float m = l - c / 2.0;
        
        vec3 rgb;
        if (h < 1.0/6.0) {
            rgb = vec3(c, x, 0.0);
        } else if (h < 2.0/6.0) {
            rgb = vec3(x, c, 0.0);
        } else if (h < 3.0/6.0) {
            rgb = vec3(0.0, c, x);
        } else if (h < 4.0/6.0) {
            rgb = vec3(0.0, x, c);
        } else if (h < 5.0/6.0) {
            rgb = vec3(x, 0.0, c);
        } else {
            rgb = vec3(c, 0.0, x);
        }
        
        return rgb + m;
    }
    
    void main() {
        vec4 texel = texture2D(u_texture, v_texCoord);
        float phase = texel.r; // Fase almacenada en canal R
        
        vec3 color = phaseToHSL(phase);
        gl_FragColor = vec4(color, 1.0);
    }
`;

/**
 * Shader de fragment para visualización de energía (energy = |∇ψ|²)
 */
export const FRAGMENT_SHADER_ENERGY = `
    precision mediump float;
    
    uniform sampler2D u_texture;
    uniform vec2 u_textureSize;
    uniform float u_minValue;
    uniform float u_maxValue;
    uniform float u_gamma;
    uniform int u_colormap;
    
    varying vec2 v_texCoord;
    
    // Colormap Viridis (reutilizado de density shader)
    vec3 viridis(float t) {
        t = clamp(t, 0.0, 1.0);
        vec3 c0 = vec3(0.267004, 0.004874, 0.329415);
        vec3 c1 = vec3(0.127568, 0.566949, 0.550556);
        vec3 c2 = vec3(0.369214, 0.788888, 0.382914);
        vec3 c3 = vec3(0.993248, 0.906157, 0.143936);
        
        if (t < 0.33) {
            return mix(c0, c1, t * 3.0);
        } else if (t < 0.66) {
            return mix(c1, c2, (t - 0.33) * 3.0);
        } else {
            return mix(c2, c3, (t - 0.66) * 3.0);
        }
    }
    
    // Colormap Plasma (reutilizado)
    vec3 plasma(float t) {
        t = clamp(t, 0.0, 1.0);
        vec3 c0 = vec3(0.050383, 0.029803, 0.527975);
        vec3 c1 = vec3(0.546989, 0.127025, 0.513125);
        vec3 c2 = vec3(0.998156, 0.401833, 0.255082);
        vec3 c3 = vec3(0.988362, 0.998364, 0.644924);
        
        if (t < 0.33) {
            return mix(c0, c1, t * 3.0);
        } else if (t < 0.66) {
            return mix(c1, c2, (t - 0.33) * 3.0);
        } else {
            return mix(c2, c3, (t - 0.66) * 3.0);
        }
    }
    
    void main() {
        vec4 texel = texture2D(u_texture, v_texCoord);
        // La textura almacena datos normalizados a [0, 255] (UNSIGNED_BYTE)
        // WebGL lee esto automáticamente como [0, 1] cuando usamos LUMINANCE + UNSIGNED_BYTE
        // Por lo tanto, texel.r ya está en [0, 1] y NO necesitamos normalizar de nuevo
        float normalized = texel.r;
        normalized = clamp(normalized, 0.0, 1.0);
        
        // Aplicar corrección gamma
        normalized = pow(normalized, u_gamma);
        
        // Aplicar colormap
        vec3 color;
        if (u_colormap == 0) {
            color = viridis(normalized);
        } else if (u_colormap == 1) {
            color = plasma(normalized);
        } else {
            // Default: grayscale
            color = vec3(normalized);
        }
        
        gl_FragColor = vec4(color, 1.0);
    }
`;

/**
 * Shader de fragment para visualización de parte real (real = Re(ψ))
 */
export const FRAGMENT_SHADER_REAL = `
    precision mediump float;
    
    uniform sampler2D u_texture;
    uniform vec2 u_textureSize;
    uniform float u_minValue;
    uniform float u_maxValue;
    uniform float u_gamma;
    uniform int u_colormap;
    
    varying vec2 v_texCoord;
    
    // Colormap Viridis
    vec3 viridis(float t) {
        t = clamp(t, 0.0, 1.0);
        vec3 c0 = vec3(0.267004, 0.004874, 0.329415);
        vec3 c1 = vec3(0.127568, 0.566949, 0.550556);
        vec3 c2 = vec3(0.369214, 0.788888, 0.382914);
        vec3 c3 = vec3(0.993248, 0.906157, 0.143936);
        
        if (t < 0.33) {
            return mix(c0, c1, t * 3.0);
        } else if (t < 0.66) {
            return mix(c1, c2, (t - 0.33) * 3.0);
        } else {
            return mix(c2, c3, (t - 0.66) * 3.0);
        }
    }
    
    void main() {
        vec4 texel = texture2D(u_texture, v_texCoord);
        // La textura ya está normalizada a [0, 1] desde [dataMin, dataMax]
        // Usamos directamente texel.r que ya está en [0, 1]
        float normalized = texel.r;
        normalized = clamp(normalized, 0.0, 1.0);
        
        // Aplicar corrección gamma
        normalized = pow(normalized, u_gamma);
        
        // Aplicar colormap
        vec3 color = viridis(normalized);
        
        gl_FragColor = vec4(color, 1.0);
    }
`;

/**
 * Shader de fragment para visualización de parte imaginaria (imag = Im(ψ))
 */
export const FRAGMENT_SHADER_IMAG = `
    precision mediump float;
    
    uniform sampler2D u_texture;
    uniform vec2 u_textureSize;
    uniform float u_minValue;
    uniform float u_maxValue;
    uniform float u_gamma;
    uniform int u_colormap;
    
    varying vec2 v_texCoord;
    
    // Colormap Viridis
    vec3 viridis(float t) {
        t = clamp(t, 0.0, 1.0);
        vec3 c0 = vec3(0.267004, 0.004874, 0.329415);
        vec3 c1 = vec3(0.127568, 0.566949, 0.550556);
        vec3 c2 = vec3(0.369214, 0.788888, 0.382914);
        vec3 c3 = vec3(0.993248, 0.906157, 0.143936);
        
        if (t < 0.33) {
            return mix(c0, c1, t * 3.0);
        } else if (t < 0.66) {
            return mix(c1, c2, (t - 0.33) * 3.0);
        } else {
            return mix(c2, c3, (t - 0.66) * 3.0);
        }
    }
    
    void main() {
        vec4 texel = texture2D(u_texture, v_texCoord);
        // La textura ya está normalizada a [0, 1] desde [dataMin, dataMax]
        // Usamos directamente texel.r que ya está en [0, 1]
        float normalized = texel.r;
        normalized = clamp(normalized, 0.0, 1.0);
        
        // Aplicar corrección gamma
        normalized = pow(normalized, u_gamma);
        
        // Aplicar colormap
        vec3 color = viridis(normalized);
        
        gl_FragColor = vec4(color, 1.0);
    }
`;

/**
 * Shader de fragment para visualización HSV de fase
 * Mapea la fase a HSV: Hue = fase, Saturation = 1.0, Value = densidad/intensidad
 * El backend envía la fase ya normalizada a [0, 1] donde representa phase_normalized = (phase + π) / (2π)
 */
export const FRAGMENT_SHADER_HSV = `
    precision mediump float;
    
    uniform sampler2D u_texture;
    uniform vec2 u_textureSize;
    
    varying vec2 v_texCoord;
    
    // Convertir HSV a RGB
    // h: hue [0, 1], s: saturation [0, 1], v: value [0, 1]
    vec3 hsvToRgb(float h, float s, float v) {
        h = mod(h, 1.0) * 6.0; // h en [0, 6]
        float c = v * s;       // chroma
        float x = c * (1.0 - abs(mod(h, 2.0) - 1.0));
        float m = v - c;
        
        vec3 rgb;
        if (h < 1.0) {
            rgb = vec3(c, x, 0.0);
        } else if (h < 2.0) {
            rgb = vec3(x, c, 0.0);
        } else if (h < 3.0) {
            rgb = vec3(0.0, c, x);
        } else if (h < 4.0) {
            rgb = vec3(0.0, x, c);
        } else if (h < 5.0) {
            rgb = vec3(x, 0.0, c);
        } else {
            rgb = vec3(c, 0.0, x);
        }
        
        return rgb + m;
    }
    
    void main() {
        vec4 texel = texture2D(u_texture, v_texCoord);
        
        // La textura contiene phase_normalized en canal R [0, 1]
        // donde phase_normalized = (phase + π) / (2π)
        float hue = texel.r;
        
        // Usar saturation = 1.0 y value = 1.0 para colores vibrantes
        // Si quisiéramos modular brightness por densidad, podríamos usar texel.g
        float saturation = 1.0;
        float value = 1.0;
        
        vec3 color = hsvToRgb(hue, saturation, value);
        gl_FragColor = vec4(color, 1.0);
    }
`;

/**
 * Shader de fragment para visualización holográfica (Poincaré Disk)
 * Mapea el grid 2D (Euclidiano) al Disco de Poincaré (Hiperbólico)
 * Simula la correspondencia AdS/CFT donde el grid es el Boundary y el disco es el Bulk
 */
export const FRAGMENT_SHADER_POINCARE = `
    precision mediump float;
    
    uniform sampler2D u_texture;
    uniform vec2 u_resolution;
    uniform float u_minValue;
    uniform float u_maxValue;
    uniform float u_gamma;
    uniform int u_colormap;
    
    varying vec2 v_texCoord;
    
    // Constantes para el modelo de Poincaré
    const float DISK_RADIUS = 0.95;
    
    // Colormap Viridis
    vec3 viridis(float t) {
        t = clamp(t, 0.0, 1.0);
        vec3 c0 = vec3(0.267004, 0.004874, 0.329415);
        vec3 c1 = vec3(0.127568, 0.566949, 0.550556);
        vec3 c2 = vec3(0.369214, 0.788888, 0.382914);
        vec3 c3 = vec3(0.993248, 0.906157, 0.143936);
        
        if (t < 0.33) {
            return mix(c0, c1, t * 3.0);
        } else if (t < 0.66) {
            return mix(c1, c2, (t - 0.33) * 3.0);
        } else {
            return mix(c2, c3, (t - 0.66) * 3.0);
        }
    }
    
    // Colormap Plasma
    vec3 plasma(float t) {
        t = clamp(t, 0.0, 1.0);
        vec3 c0 = vec3(0.050383, 0.029803, 0.527975);
        vec3 c1 = vec3(0.546989, 0.127025, 0.513125);
        vec3 c2 = vec3(0.998156, 0.401833, 0.255082);
        vec3 c3 = vec3(0.988362, 0.998364, 0.644924);
        
        if (t < 0.33) {
            return mix(c0, c1, t * 3.0);
        } else if (t < 0.66) {
            return mix(c1, c2, (t - 0.33) * 3.0);
        } else {
            return mix(c2, c3, (t - 0.66) * 3.0);
        }
    }
    
    void main() {
        // Coordenadas normalizadas [-1, 1] centradas
        // v_texCoord va de [0, 0] a [1, 1]
        vec2 uv = v_texCoord * 2.0 - 1.0;
        
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
        
        // Efecto de "Lente Gravitacional" / Proyección Estereográfica Inversa
        // Distorsionamos las coordenadas UV basándonos en la distancia al centro
        // para simular la curvatura del espacio AdS.
        // r' = r^alpha (donde alpha < 1 expande el centro, alpha > 1 expande el borde)
        float r_distorted = pow(r, 0.7); // Expandir ligeramente el centro
        
        float theta = atan(uv.y, uv.x);
        vec2 texCoord = vec2(
            0.5 + 0.5 * r_distorted * cos(theta),
            0.5 + 0.5 * r_distorted * sin(theta)
        );
        
        // Sampling de la textura (el estado del universo 2D)
        vec4 texel = texture2D(u_texture, clamp(texCoord, 0.0, 1.0));
        float normalized = texel.r;
        
        // Aplicar corrección gamma
        normalized = pow(normalized, u_gamma);
        
        // Aplicar colormap
        vec3 color;
        if (u_colormap == 0) {
            color = viridis(normalized);
        } else if (u_colormap == 1) {
            color = plasma(normalized);
        } else {
            color = vec3(normalized);
        }
        
        // Efecto de "Atmósfera" o "Glow" basado en la intensidad
        color += normalized * normalized * 0.2;
        
        // Viñeteado suave hacia el borde del disco para dar sensación de profundidad
        float vignette = smoothstep(DISK_RADIUS, 0.0, r);
        color *= (0.5 + 0.5 * vignette);
        
        gl_FragColor = vec4(color, 1.0);
    }
`;

/**
 * Shader de fragment para visualización de Campos (Field Theory)
 * Mapea los canales RGB a campos físicos:
 * R: Campo Electromagnético (Energía)
 * G: Campo Gravitatorio (Fase/Curvatura)
 * B: Campo de Higgs (Masa)
 */
export const FRAGMENT_SHADER_FIELDS = `
    precision mediump float;
    
    uniform sampler2D u_texture;
    uniform vec2 u_resolution;
    uniform int u_channel_mode; // 0=Composite, 1=Ch0(R), 2=Ch1(G), 3=Ch2(B)
    uniform float u_gamma;
    
    varying vec2 v_texCoord;
    
    void main() {
        vec4 texel = texture2D(u_texture, v_texCoord);
        vec3 color = vec3(0.0);
        
        if (u_channel_mode == 0) {
            // Composite: Mostrar RGB tal cual
            color = texel.rgb;
            
            // Aplicar gamma a cada canal
            color = pow(color, vec3(u_gamma));
        } else if (u_channel_mode == 1) {
            // Channel 0 (Red) - Mostrar en Rojo o Blanco?
            // Mejor mostrar en su color correspondiente (Rojo)
            float val = pow(texel.r, u_gamma);
            color = vec3(val, 0.0, 0.0);
        } else if (u_channel_mode == 2) {
            // Channel 1 (Green)
            float val = pow(texel.g, u_gamma);
            color = vec3(0.0, val, 0.0);
        } else if (u_channel_mode == 3) {
            // Channel 2 (Blue)
            float val = pow(texel.b, u_gamma);
            color = vec3(0.0, 0.0, val);
        }
        
        gl_FragColor = vec4(color, 1.0);
    }
`;

/**
 * Crea un programa de shader WebGL
 */
export function createShaderProgram(
  gl: WebGLRenderingContext,
  vertexSource: string,
  fragmentSource: string
): WebGLProgram | null {
  const vertexShader = compileShader(gl, gl.VERTEX_SHADER, vertexSource);
  const fragmentShader = compileShader(gl, gl.FRAGMENT_SHADER, fragmentSource);

  if (!vertexShader || !fragmentShader) {
    return null;
  }

  const program = gl.createProgram();
  if (!program) {
    return null;
  }

  gl.attachShader(program, vertexShader);
  gl.attachShader(program, fragmentShader);
  gl.linkProgram(program);

  if (!gl.getProgramParameter(program, gl.LINK_STATUS)) {
    console.error(
      "Error linking shader program:",
      gl.getProgramInfoLog(program)
    );
    gl.deleteProgram(program);
    return null;
  }

  return program;
}

/**
 * Compila un shader individual
 */
function compileShader(
  gl: WebGLRenderingContext,
  type: number,
  source: string
): WebGLShader | null {
  const shader = gl.createShader(type);
  if (!shader) {
    return null;
  }

  gl.shaderSource(shader, source);
  gl.compileShader(shader);

  if (!gl.getShaderParameter(shader, gl.COMPILE_STATUS)) {
    console.error("Error compiling shader:", gl.getShaderInfoLog(shader));
    gl.deleteShader(shader);
    return null;
  }

  return shader;
}

/**
 * Crea una textura WebGL desde datos de array 2D
 *
 * NOTA: Esta función NO normaliza los datos. Los datos se pasan directamente
 * a la textura, y la normalización se hace en el shader usando u_minValue y u_maxValue.
 * Esto evita doble normalización que puede causar visualización incorrecta.
 */
export function createTextureFromData(
  gl: WebGLRenderingContext,
  data: any[][], // number[][] or number[][][]
  width: number,
  height: number,
  minValue?: number,
  maxValue?: number
): WebGLTexture | null {
  const texture = gl.createTexture();
  if (!texture) return null;

  gl.bindTexture(gl.TEXTURE_2D, texture);

  // Detect 3D data (RGB)
  const is3D = data.length > 0 && Array.isArray(data[0][0]);

  // Flatten array
  // If 3D, we need 3 channels. If 2D, 1 channel.
  const channels = is3D ? 3 : 1;
  const flatData = new Float32Array(width * height * channels);

  for (let y = 0; y < height; y++) {
    for (let x = 0; x < width; x++) {
      if (is3D) {
        const pixel = (data[y]?.[x] as unknown as number[]) || [0, 0, 0];
        const idx = (y * width + x) * 3;
        flatData[idx] = pixel[0] || 0;
        flatData[idx + 1] = pixel[1] || 0;
        flatData[idx + 2] = pixel[2] || 0;
      } else {
        flatData[y * width + x] = (data[y]?.[x] as number) ?? 0;
      }
    }
  }

  // Determine normalization
  const isPreNormalized =
    (minValue === 0 && maxValue === 1) ||
    (minValue === undefined && maxValue === undefined);
  let dataMin = minValue;
  let dataMax = maxValue;
  if (dataMin === undefined || dataMax === undefined) {
    let minVal = Infinity;
    let maxVal = -Infinity;
    for (let i = 0; i < flatData.length; i++) {
      const v = flatData[i];
      if (isFinite(v)) {
        minVal = Math.min(minVal, v);
        maxVal = Math.max(maxVal, v);
      }
    }
    dataMin = minVal === Infinity ? 0 : minVal;
    dataMax = maxVal === -Infinity ? 1 : maxVal;
    if (dataMin >= 0 && dataMax <= 1 && dataMax - dataMin < 1.1) {
      dataMin = 0;
      dataMax = 1;
    }
  }
  const dataRange = dataMax - dataMin || 1;
  const isNormalized =
    (dataMin === 0 && dataMax === 1) ||
    (Math.abs(dataMin) < 0.01 && Math.abs(dataMax - 1) < 0.01);

  // Build RGBA Uint8Array
  const rgbaData = new Uint8Array(width * height * 4);

  if (is3D) {
    // Handle 3D RGB data
    // Assuming data is already normalized [0, 1] from backend for 'fields' viz
    for (let i = 0; i < width * height; i++) {
      const srcIdx = i * 3;
      const dstIdx = i * 4;

      rgbaData[dstIdx] = Math.round(
        Math.max(0, Math.min(1, flatData[srcIdx])) * 255
      ); // R
      rgbaData[dstIdx + 1] = Math.round(
        Math.max(0, Math.min(1, flatData[srcIdx + 1])) * 255
      ); // G
      rgbaData[dstIdx + 2] = Math.round(
        Math.max(0, Math.min(1, flatData[srcIdx + 2])) * 255
      ); // B
      rgbaData[dstIdx + 3] = 255; // A
    }
  } else {
    // Handle 2D Grayscale data
    for (let i = 0; i < flatData.length; i++) {
      const val = flatData[i];
      let normalized: number;
      if (isFinite(val)) {
        if (isNormalized || isPreNormalized) {
          normalized = Math.max(0, Math.min(1, val));
        } else {
          normalized = (val - dataMin) / dataRange;
          normalized = Math.max(0, Math.min(1, normalized));
        }
      } else {
        normalized = 0;
      }
      const byte = Math.round(normalized * 255);
      const idx = i * 4;
      rgbaData[idx] = byte; // R
      rgbaData[idx + 1] = 0; // G
      rgbaData[idx + 2] = 0; // B
      rgbaData[idx + 3] = 255; // A
    }
  }

  // Upload texture as RGBA
  gl.texImage2D(
    gl.TEXTURE_2D,
    0,
    gl.RGBA,
    width,
    height,
    0,
    gl.RGBA,
    gl.UNSIGNED_BYTE,
    rgbaData
  );

  // Set texture parameters
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.LINEAR);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.LINEAR);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);

  return texture;
}

/**
 * Renderiza datos usando shaders WebGL
 */
export function renderWithShader(
  gl: WebGLRenderingContext,
  program: WebGLProgram,
  texture: WebGLTexture,
  config: ShaderConfig,
  canvasWidth: number,
  canvasHeight: number
): void {
  gl.useProgram(program);

  // Configurar viewport
  gl.viewport(0, 0, canvasWidth, canvasHeight);
  gl.clearColor(0.1, 0.1, 0.1, 1.0);
  gl.clear(gl.COLOR_BUFFER_BIT);

  // Activar textura
  gl.activeTexture(gl.TEXTURE0);
  gl.bindTexture(gl.TEXTURE_2D, texture);

  // Configurar uniforms
  const resolutionLocation = gl.getUniformLocation(program, "u_resolution");
  const textureLocation = gl.getUniformLocation(program, "u_texture");
  const minValueLocation = gl.getUniformLocation(program, "u_minValue");
  const maxValueLocation = gl.getUniformLocation(program, "u_maxValue");
  const gammaLocation = gl.getUniformLocation(program, "u_gamma");
  const colormapLocation = gl.getUniformLocation(program, "u_colormap");
  const channelModeLocation = gl.getUniformLocation(program, "u_channel_mode");

  if (resolutionLocation) {
    gl.uniform2f(resolutionLocation, canvasWidth, canvasHeight);
  }
  if (textureLocation) {
    gl.uniform1i(textureLocation, 0);
  }
  // CRÍTICO: Los datos ya están normalizados [0, 1] en la textura
  // El shader NO necesita usar u_minValue/u_maxValue porque lee directamente texel.r [0, 1]
  // Pasamos 0 y 1 para compatibilidad, pero el shader no los usa
  if (minValueLocation) {
    gl.uniform1f(minValueLocation, 0); // Shader no usa este valor, pero lo pasamos por compatibilidad
  }
  if (maxValueLocation) {
    gl.uniform1f(maxValueLocation, 1); // Shader no usa este valor, pero lo pasamos por compatibilidad
  }
  if (gammaLocation) {
    gl.uniform1f(gammaLocation, config.gamma || 1.0);
  }
  if (colormapLocation) {
    const colormapIndex = config.colormap === "plasma" ? 1 : 0;
    gl.uniform1i(colormapLocation, colormapIndex);
  }
  if (channelModeLocation) {
    gl.uniform1i(channelModeLocation, config.channelMode || 0);
  }

  // Renderizar cuadrilátero completo
  const positionBuffer = gl.createBuffer();
  gl.bindBuffer(gl.ARRAY_BUFFER, positionBuffer);
  gl.bufferData(
    gl.ARRAY_BUFFER,
    new Float32Array([-1, -1, 0, 1, 1, -1, 1, 1, -1, 1, 0, 0, 1, 1, 1, 0]),
    gl.STATIC_DRAW
  );

  const positionLocation = gl.getAttribLocation(program, "a_position");
  const texCoordLocation = gl.getAttribLocation(program, "a_texCoord");

  if (positionLocation >= 0) {
    gl.enableVertexAttribArray(positionLocation);
    gl.vertexAttribPointer(positionLocation, 2, gl.FLOAT, false, 16, 0);
  }
  if (texCoordLocation >= 0) {
    gl.enableVertexAttribArray(texCoordLocation);
    gl.vertexAttribPointer(texCoordLocation, 2, gl.FLOAT, false, 16, 8);
  }

  gl.drawArrays(gl.TRIANGLE_STRIP, 0, 4);
}
