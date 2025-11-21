/**
 * Utilidades para visualización con Shaders WebGL/GPU
 * 
 * Este módulo implementa visualizaciones procesadas en GPU del navegador
 * para reducir el overhead del backend y mejorar el rendimiento.
 */

export interface ShaderConfig {
    type: 'density' | 'phase' | 'energy' | 'complex';
    colormap?: 'viridis' | 'plasma' | 'inferno' | 'magma' | 'turbo';
    minValue?: number;
    maxValue?: number;
    gamma?: number; // Corrección gamma para colormap
}

/**
 * Detecta si WebGL está disponible en el navegador
 */
export function isWebGLAvailable(): boolean {
    try {
        const canvas = document.createElement('canvas');
        const gl = canvas.getContext('webgl') || canvas.getContext('experimental-webgl');
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
        const canvas = document.createElement('canvas');
        const gl = canvas.getContext('webgl2');
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
    
    uniform vec2 u_resolution;
    uniform vec2 u_scale;
    uniform vec2 u_offset;
    
    varying vec2 v_texCoord;
    
    void main() {
        // Escalar y desplazar posición
        vec2 position = (a_position * u_scale + u_offset) / u_resolution * 2.0 - 1.0;
        // Invertir Y para canvas
        position.y = -position.y;
        gl_Position = vec4(position, 0.0, 1.0);
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
        vec4 texel = texture2D(u_texture, v_texCoord);
        float density = texel.r; // Asumir que densidad está en canal R
        
        // Normalizar a rango [0, 1]
        float normalized = (density - u_minValue) / (u_maxValue - u_minValue);
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
        console.error('Error linking shader program:', gl.getProgramInfoLog(program));
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
        console.error('Error compiling shader:', gl.getShaderInfoLog(shader));
        gl.deleteShader(shader);
        return null;
    }
    
    return shader;
}

/**
 * Crea una textura WebGL desde datos de array 2D
 */
export function createTextureFromData(
    gl: WebGLRenderingContext,
    data: number[][],
    width: number,
    height: number
): WebGLTexture | null {
    const texture = gl.createTexture();
    if (!texture) {
        return null;
    }
    
    gl.bindTexture(gl.TEXTURE_2D, texture);
    
    // Convertir array 2D a array plano (row-major order)
    const flatData = new Float32Array(width * height);
    for (let y = 0; y < height; y++) {
        for (let x = 0; x < width; x++) {
            flatData[y * width + x] = data[y]?.[x] || 0;
        }
    }
    
    // Crear textura (usar formato R32F para un solo canal de float)
    // Si WebGL2 no está disponible, usar RGBA8 como fallback
    gl.texImage2D(
        gl.TEXTURE_2D,
        0,
        gl.LUMINANCE, // Un solo canal
        width,
        height,
        0,
        gl.LUMINANCE,
        gl.FLOAT,
        flatData
    );
    
    // Configurar filtrado
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
    const resolutionLocation = gl.getUniformLocation(program, 'u_resolution');
    const textureLocation = gl.getUniformLocation(program, 'u_texture');
    const minValueLocation = gl.getUniformLocation(program, 'u_minValue');
    const maxValueLocation = gl.getUniformLocation(program, 'u_maxValue');
    const gammaLocation = gl.getUniformLocation(program, 'u_gamma');
    const colormapLocation = gl.getUniformLocation(program, 'u_colormap');
    
    if (resolutionLocation) {
        gl.uniform2f(resolutionLocation, canvasWidth, canvasHeight);
    }
    if (textureLocation) {
        gl.uniform1i(textureLocation, 0);
    }
    if (minValueLocation) {
        gl.uniform1f(minValueLocation, config.minValue || 0);
    }
    if (maxValueLocation) {
        gl.uniform1f(maxValueLocation, config.maxValue || 1);
    }
    if (gammaLocation) {
        gl.uniform1f(gammaLocation, config.gamma || 1.0);
    }
    if (colormapLocation) {
        const colormapIndex = config.colormap === 'plasma' ? 1 : 0;
        gl.uniform1i(colormapLocation, colormapIndex);
    }
    
    // Renderizar cuadrilátero completo
    const positionBuffer = gl.createBuffer();
    gl.bindBuffer(gl.ARRAY_BUFFER, positionBuffer);
    gl.bufferData(
        gl.ARRAY_BUFFER,
        new Float32Array([
            -1, -1, 0, 1,
             1, -1, 1, 1,
            -1,  1, 0, 0,
             1,  1, 1, 0
        ]),
        gl.STATIC_DRAW
    );
    
    const positionLocation = gl.getAttribLocation(program, 'a_position');
    const texCoordLocation = gl.getAttribLocation(program, 'a_texCoord');
    
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

