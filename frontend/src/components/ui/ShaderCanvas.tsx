/**
 * Componente de Canvas con renderizado WebGL usando Shaders
 * 
 * Este componente usa shaders para procesar visualizaciones en GPU,
 * eliminando el overhead de procesamiento en CPU.
 */

import React, { useEffect, useMemo, useRef } from 'react';
import {
    createShaderProgram,
    createTextureFromData,
    FRAGMENT_SHADER_DENSITY,
    FRAGMENT_SHADER_ENERGY,
    FRAGMENT_SHADER_HSV,
    FRAGMENT_SHADER_IMAG,
    FRAGMENT_SHADER_PHASE,
    FRAGMENT_SHADER_REAL,
    isWebGLAvailable,
    renderWithShader,
    VERTEX_SHADER_2D,
    type ShaderConfig
} from '../../utils/shaderVisualization';

interface ShaderCanvasProps {
    mapData: number[][];
    width: number;
    height: number;
    selectedViz: string;
    minValue?: number;
    maxValue?: number;
    className?: string;
    style?: React.CSSProperties;
}

export const ShaderCanvas: React.FC<ShaderCanvasProps> = ({
    mapData,
    width,
    height,
    selectedViz,
    minValue,
    maxValue,
    className,
    style
}) => {
    const canvasRef = useRef<HTMLCanvasElement>(null);
    const glRef = useRef<WebGLRenderingContext | null>(null);
    const programRef = useRef<WebGLProgram | null>(null);
    const textureRef = useRef<WebGLTexture | null>(null);
    
    // Detectar si WebGL está disponible
    const webglAvailable = useMemo(() => isWebGLAvailable(), []);
    
    // Inicializar WebGL
    useEffect(() => {
        if (!webglAvailable || !canvasRef.current) return;
        
        const canvas = canvasRef.current;
        canvas.width = width;
        canvas.height = height;
        
        const gl = canvas.getContext('webgl') || canvas.getContext('experimental-webgl') as WebGLRenderingContext | null;
        if (!gl || !(gl instanceof WebGLRenderingContext)) {
            console.warn('WebGL no disponible, usando renderizado por defecto');
            return;
        }
        
        glRef.current = gl;
        
        // Crear programa de shader según el tipo de visualización
        let fragmentShader: string;
        if (selectedViz === 'phase') {
            fragmentShader = FRAGMENT_SHADER_PHASE;
        } else if (selectedViz === 'energy') {
            fragmentShader = FRAGMENT_SHADER_ENERGY;
        } else if (selectedViz === 'real') {
            fragmentShader = FRAGMENT_SHADER_REAL;
        } else if (selectedViz === 'imag') {
            fragmentShader = FRAGMENT_SHADER_IMAG;
        } else if (selectedViz === 'phase_hsv') {
            fragmentShader = FRAGMENT_SHADER_HSV;
        } else {
            // Default: density
            fragmentShader = FRAGMENT_SHADER_DENSITY;
        }
        
        const program = createShaderProgram(gl, VERTEX_SHADER_2D, fragmentShader);
        if (!program) {
            console.error('Error creando programa de shader');
            return;
        }
        
        programRef.current = program;
        
        return () => {
            // Cleanup
            if (textureRef.current) {
                gl.deleteTexture(textureRef.current);
            }
            if (program) {
                gl.deleteProgram(program);
            }
        };
    }, [webglAvailable, width, height, selectedViz]);
    
    // Renderizar con shaders
    useEffect(() => {
        if (!webglAvailable || !glRef.current || !programRef.current || !mapData || mapData.length === 0) {
            if (process.env.NODE_ENV === 'development') {
                if (!webglAvailable) console.warn('ShaderCanvas: WebGL no disponible');
                if (!glRef.current) console.warn('ShaderCanvas: glRef.current es null');
                if (!programRef.current) console.warn('ShaderCanvas: programRef.current es null');
                if (!mapData || mapData.length === 0) console.warn('ShaderCanvas: mapData vacío o inválido', { mapData, length: mapData?.length });
            }
            return;
        }
        
        const gl = glRef.current;
        const program = programRef.current;
        
        // DEBUG: Verificar que mapData tenga datos válidos
        if (process.env.NODE_ENV === 'development') {
            const mapDataHeight = mapData.length;
            const mapDataWidth = mapData[0]?.length || 0;
            let validCount = 0;
            let minVal = Infinity;
            let maxVal = -Infinity;
            
            for (let y = 0; y < Math.min(10, mapDataHeight); y++) {
                for (let x = 0; x < Math.min(10, mapDataWidth); x++) {
                    const val = mapData[y]?.[x];
                    if (typeof val === 'number' && !isNaN(val)) {
                        validCount++;
                        minVal = Math.min(minVal, val);
                        maxVal = Math.max(maxVal, val);
                    }
                }
            }
            
            console.log(`🔍 ShaderCanvas: mapData stats - shape: [${mapDataHeight}, ${mapDataWidth}], valid samples: ${validCount}, min: ${minVal}, max: ${maxVal}`);
        }
        
        // CRÍTICO: Si minValue y maxValue están definidos explícitamente (especialmente 0 y 1),
        // significa que los datos ya están normalizados desde el backend
        // En ese caso, NO calcular min/max de los datos, usar los valores proporcionados
        const dataMin = minValue !== undefined ? minValue : (() => {
            // Implementación robusta usando percentiles (igual que PanZoomCanvas)
            // para evitar que outliers o ruido de vacío distorsionen la visualización
            const values: number[] = [];
            // Muestreo para rendimiento en grids grandes
            const step = Math.max(1, Math.floor(mapData.length * mapData[0].length / 10000));
            
            for (let y = 0; y < mapData.length; y++) {
                for (let x = 0; x < mapData[y]?.length || 0; x += step) {
                    const val = mapData[y]?.[x];
                    if (typeof val === 'number' && !isNaN(val) && isFinite(val)) {
                        values.push(val);
                    }
                }
            }
            
            if (values.length === 0) return 0;
            
            // Si hay pocos valores, usar min/max directo
            if (values.length < 100) {
                return Math.min(...values);
            }
            
            // Ordenar para percentiles
            values.sort((a, b) => a - b);
            const p1Index = Math.floor(values.length * 0.01);
            return values[p1Index];
        })();
        
        const dataMax = maxValue !== undefined ? maxValue : (() => {
            // Implementación robusta usando percentiles
            const values: number[] = [];
            const step = Math.max(1, Math.floor(mapData.length * mapData[0].length / 10000));
            
            for (let y = 0; y < mapData.length; y++) {
                for (let x = 0; x < mapData[y]?.length || 0; x += step) {
                    const val = mapData[y]?.[x];
                    if (typeof val === 'number' && !isNaN(val) && isFinite(val)) {
                        values.push(val);
                    }
                }
            }
            
            if (values.length === 0) return 1;
            
            if (values.length < 100) {
                return Math.max(...values);
            }
            
            values.sort((a, b) => a - b);
            const p99Index = Math.floor(values.length * 0.99);
            return values[p99Index];
        })();
        
        if (process.env.NODE_ENV === 'development') {
            console.log(`🔍 ShaderCanvas: Normalización - dataMin: ${dataMin}, dataMax: ${dataMax}, range: ${dataMax - dataMin}, isPreNormalized: ${minValue === 0 && maxValue === 1}`);
        }
        
        // Limpiar textura anterior para evitar fugas de memoria y artefactos grises
        if (textureRef.current) {
          gl.deleteTexture(textureRef.current);
          textureRef.current = null;
        }
        // Crear/actualizar textura
        const texture = createTextureFromData(gl, mapData, width, height, dataMin, dataMax);
        if (!texture) {
          console.error('❌ ShaderCanvas: Error creando textura WebGL');
          return;
        }
        textureRef.current = texture;
        
        if (process.env.NODE_ENV === 'development') {
            console.log(`✅ ShaderCanvas: Textura creada exitosamente - width: ${width}, height: ${height}`);
        }
        
        // Configurar shader
        // IMPORTANTE: Si los datos ya están normalizados [0, 1], el shader NO necesita minValue/maxValue
        // porque la textura ya contiene valores normalizados y el shader los lee directamente
        const config: ShaderConfig = {
            type: selectedViz === 'phase' ? 'phase' : 'density',
            colormap: 'viridis',
            minValue: 0,  // Shader no usa estos valores cuando datos ya están normalizados, pero los pasamos para compatibilidad
            maxValue: 1,  // Shader no usa estos valores cuando datos ya están normalizados, pero los pasamos para compatibilidad
            gamma: 1.0
        };
        
        // Renderizar
        try {
            renderWithShader(gl, program, texture, config, width, height);
            if (process.env.NODE_ENV === 'development') {
                console.log(`✅ ShaderCanvas: Renderizado completado para viz_type=${selectedViz}, canvas size: ${width}x${height}`);
                
                // Verificar si el canvas es visible
                if (canvasRef.current) {
                    const rect = canvasRef.current.getBoundingClientRect();
                    const style = window.getComputedStyle(canvasRef.current);
                    console.log(`🔍 ShaderCanvas: Canvas visibility - rect: ${rect.width}x${rect.height}, visible: ${rect.width > 0 && rect.height > 0}, display: ${style.display}, visibility: ${style.visibility}, opacity: ${style.opacity}`);
                }
            }
        } catch (error) {
            console.error('❌ ShaderCanvas: Error durante renderizado:', error);
        }
    }, [webglAvailable, mapData, width, height, selectedViz, minValue, maxValue]);
    
    // Si WebGL no está disponible, renderizar canvas vacío (el componente padre usará Canvas2D)
    if (!webglAvailable) {
        return null;
    }
    
    return (
        <canvas
            ref={canvasRef}
            className={className}
            style={style}
            width={width}
            height={height}
        />
    );
};

