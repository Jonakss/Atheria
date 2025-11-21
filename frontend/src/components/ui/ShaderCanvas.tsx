/**
 * Componente de Canvas con renderizado WebGL usando Shaders
 * 
 * Este componente usa shaders para procesar visualizaciones en GPU,
 * eliminando el overhead de procesamiento en CPU.
 */

import React, { useRef, useEffect, useMemo } from 'react';
import {
    isWebGLAvailable,
    isWebGL2Available,
    createShaderProgram,
    createTextureFromData,
    renderWithShader,
    VERTEX_SHADER_2D,
    FRAGMENT_SHADER_DENSITY,
    FRAGMENT_SHADER_PHASE,
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
        } else {
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
            return;
        }
        
        const gl = glRef.current;
        const program = programRef.current;
        
        // Calcular min/max si no se proporcionan
        const dataMin = minValue ?? (() => {
            let min = Infinity;
            for (let y = 0; y < mapData.length; y++) {
                for (let x = 0; x < mapData[y]?.length || 0; x++) {
                    const val = mapData[y]?.[x];
                    if (typeof val === 'number' && !isNaN(val)) {
                        min = Math.min(min, val);
                    }
                }
            }
            return min === Infinity ? 0 : min;
        })();
        
        const dataMax = maxValue ?? (() => {
            let max = -Infinity;
            for (let y = 0; y < mapData.length; y++) {
                for (let x = 0; x < mapData[y]?.length || 0; x++) {
                    const val = mapData[y]?.[x];
                    if (typeof val === 'number' && !isNaN(val)) {
                        max = Math.max(max, val);
                    }
                }
            }
            return max === -Infinity ? 1 : max;
        })();
        
        // Crear/actualizar textura
        if (textureRef.current) {
            gl.deleteTexture(textureRef.current);
        }
        
        const texture = createTextureFromData(gl, mapData, width, height);
        if (!texture) {
            console.error('Error creando textura WebGL');
            return;
        }
        
        textureRef.current = texture;
        
        // Configurar shader
        const config: ShaderConfig = {
            type: selectedViz === 'phase' ? 'phase' : 'density',
            colormap: 'viridis',
            minValue: dataMin,
            maxValue: dataMax,
            gamma: 1.0
        };
        
        // Renderizar
        renderWithShader(gl, program, texture, config, width, height);
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

