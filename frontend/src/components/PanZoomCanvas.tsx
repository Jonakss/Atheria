// frontend/src/components/PanZoomCanvas.tsx
import { useRef, useEffect } from 'react';
import { Box, Text } from '@mantine/core';
import { TransformWrapper, TransformComponent } from 'react-zoom-pan-pinch';
import { useWebSocket } from '../context/WebSocketContext';
import classes from './PanZoomCanvas.module.css';

export function PanZoomCanvas() {
    const canvasRef = useRef<HTMLCanvasElement>(null);
    const { simData } = useWebSocket();

    useEffect(() => {
        const canvas = canvasRef.current;
        // Si no hay datos de simulación, no hacer nada
        if (!canvas || !simData || !simData.frame_data || !simData.frame_data.density_map) {
            return;
        }

        const ctx = canvas.getContext('2d');
        if (!ctx) return;

        const densityMap = simData.frame_data.density_map;
        const gridSize = densityMap.length;
        const cellSize = canvas.width / gridSize; // Asumimos canvas cuadrado

        // Limpiar el canvas antes de dibujar
        ctx.clearRect(0, 0, canvas.width, canvas.height);

        // Dibujar la densidad
        for (let y = 0; y < gridSize; y++) {
            for (let x = 0; x < gridSize; x++) {
                const value = densityMap[y][x];
                // Mapeo de color: 0 = azul (frío), 1 = rojo (caliente)
                const hue = 240 * (1 - value); 
                ctx.fillStyle = `hsl(${hue}, 100%, 50%)`;
                ctx.fillRect(x * cellSize, y * cellSize, cellSize, cellSize);
            }
        }
    }, [simData]); // Redibujar solo cuando simData cambia

    return (
        <Box className={classes.canvasContainer}>
            <TransformWrapper
                limitToBounds={false}
                minScale={0.2}
                maxScale={15}
                initialScale={1}
                initialPositionX={0}
                initialPositionY={0}
            >
                <TransformComponent
                    wrapperStyle={{ width: '100%', height: '100%' }}
                    contentStyle={{ width: '100%', height: '100%', display: 'flex', justifyContent: 'center', alignItems: 'center' }}
                >
                    <canvas ref={canvasRef} width="800" height="800" className={classes.canvas} />
                </TransformComponent>
            </TransformWrapper>
            
            {/* Mensaje de espera si no hay datos de simulación */}
            {!simData && (
                <Text c="dimmed" style={{ position: 'absolute', top: '50%', left: '50%', transform: 'translate(-50%, -50%)' }}>
                    Esperando datos de la simulación...
                </Text>
            )}
        </Box>
    );
}
