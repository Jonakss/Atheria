// frontend/src/components/PhaseAttractorViewer.tsx
import { useRef, useEffect, useState } from 'react';
import { Paper, Stack, Text, Group, Badge, Tooltip, ActionIcon } from '@mantine/core';
import { useWebSocket } from '../hooks/useWebSocket';
import { IconChartLine, IconHelpCircle } from '@tabler/icons-react';

interface PhasePoint {
    x: number;
    y: number;
    timestamp: number;
}

export function PhaseAttractorViewer() {
    const canvasRef = useRef<HTMLCanvasElement>(null);
    const { simData } = useWebSocket();
    const [attractorHistory, setAttractorHistory] = useState<PhasePoint[]>([]);
    const maxHistoryPoints = 1000; // Máximo de puntos en el historial
    
    useEffect(() => {
        const canvas = canvasRef.current;
        if (!canvas) return;
        
        canvas.width = 400;
        canvas.height = 400;
        const ctx = canvas.getContext('2d');
        if (!ctx) return;
        
        // Fondo oscuro
        ctx.fillStyle = '#1A1B1E';
        ctx.fillRect(0, 0, canvas.width, canvas.height);
        
        // Dibujar ejes
        const centerX = canvas.width / 2;
        const centerY = canvas.height / 2;
        
        ctx.strokeStyle = '#444';
        ctx.lineWidth = 1;
        ctx.beginPath();
        ctx.moveTo(0, centerY);
        ctx.lineTo(canvas.width, centerY);
        ctx.stroke();
        ctx.beginPath();
        ctx.moveTo(centerX, 0);
        ctx.lineTo(centerX, canvas.height);
        ctx.stroke();
        
        // Dibujar círculo de referencia
        ctx.strokeStyle = '#555';
        ctx.lineWidth = 0.5;
        const radius = Math.min(centerX, centerY) * 0.9;
        ctx.beginPath();
        ctx.arc(centerX, centerY, radius, 0, 2 * Math.PI);
        ctx.stroke();
        
        if (attractorHistory.length === 0) {
            // Mostrar mensaje de espera
            ctx.fillStyle = '#999';
            ctx.font = '14px sans-serif';
            ctx.textAlign = 'center';
            ctx.fillText('Esperando datos...', centerX, centerY);
            return;
        }
        
        // Encontrar el rango de datos para escalar
        const allX = attractorHistory.map(p => p.x);
        const allY = attractorHistory.map(p => p.y);
        // Usar reduce en lugar de spread operator para evitar stack overflow con arrays grandes
        const minX = allX.reduce((a, b) => Math.min(a, b), Infinity);
        const maxX = allX.reduce((a, b) => Math.max(a, b), -Infinity);
        const minY = allY.reduce((a, b) => Math.min(a, b), Infinity);
        const maxY = allY.reduce((a, b) => Math.max(a, b), -Infinity);
        
        const rangeX = maxX - minX || 1;
        const rangeY = maxY - minY || 1;
        const maxRange = Math.max(Math.abs(minX), Math.abs(maxX), Math.abs(minY), Math.abs(maxY)) || 1;
        
        // Función de escala
        const scaleX = (x: number) => centerX + (x / maxRange) * radius * 0.9;
        const scaleY = (y: number) => centerY - (y / maxRange) * radius * 0.9; // Invertir Y para coordenadas canvas
        
        // Dibujar puntos del atractor (más antiguos primero, más nuevos con mayor opacidad)
        attractorHistory.forEach((point, idx) => {
            const age = idx / attractorHistory.length;
            const alpha = 0.3 + age * 0.7; // Los puntos más recientes son más brillantes
            const size = 1 + age * 2; // Los puntos más recientes son más grandes
            
            const x = scaleX(point.x);
            const y = scaleY(point.y);
            
            if (isNaN(x) || isNaN(y) || !isFinite(x) || !isFinite(y)) return;
            
            ctx.fillStyle = `rgba(57, 175, 255, ${alpha})`;
            ctx.beginPath();
            ctx.arc(x, y, size, 0, 2 * Math.PI);
            ctx.fill();
        });
        
        // Dibujar el punto más reciente con un círculo más grande
        if (attractorHistory.length > 0) {
            const latest = attractorHistory[attractorHistory.length - 1];
            const x = scaleX(latest.x);
            const y = scaleY(latest.y);
            
            if (isFinite(x) && isFinite(y)) {
                ctx.fillStyle = 'rgba(57, 175, 255, 1)';
                ctx.beginPath();
                ctx.arc(x, y, 3, 0, 2 * Math.PI);
                ctx.fill();
                
                // Dibujar un círculo más grande alrededor
                ctx.strokeStyle = 'rgba(57, 175, 255, 0.5)';
                ctx.lineWidth = 1;
                ctx.beginPath();
                ctx.arc(x, y, 5, 0, 2 * Math.PI);
                ctx.stroke();
            }
        }
        
        // Etiquetas de ejes
        ctx.fillStyle = '#999';
        ctx.font = '10px sans-serif';
        ctx.textAlign = 'center';
        ctx.fillText('Canal 0 (Real)', centerX, canvas.height - 10);
        ctx.save();
        ctx.translate(10, centerY);
        ctx.rotate(-Math.PI / 2);
        ctx.fillText('Canal 1 (Real)', 0, 0);
        ctx.restore();
        
    }, [attractorHistory]);
    
    // Actualizar historial cuando llegan nuevos datos
    useEffect(() => {
        if (simData?.phase_attractor) {
            const { channel_0, channel_1 } = simData.phase_attractor;
            if (channel_0 && channel_1) {
                const newPoint: PhasePoint = {
                    x: channel_0.real,
                    y: channel_1.real,
                    timestamp: Date.now()
                };
                
                setAttractorHistory(prev => {
                    const updated = [...prev, newPoint];
                    // Mantener solo los últimos maxHistoryPoints
                    return updated.slice(-maxHistoryPoints);
                });
            }
        }
    }, [simData?.phase_attractor]);
    
    const latestPoint = attractorHistory.length > 0 ? attractorHistory[attractorHistory.length - 1] : null;
    
    return (
        <Paper p="md" withBorder style={{ backgroundColor: 'var(--mantine-color-dark-7)' }}>
            <Stack gap="sm">
                <Group justify="space-between">
                    <Group gap="xs">
                        <IconChartLine size={18} />
                        <Text size="sm" fw={600}>Atractor de Fase</Text>
                        <Tooltip 
                            label="Evolución temporal de la célula central en el espacio de fase (canales 0 y 1). Útil para detectar oscilaciones, ciclos límite y comportamientos periódicos. Referencia: Sistemas dinámicos, teoría de atractores."
                            multiline
                            width={300}
                            withArrow
                        >
                            <ActionIcon size="xs" variant="subtle" color="gray">
                                <IconHelpCircle size={14} />
                            </ActionIcon>
                        </Tooltip>
                    </Group>
                    <Badge size="sm" variant="light" color="blue">
                        {attractorHistory.length} puntos
                    </Badge>
                </Group>
                
                <canvas
                    ref={canvasRef}
                    style={{
                        width: '100%',
                        height: '400px',
                        border: '1px solid var(--mantine-color-dark-4)',
                        borderRadius: 'var(--mantine-radius-sm)'
                    }}
                />
                
                {latestPoint && (
                    <Group gap="md" justify="center">
                        <Text size="xs" c="dimmed">
                            Canal 0: <Text component="span" fw={500}>{latestPoint.x.toFixed(4)}</Text>
                        </Text>
                        <Text size="xs" c="dimmed">
                            Canal 1: <Text component="span" fw={500}>{latestPoint.y.toFixed(4)}</Text>
                        </Text>
                    </Group>
                )}
                
                {simData?.map_data && (
                    <Text size="xs" c="dimmed" style={{ fontStyle: 'italic' }}>
                        Célula central (x={Math.floor((simData.map_data[0]?.length || 256) / 2)}, 
                        y={Math.floor((simData.map_data.length || 256) / 2)}) - Grid: {simData.map_data[0]?.length || 256}x{simData.map_data.length || 256}
                    </Text>
                )}
            </Stack>
        </Paper>
    );
}

