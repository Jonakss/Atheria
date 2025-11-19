// frontend/src/components/FlowViewer.tsx
import { useRef, useEffect, useState, useMemo } from 'react';
import { 
    Paper, Stack, Text, Group, Tooltip, ActionIcon, 
    Switch, NumberInput, Badge, RingProgress, Table
} from '@mantine/core';
import { useWebSocket } from '../hooks/useWebSocket';
import { IconArrowsMaximize, IconHelpCircle, IconChartBar } from '@tabler/icons-react';

interface FlowStats {
    avgMagnitude: number;
    maxMagnitude: number;
    minMagnitude: number;
    totalVectors: number;
    significantVectors: number;
    avgDirection: number; // en radianes
    stdDevMagnitude: number;
}

export function FlowViewer() {
    const canvasRef = useRef<HTMLCanvasElement>(null);
    const { simData } = useWebSocket();
    const [showVisualization, setShowVisualization] = useState(false);
    const [maxVectors, setMaxVectors] = useState(1000);
    const [magnitudeThreshold, setMagnitudeThreshold] = useState(0.01);
    
    // Calcular estadísticas del flujo
    const flowStats = useMemo<FlowStats | null>(() => {
        const flowData = simData?.flow_data;
        if (!flowData || !flowData.dx || !flowData.dy) {
            return null;
        }
        
        const dx = flowData.dx;
        const dy = flowData.dy;
        const magnitude = flowData.magnitude || [];
        
        const gridHeight = Array.isArray(dx) ? dx.length : 0;
        const gridWidth = Array.isArray(dx[0]) ? dx[0].length : 0;
        
        if (gridHeight === 0 || gridWidth === 0) return null;
        
        const magnitudes: number[] = [];
        const directions: number[] = [];
        let significantCount = 0;
        
        for (let y = 0; y < gridHeight; y++) {
            for (let x = 0; x < gridWidth; x++) {
                if (!Array.isArray(dx[y]) || !Array.isArray(dy[y])) continue;
                
                const dx_val = dx[y][x] || 0;
                const dy_val = dy[y][x] || 0;
                
                const mag = (Array.isArray(magnitude[y]) && typeof magnitude[y][x] === 'number')
                    ? magnitude[y][x]
                    : Math.sqrt(dx_val * dx_val + dy_val * dy_val);
                
                magnitudes.push(mag);
                if (mag > 0) {
                    directions.push(Math.atan2(dy_val, dx_val));
                }
                if (mag > magnitudeThreshold) {
                    significantCount++;
                }
            }
        }
        
        if (magnitudes.length === 0) return null;
        
        const sum = magnitudes.reduce((a, b) => a + b, 0);
        const avg = sum / magnitudes.length;
        const variance = magnitudes.reduce((acc, m) => acc + Math.pow(m - avg, 2), 0) / magnitudes.length;
        const stdDev = Math.sqrt(variance);
        
        // Promedio de dirección (vector promedio)
        let avgDirX = 0, avgDirY = 0;
        directions.forEach(dir => {
            avgDirX += Math.cos(dir);
            avgDirY += Math.sin(dir);
        });
        const avgDirection = directions.length > 0 ? Math.atan2(avgDirY / directions.length, avgDirX / directions.length) : 0;
        
        // Usar reduce en lugar de spread operator para evitar stack overflow con arrays grandes
        const maxMagnitude = magnitudes.reduce((a, b) => Math.max(a, b), -Infinity);
        const minMagnitude = magnitudes.reduce((a, b) => Math.min(a, b), Infinity);
        
        return {
            avgMagnitude: avg,
            maxMagnitude,
            minMagnitude,
            totalVectors: gridHeight * gridWidth,
            significantVectors: significantCount,
            avgDirection,
            stdDevMagnitude: stdDev
        };
    }, [simData?.flow_data, magnitudeThreshold]);
    
    // Renderizar visualización (solo si está habilitada y con límites)
    useEffect(() => {
        if (!showVisualization || !flowStats) return;
        
        const canvas = canvasRef.current;
        if (!canvas) return;
        
        const flowData = simData?.flow_data;
        if (!flowData || !flowData.dx || !flowData.dy) return;
        
        const dx = flowData.dx;
        const dy = flowData.dy;
        const magnitude = flowData.magnitude || [];
        
        const gridHeight = Array.isArray(dx) ? dx.length : 0;
        const gridWidth = Array.isArray(dx[0]) ? dx[0].length : 0;
        
        if (gridHeight === 0 || gridWidth === 0) return;
        
        canvas.width = gridWidth;
        canvas.height = gridHeight;
        const ctx = canvas.getContext('2d');
        if (!ctx) return;
        
        // Fondo negro
        ctx.fillStyle = '#000000';
        ctx.fillRect(0, 0, canvas.width, canvas.height);
        
        // Calcular stepSize para limitar número de vectores
        const totalCells = gridHeight * gridWidth;
        const stepSize = Math.max(1, Math.ceil(Math.sqrt(totalCells / maxVectors)));
        
        // Encontrar el rango de magnitudes
        let maxMag = 0;
        for (let y = 0; y < gridHeight; y += stepSize) {
            for (let x = 0; x < gridWidth; x += stepSize) {
                if (Array.isArray(magnitude[y]) && typeof magnitude[y][x] === 'number') {
                    maxMag = Math.max(maxMag, magnitude[y][x]);
                }
            }
        }
        if (maxMag === 0) maxMag = 1;
        
        // Dibujar solo vectores significativos y limitados
        let vectorsDrawn = 0;
        for (let y = 0; y < gridHeight && vectorsDrawn < maxVectors; y += stepSize) {
            for (let x = 0; x < gridWidth && vectorsDrawn < maxVectors; x += stepSize) {
                if (!Array.isArray(dx[y]) || typeof dx[y][x] !== 'number') continue;
                if (!Array.isArray(dy[y]) || typeof dy[y][x] !== 'number') continue;
                
                const dx_val = dx[y][x];
                const dy_val = dy[y][x];
                const mag = (Array.isArray(magnitude[y]) && typeof magnitude[y][x] === 'number')
                    ? magnitude[y][x] / maxMag
                    : Math.sqrt(dx_val * dx_val + dy_val * dy_val);
                
                // Solo dibujar si excede el threshold
                if (mag < magnitudeThreshold) continue;
                
                const rawLength = Math.sqrt(dx_val * dx_val + dy_val * dy_val);
                if (rawLength === 0) continue;
                
                const minLength = stepSize * 0.2;
                const maxLength = stepSize * 0.6;
                const scaledLength = Math.max(minLength, Math.min(maxLength, rawLength * stepSize * 1.5));
                
                const dirX = (dx_val / rawLength) * scaledLength;
                const dirY = -(dy_val / rawLength) * scaledLength;
                
                const intensity = Math.min(mag, 1.0);
                const hue = 200 - intensity * 60;
                const lightness = 40 + intensity * 40;
                ctx.strokeStyle = `hsl(${hue}, 100%, ${lightness}%)`;
                ctx.fillStyle = `hsl(${hue}, 100%, ${lightness}%)`;
                ctx.lineWidth = 1 + intensity * 1.5;
                
                const endX = x + dirX;
                const endY = y + dirY;
                
                ctx.beginPath();
                ctx.moveTo(x, y);
                ctx.lineTo(endX, endY);
                ctx.stroke();
                
                if (scaledLength > 2) {
                    const arrowAngle = Math.atan2(dirY, dirX);
                    const arrowLength = Math.min(scaledLength * 0.3, 5);
                    
                    ctx.beginPath();
                    ctx.moveTo(endX, endY);
                    ctx.lineTo(endX - arrowLength * Math.cos(arrowAngle - Math.PI / 6), endY - arrowLength * Math.sin(arrowAngle - Math.PI / 6));
                    ctx.lineTo(endX, endY);
                    ctx.lineTo(endX - arrowLength * Math.cos(arrowAngle + Math.PI / 6), endY - arrowLength * Math.sin(arrowAngle + Math.PI / 6));
                    ctx.closePath();
                    ctx.fill();
                }
                
                vectorsDrawn++;
            }
        }
    }, [simData?.flow_data, showVisualization, maxVectors, magnitudeThreshold, flowStats]);
    
    if (!flowStats) {
        return (
            <Paper p="md" withBorder style={{ backgroundColor: 'var(--mantine-color-dark-7)' }}>
                <Stack gap="sm">
                    <Group gap="xs">
                        <IconChartBar size={18} />
                        <Text size="sm" fw={600}>Estadísticas de Flujo</Text>
                        <Tooltip 
                            label="Análisis estadístico del campo vectorial delta_psi. Muestra métricas sobre la magnitud, dirección y distribución del cambio en el estado cuántico."
                            multiline
                            width={300}
                            withArrow
                        >
                            <ActionIcon size="xs" variant="subtle" color="gray">
                                <IconHelpCircle size={14} />
                            </ActionIcon>
                        </Tooltip>
                    </Group>
                    <Text size="xs" c="dimmed">
                        No hay datos de flujo disponibles. El flujo se calcula automáticamente durante la simulación.
                    </Text>
                </Stack>
            </Paper>
        );
    }
    
    const significantPercent = (flowStats.significantVectors / flowStats.totalVectors) * 100;
    const directionDegrees = (flowStats.avgDirection * 180 / Math.PI).toFixed(1);
    
    return (
        <Paper p="md" withBorder style={{ backgroundColor: 'var(--mantine-color-dark-7)' }}>
            <Stack gap="md">
                <Group justify="space-between">
                    <Group gap="xs">
                        <IconChartBar size={18} />
                        <Text size="sm" fw={600}>Estadísticas de Flujo</Text>
                        <Tooltip 
                            label="Análisis estadístico del campo vectorial delta_psi. Muestra métricas sobre la magnitud, dirección y distribución del cambio en el estado cuántico. La visualización es opcional para mejor rendimiento."
                            multiline
                            width={300}
                            withArrow
                        >
                            <ActionIcon size="xs" variant="subtle" color="gray">
                                <IconHelpCircle size={14} />
                            </ActionIcon>
                        </Tooltip>
                    </Group>
                    <Switch
                        label="Mostrar visualización"
                        checked={showVisualization}
                        onChange={(e) => setShowVisualization(e.currentTarget.checked)}
                        size="sm"
                    />
                </Group>
                
                {/* Estadísticas principales */}
                <Group grow>
                    <Paper p="sm" withBorder style={{ backgroundColor: 'var(--mantine-color-dark-6)' }}>
                        <Stack gap="xs" align="center">
                            <Text size="xs" c="dimmed">Magnitud Promedio</Text>
                            <Text size="lg" fw={700}>{flowStats.avgMagnitude.toExponential(3)}</Text>
                            <Text size="xs" c="dimmed">
                                Min: {flowStats.minMagnitude.toExponential(3)} | 
                                Max: {flowStats.maxMagnitude.toExponential(3)}
                            </Text>
                        </Stack>
                    </Paper>
                    
                    <Paper p="sm" withBorder style={{ backgroundColor: 'var(--mantine-color-dark-6)' }}>
                        <Stack gap="xs" align="center">
                            <Text size="xs" c="dimmed">Desviación Estándar</Text>
                            <Text size="lg" fw={700}>{flowStats.stdDevMagnitude.toExponential(3)}</Text>
                            <Text size="xs" c="dimmed">Variabilidad del flujo</Text>
                        </Stack>
                    </Paper>
                    
                    <Paper p="sm" withBorder style={{ backgroundColor: 'var(--mantine-color-dark-6)' }}>
                        <Stack gap="xs" align="center">
                            <Text size="xs" c="dimmed">Dirección Promedio</Text>
                            <Text size="lg" fw={700}>{directionDegrees}°</Text>
                            <Text size="xs" c="dimmed">Orientación dominante</Text>
                        </Stack>
                    </Paper>
                </Group>
                
                {/* Vectores significativos */}
                <Group grow>
                    <RingProgress
                        size={120}
                        thickness={12}
                        sections={[{ value: significantPercent, color: 'blue' }]}
                        label={
                            <Text size="xs" ta="center" c="dimmed">
                                {significantPercent.toFixed(1)}%
                            </Text>
                        }
                    />
                    <Stack gap="xs" style={{ flex: 1 }}>
                        <Group justify="space-between">
                            <Text size="sm" fw={500}>Vectores Significativos</Text>
                            <Badge size="lg" variant="light" color="blue">
                                {flowStats.significantVectors} / {flowStats.totalVectors}
                            </Badge>
                        </Group>
                        <Text size="xs" c="dimmed">
                            Vectores con magnitud {'>'} {magnitudeThreshold.toExponential(2)}
                        </Text>
                    </Stack>
                </Group>
                
                {/* Tabla de métricas detalladas */}
                <Table size="sm" withTableBorder withColumnBorders>
                    <Table.Tbody>
                        <Table.Tr>
                            <Table.Td><Text size="xs" fw={500}>Total de Celdas</Text></Table.Td>
                            <Table.Td><Text size="xs">{flowStats.totalVectors.toLocaleString()}</Text></Table.Td>
                        </Table.Tr>
                        <Table.Tr>
                            <Table.Td><Text size="xs" fw={500}>Vectores Significativos</Text></Table.Td>
                            <Table.Td><Text size="xs">{flowStats.significantVectors.toLocaleString()} ({significantPercent.toFixed(2)}%)</Text></Table.Td>
                        </Table.Tr>
                        <Table.Tr>
                            <Table.Td><Text size="xs" fw={500}>Magnitud Promedio</Text></Table.Td>
                            <Table.Td><Text size="xs">{flowStats.avgMagnitude.toExponential(4)}</Text></Table.Td>
                        </Table.Tr>
                        <Table.Tr>
                            <Table.Td><Text size="xs" fw={500}>Magnitud Máxima</Text></Table.Td>
                            <Table.Td><Text size="xs">{flowStats.maxMagnitude.toExponential(4)}</Text></Table.Td>
                        </Table.Tr>
                        <Table.Tr>
                            <Table.Td><Text size="xs" fw={500}>Magnitud Mínima</Text></Table.Td>
                            <Table.Td><Text size="xs">{flowStats.minMagnitude.toExponential(4)}</Text></Table.Td>
                        </Table.Tr>
                        <Table.Tr>
                            <Table.Td><Text size="xs" fw={500}>Desviación Estándar</Text></Table.Td>
                            <Table.Td><Text size="xs">{flowStats.stdDevMagnitude.toExponential(4)}</Text></Table.Td>
                        </Table.Tr>
                        <Table.Tr>
                            <Table.Td><Text size="xs" fw={500}>Dirección Promedio</Text></Table.Td>
                            <Table.Td><Text size="xs">{directionDegrees}° ({flowStats.avgDirection.toFixed(4)} rad)</Text></Table.Td>
                        </Table.Tr>
                    </Table.Tbody>
                </Table>
                
                {/* Controles de visualización */}
                {showVisualization && (
                    <Stack gap="sm">
                        <Group grow>
                            <NumberInput
                                label="Máximo de vectores"
                                description="Limita el número de flechas renderizadas"
                                value={maxVectors}
                                onChange={(val) => setMaxVectors(typeof val === 'number' ? val : 1000)}
                                min={100}
                                max={10000}
                                step={100}
                                size="sm"
                            />
                            <NumberInput
                                label="Threshold de magnitud"
                                description="Solo renderiza vectores con magnitud mayor"
                                value={magnitudeThreshold}
                                onChange={(val) => setMagnitudeThreshold(typeof val === 'number' ? val : 0.01)}
                                min={0}
                                max={1}
                                step={0.001}
                                decimalScale={4}
                                size="sm"
                            />
                        </Group>
                        <canvas
                            ref={canvasRef}
                            style={{
                                width: '100%',
                                height: 'auto',
                                border: '1px solid var(--mantine-color-dark-4)',
                                borderRadius: 'var(--mantine-radius-sm)',
                                imageRendering: 'pixelated'
                            }}
                        />
                        <Text size="xs" c="dimmed" style={{ fontStyle: 'italic' }}>
                            Visualización del campo vectorial delta_psi. Mostrando solo vectores significativos para mejor rendimiento.
                        </Text>
                    </Stack>
                )}
            </Stack>
        </Paper>
    );
}
