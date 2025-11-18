// frontend/src/components/CanvasOverlays.tsx
import { useRef, useEffect } from 'react';
import { Box, Stack, Group, Text, Switch, Select, Badge } from '@mantine/core';
import { useWebSocket } from '../hooks/useWebSocket';

export interface OverlayConfig {
    showGrid: boolean;
    showCoordinates: boolean;
    showQuadtree: boolean;
    showStats: boolean;
    gridSize: number;
    quadtreeThreshold: number;
}

interface CanvasOverlaysProps {
    canvasRef: React.RefObject<HTMLCanvasElement>;
    mapData?: number[][];
    pan: { x: number; y: number };
    zoom: number;
    config: OverlayConfig;
}

export function CanvasOverlays({ canvasRef, mapData, pan, zoom, config }: CanvasOverlaysProps) {
    const overlayCanvasRef = useRef<HTMLCanvasElement>(null);
    
    useEffect(() => {
        const canvas = canvasRef.current;
        const overlayCanvas = overlayCanvasRef.current;
        if (!canvas || !overlayCanvas) return;
        
        // Obtener dimensiones y posición del canvas principal
        const rect = canvas.getBoundingClientRect();
        const parent = canvas.parentElement;
        if (!parent) return;
        
        // Sincronizar tamaño del canvas overlay con el principal
        overlayCanvas.width = canvas.width;
        overlayCanvas.height = canvas.height;
        
        // Sincronizar estilos CSS (posición y tamaño, pero NO transform)
        overlayCanvas.style.width = `${rect.width}px`;
        overlayCanvas.style.height = `${rect.height}px`;
        
        // Posicionar overlay exactamente sobre el canvas principal (sin transform CSS)
        const parentRect = parent.getBoundingClientRect();
        overlayCanvas.style.top = `${rect.top - parentRect.top}px`;
        overlayCanvas.style.left = `${rect.left - parentRect.left}px`;
        overlayCanvas.style.transform = 'none'; // NO usar transform CSS, aplicar directamente en canvas
        
        const ctx = overlayCanvas.getContext('2d');
        if (!ctx) return;
        
        // Limpiar canvas
        ctx.clearRect(0, 0, overlayCanvas.width, overlayCanvas.height);
        
        if (!mapData || mapData.length === 0) return;
        
        const gridHeight = mapData.length;
        const gridWidth = mapData[0]?.length || 0;
        
        if (gridWidth === 0 || gridHeight === 0) return;
        
        // Calcular escala base (igual que PanZoomCanvas)
        const scaleX = overlayCanvas.width / gridWidth;
        const scaleY = overlayCanvas.height / gridHeight;
        const baseScale = Math.min(scaleX, scaleY);
        
        // Aplicar transformaciones directamente en el canvas (no CSS)
        ctx.save();
        
        // 1. Mover al centro del canvas
        const centerX = overlayCanvas.width / 2;
        const centerY = overlayCanvas.height / 2;
        ctx.translate(centerX, centerY);
        
        // 2. Aplicar zoom (igual que el canvas principal)
        ctx.scale(zoom, zoom);
        
        // 3. Aplicar pan (igual que el canvas principal)
        ctx.translate(pan.x, pan.y);
        
        // 4. Aplicar escala base y centrar la grilla
        const scaledGridWidth = gridWidth * baseScale;
        const scaledGridHeight = gridHeight * baseScale;
        const offsetX = -scaledGridWidth / 2;
        const offsetY = -scaledGridHeight / 2;
        
        ctx.translate(offsetX, offsetY);
        ctx.scale(baseScale, baseScale);
        
        // Grid overlay
        if (config.showGrid) {
            ctx.strokeStyle = 'rgba(100, 150, 255, 0.2)';
            // Ajustar lineWidth según el zoom para que sea consistente visualmente
            ctx.lineWidth = 0.5 / (baseScale * zoom);
            
            const step = config.gridSize;
            // Dibujar líneas verticales
            for (let x = 0; x <= gridWidth; x += step) {
                ctx.beginPath();
                ctx.moveTo(x, 0);
                ctx.lineTo(x, gridHeight);
                ctx.stroke();
            }
            // Dibujar líneas horizontales
            for (let y = 0; y <= gridHeight; y += step) {
                ctx.beginPath();
                ctx.moveTo(0, y);
                ctx.lineTo(gridWidth, y);
                ctx.stroke();
            }
        }
        
        // Quadtree overlay (visualización de estructura)
        if (config.showQuadtree && mapData) {
            ctx.strokeStyle = 'rgba(0, 255, 255, 0.5)';
            // Ajustar lineWidth según el zoom para que sea consistente visualmente
            ctx.lineWidth = 1.5 / (baseScale * zoom);
            
            // Visualizar estructura quadtree (regiones con datos significativos)
            const threshold = config.quadtreeThreshold;
            const drawQuadtree = (minX: number, minY: number, maxX: number, maxY: number, depth: number) => {
                if (depth > 6) return; // Limitar profundidad
                
                const width = maxX - minX;
                const height = maxY - minY;
                
                // Calcular si hay datos significativos en esta región
                let hasData = false;
                let maxValue = 0;
                
                for (let y = Math.floor(minY); y < Math.ceil(maxY) && y < gridHeight; y++) {
                    for (let x = Math.floor(minX); x < Math.ceil(maxX) && x < gridWidth; x++) {
                        const value = Math.abs(mapData[y]?.[x] || 0);
                        if (value > threshold) {
                            hasData = true;
                            maxValue = Math.max(maxValue, value);
                        }
                    }
                }
                
                if (hasData) {
                    // Dibujar borde de región
                    ctx.strokeRect(minX, minY, width, height);
                    
                    // Si la región es suficientemente grande, subdividir
                    if (width > 4 && height > 4) {
                        const midX = (minX + maxX) / 2;
                        const midY = (minY + maxY) / 2;
                        
                        drawQuadtree(minX, minY, midX, midY, depth + 1);
                        drawQuadtree(midX, minY, maxX, midY, depth + 1);
                        drawQuadtree(minX, midY, midX, maxY, depth + 1);
                        drawQuadtree(midX, midY, maxX, maxY, depth + 1);
                    }
                }
            };
            
            drawQuadtree(0, 0, gridWidth, gridHeight, 0);
        }
        
        // Coordenadas overlay
        if (config.showCoordinates) {
            ctx.fillStyle = 'rgba(255, 255, 255, 0.95)';
            // Ajustar fontSize según el zoom para que sea legible
            const fontSize = Math.max(8, 12 / (baseScale * zoom));
            ctx.font = `${fontSize}px monospace`;
            ctx.textBaseline = 'top';
            
            // Mostrar coordenadas en las esquinas
            const corners = [
                { x: 0, y: 0, label: `(0, 0)` },
                { x: gridWidth, y: 0, label: `(${gridWidth}, 0)` },
                { x: 0, y: gridHeight, label: `(0, ${gridHeight})` },
                { x: gridWidth, y: gridHeight, label: `(${gridWidth}, ${gridHeight})` }
            ];
            
            corners.forEach(corner => {
                const padding = Math.max(1, 2 / (baseScale * zoom));
                ctx.fillText(corner.label, corner.x + padding, corner.y + padding);
            });
        }
        
        ctx.restore();
        
        // Estadísticas overlay (fuera de la transformación)
        if (config.showStats && mapData) {
            ctx.save();
            ctx.resetTransform();
            
            // Calcular estadísticas
            let sum = 0;
            let count = 0;
            let min = Infinity;
            let max = -Infinity;
            
            mapData.forEach(row => {
                row.forEach(value => {
                    if (typeof value === 'number' && isFinite(value)) {
                        sum += value;
                        count++;
                        min = Math.min(min, value);
                        max = Math.max(max, value);
                    }
                });
            });
            
            const avg = count > 0 ? sum / count : 0;
            
            // Dibujar panel de estadísticas
            ctx.fillStyle = 'rgba(0, 0, 0, 0.7)';
            ctx.fillRect(10, 10, 200, 100);
            
            ctx.fillStyle = 'rgba(255, 255, 255, 0.9)';
            ctx.font = '12px monospace';
            ctx.fillText(`Min: ${min.toFixed(4)}`, 15, 25);
            ctx.fillText(`Max: ${max.toFixed(4)}`, 15, 40);
            ctx.fillText(`Avg: ${avg.toFixed(4)}`, 15, 55);
            ctx.fillText(`Grid: ${gridWidth}x${gridHeight}`, 15, 70);
            ctx.fillText(`Zoom: ${zoom.toFixed(2)}x`, 15, 85);
            
            ctx.restore();
        }
    }, [mapData, pan, zoom, config, canvasRef]);
    
    // Sincronizar posición y tamaño del overlay cuando cambia el canvas principal
    useEffect(() => {
        const canvas = canvasRef.current;
        const overlayCanvas = overlayCanvasRef.current;
        if (!canvas || !overlayCanvas) return;
        
        const updatePosition = () => {
            const rect = canvas.getBoundingClientRect();
            const parent = canvas.parentElement;
            if (!parent) return;
            
            const parentRect = parent.getBoundingClientRect();
            
            // Sincronizar posición absoluta
            overlayCanvas.style.top = `${rect.top - parentRect.top}px`;
            overlayCanvas.style.left = `${rect.left - parentRect.left}px`;
            
            // Sincronizar tamaño
            overlayCanvas.style.width = `${rect.width}px`;
            overlayCanvas.style.height = `${rect.height}px`;
        };
        
        // Actualizar inmediatamente
        updatePosition();
        
        // Observar cambios de tamaño/posición del canvas principal
        const resizeObserver = new ResizeObserver(updatePosition);
        resizeObserver.observe(canvas);
        
        // También observar el parent para cambios de layout
        if (canvas.parentElement) {
            resizeObserver.observe(canvas.parentElement);
        }
        
        return () => {
            resizeObserver.disconnect();
        };
    }, [canvasRef]);
    
    return (
        <canvas
            ref={overlayCanvasRef}
            style={{
                position: 'absolute',
                pointerEvents: 'none',
                zIndex: 10,
                imageRendering: 'crisp-edges', // Cambiar de pixelated a crisp-edges para overlays más suaves
                mixBlendMode: 'screen' // Mejorar visibilidad sobre el canvas principal
            }}
        />
    );
}

interface OverlayControlsProps {
    config: OverlayConfig;
    onConfigChange: (config: OverlayConfig) => void;
}

export function OverlayControls({ config, onConfigChange }: OverlayControlsProps) {
    return (
        <Stack gap="xs" p="sm" style={{ backgroundColor: 'var(--mantine-color-dark-8)', borderRadius: '4px' }}>
            <Text size="xs" fw={600}>Overlays</Text>
            
            <Switch
                label="Grid"
                checked={config.showGrid}
                onChange={(e) => onConfigChange({ ...config, showGrid: e.currentTarget.checked })}
                size="xs"
            />
            
            <Switch
                label="Coordenadas"
                checked={config.showCoordinates}
                onChange={(e) => onConfigChange({ ...config, showCoordinates: e.currentTarget.checked })}
                size="xs"
            />
            
            <Switch
                label="Quadtree"
                checked={config.showQuadtree}
                onChange={(e) => onConfigChange({ ...config, showQuadtree: e.currentTarget.checked })}
                size="xs"
            />
            
            <Switch
                label="Estadísticas"
                checked={config.showStats}
                onChange={(e) => onConfigChange({ ...config, showStats: e.currentTarget.checked })}
                size="xs"
            />
            
            {config.showGrid && (
                <Box>
                    <Text size="xs" c="dimmed" mb={4}>Tamaño Grid</Text>
                    <Select
                        value={config.gridSize.toString()}
                        onChange={(val) => onConfigChange({ ...config, gridSize: parseInt(val || '10') })}
                        data={[
                            { value: '5', label: '5' },
                            { value: '10', label: '10' },
                            { value: '20', label: '20' },
                            { value: '50', label: '50' }
                        ]}
                        size="xs"
                    />
                </Box>
            )}
            
            {config.showQuadtree && (
                <Box>
                    <Text size="xs" c="dimmed" mb={4}>Threshold Quadtree</Text>
                    <Select
                        value={config.quadtreeThreshold.toString()}
                        onChange={(val) => onConfigChange({ ...config, quadtreeThreshold: parseFloat(val || '0.01') })}
                        data={[
                            { value: '0.001', label: '0.001' },
                            { value: '0.01', label: '0.01' },
                            { value: '0.1', label: '0.1' },
                            { value: '0.5', label: '0.5' }
                        ]}
                        size="xs"
                    />
                </Box>
            )}
        </Stack>
    );
}

