// frontend/src/components/CanvasOverlays.tsx
import { useEffect, useRef } from 'react';
import { Box } from '../../modules/Dashboard/components/Box';
import { Select } from '../../modules/Dashboard/components/Select';
import { Stack } from '../../modules/Dashboard/components/Stack';
import { Switch } from '../../modules/Dashboard/components/Switch';
import { Text } from '../../modules/Dashboard/components/Text';

// Helper para obtener color de un valor normalizado [0,1]
function getColor(value: number): string {
    if (typeof value !== 'number' || isNaN(value) || !isFinite(value)) {
        return 'rgb(68, 1, 84)'; // Color por defecto
    }
    
    const colors = [
        [68, 1, 84], [72, 40, 120], [62, 74, 137], [49, 104, 142],
        [38, 130, 142], [31, 158, 137], [53, 183, 121], [109, 205, 89],
        [180, 222, 44], [253, 231, 37], [255, 200, 0], [255, 150, 0],
        [255, 100, 0], [255, 50, 0], [255, 0, 0]
    ];
    
    const normalizedValue = Math.max(0, Math.min(1, value));
    const i = Math.min(Math.max(Math.floor(normalizedValue * (colors.length - 1)), 0), colors.length - 1);
    const c = colors[i];
    
    if (!c || !Array.isArray(c) || c.length < 3) {
        return 'rgb(68, 1, 84)';
    }
    
    return `rgb(${c[0]}, ${c[1]}, ${c[2]})`;
}

export interface OverlayConfig {
    showGrid: boolean;
    showCoordinates: boolean;
    showQuadtree: boolean;
    showStats: boolean;
    showToroidalBorders: boolean; // Mostrar bordes toroidales (conectividad de bordes)
    gridSize: number;
    quadtreeThreshold: number;
}

interface CanvasOverlaysProps {
    canvasRef: React.RefObject<HTMLCanvasElement>;
    mapData?: number[][];
    pan: { x: number; y: number };
    zoom: number;
    config: OverlayConfig;
    roiInfo?: {
        enabled: boolean;
        x: number;
        y: number;
        width: number;
        height: number;
    } | null;
}

export function CanvasOverlays({ canvasRef, mapData, pan, zoom, config, roiInfo }: CanvasOverlaysProps) {
    const overlayCanvasRef = useRef<HTMLCanvasElement>(null);
    
    useEffect(() => {
        const canvas = canvasRef.current;
        const overlayCanvas = overlayCanvasRef.current;
        if (!canvas || !overlayCanvas) return;
        
        // Función para sincronizar posición y tamaño
        const syncPosition = () => {
            const rect = canvas.getBoundingClientRect();
            const parent = canvas.parentElement;
            if (!parent) return;
            
            // Sincronizar tamaño del canvas overlay con el principal
            overlayCanvas.width = canvas.width;
            overlayCanvas.height = canvas.height;
            
            // Sincronizar estilos CSS (posición y tamaño)
            overlayCanvas.style.width = `${rect.width}px`;
            overlayCanvas.style.height = `${rect.height}px`;
            
            // Posicionar overlay exactamente sobre el canvas principal
            const parentRect = parent.getBoundingClientRect();
            overlayCanvas.style.position = 'absolute';
            overlayCanvas.style.top = `${rect.top - parentRect.top}px`;
            overlayCanvas.style.left = `${rect.left - parentRect.left}px`;
            overlayCanvas.style.pointerEvents = 'none'; // Permitir que los eventos pasen al canvas principal
            overlayCanvas.style.transform = 'none'; // NO usar transform CSS, aplicar directamente en canvas
        };
        
        // Sincronizar inicialmente
        syncPosition();
        
        // Usar ResizeObserver para sincronizar cuando cambia el tamaño del canvas
        const resizeObserver = new ResizeObserver(() => {
            syncPosition();
        });
        resizeObserver.observe(canvas);
        
        // También observar cambios en el parent
        const parent = canvas.parentElement;
        if (parent) {
            resizeObserver.observe(parent);
        }
        
        return () => {
            resizeObserver.disconnect();
        };
    }, [canvasRef]);
    
    useEffect(() => {
        const canvas = canvasRef.current;
        const overlayCanvas = overlayCanvasRef.current;
        if (!canvas || !overlayCanvas) return;
        
        const ctx = overlayCanvas.getContext('2d');
        if (!ctx) return;
        
        // Limpiar canvas
        ctx.clearRect(0, 0, overlayCanvas.width, overlayCanvas.height);
        
        if (!mapData || mapData.length === 0) return;
        
        const gridHeight = mapData.length;
        const gridWidth = mapData[0]?.length || 0;
        
        if (gridWidth === 0 || gridHeight === 0) return;
        
        // Actualizar dependencias del efecto para incluir roiInfo
        
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
            
            // Ajustar el inicio para que sea el primer múltiplo de step dentro de la vista
            const firstX = roiInfo?.enabled ? (step - (roiInfo.x % step)) % step : 0;
            const firstY = roiInfo?.enabled ? (step - (roiInfo.y % step)) % step : 0;

            // Dibujar líneas verticales
            for (let x = firstX; x <= gridWidth; x += step) {
                ctx.beginPath();
                ctx.moveTo(x, 0);
                ctx.lineTo(x, gridHeight);
                ctx.stroke();
            }
            // Dibujar líneas horizontales
            for (let y = firstY; y <= gridHeight; y += step) {
                ctx.beginPath();
                ctx.moveTo(0, y);
                ctx.lineTo(gridWidth, y);
                ctx.stroke();
            }
        }
        
        // Quadtree overlay (visualización de estructura) - OPTIMIZADO Y MEJORADO
        if (config.showQuadtree && mapData) {
            // OPTIMIZACIÓN: Deshabilitar quadtree automáticamente para grids muy grandes
            const totalCells = gridWidth * gridHeight;
            const maxCellsForQuadtree = 256 * 256; // Aumentado a 256x256 para permitir más casos
            
            if (totalCells > maxCellsForQuadtree) {
                // Para grids muy grandes, mostrar solo un mensaje o deshabilitar
                ctx.save();
                ctx.resetTransform();
                ctx.fillStyle = 'rgba(255, 200, 0, 0.8)';
                ctx.font = '12px monospace';
                ctx.fillText('Quadtree deshabilitado para grids muy grandes (>256x256)', 10, 30);
                ctx.restore();
            } else {
                ctx.strokeStyle = 'rgba(0, 255, 255, 0.6)';
                // Ajustar lineWidth según el zoom para que sea consistente visualmente
                const baseLineWidth = 1.5;
                ctx.lineWidth = Math.max(0.5, baseLineWidth / (baseScale * zoom));
                
                // Visualizar estructura quadtree (regiones con datos significativos)
                const threshold = Math.max(0.0001, config.quadtreeThreshold); // Asegurar threshold mínimo
                
                // CALCULAR NIVEL DE DETALLE (LOD) SEGÚN EL ZOOM
                // Zoom alto = más detalle (más profundidad, regiones más pequeñas)
                // Zoom bajo = menos detalle (menos profundidad, regiones más grandes)
                const baseMaxDepth = 8; // Profundidad base por defecto
                const baseMinRegionSize = Math.max(2, Math.min(8, Math.floor(Math.sqrt(totalCells) / 32)));
                
                // Ajustar profundidad máxima según zoom
                // Zoom 1.0 = profundidad base reducida (menos detalle)
                // Zoom > 2.0 = profundidad máxima (máximo detalle)
                // Interpolación logarítmica para suavizar la transición
                const zoomFactor = Math.max(0.5, Math.min(2.0, zoom)); // Limitar zoom entre 0.5x y 2.0x
                const depthMultiplier = Math.log(zoomFactor + 0.5) / Math.log(2.5); // Escala logarítmica 0-1
                
                let maxDepth = Math.floor(baseMaxDepth * (0.5 + depthMultiplier * 0.5)); // Entre 50% y 100% de profundidad base
                if (totalCells > 128 * 128) {
                    maxDepth = Math.max(4, Math.floor(maxDepth * 0.75)); // Reducir más para grids grandes
                } else if (totalCells > 64 * 64) {
                    maxDepth = Math.max(5, Math.floor(maxDepth * 0.875));
                }
                maxDepth = Math.max(4, Math.min(10, maxDepth)); // Limitar entre 4 y 10
                
                // Ajustar tamaño mínimo de región según zoom
                // Zoom alto = regiones más pequeñas (más detalle)
                // Zoom bajo = regiones más grandes (menos detalle)
                const regionSizeMultiplier = 1 / Math.max(0.5, zoomFactor); // Inversamente proporcional al zoom
                const minRegionSize = Math.max(1, Math.floor(baseMinRegionSize * regionSizeMultiplier));
                
                // Si hay ROI activa, solo procesar la región visible
                const roiBounds = roiInfo?.enabled ? {
                    minX: roiInfo.x,
                    minY: roiInfo.y,
                    maxX: roiInfo.x + roiInfo.width,
                    maxY: roiInfo.y + roiInfo.height
                } : null;
                
                const drawQuadtree = (minX: number, minY: number, maxX: number, maxY: number, depth: number) => {
                    if (depth > maxDepth) return; // Limitar profundidad
                    
                    // Si hay ROI activa, solo procesar regiones que intersecten con la ROI
                    if (roiBounds) {
                        // Verificar si la región intersecta con la ROI
                        if (maxX < roiBounds.minX || minX > roiBounds.maxX || 
                            maxY < roiBounds.minY || minY > roiBounds.maxY) {
                            return; // Región fuera de la ROI, ignorar
                        }
                        // Clamp a la ROI
                        minX = Math.max(minX, roiBounds.minX);
                        minY = Math.max(minY, roiBounds.minY);
                        maxX = Math.min(maxX, roiBounds.maxX);
                        maxY = Math.min(maxY, roiBounds.maxY);
                    }
                    
                    const width = maxX - minX;
                    const height = maxY - minY;
                    
                    // Si la región es muy pequeña según el LOD actual, no subdividir más
                    // El tamaño mínimo se ajusta según el zoom (más zoom = regiones más pequeñas visibles)
                    const currentMinSize = minRegionSize / zoom; // Ajustar según zoom actual
                    if (width < currentMinSize || height < currentMinSize) {
                        // Verificar si tiene datos significativos en esta región pequeña
                        let hasData = false;
                        for (let y = Math.floor(minY); y < Math.ceil(maxY) && y < gridHeight; y++) {
                            for (let x = Math.floor(minX); x < Math.ceil(maxX) && x < gridWidth; x++) {
                                const value = Math.abs(mapData[y]?.[x] || 0);
                                if (value > threshold) {
                                    hasData = true;
                                    break;
                                }
                            }
                            if (hasData) break;
                        }
                        
                        if (hasData) {
                            // Dibujar borde de región pequeña
                            ctx.strokeRect(minX, minY, width, height);
                        }
                        return;
                    }
                    
                    // Para regiones más grandes, usar muestreo inteligente
                    let hasData = false;
                    let maxValue = 0;
                    let sampleCount = 0;
                    const maxSamples = Math.min(64, Math.floor(width * height / 4)); // Muestrear hasta 64 puntos
                    
                    // Calcular paso de muestreo para cubrir la región eficientemente
                    const sampleStepX = Math.max(1, Math.floor(width / Math.sqrt(maxSamples)));
                    const sampleStepY = Math.max(1, Math.floor(height / Math.sqrt(maxSamples)));
                    
                    for (let y = Math.floor(minY); y < Math.ceil(maxY) && y < gridHeight && sampleCount < maxSamples; y += sampleStepY) {
                        for (let x = Math.floor(minX); x < Math.ceil(maxX) && x < gridWidth && sampleCount < maxSamples; x += sampleStepX) {
                            const value = Math.abs(mapData[y]?.[x] || 0);
                            if (value > threshold) {
                                hasData = true;
                                maxValue = Math.max(maxValue, value);
                            }
                            sampleCount++;
                        }
                    }
                    
                    if (hasData) {
                        // Dibujar borde de región
                        ctx.strokeRect(minX, minY, width, height);
                        
                        // Subdividir si la región es suficientemente grande y no hemos alcanzado la profundidad máxima
                        // Subdividir si la región es suficientemente grande y no hemos alcanzado la profundidad máxima
                        // El tamaño mínimo para subdividir se ajusta según el zoom (más zoom = subdivisiones más pequeñas)
                        const subdivisionThreshold = currentMinSize * 2; // Usar el tamaño mínimo ajustado al zoom
                        if (width > subdivisionThreshold && height > subdivisionThreshold && depth < maxDepth) {
                            const midX = (minX + maxX) / 2;
                            const midY = (minY + maxY) / 2;
                            
                            // Subdividir recursivamente en 4 cuadrantes
                            drawQuadtree(minX, minY, midX, midY, depth + 1);
                            drawQuadtree(midX, minY, maxX, midY, depth + 1);
                            drawQuadtree(minX, midY, midX, maxY, depth + 1);
                            drawQuadtree(midX, midY, maxX, maxY, depth + 1);
                        }
                    }
                };
                
                // Iniciar quadtree desde el grid completo o desde la ROI si está activa
                if (roiBounds) {
                    drawQuadtree(roiBounds.minX, roiBounds.minY, roiBounds.maxX, roiBounds.maxY, 0);
                } else {
                drawQuadtree(0, 0, gridWidth, gridHeight, 0);
                }
            }
        }
        
        // Bordes Toroidales overlay - Mostrar dónde se conectan los bordes
        if (config.showToroidalBorders) {
            ctx.strokeStyle = 'rgba(255, 100, 0, 0.8)'; // Color naranja visible
            ctx.lineWidth = Math.max(1, 2 / (baseScale * zoom)); // Línea más gruesa, ajustada al zoom
            
            // Dibujar bordes del grid con estilo especial para indicar conectividad toroidal
            // Borde superior e inferior (conectan)
            ctx.setLineDash([5, 5]); // Línea punteada para indicar conectividad
            ctx.beginPath();
            ctx.moveTo(0, 0);
            ctx.lineTo(gridWidth, 0);
            ctx.moveTo(0, gridHeight);
            ctx.lineTo(gridWidth, gridHeight);
            ctx.stroke();
            
            // Borde izquierdo y derecho (conectan)
            ctx.beginPath();
            ctx.moveTo(0, 0);
            ctx.lineTo(0, gridHeight);
            ctx.moveTo(gridWidth, 0);
            ctx.lineTo(gridWidth, gridHeight);
            ctx.stroke();
            
            // Restaurar línea sólida
            ctx.setLineDash([]);
            
            // Mostrar indicadores visuales en las esquinas para indicar que se conectan
            // Esquina superior izquierda conecta con inferior derecha
            // Esquina superior derecha conecta con inferior izquierda
            const cornerSize = Math.max(2, 4 / (baseScale * zoom));
            ctx.fillStyle = 'rgba(255, 100, 0, 0.6)';
            
            // Esquinas con indicador de conectividad
            // Superior izquierda
            ctx.fillRect(0, 0, cornerSize, cornerSize);
            // Superior derecha
            ctx.fillRect(gridWidth - cornerSize, 0, cornerSize, cornerSize);
            // Inferior izquierda
            ctx.fillRect(0, gridHeight - cornerSize, cornerSize, cornerSize);
            // Inferior derecha
            ctx.fillRect(gridWidth - cornerSize, gridHeight - cornerSize, cornerSize, cornerSize);
            
                // Opcional: Mostrar pequeña copia "wrap" de los bordes para visualizar conectividad
            // Solo cuando el zoom es suficiente para verlo claramente
            if (zoom > 1.5 && baseScale * zoom > 0.5 && mapData) {
                // Calcular rango de valores para normalización
                let minVal = Infinity;
                let maxVal = -Infinity;
                for (let y = 0; y < gridHeight; y++) {
                    for (let x = 0; x < gridWidth; x++) {
                        const val = mapData[y]?.[x];
                        if (typeof val === 'number' && isFinite(val)) {
                            minVal = Math.min(minVal, val);
                            maxVal = Math.max(maxVal, val);
                        }
                    }
                }
                const range = maxVal - minVal || 1;
                
                const wrapWidth = Math.min(5, gridWidth * 0.05); // 5% del ancho o máximo 5 celdas
                const wrapHeight = Math.min(5, gridHeight * 0.05); // 5% del alto o máximo 5 celdas
                const wrapAlpha = 0.3;
                
                // Helper para normalizar valor y obtener color
                const getNormalizedColor = (val: number): string => {
                    const normalized = (val - minVal) / range;
                    return getColor(normalized);
                };
                
                // Copia del borde superior en la parte inferior (para mostrar que se conectan)
                if (mapData[0]) {
                    ctx.globalAlpha = wrapAlpha;
                    for (let x = 0; x < Math.min(wrapWidth, gridWidth); x++) {
                        const sourceY = 0;
                        const targetY = gridHeight - wrapHeight;
                        if (mapData[sourceY] && typeof mapData[sourceY][x] === 'number') {
                            const value = mapData[sourceY][x];
                            ctx.fillStyle = getNormalizedColor(value);
                            ctx.fillRect(x, targetY, 1, wrapHeight);
                        }
                    }
                    ctx.globalAlpha = 1.0;
                }
                
                // Copia del borde inferior en la parte superior
                if (mapData[gridHeight - 1]) {
                    ctx.globalAlpha = wrapAlpha;
                    for (let x = 0; x < Math.min(wrapWidth, gridWidth); x++) {
                        const sourceY = gridHeight - 1;
                        const targetY = 0;
                        if (mapData[sourceY] && typeof mapData[sourceY][x] === 'number') {
                            const value = mapData[sourceY][x];
                            ctx.fillStyle = getNormalizedColor(value);
                            ctx.fillRect(x, targetY, 1, wrapHeight);
                        }
                    }
                    ctx.globalAlpha = 1.0;
                }
                
                // Copia del borde izquierdo en la parte derecha
                ctx.globalAlpha = wrapAlpha;
                for (let y = 0; y < Math.min(wrapHeight, gridHeight); y++) {
                    const sourceX = 0;
                    const targetX = gridWidth - wrapWidth;
                    if (mapData[y] && typeof mapData[y][sourceX] === 'number') {
                        const value = mapData[y][sourceX];
                        ctx.fillStyle = getNormalizedColor(value);
                        ctx.fillRect(targetX, y, wrapWidth, 1);
                    }
                }
                ctx.globalAlpha = 1.0;
                
                // Copia del borde derecho en la parte izquierda
                ctx.globalAlpha = wrapAlpha;
                for (let y = 0; y < Math.min(wrapHeight, gridHeight); y++) {
                    const sourceX = gridWidth - 1;
                    const targetX = 0;
                    if (mapData[y] && typeof mapData[y][sourceX] === 'number') {
                        const value = mapData[y][sourceX];
                        ctx.fillStyle = getNormalizedColor(value);
                        ctx.fillRect(targetX, y, wrapWidth, 1);
                    }
                }
                ctx.globalAlpha = 1.0;
            }
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
        
        // ROI Box overlay - Mostrar el rectángulo de Region of Interest
        if (roiInfo && roiInfo.enabled) {
            ctx.strokeStyle = 'rgba(0, 255, 255, 0.8)'; // Cyan brillante
            ctx.lineWidth = Math.max(1.5, 2 / (baseScale * zoom)); // Línea más gruesa, ajustada al zoom
            ctx.setLineDash([8, 4]); // Línea punteada para distinguirlo de otros overlays
            
            // Dibujar rectángulo del ROI
            ctx.strokeRect(
                roiInfo.x,
                roiInfo.y,
                roiInfo.width,
                roiInfo.height
            );
            
            // Restaurar línea sólida
            ctx.setLineDash([]);
            
            // Agregar etiqueta "ROI" en la esquina superior izquierda del ROI
            const fontSize = Math.max(10, 14 / (baseScale * zoom));
            ctx.font = `bold ${fontSize}px monospace`;
            ctx.fillStyle = 'rgba(0, 255, 255, 0.9)';
            ctx.textBaseline = 'top';
            const padding = Math.max(2, 3 / (baseScale * zoom));
            ctx.fillText('ROI', roiInfo.x + padding, roiInfo.y + padding);
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
    }, [mapData, pan, zoom, config, canvasRef, roiInfo]);
    
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
        <Stack gap={2} className="p-3 bg-[#080808] rounded">
            <Text size="xs" weight="bold">Overlays</Text>
            
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
            
            <Switch
                label="Bordes Toroidales"
                description="Mostrar bordes toroidales (indica dónde se conectan los bordes del grid)"
                checked={config.showToroidalBorders}
                onChange={(e) => onConfigChange({ ...config, showToroidalBorders: e.currentTarget.checked })}
                size="xs"
            />
            
            {config.showGrid && (
                <Box className="mt-2">
                    <Text size="xs" color="dimmed" className="block mb-1">Tamaño Grid</Text>
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
                <Box className="mt-2">
                    <Text size="xs" color="dimmed" className="block mb-1">Threshold Quadtree</Text>
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



