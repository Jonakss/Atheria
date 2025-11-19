// frontend/src/components/PanZoomCanvas.tsx
import { useRef, useEffect, useState } from 'react';
import { useWebSocket } from '../../hooks/useWebSocket';
import { usePanZoom } from '../../hooks/usePanZoom';
import { CanvasOverlays, OverlayControls, OverlayConfig } from './CanvasOverlays';
import { Button, Group, ActionIcon, Tooltip, Switch, Paper, Stack, Text } from '@mantine/core';
import { IconZoomReset, IconSettings } from '@tabler/icons-react';
import classes from './PanZoomCanvas.module.css';

function getColor(value: number) {
    // Validar que value sea un número válido
    if (typeof value !== 'number' || isNaN(value) || !isFinite(value)) {
        return 'rgb(68, 1, 84)'; // Color por defecto (primero de la paleta)
    }
    
    const colors = [
        [68, 1, 84], [72, 40, 120], [62, 74, 137], [49, 104, 142],
        [38, 130, 142], [31, 158, 137], [53, 183, 121], [109, 205, 89],
        [180, 222, 44], [253, 231, 37], [255, 200, 0], [255, 150, 0],
        [255, 100, 0], [255, 50, 0], [255, 0, 0]
    ];
    
    // Asegurar que value esté en el rango [0, 1]
    const normalizedValue = Math.max(0, Math.min(1, value));
    const i = Math.min(Math.max(Math.floor(normalizedValue * (colors.length - 1)), 0), colors.length - 1);
    const c = colors[i];
    
    // Validar que el color existe
    if (!c || !Array.isArray(c) || c.length < 3) {
        return 'rgb(68, 1, 84)'; // Color por defecto
    }
    
    return `rgb(${c[0]}, ${c[1]}, ${c[2]})`;
}

interface PanZoomCanvasProps {
    historyFrame?: {
        step: number;
        timestamp: string;
        map_data: number[][];
    } | null;
}

export function PanZoomCanvas({ historyFrame }: PanZoomCanvasProps = {}) {
    const canvasRef = useRef<HTMLCanvasElement>(null);
    const { simData, selectedViz, inferenceStatus, sendCommand } = useWebSocket();
    
    // Usar historyFrame si está disponible, sino usar simData actual
    const dataToRender = historyFrame ? historyFrame : simData;
    
    // Obtener dimensiones de la grilla
    const mapData = dataToRender?.map_data;
    const gridWidth = mapData?.[0]?.length || 0;
    const gridHeight = mapData?.length || 0;
    
    const { pan, zoom, handleMouseDown, handleMouseMove, handleMouseUp, handleWheel, resetView } = usePanZoom(canvasRef, gridWidth, gridHeight);
    
    // Estado de overlays
    const [overlayConfig, setOverlayConfig] = useState<OverlayConfig>({
        showGrid: false,
        showCoordinates: false,
        showQuadtree: false,
        showStats: false,
        gridSize: 10,
        quadtreeThreshold: 0.01
    });
    const [showOverlayControls, setShowOverlayControls] = useState(false);
    const [autoROIEnabled, setAutoROIEnabled] = useState(false);
    const lastROIUpdate = useRef<number>(0);
    const ROIUpdateThrottle = 500; // Actualizar ROI máximo cada 500ms

    // Sincronizar ROI automáticamente con la vista visible del canvas
    useEffect(() => {
        if (!autoROIEnabled || !canvasRef.current || !mapData || gridWidth === 0 || gridHeight === 0) return;
        
        const now = Date.now();
        if (now - lastROIUpdate.current < ROIUpdateThrottle) return;
        
        const canvas = canvasRef.current;
        const canvasWidth = canvas.width;
        const canvasHeight = canvas.height;
        
        // Calcular escala base
        const scaleX = canvasWidth / gridWidth;
        const scaleY = canvasHeight / gridHeight;
        const baseScale = Math.min(scaleX, scaleY);
        
        // Calcular región visible en coordenadas del grid
        // El canvas está centrado y escalado, luego se aplica pan y zoom
        const centerX = canvasWidth / 2;
        const centerY = canvasHeight / 2;
        
        // Calcular el tamaño visible del grid en píxeles del canvas
        const visibleGridWidth = canvasWidth / (baseScale * zoom);
        const visibleGridHeight = canvasHeight / (baseScale * zoom);
        
        // Convertir pan (en píxeles del canvas) a coordenadas del grid
        // El pan está en coordenadas del canvas transformado
        const gridCenterX = gridWidth / 2;
        const gridCenterY = gridHeight / 2;
        
        // Calcular offset del grid desde el centro
        const offsetX = -pan.x / (baseScale * zoom);
        const offsetY = -pan.y / (baseScale * zoom);
        
        // Calcular región visible en coordenadas del grid
        const visibleX = Math.max(0, Math.floor(gridCenterX + offsetX - visibleGridWidth / 2));
        const visibleY = Math.max(0, Math.floor(gridCenterY + offsetY - visibleGridHeight / 2));
        const visibleWidth = Math.min(gridWidth - visibleX, Math.ceil(visibleGridWidth));
        const visibleHeight = Math.min(gridHeight - visibleY, Math.ceil(visibleGridHeight));
        
        // Solo actualizar ROI si la región visible es significativamente diferente del grid completo
        // y si el zoom es > 1 (estamos haciendo zoom in)
        if (zoom > 1.1 && (visibleWidth < gridWidth * 0.9 || visibleHeight < gridHeight * 0.9)) {
            lastROIUpdate.current = now;
            sendCommand('simulation', 'set_roi', {
                enabled: true,
                x: visibleX,
                y: visibleY,
                width: visibleWidth,
                height: visibleHeight
            });
        } else if (zoom <= 1.1) {
            // Si estamos en zoom out, desactivar ROI
            lastROIUpdate.current = now;
            sendCommand('simulation', 'set_roi', { enabled: false });
        }
    }, [pan, zoom, gridWidth, gridHeight, mapData, autoROIEnabled, sendCommand]);

    // Manejar wheel event con listener no pasivo para prevenir el comportamiento por defecto
    useEffect(() => {
        const canvas = canvasRef.current;
        if (!canvas) return;
        
        const wheelHandler = (e: WheelEvent) => {
            e.preventDefault();
            handleWheel(e as any);
        };
        
        // Agregar listener con opciones para que no sea pasivo
        canvas.addEventListener('wheel', wheelHandler, { passive: false });
        
        return () => {
            canvas.removeEventListener('wheel', wheelHandler);
        };
    }, [handleWheel]);

    useEffect(() => {
        const canvas = canvasRef.current;
        if (!canvas) return;
        const ctx = canvas.getContext('2d');
        if (!ctx) return;

        if (selectedViz === 'poincare') {
            // Poincaré solo funciona con datos en tiempo real
            const coords = simData?.poincare_coords;
            if (!coords || !Array.isArray(coords) || coords.length === 0) {
                ctx.clearRect(0, 0, canvas.width, canvas.height);
                return;
            }
            
            canvas.width = 512;
            canvas.height = 512;
            const centerX = canvas.width / 2;
            const centerY = canvas.height / 2;
            const radius = Math.min(centerX, centerY) * 0.95;

            ctx.clearRect(0, 0, canvas.width, canvas.height);
            ctx.fillStyle = '#1A1B1E';
            ctx.fillRect(0, 0, canvas.width, canvas.height);
            
            ctx.beginPath();
            ctx.arc(centerX, centerY, radius, 0, 2 * Math.PI, false);
            ctx.strokeStyle = '#444';
            ctx.stroke();

            // Mejorar renderizado de Poincaré
            const pointCount = coords.length;
            const pointSize = Math.max(1, Math.min(3, 512 / Math.sqrt(pointCount)));
            
            coords.forEach((point, index) => {
                // Validar que point sea un array con al menos 2 elementos
                if (!Array.isArray(point) || point.length < 2) {
                    return;
                }
                const x = centerX + (typeof point[0] === 'number' ? point[0] : 0) * radius;
                const y = centerY + (typeof point[1] === 'number' ? point[1] : 0) * radius;
                // Validar que las coordenadas sean números válidos
                if (isNaN(x) || isNaN(y) || !isFinite(x) || !isFinite(y)) {
                    return;
                }
                
                // Color basado en posición (gradiente)
                const t = index / pointCount;
                const hue = t * 240; // De azul a rojo
                ctx.beginPath();
                ctx.arc(x, y, pointSize, 0, 2 * Math.PI, false);
                
                // Gradiente de color más visible
                const alpha = 0.8;
                ctx.fillStyle = `hsla(${hue}, 70%, 60%, ${alpha})`;
                ctx.fill();
                
                // Borde para mejor visibilidad
                ctx.strokeStyle = `hsla(${hue}, 70%, 50%, 0.5)`;
                ctx.lineWidth = 0.5;
                ctx.stroke();
            });

        } else if (selectedViz === 'flow') {
            // Visualización de flujo (quiver plot) - solo funciona con datos en tiempo real
            const flowData = simData?.flow_data;
            if (!flowData || !flowData.dx || !flowData.dy) {
                ctx.clearRect(0, 0, canvas.width, canvas.height);
                return;
            }
            
            const dx = flowData.dx;
            const dy = flowData.dy;
            const magnitude = flowData.magnitude || [];
            
            const gridHeight = Array.isArray(dx) ? dx.length : 0;
            const gridWidth = Array.isArray(dx[0]) ? dx[0].length : 0;
            
            if (gridHeight === 0 || gridWidth === 0) return;
            
            canvas.width = gridWidth;
            canvas.height = gridHeight;
            
            // Fondo negro con overlay de densidad
            const mapData = simData?.map_data;
            if (mapData && Array.isArray(mapData) && mapData.length > 0) {
                for (let y = 0; y < Math.min(gridHeight, mapData.length); y++) {
                    for (let x = 0; x < Math.min(gridWidth, mapData[y]?.length || 0); x++) {
                        const value = mapData[y]?.[x];
                        if (typeof value === 'number' && !isNaN(value)) {
                            ctx.fillStyle = getColor(value * 0.3); // Más oscuro para ver las flechas
                            ctx.fillRect(x, y, 1, 1);
                        }
                    }
                }
            } else {
                ctx.fillStyle = '#000000';
                ctx.fillRect(0, 0, gridWidth, gridHeight);
            }
            
            // Dibujar flechas
            const stepSize = Math.max(1, Math.floor(Math.max(gridWidth, gridHeight) / 40));
            
            for (let y = 0; y < gridHeight; y += stepSize) {
                for (let x = 0; x < gridWidth; x += stepSize) {
                    if (!Array.isArray(dx[y]) || typeof dx[y][x] !== 'number') continue;
                    if (!Array.isArray(dy[y]) || typeof dy[y][x] !== 'number') continue;
                    
                    const dx_val = dx[y][x];
                    const dy_val = dy[y][x];
                    
                    // Normalizar magnitud
                    const rawMag = (Array.isArray(magnitude[y]) && typeof magnitude[y][x] === 'number') 
                        ? magnitude[y][x] 
                        : Math.sqrt(dx_val * dx_val + dy_val * dy_val);
                    
                    // Encontrar maxMag para normalizar
                    let maxMag = 0;
                    for (let yy = 0; yy < gridHeight; yy += stepSize) {
                        for (let xx = 0; xx < gridWidth; xx += stepSize) {
                            if (Array.isArray(magnitude[yy]) && typeof magnitude[yy][xx] === 'number') {
                                maxMag = Math.max(maxMag, magnitude[yy][xx]);
                            }
                        }
                    }
                    if (maxMag === 0) maxMag = 1;
                    const mag = rawMag / maxMag;
                    
                    // Calcular longitud de la flecha
                    const rawLength = Math.sqrt(dx_val * dx_val + dy_val * dy_val);
                    if (rawLength === 0) continue;
                    
                    const minVisibleLength = stepSize * 0.3;
                    const maxLength = stepSize * 0.7;
                    const scaledLength = Math.max(minVisibleLength, Math.min(maxLength, rawLength * stepSize * 2));
                    
                    const dirX = (dx_val / rawLength) * scaledLength;
                    const dirY = -(dy_val / rawLength) * scaledLength;
                    
                    // Color más brillante y visible
                    const intensity = Math.min(mag, 1.0);
                    const hue = 200 - intensity * 60;
                    const lightness = 40 + intensity * 40;
                    ctx.strokeStyle = `hsl(${hue}, 100%, ${lightness}%)`;
                    ctx.fillStyle = `hsl(${hue}, 100%, ${lightness}%)`;
                    ctx.lineWidth = 1 + intensity * 2;
                    
                    const endX = x + dirX;
                    const endY = y + dirY;
                    
                    ctx.beginPath();
                    ctx.moveTo(x, y);
                    ctx.lineTo(endX, endY);
                    ctx.stroke();
                    
                    if (scaledLength > 1) {
                        const angle = Math.atan2(dirY, dirX);
                        const arrowLength = Math.min(scaledLength * 0.4, 6);
                        
                        ctx.beginPath();
                        ctx.moveTo(endX, endY);
                        ctx.lineTo(endX - arrowLength * Math.cos(angle - Math.PI / 6), endY - arrowLength * Math.sin(angle - Math.PI / 6));
                        ctx.lineTo(endX, endY);
                        ctx.lineTo(endX - arrowLength * Math.cos(angle + Math.PI / 6), endY - arrowLength * Math.sin(angle + Math.PI / 6));
                        ctx.closePath();
                        ctx.fill();
                    }
                }
            }
        } else {
            // Visualización normal (density, phase, etc.)
            const mapData = dataToRender?.map_data;
            if (!mapData || !Array.isArray(mapData) || mapData.length === 0 || !Array.isArray(mapData[0]) || mapData[0].length === 0) {
                ctx.clearRect(0, 0, canvas.width, canvas.height);
                return;
            }
            
            const gridHeight = mapData.length;
            const gridWidth = mapData[0].length;

            if (canvas.width !== gridWidth || canvas.height !== gridHeight) {
                if (gridWidth > 0 && gridHeight > 0) {
                canvas.width = gridWidth;
                canvas.height = gridHeight;
            }
            }

            ctx.clearRect(0, 0, canvas.width, canvas.height);

            // Si no hay datos de mapa (por ejemplo, live feed desactivado), terminar aquí
            if (!mapData || mapData.length === 0) {
                return;
            }
            
            // Verificar si es visualización HSV (solo funciona con datos en tiempo real)
            if (selectedViz === 'phase_hsv' && simData?.phase_hsv_data) {
                const hsvData = simData.phase_hsv_data;
                const hue = hsvData.hue || mapData;
                const saturation = hsvData.saturation || mapData.map(() => mapData[0].map(() => 1));
                const value = hsvData.value || mapData;
                
                for (let y = 0; y < gridHeight; y++) {
                    for (let x = 0; x < gridWidth; x++) {
                        if (!hue[y] || typeof hue[y][x] === 'undefined') continue;
                        const h = hue[y][x] * 360; // Convertir a grados [0, 360]
                        const s = saturation[y]?.[x] ?? 1.0;
                        const v = value[y]?.[x] ?? 1.0;
                        
                        // Convertir HSV a RGB
                        const c = v * s;
                        const x_h = c * (1 - Math.abs(((h / 60) % 2) - 1));
                        const m = v - c;
                        
                        let r = 0, g = 0, b = 0;
                        if (h < 60) { r = c; g = x_h; b = 0; }
                        else if (h < 120) { r = x_h; g = c; b = 0; }
                        else if (h < 180) { r = 0; g = c; b = x_h; }
                        else if (h < 240) { r = 0; g = x_h; b = c; }
                        else if (h < 300) { r = x_h; g = 0; b = c; }
                        else { r = c; g = 0; b = x_h; }
                        
                        ctx.fillStyle = `rgb(${Math.round((r + m) * 255)}, ${Math.round((g + m) * 255)}, ${Math.round((b + m) * 255)})`;
                        ctx.fillRect(x, y, 1, 1);
                    }
                }
            } else {
                // Visualización normal con colormap
                // Calcular estadísticas para normalización robusta (usa percentiles para evitar outliers)
                const values: number[] = [];
                for (let y = 0; y < gridHeight; y++) {
                    for (let x = 0; x < gridWidth; x++) {
                        if (!mapData[y] || typeof mapData[y][x] === 'undefined') continue;
                        const val = mapData[y][x];
                        if (typeof val === 'number' && !isNaN(val) && isFinite(val)) {
                            values.push(val);
                        }
                    }
                }
                
                if (values.length === 0) {
                    ctx.clearRect(0, 0, gridWidth, gridHeight);
                    return;
                }
                
                // Ordenar valores para calcular percentiles
                values.sort((a, b) => a - b);
                const minVal = values[0];
                const maxVal = values[values.length - 1];
                
                // Usar percentiles para normalización robusta (evita que un outlier comprima todo)
                // Percentil 1% como mínimo y 99% como máximo para visualizaciones con outliers
                const p1Index = Math.floor(values.length * 0.01);
                const p99Index = Math.floor(values.length * 0.99);
                const robustMin = values[p1Index] || minVal;
                const robustMax = values[p99Index] || maxVal;
                
                // Si hay mucha diferencia entre min/max y los percentiles, usar percentiles
                // Esto es especialmente útil para FFT/spectral que tiene picos extremos
                const useRobust = (maxVal - minVal) > (robustMax - robustMin) * 2;
                const rangeMin = useRobust ? robustMin : minVal;
                const rangeMax = useRobust ? robustMax : maxVal;
                const range = rangeMax - rangeMin || 1;
                
                for (let y = 0; y < gridHeight; y++) {
                    for (let x = 0; x < gridWidth; x++) {
                        // Validar que mapData[y] existe y tiene el elemento x
                        if (!mapData[y] || typeof mapData[y][x] === 'undefined') {
                            continue;
                        }
                        const value = mapData[y][x];
                        // Validar que value sea un número válido
                        if (typeof value !== 'number' || isNaN(value) || !isFinite(value)) {
                            continue;
                        }
                        // Normalizar usando rango robusto (percentiles)
                        let normalizedValue = (value - rangeMin) / range;
                        // Clipear valores fuera del rango robusto (saturar en los extremos)
                        normalizedValue = Math.max(0, Math.min(1, normalizedValue));
                        ctx.fillStyle = getColor(normalizedValue);
                        ctx.fillRect(x, y, 1, 1);
                    }
                }
            }
        }
    }, [dataToRender, simData, selectedViz, pan, zoom, historyFrame, gridWidth, gridHeight]);

    return (
        <div className={classes.canvasContainer} style={{ position: 'relative' }}>
            {(!dataToRender?.map_data && !simData?.map_data) && (
                <div style={{
                    position: 'absolute',
                    top: '50%',
                    left: '50%',
                    transform: 'translate(-50%, -50%)',
                    textAlign: 'center',
                    color: '#999',
                    zIndex: 1,
                    pointerEvents: 'none'
                }}>
                    <p style={{ margin: 0, fontSize: '1.1rem' }}>Esperando datos de simulación...</p>
                    <p style={{ margin: '0.5rem 0 0 0', fontSize: '0.9rem' }}>
                        {inferenceStatus === 'running' 
                            ? 'Carga un modelo desde el panel lateral para ver la simulación'
                            : 'Inicia la simulación o carga un modelo para ver datos'}
                    </p>
                </div>
            )}

            {simData && simData.simulation_info?.live_feed_enabled === false && (
                <div style={{
                    position: 'absolute',
                    top: '50%',
                    left: '50%',
                    transform: 'translate(-50%, -50%)',
                    textAlign: 'center',
                    color: '#999',
                    zIndex: 1,
                    pointerEvents: 'none', // Permitir interacción con el canvas
                    backgroundColor: 'rgba(0, 0, 0, 0.7)',
                    padding: '1rem',
                    borderRadius: '8px'
                }}>
                    <p style={{ margin: 0, fontSize: '1.2rem', fontWeight: 'bold', color: '#fff' }}>Live Feed Pausado</p>
                    <p style={{ margin: '0.5rem 0 0 0', fontSize: '0.9rem', color: '#ccc' }}>
                        Simulación en ejecución (Paso: {simData.step || simData.simulation_info?.step || '...'})
                    </p>
                </div>
            )}
            
            {/* Controles de overlay */}
            <div style={{
                position: 'absolute',
                top: 10,
                right: 10,
                zIndex: 20,
                display: 'flex',
                flexDirection: 'column',
                gap: 8
            }}>
                <Group gap={4}>
                    <Tooltip label="Resetear vista">
                        <ActionIcon
                            variant="filled"
                            color="blue"
                            onClick={resetView}
                            size="sm"
                        >
                            <IconZoomReset size={16} />
                        </ActionIcon>
                    </Tooltip>
                    <Tooltip label="Configurar overlays">
                        <ActionIcon
                            variant={showOverlayControls ? "filled" : "light"}
                            color="gray"
                            onClick={() => setShowOverlayControls(!showOverlayControls)}
                            size="sm"
                        >
                            <IconSettings size={16} />
                        </ActionIcon>
                    </Tooltip>
                </Group>
                
                {showOverlayControls && (
                    <div style={{ 
                        position: 'absolute',
                        top: 40,
                        right: 0,
                        zIndex: 21
                    }}>
                        <Paper p="sm" withBorder style={{ backgroundColor: 'var(--mantine-color-dark-7)' }}>
                            <Stack gap="xs">
                        <OverlayControls
                            config={overlayConfig}
                            onConfigChange={setOverlayConfig}
                        />
                                <div style={{ borderTop: '1px solid var(--mantine-color-dark-4)', paddingTop: '8px', marginTop: '4px' }}>
                                    <Switch
                                        label="ROI Automático"
                                        description="Sincronizar ROI con la vista visible (solo procesar lo que ves)"
                                        checked={autoROIEnabled}
                                        onChange={(e) => setAutoROIEnabled(e.currentTarget.checked)}
                                        size="sm"
                                    />
                                    {autoROIEnabled && (
                                        <Text size="xs" c="dimmed" mt={4}>
                                            El ROI se actualiza automáticamente según el zoom y pan. Solo se procesa la región visible.
                                        </Text>
                                    )}
                                </div>
                            </Stack>
                        </Paper>
                    </div>
                )}
            </div>
            
            <canvas
                ref={canvasRef}
                className={classes.canvas}
                onMouseDown={handleMouseDown}
                onMouseMove={handleMouseMove}
                onMouseUp={handleMouseUp}
                onMouseLeave={handleMouseUp}
                style={{ 
                    transform: `translate(calc(50% + ${pan.x}px), calc(50% + ${pan.y}px)) scale(${zoom})`,
                    transformOrigin: 'center center',
                    visibility: (dataToRender?.map_data || simData?.map_data) ? 'visible' : 'hidden',
                    position: 'absolute',
                    left: gridWidth > 0 ? `calc(50% - ${gridWidth / 2}px)` : '50%',
                    top: gridHeight > 0 ? `calc(50% - ${gridHeight / 2}px)` : '50%'
                }}
            />
            
            {/* Overlays */}
            {(overlayConfig.showGrid || overlayConfig.showCoordinates || overlayConfig.showQuadtree || overlayConfig.showStats) && (
                <CanvasOverlays
                    canvasRef={canvasRef}
                    mapData={mapData}
                    pan={pan}
                    zoom={zoom}
                    config={overlayConfig}
                    roiInfo={simData?.roi_info || null}
                />
            )}
        </div>
    );
}
