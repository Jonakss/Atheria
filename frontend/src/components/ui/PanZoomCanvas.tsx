// frontend/src/components/PanZoomCanvas.tsx
import { useRef, useEffect, useState, useCallback, useMemo } from 'react';
import { useWebSocket } from '../../hooks/useWebSocket';
import { usePanZoom } from '../../hooks/usePanZoom';
import { CanvasOverlays, OverlayControls, OverlayConfig } from './CanvasOverlays';
import { ZoomOut, Settings, Eye, EyeOff, Maximize2, Minimize2 } from 'lucide-react';
import { Group } from '../../modules/Dashboard/components/Group';
import { ActionIcon } from '../../modules/Dashboard/components/ActionIcon';
import { Tooltip } from '../../modules/Dashboard/components/Tooltip';
import { Switch } from '../../modules/Dashboard/components/Switch';
import { GlassPanel } from '../../modules/Dashboard/components/GlassPanel';
import { Stack } from '../../modules/Dashboard/components/Stack';
import { Text } from '../../modules/Dashboard/components/Text';
import { Box } from '../../modules/Dashboard/components/Box';

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
    
    // Estado de overlays (declarar ANTES de usarlo)
    const [overlayConfig, setOverlayConfig] = useState<OverlayConfig>({
        showGrid: false,
        showCoordinates: false,
        showQuadtree: false,
        showStats: false,
        showToroidalBorders: false, // Por defecto oculto, pero disponible para verificar conectividad
        gridSize: 10,
        quadtreeThreshold: 0.01
    });
    
    const toroidalMode = overlayConfig.showToroidalBorders; // Usar overlay de bordes toroidales como indicador
    const { pan, zoom, handleMouseDown, handleMouseMove, handleMouseUp, handleWheel, resetView, isPanning, isZooming, zoomCenter } = usePanZoom(canvasRef, gridWidth, gridHeight, toroidalMode);
    const [showOverlayControls, setShowOverlayControls] = useState(false);
    const [autoROIEnabled, setAutoROIEnabled] = useState(false);
    const lastROIUpdate = useRef<number>(0);
    const roiUpdateTimeoutRef = useRef<NodeJS.Timeout | null>(null);
    const ROIUpdateThrottle = 300; // Throttle: mínimo tiempo entre actualizaciones (300ms)
    const ROIDebounceDelay = 500; // Debounce: esperar 500ms después de la última interacción antes de actualizar
    
    // Estado para tooltip de información del punto
    const [tooltipData, setTooltipData] = useState<{
        x: number;
        y: number;
        gridX: number;
        gridY: number;
        value: number | null;
        visible: boolean;
    } | null>(null);
    
    // Handler para mover el mouse y mostrar información del punto
    const handleCanvasMouseMove = useCallback((e: React.MouseEvent<HTMLDivElement>) => {
        if (!canvasRef.current || !mapData || gridWidth === 0 || gridHeight === 0) {
            if (tooltipData) setTooltipData(null);
            return;
        }
        
        const canvas = canvasRef.current;
        const container = canvas.parentElement;
        if (!container) return;
        
        const containerRect = container.getBoundingClientRect();
        
        // Coordenadas del mouse relativas al contenedor
        const mouseX = e.clientX - containerRect.left;
        const mouseY = e.clientY - containerRect.top;
        
        // El canvas está centrado con:
        // - left: 50%, top: 50%
        // - marginLeft: -(gridWidth/2) + (pan.x/zoom)
        // - marginTop: -(gridHeight/2) + (pan.y/zoom)
        // - transform: scale(zoom)
        // - transformOrigin: center center
        
        // Coordenadas relativas al centro del contenedor
        const containerWidth = containerRect.width;
        const containerHeight = containerRect.height;
        const mouseRelToCenterX = mouseX - containerWidth / 2;
        const mouseRelToCenterY = mouseY - containerHeight / 2;
        
        // El canvas está desplazado por el margen antes del scale
        // El margen efectivo es: -(gridWidth/2) + (pan.x/zoom) en X
        // Después del scale, este desplazamiento se multiplica por zoom
        // Entonces, la posición del centro del canvas (antes del scale) es:
        const canvasCenterOffsetX = -(gridWidth / 2) + (pan.x / zoom);
        const canvasCenterOffsetY = -(gridHeight / 2) + (pan.y / zoom);
        
        // Coordenadas del mouse relativas al centro del canvas (antes del scale)
        // Necesitamos deshacer el scale primero
        let mouseRelToCanvasCenterX = (mouseRelToCenterX / zoom) - canvasCenterOffsetX;
        let mouseRelToCanvasCenterY = (mouseRelToCenterY / zoom) - canvasCenterOffsetY;
        
        // Verificar si estamos en modo toroidal
        const toroidalMode = overlayConfig.showToroidalBorders;
        
        // En modo toroidal, aplicar wraparound a las coordenadas
        if (toroidalMode && gridWidth > 0 && gridHeight > 0) {
            // Aplicar wraparound: las coordenadas se "envuelven" alrededor del grid
            mouseRelToCanvasCenterX = ((mouseRelToCanvasCenterX % gridWidth) + gridWidth) % gridWidth;
            mouseRelToCanvasCenterY = ((mouseRelToCanvasCenterY % gridHeight) + gridHeight) % gridHeight;
        }
        
        // Ahora convertir a coordenadas del grid
        const gridX = Math.floor(mouseRelToCanvasCenterX);
        const gridY = Math.floor(mouseRelToCanvasCenterY);
        
        // Verificar si estamos dentro del grid (en modo toroidal siempre está "dentro")
        const margin = 0.5; // Permitir un pequeño margen para bordes
        const isInside = toroidalMode || (gridX >= -margin && gridX < gridWidth + margin && gridY >= -margin && gridY < gridHeight + margin);
        
        if (isInside && gridWidth > 0 && gridHeight > 0) {
            // En modo toroidal, las coordenadas ya están wrappeadas (siempre válidas)
            // En modo normal, clampear a los límites válidos del grid
            let finalX, finalY;
            if (toroidalMode) {
                // Wraparound final en modo toroidal
                finalX = ((gridX % gridWidth) + gridWidth) % gridWidth;
                finalY = ((gridY % gridHeight) + gridHeight) % gridHeight;
            } else {
                // Clampear a límites válidos en modo normal
                finalX = Math.max(0, Math.min(gridWidth - 1, gridX));
                finalY = Math.max(0, Math.min(gridHeight - 1, gridY));
            }
            
            const value = mapData[finalY]?.[finalX];
            const numValue = typeof value === 'number' && !isNaN(value) ? value : null;
            
            setTooltipData({
                x: e.clientX,
                y: e.clientY,
                gridX: finalX,
                gridY: finalY,
                value: numValue,
                visible: true
            });
        } else {
            setTooltipData(null);
        }
    }, [mapData, gridWidth, gridHeight, pan, zoom, tooltipData, overlayConfig.showToroidalBorders]);
    
    const handleCanvasMouseLeave = useCallback(() => {
        setTooltipData(null);
    }, []);

    // Sincronizar ROI automáticamente con la vista visible del canvas
    // Usa debounce para evitar actualizaciones excesivas durante pan/zoom activos
    useEffect(() => {
        if (!autoROIEnabled || !canvasRef.current || !mapData || gridWidth === 0 || gridHeight === 0) {
            // Si ROI está desactivado, cancelar cualquier actualización pendiente
            if (roiUpdateTimeoutRef.current) {
                clearTimeout(roiUpdateTimeoutRef.current);
                roiUpdateTimeoutRef.current = null;
            }
            // Desactivar ROI si estaba activo
            if (autoROIEnabled === false && lastROIUpdate.current > 0) {
                sendCommand('simulation', 'set_roi', { enabled: false });
                lastROIUpdate.current = 0;
            }
            return;
        }
        
        // Limpiar timeout anterior (debounce)
        if (roiUpdateTimeoutRef.current) {
            clearTimeout(roiUpdateTimeoutRef.current);
        }
        
        // Crear nuevo timeout con debounce
        roiUpdateTimeoutRef.current = setTimeout(() => {
            const now = Date.now();
            
            // Verificar throttle (evitar actualizaciones demasiado frecuentes)
            if (now - lastROIUpdate.current < ROIUpdateThrottle) {
                return;
            }
            
            const canvas = canvasRef.current;
            if (!canvas) return;
            const container = canvas.parentElement;
            if (!container) return;
            
            const containerRect = container.getBoundingClientRect();
            const containerWidth = containerRect.width;
            const containerHeight = containerRect.height;
            
            if (containerWidth === 0 || containerHeight === 0) return;
            
            // El canvas está centrado en el contenedor con CSS
            // El origen del transform (0,0) está en el centro del contenedor
            // Después del scale(zoom) translate(pan.x, pan.y):
            // - El punto (gridX, gridY) del canvas está en: (containerCenter + pan.x + gridX*zoom, containerCenter + pan.y + gridY*zoom)
            // - Para convertir coordenadas del contenedor a coordenadas del grid:
            //   * containerX = containerWidth/2 + pan.x + gridX*zoom
            //   * gridX = (containerX - containerWidth/2 - pan.x) / zoom
            
            // Calcular qué región del contenedor está visible (toda la región visible)
            // La esquina superior izquierda visible: (0, 0) en coordenadas del contenedor
            // La esquina inferior derecha visible: (containerWidth, containerHeight)
            
            // Convertir a coordenadas del grid
            const topLeftX = (0 - containerWidth/2 - pan.x) / zoom;
            const topLeftY = (0 - containerHeight/2 - pan.y) / zoom;
            const bottomRightX = (containerWidth - containerWidth/2 - pan.x) / zoom;
            const bottomRightY = (containerHeight - containerHeight/2 - pan.y) / zoom;
        
            // El canvas está centrado, así que las coordenadas del grid son relativas al centro
            // El centro del grid (gridWidth/2, gridHeight/2) está en el origen del transform
            // Entonces: gridX = (containerX - containerWidth/2 - pan.x) / zoom + gridWidth/2
            
            const visibleX = Math.max(0, Math.floor((topLeftX + gridWidth/2)));
            const visibleY = Math.max(0, Math.floor((topLeftY + gridHeight/2)));
            const visibleRightX = Math.min(gridWidth, Math.ceil((bottomRightX + gridWidth/2)));
            const visibleBottomY = Math.min(gridHeight, Math.ceil((bottomRightY + gridHeight/2)));
            
            const visibleWidth = Math.max(1, visibleRightX - visibleX);
            const visibleHeight = Math.max(1, visibleBottomY - visibleY);
        
            // Solo actualizar ROI si la región visible es significativamente menor que el grid completo
            // y si el zoom es > 1.1 (estamos haciendo zoom in)
            const visibleRatio = (visibleWidth * visibleHeight) / (gridWidth * gridHeight);
            if (zoom > 1.1 && visibleRatio < 0.9 && visibleWidth > 16 && visibleHeight > 16) {
            lastROIUpdate.current = now;
            sendCommand('simulation', 'set_roi', {
                enabled: true,
                x: visibleX,
                y: visibleY,
                width: visibleWidth,
                height: visibleHeight
            });
            } else if (zoom <= 1.1 || visibleRatio >= 0.9) {
                // Si estamos en zoom out o mostrando más del 90%, desactivar ROI
            lastROIUpdate.current = now;
            sendCommand('simulation', 'set_roi', { enabled: false });
        }
        }, ROIDebounceDelay); // Debounce: esperar 500ms después de la última interacción
        
        // Cleanup: cancelar timeout si el componente se desmonta o cambian las dependencias
        return () => {
            if (roiUpdateTimeoutRef.current) {
                clearTimeout(roiUpdateTimeoutRef.current);
                roiUpdateTimeoutRef.current = null;
            }
        };
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
                
                // RENDERIZADO ADAPTATIVO SEGÚN ZOOM (LOD) - Igual que visualización normal
                const maxZoomForQuality = 2.0;
                const minQuality = 0.25;
                const maxQuality = 1.0;
                
                let renderQuality = maxQuality;
                if (zoom > maxZoomForQuality) {
                    const qualityDrop = (zoom - maxZoomForQuality) / (5.0 - maxZoomForQuality);
                    renderQuality = Math.max(minQuality, maxQuality - qualityDrop * (maxQuality - minQuality));
                }
                
                const sampleStep = Math.max(1, Math.floor(1 / renderQuality));
                
                // Renderizar con muestreo adaptativo
                for (let y = 0; y < gridHeight; y += sampleStep) {
                    for (let x = 0; x < gridWidth; x += sampleStep) {
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
                        // Dibujar bloque de píxeles (más grande cuando hay downsampling)
                        ctx.fillRect(x, y, sampleStep, sampleStep);
                    }
                }
            } else {
                // Visualización normal con colormap
                // OPTIMIZACIÓN: Usar muestreo para cálculos de estadísticas (reducir de O(N²) a O(samples))
                // En grids grandes (>256x256), calcular estadísticas solo sobre un subconjunto
                const STAT_SAMPLE_SIZE = 10000; // Máximo de muestras para calcular estadísticas
                const totalCells = gridWidth * gridHeight;
                const needsSampling = totalCells > STAT_SAMPLE_SIZE;
                const statSampleStep = needsSampling ? Math.max(1, Math.floor(Math.sqrt(totalCells / STAT_SAMPLE_SIZE))) : 1;
                
                // Calcular estadísticas para normalización robusta (usa percentiles para evitar outliers)
                const values: number[] = [];
                const targetSampleCount = Math.min(STAT_SAMPLE_SIZE, totalCells);
                
                for (let y = 0; y < gridHeight; y += statSampleStep) {
                    for (let x = 0; x < gridWidth; x += statSampleStep) {
                        if (!mapData[y] || typeof mapData[y][x] === 'undefined') continue;
                        const val = mapData[y][x];
                        if (typeof val === 'number' && !isNaN(val) && isFinite(val)) {
                            values.push(val);
                            if (values.length >= targetSampleCount) break;
                        }
                    }
                    if (values.length >= targetSampleCount) break;
                }
                
                if (values.length === 0) {
                    ctx.clearRect(0, 0, gridWidth, gridHeight);
                    return;
                }
                
                // OPTIMIZACIÓN: Ordenar solo si tenemos suficientes valores para percentiles
                // Para muestras pequeñas, usar min/max directo (más rápido)
                let rangeMin: number, rangeMax: number, range: number;
                
                if (values.length < 100) {
                    // Para muestras pequeñas, usar min/max directo (más rápido, O(N))
                    rangeMin = Math.min(...values);
                    rangeMax = Math.max(...values);
                    range = rangeMax - rangeMin || 1;
                } else {
                    // Para muestras grandes, calcular percentiles (más robusto, O(N log N))
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
                    rangeMin = useRobust ? robustMin : minVal;
                    rangeMax = useRobust ? robustMax : maxVal;
                    range = rangeMax - rangeMin || 1;
                }
                
                // RENDERIZADO ADAPTATIVO SEGÚN ZOOM (LOD)
                // Zoom alto (> 2.0x) = menor calidad (más rápido)
                // Zoom bajo (< 1.0x) = calidad completa
                // Interpolación suave entre calidades
                const maxZoomForQuality = 2.0; // Zoom máximo para calidad completa
                const minQuality = 0.25; // Calidad mínima (25% de píxeles)
                const maxQuality = 1.0; // Calidad máxima (100% de píxeles)
                
                // Calcular factor de calidad basado en zoom
                // Zoom 1.0x = calidad 100%
                // Zoom > 2.0x = calidad degradada progresivamente
                let renderQuality = maxQuality;
                if (zoom > maxZoomForQuality) {
                    // Degradar calidad progresivamente después de 2.0x
                    const qualityDrop = (zoom - maxZoomForQuality) / (5.0 - maxZoomForQuality); // Interpolación hasta zoom 5.0x
                    renderQuality = Math.max(minQuality, maxQuality - qualityDrop * (maxQuality - minQuality));
                }
                
                // Calcular paso de muestreo basado en calidad
                // renderQuality = 1.0 => sampleStep = 1 (todos los píxeles)
                // renderQuality = 0.25 => sampleStep = 4 (1 de cada 4 píxeles)
                const sampleStep = Math.max(1, Math.floor(1 / renderQuality));
                
                // Renderizar con muestreo adaptativo
                for (let y = 0; y < gridHeight; y += sampleStep) {
                    for (let x = 0; x < gridWidth; x += sampleStep) {
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
                        
                        // Dibujar bloque de píxeles (más grande cuando hay downsampling)
                        ctx.fillRect(x, y, sampleStep, sampleStep);
                    }
                }
            }
        }
    }, [dataToRender, simData, selectedViz, pan, zoom, historyFrame, gridWidth, gridHeight]);

    // Atajos de teclado para vistas rápidas
    useEffect(() => {
        const handleKeyPress = (e: KeyboardEvent) => {
            // Solo procesar si no estamos escribiendo en un input
            if (e.target instanceof HTMLInputElement || e.target instanceof HTMLTextAreaElement) {
                return;
            }
            
            // Ctrl/Cmd + números para cambiar visualización rápidamente
            if ((e.ctrlKey || e.metaKey) && e.key >= '1' && e.key <= '5') {
                e.preventDefault();
                const vizIndex = parseInt(e.key) - 1;
                const vizOptions = ['density', 'phase', 'flow', 'spectral', 'poincare'];
                if (vizOptions[vizIndex]) {
                    sendCommand('simulation', 'set_viz', { viz_type: vizOptions[vizIndex] });
                }
            }
            
            // R para resetear vista
            if (e.key === 'r' || e.key === 'R') {
                if (!(e.ctrlKey || e.metaKey || e.altKey)) {
                    e.preventDefault();
                    resetView();
                }
            }
        };
        
        window.addEventListener('keydown', handleKeyPress);
        return () => window.removeEventListener('keydown', handleKeyPress);
    }, [sendCommand, resetView]);

    return (
        <Box 
            className="relative w-full h-full"
            onMouseMove={handleCanvasMouseMove}
            onMouseLeave={handleCanvasMouseLeave}
        >
            {/* Overlay de zoom centrado en el mouse */}
            {isZooming && zoomCenter && (
                <div
                    className="absolute pointer-events-none z-50"
                    style={{
                        left: `${zoomCenter.x}px`,
                        top: `${zoomCenter.y}px`,
                        transform: 'translate(-50%, -50%)',
                    }}
                >
                    <div className="relative">
                        {/* Círculo exterior pulsante */}
                        <div 
                            className="absolute inset-0 rounded-full border-2 border-teal-400/50 animate-ping"
                            style={{
                                width: '60px',
                                height: '60px',
                                margin: '-30px 0 0 -30px',
                            }}
                        />
                        {/* Círculo interior fijo */}
                        <div 
                            className="absolute inset-0 rounded-full border-2 border-teal-400 shadow-glow-teal"
                            style={{
                                width: '40px',
                                height: '40px',
                                margin: '-20px 0 0 -20px',
                            }}
                        />
                        {/* Indicador de zoom */}
                        <div 
                            className="absolute inset-0 flex items-center justify-center text-teal-400 text-xs font-mono font-bold"
                            style={{
                                width: '40px',
                                height: '40px',
                                margin: '-20px 0 0 -20px',
                            }}
                        >
                            {zoom.toFixed(1)}x
                        </div>
                    </div>
                </div>
            )}
            {(!dataToRender?.map_data && !simData?.map_data) && (
                <Box className="absolute top-1/2 left-1/2 -translate-x-1/2 -translate-y-1/2 text-center text-gray-500 z-10 pointer-events-none">
                    <Text size="lg" className="block mb-2">Esperando datos de simulación...</Text>
                    <Text size="sm" color="dimmed">
                        {inferenceStatus === 'running' 
                            ? 'Carga un modelo desde el panel lateral para ver la simulación'
                            : 'Inicia la simulación o carga un modelo para ver datos'}
                    </Text>
                </Box>
            )}

            {simData && simData.simulation_info?.live_feed_enabled === false && (
                <Box className="absolute top-1/2 left-1/2 -translate-x-1/2 -translate-y-1/2 text-center text-gray-500 z-10 pointer-events-none bg-black/70 p-4 rounded-lg">
                    <Text size="lg" weight="bold" className="block mb-2 text-white">Live Feed Pausado</Text>
                    <Text size="sm" color="muted">
                        Simulación en ejecución (Paso: {simData.step || simData.simulation_info?.step || '...'})
                    </Text>
                </Box>
            )}
            
            {/* Controles de overlay */}
            <Box className="absolute top-2.5 right-2.5 z-20 flex flex-col gap-2">
                <Group gap={1}>
                    <Tooltip label="Resetear vista">
                        <ActionIcon
                            variant="filled"
                            color="blue"
                            onClick={resetView}
                            size="sm"
                        >
                            <ZoomOut size={16} />
                        </ActionIcon>
                    </Tooltip>
                    <Tooltip label="Configurar overlays">
                        <ActionIcon
                            variant={showOverlayControls ? "filled" : "light"}
                            color="gray"
                            onClick={() => setShowOverlayControls(!showOverlayControls)}
                            size="sm"
                        >
                            <Settings size={16} />
                        </ActionIcon>
                    </Tooltip>
                </Group>
                
                {showOverlayControls && (
                    <Box className="absolute top-10 right-0 z-30">
                        <GlassPanel className="p-3">
                            <Stack gap={2}>
                        <OverlayControls
                            config={overlayConfig}
                            onConfigChange={setOverlayConfig}
                        />
                                <Box className="border-t border-white/10 pt-2 mt-1">
                                    <Switch
                                        label="ROI Automático"
                                        description="Sincronizar ROI con la vista visible (solo procesar lo que ves)"
                                        checked={autoROIEnabled}
                                        onChange={(e) => setAutoROIEnabled(e.currentTarget.checked)}
                                        size="sm"
                                    />
                                    {autoROIEnabled && (
                                        <Text size="xs" color="dimmed" className="mt-1 block">
                                            El ROI se actualiza automáticamente según el zoom y pan. Solo se procesa la región visible.
                                        </Text>
                                    )}
                                </Box>
                            </Stack>
                        </GlassPanel>
                    </Box>
                )}
            </Box>
            
            {/* Canvas principal - En modo toroidal, se puede mover infinitamente */}
            <canvas
                ref={canvasRef}
                onMouseDown={handleMouseDown}
                onMouseMove={handleMouseMove}
                onMouseUp={handleMouseUp}
                onMouseLeave={handleMouseUp}
                style={{ 
                    position: 'absolute',
                    left: '50%',
                    top: '50%',
                    marginLeft: `${-(gridWidth / 2) + (pan.x / zoom)}px`,
                    marginTop: `${-(gridHeight / 2) + (pan.y / zoom)}px`,
                    width: `${gridWidth}px`,
                    height: `${gridHeight}px`,
                    transform: `scale(${zoom})`,
                    transformOrigin: 'center center',
                    imageRendering: zoom > 2 ? 'pixelated' : 'auto',
                    visibility: (dataToRender?.map_data || simData?.map_data) ? 'visible' : 'hidden',
                    cursor: isPanning ? 'grabbing' : 'grab',
                    pointerEvents: 'auto'
                }}
            />
            
            {/* Overlays */}
            {(overlayConfig.showGrid || overlayConfig.showCoordinates || overlayConfig.showQuadtree || overlayConfig.showStats || overlayConfig.showToroidalBorders) && (
                <CanvasOverlays
                    canvasRef={canvasRef}
                    mapData={mapData}
                    pan={pan}
                    zoom={zoom}
                    config={overlayConfig}
                    roiInfo={simData?.roi_info || null}
                />
            )}
            
            {/* Tooltip de información del punto (tipo Google Maps) */}
            {tooltipData && tooltipData.visible && (
                <div
                    className="fixed z-[100] pointer-events-none"
                    style={{
                        left: `${Math.min(tooltipData.x + 10, window.innerWidth - 200)}px`,
                        top: `${Math.max(tooltipData.y - 120, 10)}px`
                    }}
                >
                    <GlassPanel className="p-2.5 shadow-xl border border-white/20 min-w-[160px]">
                        <div className="text-[10px] space-y-1.5">
                            <div className="flex items-center justify-between gap-3">
                                <span className="text-gray-400 font-mono uppercase text-[9px] tracking-wider">Posición</span>
                                <span className="text-gray-200 font-mono font-bold">
                                    ({tooltipData.gridX}, {tooltipData.gridY})
                                </span>
                            </div>
                            {tooltipData.value !== null && (
                                <div className="flex items-center justify-between gap-3">
                                    <span className="text-gray-400 font-mono uppercase text-[9px] tracking-wider">Valor</span>
                                    <span className="text-emerald-400 font-mono font-bold">
                                        {tooltipData.value.toFixed(4)}
                                    </span>
                                </div>
                            )}
                            {simData?.step !== undefined && simData.step !== null && (
                                <div className="flex items-center justify-between gap-3">
                                    <span className="text-gray-400 font-mono uppercase text-[9px] tracking-wider">Paso</span>
                                    <span className="text-blue-400 font-mono font-bold">
                                        {simData.step.toLocaleString()}
                                    </span>
                                </div>
                            )}
                            {zoom && (
                                <div className="flex items-center justify-between gap-3">
                                    <span className="text-gray-400 font-mono uppercase text-[9px] tracking-wider">Zoom</span>
                                    <span className="text-amber-400 font-mono font-bold">
                                        {zoom.toFixed(2)}x
                                    </span>
                                </div>
                            )}
                            {selectedViz && (
                                <div className="flex items-center justify-between gap-3 pt-1 border-t border-white/10">
                                    <span className="text-gray-400 font-mono uppercase text-[9px] tracking-wider">Vista</span>
                                    <span className="text-purple-400 font-mono font-bold text-[9px]">
                                        {selectedViz.toUpperCase()}
                                    </span>
                                </div>
                            )}
                        </div>
                    </GlassPanel>
                </div>
            )}
            
            {/* Indicador de atajos de teclado - Vistas Rápidas (siempre visible) */}
            <Box className="absolute bottom-2.5 left-2.5 z-20 opacity-70 hover:opacity-100 transition-opacity">
                <GlassPanel className="p-2.5">
                    <div className="space-y-2">
                        <div className="text-[10px] font-bold text-gray-400 uppercase tracking-wider mb-2">
                            Vistas Rápidas
                        </div>
                        <div className="space-y-1.5 text-[10px]">
                            <div className="flex items-center gap-2">
                                <kbd className="px-1.5 py-0.5 bg-white/10 border border-white/20 rounded text-[9px] font-mono text-gray-300">
                                    Ctrl+1
                                </kbd>
                                <span className="text-gray-500">Densidad</span>
                            </div>
                            <div className="flex items-center gap-2">
                                <kbd className="px-1.5 py-0.5 bg-white/10 border border-white/20 rounded text-[9px] font-mono text-gray-300">
                                    Ctrl+2
                                </kbd>
                                <span className="text-gray-500">Fase</span>
                            </div>
                            <div className="flex items-center gap-2">
                                <kbd className="px-1.5 py-0.5 bg-white/10 border border-white/20 rounded text-[9px] font-mono text-gray-300">
                                    Ctrl+3
                                </kbd>
                                <span className="text-gray-500">Flujo</span>
                            </div>
                            <div className="flex items-center gap-2">
                                <kbd className="px-1.5 py-0.5 bg-white/10 border border-white/20 rounded text-[9px] font-mono text-gray-300">
                                    Ctrl+4
                                </kbd>
                                <span className="text-gray-500">Espectral</span>
                            </div>
                            <div className="flex items-center gap-2">
                                <kbd className="px-1.5 py-0.5 bg-white/10 border border-white/20 rounded text-[9px] font-mono text-gray-300">
                                    Ctrl+5
                                </kbd>
                                <span className="text-gray-500">Poincaré</span>
                            </div>
                            <div className="pt-1.5 border-t border-white/10 mt-1.5">
                                <div className="flex items-center gap-2">
                                    <kbd className="px-1.5 py-0.5 bg-white/10 border border-white/20 rounded text-[9px] font-mono text-gray-300">
                                        R
                                    </kbd>
                                    <span className="text-gray-500">Resetear vista</span>
                                </div>
                            </div>
                        </div>
        </div>
                </GlassPanel>
            </Box>
        </Box>
    );
}
