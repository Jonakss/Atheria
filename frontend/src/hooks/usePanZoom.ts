// frontend/src/hooks/usePanZoom.ts
import { useState, useRef, useCallback, useEffect } from 'react';

// Throttle para optimizar el pan (evitar demasiadas actualizaciones)
function throttle<T extends (...args: any[]) => void>(func: T, limit: number): T {
    let inThrottle: boolean;
    return ((...args: any[]) => {
        if (!inThrottle) {
            func(...args);
            inThrottle = true;
            setTimeout(() => inThrottle = false, limit);
        }
    }) as T;
}

export const usePanZoom = (canvasRef: React.RefObject<HTMLCanvasElement>, gridWidth?: number, gridHeight?: number) => {
    const [pan, setPan] = useState({ x: 0, y: 0 });
    const [zoom, setZoom] = useState(1);
    const isPanning = useRef(false);
    const lastMousePos = useRef({ x: 0, y: 0 });
    const panUpdateRef = useRef<{ x: number; y: number }>({ x: 0, y: 0 });
    const initializedRef = useRef(false);
    const lastGridSizeRef = useRef<{ width: number; height: number } | null>(null);

    // Función para calcular zoom y pan inicial centrado en el centro del grid
    const calculateInitialView = useCallback(() => {
        if (!canvasRef.current || !gridWidth || !gridHeight) {
            return { pan: { x: 0, y: 0 }, zoom: 1 };
        }

        const canvas = canvasRef.current;
        // Obtener el tamaño del contenedor
        const container = canvas.parentElement;
        if (!container) {
            return { pan: { x: 0, y: 0 }, zoom: 1 };
        }

        const containerRect = container.getBoundingClientRect();
        const containerWidth = containerRect.width;
        const containerHeight = containerRect.height;

        if (containerWidth === 0 || containerHeight === 0) {
            return { pan: { x: 0, y: 0 }, zoom: 1 };
        }

        // Tamaño objetivo: mostrar 256 unidades del grid (o todo si es menor)
        const targetUnits = 256;
        const unitsToShowX = Math.min(targetUnits, gridWidth);
        const unitsToShowY = Math.min(targetUnits, gridHeight);
        
        // Calcular zoom para que las unidades objetivo ocupen el contenedor
        // El canvas tiene dimensiones gridWidth x gridHeight píxeles (1 píxel = 1 unidad del grid)
        // Queremos que `unitsToShowX` unidades ocupen `containerWidth` píxeles del contenedor
        const zoomX = containerWidth / unitsToShowX;
        const zoomY = containerHeight / unitsToShowY;
        const initialZoom = Math.min(zoomX, zoomY);

        // El canvas está posicionado con left: 50%, top: 50%, marginLeft: -gridWidth/2, marginTop: -gridHeight/2
        // Esto centra el canvas en el contenedor
        // Con transformOrigin: '0 0' y transform: scale() translate(), el origen de la transformación está en (0,0) del canvas
        // Para centrar el centro del grid en el contenedor:
        // - El centro del grid está en (gridWidth/2, gridHeight/2) en coordenadas del canvas
        // - Después del scale, el centro del grid está en (gridWidth/2 * zoom, gridHeight/2 * zoom)
        // - El centro del contenedor está en (containerWidth/2, containerHeight/2)
        // - Necesitamos: pan.x + gridWidth/2 * zoom = containerWidth/2
        // - Entonces: pan.x = containerWidth/2 - gridWidth/2 * zoom
        
        // Pero como el canvas está centrado con CSS, el punto (0,0) del canvas está en el centro del contenedor
        // después de aplicar left: 50%, top: 50%, marginLeft, marginTop
        // El transform se aplica DESPUÉS de este posicionamiento
        // Entonces, para que el centro del grid esté en el centro del contenedor:
        // Necesitamos mover el centro del grid (gridWidth/2, gridHeight/2) al origen (0,0)
        // Después del scale, esto se convierte en (gridWidth/2 * zoom, gridHeight/2 * zoom)
        // Necesitamos: pan.x - gridWidth/2 * zoom = 0, entonces pan.x = gridWidth/2 * zoom
        // Pero esto no es correcto...

        // Revisando más cuidadosamente:
        // - El canvas tiene width=gridWidth, height=gridHeight
        // - Está centrado con CSS: left: 50%, top: 50%, marginLeft: -gridWidth/2, marginTop: -gridHeight/2
        // - Esto coloca el punto (0,0) del canvas en el centro del contenedor
        // - Con transformOrigin: '0 0', el origen de la transformación es (0,0) del canvas
        // - Con transform: scale(zoom) translate(pan.x, pan.y):
        //   * Primero se escala (el punto (0,0) sigue en (0,0) después del scale)
        //   * Luego se traslada (el punto (0,0) se mueve a (pan.x, pan.y))
        // - Para que el centro del grid (gridWidth/2, gridHeight/2) esté en el centro del contenedor:
        //   * Después del scale: el centro está en (gridWidth/2 * zoom, gridHeight/2 * zoom)
        //   * Después del translate: el centro está en (gridWidth/2 * zoom + pan.x, gridHeight/2 * zoom + pan.y)
        //   * Queremos que esté en (0, 0) relativo al centro del contenedor (que es donde está el origen del canvas)
        //   * Entonces: gridWidth/2 * zoom + pan.x = 0, por lo tanto pan.x = -gridWidth/2 * zoom
        
        const initialPan = {
            x: -gridWidth / 2 * initialZoom,
            y: -gridHeight / 2 * initialZoom
        };

        return { pan: initialPan, zoom: initialZoom };
    }, [canvasRef, gridWidth, gridHeight]);

    // Inicializar vista cuando cambia el tamaño del grid
    useEffect(() => {
        if (!gridWidth || !gridHeight) return;
        
        const currentGridSize = { width: gridWidth, height: gridHeight };
        const lastGridSize = lastGridSizeRef.current;
        
        // Solo inicializar si es la primera vez o si el tamaño del grid cambió
        if (!initializedRef.current || 
            !lastGridSize || 
            lastGridSize.width !== currentGridSize.width || 
            lastGridSize.height !== currentGridSize.height) {
            
            // Esperar un frame para que el contenedor tenga dimensiones
            requestAnimationFrame(() => {
                const initialView = calculateInitialView();
                setPan(initialView.pan);
                setZoom(initialView.zoom);
                initializedRef.current = true;
                lastGridSizeRef.current = currentGridSize;
            });
        }
    }, [gridWidth, gridHeight, calculateInitialView]);

    // Función para resetear vista
    const resetView = useCallback(() => {
        const initialView = calculateInitialView();
        setPan(initialView.pan);
        setZoom(initialView.zoom);
    }, [calculateInitialView]);

    // Limitar pan y zoom para mantener la grilla visible
    const constrainPanZoom = useCallback((newPan: { x: number; y: number }, newZoom: number) => {
        if (!canvasRef.current || !gridWidth || !gridHeight) {
            return { pan: newPan, zoom: newZoom };
        }

        const container = canvasRef.current.parentElement;
        if (!container) {
            return { pan: newPan, zoom: newZoom };
        }

        const containerRect = container.getBoundingClientRect();
        const containerWidth = containerRect.width;
        const containerHeight = containerRect.height;

        if (containerWidth === 0 || containerHeight === 0) {
            return { pan: newPan, zoom: newZoom };
        }
        
        // Calcular límites de zoom
        // Zoom mínimo: mostrar todo el grid o al menos 256 unidades
        const targetUnits = 256;
        const unitsToShowX = Math.min(targetUnits, gridWidth);
        const unitsToShowY = Math.min(targetUnits, gridHeight);
        const minZoomX = containerWidth / unitsToShowX;
        const minZoomY = containerHeight / unitsToShowY;
        const minZoom = Math.min(minZoomX, minZoomY) * 0.9; // 90% para dejar margen
        
        // Zoom máximo: permitir zoom hasta 20x o hasta que 1 unidad = 10 píxeles
        const maxZoom = Math.max(20, Math.min(containerWidth / 10, containerHeight / 10));
        
        const constrainedZoom = Math.max(minZoom, Math.min(newZoom, maxZoom));
        
        // Calcular límites de pan
        // El canvas está centrado en el contenedor con CSS
        // Con transformOrigin: '0 0', el origen está en el centro del contenedor
        // Después del scale y translate, el centro del grid debe estar dentro del contenedor
        // El centro del grid en coordenadas del canvas: (gridWidth/2, gridHeight/2)
        // Después del scale: (gridWidth/2 * zoom, gridHeight/2 * zoom)
        // Después del translate: (gridWidth/2 * zoom + pan.x, gridHeight/2 * zoom + pan.y)
        // Para mantener el grid visible:
        // - El borde izquierdo del grid (x=0) debe estar a la izquierda del borde derecho del contenedor
        // - gridWidth * zoom + pan.x >= -containerWidth/2 (muy aproximado)
        // - El borde derecho del grid (x=gridWidth) debe estar a la derecha del borde izquierdo del contenedor
        // - pan.x <= containerWidth/2 (muy aproximado)
        
        // Límites más conservadores: mantener al menos el 10% del grid visible
        const margin = 0.1;
        const maxPanX = (gridWidth * constrainedZoom * (1 + margin)) / 2;
        const maxPanY = (gridHeight * constrainedZoom * (1 + margin)) / 2;
        const minPanX = -maxPanX;
        const minPanY = -maxPanY;
        
        const constrainedPan = {
            x: Math.max(minPanX, Math.min(newPan.x, maxPanX)),
            y: Math.max(minPanY, Math.min(newPan.y, maxPanY))
        };
        
        return { pan: constrainedPan, zoom: constrainedZoom };
    }, [canvasRef, gridWidth, gridHeight]);

    const handleMouseDown = useCallback((e: React.MouseEvent) => {
        isPanning.current = true;
        lastMousePos.current = { x: e.clientX, y: e.clientY };
        panUpdateRef.current = { ...pan };
    }, [pan]);

    // Throttled mouse move para mejor rendimiento
    const handleMouseMoveThrottled = useCallback(
        throttle((e: React.MouseEvent) => {
        if (!isPanning.current) return;
        // El pan debe ser inversamente proporcional al zoom para mantener la velocidad constante
        const dx = (e.clientX - lastMousePos.current.x);
        const dy = (e.clientY - lastMousePos.current.y);
            const newPan = {
                x: panUpdateRef.current.x + dx,
                y: panUpdateRef.current.y + dy
            };
            const constrained = constrainPanZoom(newPan, zoom);
            panUpdateRef.current = constrained.pan;
            setPan(constrained.pan);
        lastMousePos.current = { x: e.clientX, y: e.clientY };
        }, 16), // ~60fps
        [zoom, constrainPanZoom]
    );

    const handleMouseMove = useCallback((e: React.MouseEvent) => {
        handleMouseMoveThrottled(e);
    }, [handleMouseMoveThrottled]);

    const handleMouseUp = useCallback(() => {
        isPanning.current = false;
    }, []);

    const handleWheel = useCallback((e: React.WheelEvent) => {
        if (e.deltaY !== 0) {
            e.preventDefault();
            
            // Calcular el punto del mouse en coordenadas del canvas antes del zoom
            if (!canvasRef.current || !gridWidth || !gridHeight) return;
            
            const canvas = canvasRef.current;
            const container = canvas.parentElement;
            if (!container) return;
            
            const containerRect = container.getBoundingClientRect();
            const containerWidth = containerRect.width;
            const containerHeight = containerRect.height;
            
            // Coordenadas del mouse relativas al centro del contenedor
            // El canvas está centrado con CSS, así que el punto (0,0) del canvas está en el centro del contenedor
            const mouseX = e.clientX - containerRect.left;
            const mouseY = e.clientY - containerRect.top;
            const mouseRelToCenterX = mouseX - containerWidth / 2;
            const mouseRelToCenterY = mouseY - containerHeight / 2;
            
            // Convertir el punto del mouse a coordenadas del canvas (antes del zoom)
            // Con transform: scale(zoom) translate(pan.x, pan.y) y transformOrigin: '0 0':
            // Un punto del canvas (canvasX, canvasY) se transforma a:
            //   (canvasX * zoom + pan.x, canvasY * zoom + pan.y) en coordenadas relativas al centro del contenedor
            // Inversamente, para un punto del mouse (mouseRelToCenterX, mouseRelToCenterY):
            //   canvasX = (mouseRelToCenterX - pan.x) / zoom
            //   canvasY = (mouseRelToCenterY - pan.y) / zoom
            const canvasX = (mouseRelToCenterX - pan.x) / zoom;
            const canvasY = (mouseRelToCenterY - pan.y) / zoom;
            
            // Aplicar zoom
            const zoomFactor = 1.1;
            const newZoom = e.deltaY < 0 ? zoom * zoomFactor : zoom / zoomFactor;
            const constrainedZoom = Math.max(0.1, Math.min(newZoom, 50));
            
            // Calcular nuevo pan para mantener el punto del canvas fijo bajo el mouse
            // Después del nuevo zoom, queremos que el punto (canvasX, canvasY) siga estando bajo el mouse:
            //   mouseRelToCenterX = canvasX * constrainedZoom + newPanX
            //   Por lo tanto: newPanX = mouseRelToCenterX - canvasX * constrainedZoom
            // Sustituyendo canvasX:
            //   newPanX = mouseRelToCenterX - ((mouseRelToCenterX - pan.x) / zoom) * constrainedZoom
            //   newPanX = mouseRelToCenterX - (mouseRelToCenterX - pan.x) * (constrainedZoom / zoom)
            //   newPanX = mouseRelToCenterX * (1 - constrainedZoom / zoom) + pan.x * (constrainedZoom / zoom)
            const zoomRatio = constrainedZoom / zoom;
            const newPanX = mouseRelToCenterX * (1 - zoomRatio) + pan.x * zoomRatio;
            const newPanY = mouseRelToCenterY * (1 - zoomRatio) + pan.y * zoomRatio;
            
            const constrained = constrainPanZoom({ x: newPanX, y: newPanY }, constrainedZoom);
            setZoom(constrained.zoom);
            setPan(constrained.pan);
        }
    }, [zoom, pan, constrainPanZoom, canvasRef, gridWidth, gridHeight]);

    return {
        pan,
        zoom,
        handleMouseDown,
        handleMouseMove,
        handleMouseUp,
        handleWheel,
        resetView,
    };
};
