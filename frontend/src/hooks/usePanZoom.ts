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

    // Función para calcular zoom y pan inicial centrado en (0,0) con 256 unidades visibles
    const calculateInitialView = useCallback(() => {
        if (!canvasRef.current || !gridWidth || !gridHeight) {
            return { pan: { x: 0, y: 0 }, zoom: 1 };
        }

        const canvas = canvasRef.current;
        // Obtener el tamaño del contenedor (no el canvas en sí, que puede ser más grande)
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

        // Tamaño objetivo: 256 unidades del grid deben ocupar el tamaño del contenedor
        const targetUnits = 256;
        
        // Determinar cuántas unidades del grid queremos mostrar en cada dimensión
        // Si el grid es menor a 256, mostrar todo el grid
        // Si el grid es mayor a 256, mostrar 256 unidades centradas en (0,0)
        const unitsToShowX = Math.min(targetUnits, gridWidth);
        const unitsToShowY = Math.min(targetUnits, gridHeight);
        
        // Calcular el zoom necesario para que las unidades objetivo ocupen el contenedor
        // El canvas tiene dimensiones gridWidth x gridHeight píxeles
        // Necesitamos escalar para que `unitsToShowX` y `unitsToShowY` unidades ocupen el contenedor
        const scaleX = containerWidth / unitsToShowX;
        const scaleY = containerHeight / unitsToShowY;
        // Usar el mínimo para asegurar que ambas dimensiones quepan
        const initialZoom = Math.min(scaleX, scaleY);

        // Centrar en (0,0) del grid
        // El punto (0,0) del grid está en la esquina superior izquierda del canvas
        // Queremos que esté en el centro del contenedor
        // El canvas tiene dimensiones gridWidth x gridHeight
        // Después del zoom, el canvas escalado tiene dimensiones gridWidth * zoom x gridHeight * zoom
        
        // Calcular el offset necesario para centrar (0,0) en el contenedor
        // El centro del contenedor está en (containerWidth/2, containerHeight/2)
        // El punto (0,0) del canvas está en (0,0) antes del transform
        // Después del transform scale(zoom) translate(pan.x, pan.y), el punto (0,0) estará en (pan.x * zoom, pan.y * zoom)
        // Queremos que esté en el centro: (containerWidth/2, containerHeight/2)
        // Entonces: pan.x * zoom = containerWidth/2 - 0, pan.y * zoom = containerHeight/2 - 0
        // Pero espera, el transform es scale(zoom) translate(pan.x, pan.y)
        // En CSS transforms, el orden es: translate primero, luego scale
        // Pero aquí el orden en el style es: scale(${zoom}) translate(${pan.x}px, ${pan.y}px)
        // Esto significa: primero scale, luego translate
        // Entonces el punto (0,0) del canvas se mueve a (pan.x, pan.y) después del scale
        
        // Revisando el código del canvas, veo que el transform es:
        // transform: `scale(${zoom}) translate(${pan.x}px, ${pan.y}px)`
        // Esto significa que primero se escala, luego se traslada
        // Entonces el punto (0,0) del canvas escalado está en (0,0) después del scale
        // Y luego se mueve a (pan.x, pan.y) después del translate
        
        // Para centrar (0,0) del grid en el contenedor:
        // El centro del contenedor está en (containerWidth/2, containerHeight/2)
        // El punto (0,0) del canvas escalado está en (0,0) en coordenadas del contenedor
        // Necesitamos moverlo a (containerWidth/2, containerHeight/2)
        // Entonces: pan.x = containerWidth/2, pan.y = containerHeight/2
        
        // Pero espera, el canvas puede ser más grande que el contenedor
        // El canvas tiene dimensiones gridWidth x gridHeight
        // Después del zoom, el canvas escalado tiene dimensiones gridWidth * zoom x gridHeight * zoom
        // El contenedor tiene dimensiones containerWidth x containerHeight
        
        // Para centrar el punto (0,0) del grid en el contenedor:
        // El canvas tiene dimensiones gridWidth x gridHeight píxeles
        // Después del scale(zoom), el canvas escalado tiene dimensiones gridWidth * zoom x gridHeight * zoom
        // El punto (0,0) del canvas está en la esquina superior izquierda
        // Después del scale(zoom), ese punto sigue en (0,0) en coordenadas del contenedor
        // Después del translate(pan.x, pan.y), ese punto se mueve a (pan.x, pan.y)
        // Queremos que el punto (0,0) del grid esté en el centro del contenedor
        // Entonces: pan.x = containerWidth/2, pan.y = containerHeight/2
        
        // Pero necesitamos considerar que el canvas puede ser más grande que el contenedor
        // El canvas escalado tiene dimensiones gridWidth * zoom x gridHeight * zoom
        // El contenedor tiene dimensiones containerWidth x containerHeight
        // Para centrar el punto (0,0) en el contenedor, necesitamos:
        // pan.x = containerWidth/2, pan.y = containerHeight/2
        
        const initialPan = {
            x: containerWidth / 2,
            y: containerHeight / 2
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

        const canvas = canvasRef.current;
        const canvasWidth = canvas.width;
        const canvasHeight = canvas.height;
        
        // Calcular tamaño de la grilla escalada
        const scaleX = canvasWidth / gridWidth;
        const scaleY = canvasHeight / gridHeight;
        const scale = Math.min(scaleX, scaleY);
        
        const scaledGridWidth = gridWidth * scale * newZoom;
        const scaledGridHeight = gridHeight * scale * newZoom;
        
        // Limitar zoom para que la grilla siempre sea visible
        const minZoom = Math.min(canvasWidth / scaledGridWidth, canvasHeight / scaledGridHeight) * 0.9;
        const maxZoom = 20;
        const constrainedZoom = Math.max(minZoom, Math.min(newZoom, maxZoom));
        
        // Limitar pan para que la grilla no se salga completamente
        const maxPanX = scaledGridWidth / 2;
        const maxPanY = scaledGridHeight / 2;
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
        const dx = (e.clientX - lastMousePos.current.x) / zoom;
        const dy = (e.clientY - lastMousePos.current.y) / zoom;
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
            const zoomFactor = 1.1;
            const newZoom = e.deltaY < 0 ? zoom * zoomFactor : zoom / zoomFactor;
            const constrained = constrainPanZoom(pan, newZoom);
            setZoom(constrained.zoom);
            setPan(constrained.pan);
        }
    }, [zoom, pan, constrainPanZoom]);

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
