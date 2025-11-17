// frontend/src/hooks/usePanZoom.ts
import { useState, useRef, useCallback } from 'react';

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

    // Función para resetear vista
    const resetView = useCallback(() => {
        setPan({ x: 0, y: 0 });
        setZoom(1);
    }, []);

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
