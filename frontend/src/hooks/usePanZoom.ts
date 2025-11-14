// frontend/src/hooks/usePanZoom.ts
import { useState, useRef, useCallback } from 'react';

export const usePanZoom = (canvasRef: React.RefObject<HTMLCanvasElement>) => {
    const [pan, setPan] = useState({ x: 0, y: 0 });
    const [zoom, setZoom] = useState(1);
    const isPanning = useRef(false);
    const lastMousePos = useRef({ x: 0, y: 0 });

    const handleMouseDown = useCallback((e: React.MouseEvent) => {
        isPanning.current = true;
        lastMousePos.current = { x: e.clientX, y: e.clientY };
    }, []);

    const handleMouseMove = useCallback((e: React.MouseEvent) => {
        if (!isPanning.current) return;
        const dx = (e.clientX - lastMousePos.current.x) / zoom;
        const dy = (e.clientY - lastMousePos.current.y) / zoom;
        setPan(prevPan => ({ x: prevPan.x + dx, y: prevPan.y + dy }));
        lastMousePos.current = { x: e.clientX, y: e.clientY };
    }, [zoom]);

    const handleMouseUp = useCallback(() => {
        isPanning.current = false;
    }, []);

    const handleWheel = useCallback((e: React.WheelEvent) => {
        e.preventDefault();
        const zoomFactor = 1.1;
        const newZoom = e.deltaY < 0 ? zoom * zoomFactor : zoom / zoomFactor;
        setZoom(Math.max(0.1, Math.min(newZoom, 20))); // Limitar el zoom
    }, [zoom]);

    return {
        pan,
        zoom,
        handleMouseDown,
        handleMouseMove,
        handleMouseUp,
        handleWheel,
    };
};
