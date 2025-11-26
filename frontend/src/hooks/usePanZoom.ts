// frontend/src/hooks/usePanZoom.ts
import { useCallback, useEffect, useRef, useState } from "react";

// Throttle para optimizar el pan (evitar demasiadas actualizaciones)
function throttle<T extends (...args: any[]) => void>(
  func: T,
  limit: number
): T {
  let inThrottle: boolean;
  return ((...args: any[]) => {
    if (!inThrottle) {
      func(...args);
      inThrottle = true;
      setTimeout(() => (inThrottle = false), limit);
    }
  }) as T;
}

export const usePanZoom = (
  canvasRef: React.RefObject<HTMLCanvasElement>,
  gridWidth?: number,
  gridHeight?: number,
  toroidalMode: boolean = false
) => {
  const [pan, setPan] = useState({ x: 0, y: 0 });
  const [zoom, setZoom] = useState(1);
  const [isPanning, setIsPanning] = useState(false);
  const [isZooming, setIsZooming] = useState(false);
  const [zoomCenter, setZoomCenter] = useState<{ x: number; y: number } | null>(
    null
  );
  const isPanningRef = useRef(false);
  const lastMousePos = useRef({ x: 0, y: 0 });
  const panUpdateRef = useRef<{ x: number; y: number }>({ x: 0, y: 0 });
  const initializedRef = useRef(false);
  const lastGridSizeRef = useRef<{ width: number; height: number } | null>(
    null
  );
  const zoomTimeoutRef = useRef<NodeJS.Timeout | null>(null);

  // Función para calcular zoom y pan inicial centrado en el centro del grid
  const calculateInitialView = useCallback(() => {
    // SIMPLIFICADO: Siempre empezar centrado con zoom 1 y pan 0
    // Esto es consistente, predecible, y evita problemas de timing con dimensiones del contenedor
    // El usuario puede hacer zoom out con la rueda del ratón si necesita ver más
    return { pan: { x: 0, y: 0 }, zoom: 1 };
  }, []);

  // Inicializar vista cuando cambia el tamaño del grid
  useEffect(() => {
    if (!gridWidth || !gridHeight) return;

    const currentGridSize = { width: gridWidth, height: gridHeight };
    const lastGridSize = lastGridSizeRef.current;

    // Solo inicializar si es la primera vez o si el tamaño del grid cambió
    if (
      !initializedRef.current ||
      !lastGridSize ||
      lastGridSize.width !== currentGridSize.width ||
      lastGridSize.height !== currentGridSize.height
    ) {
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
  const constrainPanZoom = useCallback(
    (newPan: { x: number; y: number }, newZoom: number) => {
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

      const absoluteMinZoom = 0.1;
      const finalMinZoom = absoluteMinZoom;
      const maxZoom = 10;
      const constrainedZoom = Math.max(finalMinZoom, Math.min(newZoom, maxZoom));
      const scaledWidth = gridWidth * constrainedZoom;
      const scaledHeight = gridHeight * constrainedZoom;

      let maxPanX, maxPanY, minPanX, minPanY;

      if (toroidalMode) {
        const veryLargeValue = 1e6;
        maxPanX = veryLargeValue;
        minPanX = -veryLargeValue;
        maxPanY = veryLargeValue;
        minPanY = -veryLargeValue;
      } else {
        if (scaledWidth > containerWidth) {
          const extraMargin = scaledWidth * 0.5;
          maxPanX = (scaledWidth - containerWidth) / 2 + extraMargin;
          minPanX = -maxPanX;
        } else {
          const extraMargin = scaledWidth * 2;
          maxPanX = (containerWidth - scaledWidth) / 2 + extraMargin;
          minPanX = -maxPanX;
        }

        if (scaledHeight > containerHeight) {
          const extraMargin = scaledHeight * 0.5;
          maxPanY = (scaledHeight - containerHeight) / 2 + extraMargin;
          minPanY = -maxPanY;
        } else {
          const extraMargin = scaledHeight * 2;
          maxPanY = (containerHeight - scaledHeight) / 2 + extraMargin;
          minPanY = -maxPanY;
        }
      }

      const constrainedPan = {
        x: Math.max(minPanX, Math.min(newPan.x, maxPanX)),
        y: Math.max(minPanY, Math.min(newPan.y, maxPanY)),
      };

      return { pan: constrainedPan, zoom: constrainedZoom };
    },
    [canvasRef, gridWidth, gridHeight, toroidalMode]
  );

  const handleMouseDown = useCallback(
    (e: React.MouseEvent) => {
      isPanningRef.current = true;
      setIsPanning(true);
      lastMousePos.current = { x: e.clientX, y: e.clientY };
      panUpdateRef.current = { ...pan };
    },
    [pan]
  );

  const handleMouseMove = useCallback(
    throttle((e: React.MouseEvent) => {
      if (!isPanningRef.current) return;
      const dx = e.clientX - lastMousePos.current.x;
      const dy = e.clientY - lastMousePos.current.y;
      const newPan = {
        x: panUpdateRef.current.x + dx,
        y: panUpdateRef.current.y + dy,
      };
      const constrained = constrainPanZoom(newPan, zoom);
      panUpdateRef.current = constrained.pan;
      setPan(constrained.pan);
      lastMousePos.current = { x: e.clientX, y: e.clientY };
    }, 16),
    [zoom, constrainPanZoom]
  );

  const handleMouseUp = useCallback(() => {
    isPanningRef.current = false;
    setIsPanning(false);
  }, []);

  const handleWheel = useCallback(
    (e: React.WheelEvent) => {
      if (e.deltaY !== 0) {
        e.preventDefault();

        // Calcular el punto del mouse en coordenadas del canvas antes del zoom
        if (!gridWidth || !gridHeight) return;

        // Obtener el contenedor (puede ser el div que envuelve el canvas o el canvas mismo)
        let container: HTMLElement | null = null;
        if (canvasRef.current) {
          container = canvasRef.current.parentElement;
        }

        // Si no hay contenedor, usar el target del evento
        if (!container && e.target) {
          container = (e.target as HTMLElement).parentElement;
        }

        // Si todavía no hay contenedor, usar el elemento del evento
        if (!container && e.currentTarget) {
          container = e.currentTarget as HTMLElement;
        }

        if (!container) {
          console.warn("⚠️ No se pudo obtener contenedor para zoom");
          return;
        }

        const containerRect = container.getBoundingClientRect();
        const containerWidth = containerRect.width;
        const containerHeight = containerRect.height;

        // Coordenadas del mouse relativas al contenedor
        const mouseX = e.clientX - containerRect.left;
        const mouseY = e.clientY - containerRect.top;

        // Mostrar overlay de zoom centrado en el mouse
        setIsZooming(true);
        setZoomCenter({ x: mouseX, y: mouseY });

        // Limpiar timeout anterior
        if (zoomTimeoutRef.current) {
          clearTimeout(zoomTimeoutRef.current);
        }

        // Ocultar overlay después de 300ms sin zoom
        zoomTimeoutRef.current = setTimeout(() => {
          setIsZooming(false);
          setZoomCenter(null);
        }, 300);

        // Coordenadas del mouse relativas al centro del contenedor
        const mouseRelToCenterX = mouseX - containerWidth / 2;
        const mouseRelToCenterY = mouseY - containerHeight / 2;

        // Aplicar zoom con factor más suave y progresivo
        const zoomFactor = 1.1; // Más suave para mejor control
        const delta = Math.abs(e.deltaY);
        const smoothFactor = Math.min(1 + (delta / 100) * 0.05, 1.2); // Factor suave basado en velocidad
        const newZoom =
          e.deltaY < 0
            ? zoom * zoomFactor * smoothFactor
            : zoom / (zoomFactor * smoothFactor);

        // Usar constrainPanZoom para aplicar límites (incluye minZoom ajustado)
        const constrainedZoom = constrainPanZoom(pan, newZoom).zoom;

        // Calcular nuevo pan para mantener el punto del canvas fijo bajo el mouse
        const zoomRatio = constrainedZoom / zoom;
        const newPanX = mouseRelToCenterX * (1 - zoomRatio) + pan.x * zoomRatio;
        const newPanY = mouseRelToCenterY * (1 - zoomRatio) + pan.y * zoomRatio;

        const constrainedPanZoomResult = constrainPanZoom(
          { x: newPanX, y: newPanY },
          constrainedZoom
        );
        setZoom(constrainedPanZoomResult.zoom);
        setPan(constrainedPanZoomResult.pan);
      }
    },
    [zoom, pan, constrainPanZoom, canvasRef, gridWidth, gridHeight]
  );

  // Cleanup timeout al desmontar
  useEffect(() => {
    return () => {
      if (zoomTimeoutRef.current) {
        clearTimeout(zoomTimeoutRef.current);
      }
    };
  }, []);

  return {
    pan,
    zoom,
    handleMouseDown,
    handleMouseMove,
    handleMouseUp,
    handleWheel,
    resetView,
    isPanning,
    isZooming,
    zoomCenter,
  };
};
