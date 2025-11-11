import React, { useRef, useEffect, useCallback } from 'react';

interface PanZoomCanvasProps {
  imageData: string | null;
  sendCommand: (cmd: any) => void;
}

const PanZoomCanvas: React.FC<PanZoomCanvasProps> = ({ imageData, sendCommand }) => {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const viewport = useRef({ x: 0, y: 0, width: 1, height: 1 });
  const panning = useRef({ active: false, startX: 0, startY: 0 });
  const ZOOM_SENSITIVITY = 0.001;

  const sendViewportChange = useCallback(() => {
    sendCommand({ scope: 'sim', command: 'set_viewport', args: { viewport: viewport.current } });
  }, [sendCommand]);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    // Set fixed internal resolution for the canvas
    canvas.width = 512; // Assuming a default fixed size, can be made configurable
    canvas.height = 512; // Assuming a default fixed size, can be made configurable

    const img = new window.Image();
    img.onload = () => {
      ctx.clearRect(0, 0, canvas.width, canvas.height);
      // Draw the image, scaling it to fit the fixed canvas dimensions
      ctx.drawImage(img, 0, 0, canvas.width, canvas.height);
    };
    if (imageData) {
      img.src = imageData;
    } else {
      ctx.clearRect(0, 0, canvas.width, canvas.height);
    }
  }, [imageData]);

  const onMouseDown = useCallback((e: React.MouseEvent) => {
    panning.current.active = true;
    panning.current.startX = e.clientX;
    panning.current.startY = e.clientY;
  }, []);

  const onMouseMove = useCallback((e: React.MouseEvent) => {
    if (!panning.current.active) return;
    const canvas = canvasRef.current;
    if (!canvas) return;

    const dx = e.clientX - panning.current.startX;
    const dy = e.clientY - panning.current.startY;
    viewport.current.x -= (dx / canvas.clientWidth) * viewport.current.width;
    viewport.current.y -= (dy / canvas.clientHeight) * viewport.current.height;
    panning.current.startX = e.clientX;
    panning.current.startY = e.clientY;
    sendViewportChange();
  }, [sendViewportChange]);

  const onMouseUp = useCallback(() => { panning.current.active = false; }, []);
  const onMouseLeave = useCallback(() => { panning.current.active = false; }, []);

  const onWheel = useCallback((e: React.WheelEvent) => {
    e.preventDefault();
    const canvas = canvasRef.current;
    if (!canvas) return;

    const rect = canvas.getBoundingClientRect();
    const mouseX = e.clientX - rect.left;
    const mouseY = e.clientY - rect.top;
    const mouseNormX = mouseX / canvas.clientWidth;
    const mouseNormY = mouseY / canvas.clientHeight;

    const zoomAmount = e.deltaY * ZOOM_SENSITIVITY;
    const zoomFactor = 1 + zoomAmount;
    
    const newWidth = viewport.current.width * zoomFactor;
    const newHeight = viewport.current.height * zoomFactor;

    if (newWidth < 0.001 || newWidth > 50) {
        return;
    }

    const focalX = viewport.current.x + mouseNormX * viewport.current.width;
    const focalY = viewport.current.y + mouseNormY * viewport.current.height;

    viewport.current.width = newWidth;
    viewport.current.height = newHeight;
    viewport.current.x = focalX - mouseNormX * newWidth;
    viewport.current.y = focalY - mouseNormY * newHeight;

    sendViewportChange();
  }, [sendViewportChange]);

  return (
    <canvas
      ref={canvasRef}
      style={{ width: '100%', height: 'auto', maxHeight: '85vh', aspectRatio: '1 / 1', background: '#000', cursor: panning.current.active ? 'grabbing' : 'grab' }}
      onMouseDown={onMouseDown}
      onMouseMove={onMouseMove}
      onMouseUp={onMouseUp}
      onMouseLeave={onMouseLeave}
      onWheel={onWheel}
    />
  );
};

export default PanZoomCanvas;