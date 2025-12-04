import { useRef, useEffect, useState } from 'react';

interface MinimapProps {
    totalWidth: number;
    totalHeight: number;
    roi: { x: number; y: number; width: number; height: number } | null;
    onROIChange: (x: number, y: number) => void;
    className?: string;
    width?: number;
    height?: number;
}

export function Minimap({
    totalWidth,
    totalHeight,
    roi,
    onROIChange,
    className = '',
    width = 150,
    height = 150
}: MinimapProps) {
    const canvasRef = useRef<HTMLCanvasElement>(null);
    const [isDragging, setIsDragging] = useState(false);

    // Render minimap
    useEffect(() => {
        const canvas = canvasRef.current;
        if (!canvas) return;

        const ctx = canvas.getContext('2d');
        if (!ctx) return;

        // Clear
        ctx.clearRect(0, 0, canvas.width, canvas.height);

        // Draw Universe Background (Dark Void)
        ctx.fillStyle = '#0a0a0a';
        ctx.fillRect(0, 0, canvas.width, canvas.height);

        // Draw Grid hints (optional, faint lines)
        ctx.strokeStyle = '#222';
        ctx.lineWidth = 0.5;
        ctx.beginPath();
        // Crosshair
        ctx.moveTo(canvas.width/2, 0);
        ctx.lineTo(canvas.width/2, canvas.height);
        ctx.moveTo(0, canvas.height/2);
        ctx.lineTo(canvas.width, canvas.height/2);
        ctx.stroke();

        // Draw ROI Indicator
        if (roi && totalWidth > 0 && totalHeight > 0) {
            const scaleX = canvas.width / totalWidth;
            const scaleY = canvas.height / totalHeight;

            const roiX = roi.x * scaleX;
            const roiY = roi.y * scaleY;
            const roiW = Math.max(2, roi.width * scaleX); // Min 2px visibility
            const roiH = Math.max(2, roi.height * scaleY);

            ctx.strokeStyle = '#00ff9d'; // Neon Green
            ctx.lineWidth = 1.5;
            ctx.strokeRect(roiX, roiY, roiW, roiH);

            ctx.fillStyle = 'rgba(0, 255, 157, 0.2)';
            ctx.fillRect(roiX, roiY, roiW, roiH);
        }

    }, [totalWidth, totalHeight, roi]);

    const handleInteraction = (clientX: number, clientY: number) => {
        if (!canvasRef.current || totalWidth <= 0 || totalHeight <= 0) return;

        const rect = canvasRef.current.getBoundingClientRect();
        const x = clientX - rect.left;
        const y = clientY - rect.top;

        // Convert screen coords to grid coords (center of new ROI)
        const scaleX = totalWidth / canvasRef.current.width;
        const scaleY = totalHeight / canvasRef.current.height;

        let targetX = x * scaleX;
        let targetY = y * scaleY;

        // If ROI exists, center it on the click
        if (roi) {
            targetX -= roi.width / 2;
            targetY -= roi.height / 2;

            // Clamp
            targetX = Math.max(0, Math.min(targetX, totalWidth - roi.width));
            targetY = Math.max(0, Math.min(targetY, totalHeight - roi.height));

            onROIChange(Math.floor(targetX), Math.floor(targetY));
        }
    };

    const onMouseDown = (e: React.MouseEvent) => {
        setIsDragging(true);
        handleInteraction(e.clientX, e.clientY);
    };

    const onMouseMove = (e: React.MouseEvent) => {
        if (isDragging) {
            handleInteraction(e.clientX, e.clientY);
        }
    };

    const onMouseUp = () => {
        setIsDragging(false);
    };

    const onMouseLeave = () => {
        setIsDragging(false);
    };

    return (
        <div className={`relative border border-gray-700 bg-black shadow-lg rounded overflow-hidden ${className}`}>
             <div className="absolute top-0 left-0 bg-black/50 text-[10px] text-gray-400 px-1 pointer-events-none">
                MINIMAP
            </div>
            <canvas
                ref={canvasRef}
                width={width}
                height={height}
                className="cursor-crosshair block"
                onMouseDown={onMouseDown}
                onMouseMove={onMouseMove}
                onMouseUp={onMouseUp}
                onMouseLeave={onMouseLeave}
            />
        </div>
    );
}
