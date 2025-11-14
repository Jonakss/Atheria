// frontend/src/components/PanZoomCanvas.tsx
import { useRef, useEffect } from 'react';
import { useWebSocket } from '../hooks/useWebSocket';
import { usePanZoom } from '../hooks/usePanZoom';
import classes from './PanZoomCanvas.module.css';

function getColor(value: number) {
    const colors = [
        [68, 1, 84], [72, 40, 120], [62, 74, 137], [49, 104, 142],
        [38, 130, 142], [31, 158, 137], [53, 183, 121], [109, 205, 89],
        [180, 222, 44], [253, 231, 37]
    ];
    const i = Math.min(Math.max(Math.floor(value * (colors.length - 1)), 0), colors.length - 1);
    const c = colors[i];
    return `rgb(${c[0]}, ${c[1]}, ${c[2]})`;
}

export function PanZoomCanvas() {
    const canvasRef = useRef<HTMLCanvasElement>(null);
    const { simData, selectedViz } = useWebSocket();
    const { pan, zoom, handleMouseDown, handleMouseMove, handleMouseUp, handleWheel } = usePanZoom(canvasRef);

    useEffect(() => {
        const canvas = canvasRef.current;
        if (!canvas) return;
        const ctx = canvas.getContext('2d');
        if (!ctx) return;

        if (selectedViz === 'poincare') {
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

            coords.forEach(point => {
                const x = centerX + point[0] * radius;
                const y = centerY + point[1] * radius;
                ctx.beginPath();
                ctx.arc(x, y, 1.5, 0, 2 * Math.PI, false);
                ctx.fillStyle = 'rgba(57, 175, 255, 0.7)';
                ctx.fill();
            });

        } else {
            const mapData = simData?.map_data;
            if (!mapData || !Array.isArray(mapData) || mapData.length === 0 || !Array.isArray(mapData[0]) || mapData[0].length === 0) {
                ctx.clearRect(0, 0, canvas.width, canvas.height);
                return;
            }
            
            const gridHeight = mapData.length;
            const gridWidth = mapData[0].length;

            if (canvas.width !== gridWidth || canvas.height !== gridHeight) {
                canvas.width = gridWidth;
                canvas.height = gridHeight;
            }

            ctx.clearRect(0, 0, gridWidth, gridHeight);
            
            for (let y = 0; y < gridHeight; y++) {
                for (let x = 0; x < gridWidth; x++) {
                    const value = mapData[y][x];
                    ctx.fillStyle = getColor(value);
                    ctx.fillRect(x, y, 1, 1);
                }
            }
        }
    }, [simData, selectedViz, pan, zoom]);

    return (
        <div className={classes.canvasContainer}>
            <canvas
                ref={canvasRef}
                className={classes.canvas}
                onMouseDown={handleMouseDown}
                onMouseMove={handleMouseMove}
                onMouseUp={handleMouseUp}
                onMouseLeave={handleMouseUp}
                onWheel={handleWheel}
                style={{ 
                    transform: `scale(${zoom}) translate(${pan.x}px, ${pan.y}px)`,
                    visibility: simData ? 'visible' : 'hidden'
                }}
            />
        </div>
    );
}
