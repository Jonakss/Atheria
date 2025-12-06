import { Badge } from "@/components/ui/badge";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { useEffect, useMemo, useRef } from 'react';

interface DataPoint {
    x: number;
    y: number;
    step: number;
}

interface AnalysisPanelProps {
    data: DataPoint[];
    width?: number;
    height?: number;
    className?: string;
}

export function AnalysisPanel({ data, width = 300, height = 300, className }: AnalysisPanelProps) {
    const canvasRef = useRef<HTMLCanvasElement>(null);

    // Calcular límites para escalar
    const bounds = useMemo(() => {
        if (!data || data.length === 0) return { minX: -1, maxX: 1, minY: -1, maxY: 1 };
        let minX = Infinity, maxX = -Infinity, minY = Infinity, maxY = -Infinity;
        data.forEach(p => {
            if (p.x < minX) minX = p.x;
            if (p.x > maxX) maxX = p.x;
            if (p.y < minY) minY = p.y;
            if (p.y > maxY) maxY = p.y;
        });
        // Padding
        const padding = 0.5;
        return { minX: minX - padding, maxX: maxX + padding, minY: minY - padding, maxY: maxY + padding };
    }, [data]);

    useEffect(() => {
        const canvas = canvasRef.current;
        if (!canvas) return;
        const ctx = canvas.getContext('2d');
        if (!ctx) return;

        // Clear
        ctx.clearRect(0, 0, width, height);
        
        // Background grid
        ctx.strokeStyle = '#333';
        ctx.lineWidth = 0.5;
        ctx.beginPath();
        for(let i=0; i<width; i+=50) { ctx.moveTo(i,0); ctx.lineTo(i,height); }
        for(let j=0; j<height; j+=50) { ctx.moveTo(0,j); ctx.lineTo(width,j); }
        ctx.stroke();

        if (!data || data.length === 0) {
            ctx.fillStyle = '#666';
            ctx.font = '12px sans-serif';
            ctx.fillText("Waiting for analysis data...", 20, height/2);
            return;
        }

        // Helper to transform coords
        const transform = (x: number, y: number) => {
            const rangeX = bounds.maxX - bounds.minX || 1;
            const rangeY = bounds.maxY - bounds.minY || 1;
            const px = ((x - bounds.minX) / rangeX) * width;
            const py = height - ((y - bounds.minY) / rangeY) * height; // Invert Y
            return { x: px, y: py };
        };

        // Draw trajectory
        ctx.strokeStyle = 'cyan';
        ctx.lineWidth = 1.5;
        ctx.beginPath();
        if (data.length > 0) {
            const start = transform(data[0].x, data[0].y);
            ctx.moveTo(start.x, start.y);
            for(let i=1; i<data.length; i++) {
                const p = transform(data[i].x, data[i].y);
                ctx.lineTo(p.x, p.y);
            }
        }
        ctx.stroke();

        // Draw points (latest is brighter)
        data.forEach((p, i) => {
            const pos = transform(p.x, p.y);
            const alpha = 0.2 + (0.8 * (i / data.length));
            
            ctx.fillStyle = `rgba(0, 255, 255, ${alpha})`;
            ctx.beginPath();
            ctx.arc(pos.x, pos.y, i === data.length - 1 ? 4 : 2, 0, Math.PI * 2);
            ctx.fill();
        });

    }, [data, width, height, bounds]);

    return (
        <Card className={`bg-black/40 border-slate-800 ${className}`}>
            <CardHeader className="pb-2">
                <CardTitle className="text-sm font-medium flex justify-between items-center text-slate-200">
                    <span>State Space (UMAP)</span>
                    <Badge variant="outline" className="text-xs">{data ? data.length : 0} pts</Badge>
                </CardTitle>
            </CardHeader>
            <CardContent>
                <div className="rounded border border-slate-700 bg-black overflow-hidden relative" style={{ width, height }}>
                    <canvas 
                        ref={canvasRef} 
                        width={width} 
                        height={height}
                        className="block"
                    />
                </div>
                <div className="text-[10px] text-slate-500 mt-2 flex justify-between">
                    <span>X: [{bounds.minX.toFixed(1)}, {bounds.maxX.toFixed(1)}]</span>
                    <span>Y: [{bounds.minY.toFixed(1)}, {bounds.maxY.toFixed(1)}]</span>
                </div>
            </CardContent>
        </Card>
    );
}
