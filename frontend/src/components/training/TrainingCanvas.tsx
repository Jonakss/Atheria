// frontend/src/components/training/TrainingCanvas.tsx
import { useRef, useEffect, useMemo } from 'react';
import { useWebSocket } from '../../hooks/useWebSocket';
import { GlassPanel } from '../../modules/Dashboard/components/GlassPanel';

interface TrainingCanvasProps {
    width?: number;
    height?: number;
}

function getColor(value: number): string {
    if (typeof value !== 'number' || isNaN(value) || !isFinite(value)) {
        return 'rgb(68, 1, 84)';
    }
    
    const colors = [
        [68, 1, 84], [72, 40, 120], [62, 74, 137], [49, 104, 142],
        [38, 130, 142], [31, 158, 137], [53, 183, 121], [109, 205, 89],
        [180, 222, 44], [253, 231, 37], [255, 200, 0], [255, 150, 0],
        [255, 100, 0], [255, 50, 0], [255, 0, 0]
    ];
    
    const normalizedValue = Math.max(0, Math.min(1, value));
    const i = Math.min(Math.max(Math.floor(normalizedValue * (colors.length - 1)), 0), colors.length - 1);
    const c = colors[i];
    
    return `rgb(${c[0]}, ${c[1]}, ${c[2]})`;
}

export function TrainingCanvas({ width = 512, height = 512 }: TrainingCanvasProps) {
    const canvasRef = useRef<HTMLCanvasElement>(null);
    const { trainingSnapshots } = useWebSocket();
    
    // Obtener el último snapshot o el primero disponible
    const currentSnapshot = useMemo(() => {
        if (!trainingSnapshots || trainingSnapshots.length === 0) return null;
        return trainingSnapshots[trainingSnapshots.length - 1];
    }, [trainingSnapshots]);
    
    // Renderizar snapshot en el canvas
    useEffect(() => {
        const canvas = canvasRef.current;
        if (!canvas) return;
        
        const ctx = canvas.getContext('2d');
        if (!ctx) return;
        
        // Limpiar canvas
        ctx.fillStyle = '#020202';
        ctx.fillRect(0, 0, canvas.width, canvas.height);
        
        if (!currentSnapshot || !currentSnapshot.map_data) {
            // Mostrar mensaje de "esperando datos"
            ctx.fillStyle = '#666';
            ctx.font = '14px monospace';
            ctx.textAlign = 'center';
            ctx.textBaseline = 'middle';
            ctx.fillText('Esperando snapshot de entrenamiento...', canvas.width / 2, canvas.height / 2);
            return;
        }
        
        const mapData = currentSnapshot.map_data;
        const gridHeight = mapData.length;
        const gridWidth = mapData[0]?.length || 0;
        
        if (gridWidth === 0 || gridHeight === 0) return;
        
        // Calcular dimensiones de celda
        const cellWidth = canvas.width / gridWidth;
        const cellHeight = canvas.height / gridHeight;
        
        // Renderizar cada celda
        for (let y = 0; y < gridHeight; y++) {
            for (let x = 0; x < gridWidth; x++) {
                const value = mapData[y]?.[x];
                if (value === undefined || value === null) continue;
                
                // Normalizar valor (asumimos que está en [0, 1] o similar)
                const normalizedValue = Math.max(0, Math.min(1, typeof value === 'number' ? value : 0));
                const color = getColor(normalizedValue);
                
                ctx.fillStyle = color;
                ctx.fillRect(
                    x * cellWidth,
                    y * cellHeight,
                    Math.ceil(cellWidth) + 1, // +1 para evitar gaps
                    Math.ceil(cellHeight) + 1
                );
            }
        }
        
        // Dibujar overlay con información del snapshot
        if (currentSnapshot.episode !== undefined || currentSnapshot.step !== undefined) {
            ctx.fillStyle = 'rgba(0, 0, 0, 0.7)';
            ctx.fillRect(0, 0, 150, 40);
            ctx.fillStyle = '#fff';
            ctx.font = '12px monospace';
            ctx.textAlign = 'left';
            ctx.textBaseline = 'top';
            if (currentSnapshot.episode !== undefined) {
                ctx.fillText(`Episodio: ${currentSnapshot.episode}`, 5, 5);
            }
            if (currentSnapshot.step !== undefined) {
                ctx.fillText(`Paso: ${currentSnapshot.step}`, 5, 22);
            }
        }
    }, [currentSnapshot, width, height]);
    
    return (
        <GlassPanel className="relative overflow-hidden">
            <canvas
                ref={canvasRef}
                width={width}
                height={height}
                className="w-full h-full block"
                style={{ 
                    imageRendering: 'pixelated',
                    backgroundColor: '#020202'
                }}
            />
            
            {/* Controles de navegación de snapshots */}
            {trainingSnapshots && trainingSnapshots.length > 1 && (
                <div className="absolute bottom-2 left-1/2 -translate-x-1/2 flex items-center gap-2 px-3 py-1.5 bg-black/70 backdrop-blur-sm rounded border border-white/10">
                    <button
                        onClick={() => {
                            // TODO: Navegar al snapshot anterior
                        }}
                        className="p-1 text-gray-400 hover:text-gray-200 hover:bg-white/10 rounded transition-all"
                        title="Anterior"
                    >
                        ←
                    </button>
                    <span className="text-xs text-gray-300 font-mono">
                        {trainingSnapshots.findIndex(s => s === currentSnapshot) + 1} / {trainingSnapshots.length}
                    </span>
                    <button
                        onClick={() => {
                            // TODO: Navegar al siguiente snapshot
                        }}
                        className="p-1 text-gray-400 hover:text-gray-200 hover:bg-white/10 rounded transition-all"
                        title="Siguiente"
                    >
                        →
                    </button>
                </div>
            )}
            
            {!currentSnapshot && (
                <div className="absolute inset-0 flex items-center justify-center">
                    <div className="text-center">
                        <div className="text-sm text-gray-500 mb-2">📊</div>
                        <div className="text-xs text-gray-600">No hay snapshots disponibles</div>
                    </div>
                </div>
            )}
        </GlassPanel>
    );
}

