// frontend/src/components/ui/TimelineViewer.tsx
import { useState, useEffect, useCallback, useMemo, useRef } from 'react';
import { useWebSocket } from '../../hooks/useWebSocket';
import { 
        loadTimeline, 
        getTimelineStats, 
        clearTimeline,
        TimelineFrame 
    } from '../../utils/timelineStorage';
import { Play, Pause, SkipBack, SkipForward, Trash2, Clock, Database, Settings } from 'lucide-react';
import { GlassPanel } from '../../modules/Dashboard/components/GlassPanel';

interface TimelineViewerProps {
    onFrameSelect?: (frame: TimelineFrame | null) => void;
    className?: string;
}

export function TimelineViewer({ onFrameSelect, className = '' }: TimelineViewerProps) {
    const { activeExperiment, inferenceStatus } = useWebSocket();
    const [frames, setFrames] = useState<TimelineFrame[]>([]);
    const [currentFrameIndex, setCurrentFrameIndex] = useState<number>(-1);
    const [isPlaying, setIsPlaying] = useState(false);
    const [playSpeed, setPlaySpeed] = useState(10); // FPS para reproducción
    const [maxFrames, setMaxFrames] = useState(100);
    const [stats, setStats] = useState<{
        total_frames: number;
        min_step: number;
        max_step: number;
        max_frames: number;
    } | null>(null);
    const playIntervalRef = useRef<NodeJS.Timeout | null>(null);

    const loadFrames = useCallback(() => {
        const timeline = loadTimeline(activeExperiment, maxFrames);
        setFrames(timeline.frames);

        const timelineStats = getTimelineStats(activeExperiment);
        if (timelineStats) {
            setStats({
                total_frames: timelineStats.total_frames,
                min_step: timelineStats.min_step,
                max_step: timelineStats.max_step,
                max_frames: timelineStats.max_frames,
            });
        } else {
            setStats(null);
        }

        if (timeline.frames.length > 0) {
            setCurrentFrameIndex(timeline.frames.length - 1);
        } else {
            setCurrentFrameIndex(-1);
        }
    }, [activeExperiment, maxFrames]);

    useEffect(() => {
        loadFrames();
    }, [loadFrames]);

    // Cargar frames desde localStorage
    const loadFrames = useCallback(() => {
        const timeline = loadTimeline(activeExperiment, maxFrames);
        setFrames(timeline.frames);
        
        const timelineStats = getTimelineStats(activeExperiment);
        if (timelineStats) {
            setStats({
                total_frames: timelineStats.total_frames,
                min_step: timelineStats.min_step,
                max_step: timelineStats.max_step,
                max_frames: timelineStats.max_frames,
            });
        } else {
            setStats(null);
        }
        
        // Establecer índice al último frame si hay frames
        if (timeline.frames.length > 0) {
            setCurrentFrameIndex(timeline.frames.length - 1);
        } else {
            setCurrentFrameIndex(-1);
        }
    }, [activeExperiment, maxFrames]);

    // Cargar límite de frames desde localStorage
    useEffect(() => {
        const savedMaxFrames = localStorage.getItem('atheria_timeline_max_frames');
        if (savedMaxFrames) {
            setMaxFrames(parseInt(savedMaxFrames, 10));
        }
    }, []);

    // Reproducción automática
    useEffect(() => {
        if (isPlaying && frames.length > 0 && currentFrameIndex >= 0) {
            const interval = setInterval(() => {
                setCurrentFrameIndex(prev => {
                    const next = prev + 1;
                    if (next >= frames.length) {
                        setIsPlaying(false);
                        return prev;
                    }
                    return next;
                });
            }, 1000 / playSpeed); // Convertir FPS a intervalo en ms
            
            setPlayIntervalRef(interval);
            return () => {
                clearInterval(interval);
                setPlayIntervalRef(null);
            };
        } else if (playIntervalRef) {
            clearInterval(playIntervalRef);
            setPlayIntervalRef(null);
        }
    }, [isPlaying, frames.length, currentFrameIndex, playSpeed]);

    // Si la simulación está corriendo, deseleccionar frame y pausar timeline
    useEffect(() => {
        if (inferenceStatus === 'running') {
            setIsPlaying(false);
            if (currentFrameIndex >= 0 && onFrameSelect) {
                // Deseleccionar frame para mostrar datos en vivo
                onFrameSelect(null);
            }
        }
    }, [inferenceStatus, currentFrameIndex, onFrameSelect]);

    // Notificar frame seleccionado (solo si la simulación está pausada)
    useEffect(() => {
        if (inferenceStatus === 'running') {
            // No seleccionar frames del timeline cuando la simulación está corriendo
            return;
        }
        
        if (currentFrameIndex >= 0 && currentFrameIndex < frames.length) {
            const frame = frames[currentFrameIndex];
            if (onFrameSelect) {
                onFrameSelect(frame);
            }
        } else if (currentFrameIndex === -1 && onFrameSelect) {
            onFrameSelect(null);
        }
    }, [currentFrameIndex, frames, onFrameSelect, inferenceStatus]);

    // Frame actual
    const currentFrame = useMemo(() => {
        if (currentFrameIndex >= 0 && currentFrameIndex < frames.length) {
            return frames[currentFrameIndex];
        }
        return null;
    }, [currentFrameIndex, frames]);

    // Navegación
    const goToFrame = useCallback((index: number) => {
        const clampedIndex = Math.max(0, Math.min(index, frames.length - 1));
        setCurrentFrameIndex(clampedIndex);
        setIsPlaying(false);
    }, [frames.length]);

    const goToStep = useCallback((step: number) => {
        const frameIndex = frames.findIndex(f => f.step === step);
        if (frameIndex >= 0) {
            goToFrame(frameIndex);
        }
    }, [frames, goToFrame]);

    const handlePrevious = useCallback(() => {
        goToFrame(currentFrameIndex - 1);
    }, [currentFrameIndex, goToFrame]);

    const handleNext = useCallback(() => {
        goToFrame(currentFrameIndex + 1);
    }, [currentFrameIndex, goToFrame]);

    const handlePlayPause = useCallback(() => {
        if (frames.length === 0) return;
        setIsPlaying(prev => !prev);
    }, [frames.length]);

    const handleClear = useCallback(() => {
        if (confirm('¿Estás seguro de que quieres limpiar el timeline? Esto eliminará todos los frames guardados.')) {
            clearTimeline(activeExperiment);
            setFrames([]);
            setCurrentFrameIndex(-1);
            setStats(null);
            if (onFrameSelect) {
                onFrameSelect(null);
            }
        }
    }, [activeExperiment, onFrameSelect]);

    const handleMaxFramesChange = useCallback((newMax: number) => {
        setMaxFrames(newMax);
        localStorage.setItem('atheria_timeline_max_frames', newMax.toString());
        loadFrames(); // Recargar con nuevo límite
    }, [loadFrames]);

    if (frames.length === 0) {
        return (
            <GlassPanel className={`p-4 ${className}`}>
                <div className="flex items-center gap-2 text-gray-500 text-sm">
                    <Database size={16} />
                    <span>No hay frames guardados en el timeline</span>
                </div>
            </GlassPanel>
        );
    }

    return (
        <GlassPanel className={`p-4 space-y-3 ${className}`}>
            {/* Header con estadísticas */}
            <div className="flex items-center justify-between">
                <div className="flex items-center gap-2">
                    <Clock size={16} className="text-teal-400" />
                    <span className="text-sm font-bold text-gray-200">Timeline Local</span>
                </div>
                {stats && (
                    <div className="text-xs text-gray-500">
                        {stats.total_frames} frames | Step {stats.min_step} - {stats.max_step}
                    </div>
                )}
            </div>

            {/* Controles de reproducción */}
            <div className="flex items-center gap-2">
                <button
                    onClick={() => goToFrame(0)}
                    disabled={currentFrameIndex === 0}
                    className="p-1.5 bg-white/5 hover:bg-white/10 border border-white/10 rounded disabled:opacity-50 disabled:cursor-not-allowed transition-all"
                    title="Primer frame"
                >
                    <SkipBack size={14} className="text-gray-400" />
                </button>
                
                <button
                    onClick={handlePrevious}
                    disabled={currentFrameIndex <= 0}
                    className="p-1.5 bg-white/5 hover:bg-white/10 border border-white/10 rounded disabled:opacity-50 disabled:cursor-not-allowed transition-all"
                    title="Frame anterior"
                >
                    <SkipBack size={14} className="text-gray-400" />
                </button>
                
                <button
                    onClick={handlePlayPause}
                    disabled={frames.length === 0}
                    className="px-3 py-1.5 bg-teal-500/10 hover:bg-teal-500/20 border border-teal-500/30 text-teal-400 rounded transition-all disabled:opacity-50"
                    title={isPlaying ? "Pausar" : "Reproducir"}
                >
                    {isPlaying ? <Pause size={14} /> : <Play size={14} />}
                </button>
                
                <button
                    onClick={handleNext}
                    disabled={currentFrameIndex >= frames.length - 1}
                    className="p-1.5 bg-white/5 hover:bg-white/10 border border-white/10 rounded disabled:opacity-50 disabled:cursor-not-allowed transition-all"
                    title="Siguiente frame"
                >
                    <SkipForward size={14} className="text-gray-400" />
                </button>
                
                <button
                    onClick={() => goToFrame(frames.length - 1)}
                    disabled={currentFrameIndex >= frames.length - 1}
                    className="p-1.5 bg-white/5 hover:bg-white/10 border border-white/10 rounded disabled:opacity-50 disabled:cursor-not-allowed transition-all"
                    title="Último frame"
                >
                    <SkipForward size={14} className="text-gray-400" />
                </button>

                <div className="flex-1" />
                
                <button
                    onClick={handleClear}
                    className="p-1.5 bg-red-500/10 hover:bg-red-500/20 border border-red-500/30 text-red-400 rounded transition-all"
                    title="Limpiar timeline"
                >
                    <Trash2 size={14} />
                </button>
            </div>

            {/* Slider de navegación */}
            <div className="space-y-1">
                <div className="flex items-center justify-between text-xs text-gray-500">
                    <span>Frame {currentFrameIndex + 1} / {frames.length}</span>
                    {currentFrame && (
                        <span>Step {currentFrame.step}</span>
                    )}
                </div>
                <input
                    type="range"
                    min={0}
                    max={Math.max(0, frames.length - 1)}
                    value={currentFrameIndex}
                    onChange={(e) => goToFrame(parseInt(e.target.value, 10))}
                    className="w-full h-2 bg-white/5 rounded-lg appearance-none cursor-pointer accent-teal-500"
                />
            </div>

            {/* Configuración */}
            <details className="text-xs">
                <summary className="cursor-pointer text-gray-500 hover:text-gray-300 flex items-center gap-1">
                    <Settings size={12} />
                    <span>Configuración</span>
                </summary>
                <div className="mt-2 space-y-2 pt-2 border-t border-white/10">
                    <div className="flex items-center justify-between">
                        <label className="text-gray-400">Velocidad (FPS)</label>
                        <input
                            type="number"
                            min={1}
                            max={60}
                            value={playSpeed}
                            onChange={(e) => setPlaySpeed(Math.max(1, Math.min(60, parseInt(e.target.value, 10) || 10)))}
                            className="w-16 px-2 py-1 bg-white/5 border border-white/10 rounded text-xs text-gray-300"
                        />
                    </div>
                    <div className="flex items-center justify-between">
                        <label className="text-gray-400">Máx. frames</label>
                        <input
                            type="number"
                            min={10}
                            max={1000}
                            value={maxFrames}
                            onChange={(e) => handleMaxFramesChange(Math.max(10, Math.min(1000, parseInt(e.target.value, 10) || 100)))}
                            className="w-16 px-2 py-1 bg-white/5 border border-white/10 rounded text-xs text-gray-300"
                        />
                    </div>
                </div>
            </details>
        </GlassPanel>
    );
}

