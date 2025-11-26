// frontend/src/components/ExperimentInfo.tsx
import { AlertTriangle, ArrowRightLeft, Brain, Info, Settings } from 'lucide-react';
import { useEffect, useState } from 'react';
import { useWebSocket } from '../../hooks/useWebSocket';
import { GlassPanel } from '../../modules/Dashboard/components/GlassPanel';

export function ExperimentInfo() {
    const { activeExperiment, experimentsData, compileStatus } = useWebSocket();
    
    // Obtener gridSizeInference desde localStorage (sincronizado con LabSider)
    const [gridSizeInference, setGridSizeInference] = useState<number>(() => {
        const saved = localStorage.getItem('atheria_gridSizeInference');
        return saved ? parseInt(saved, 10) : 256;
    });
    
    // Escuchar cambios en localStorage para actualizar en tiempo real
    useEffect(() => {
        const handleStorageChange = () => {
            const saved = localStorage.getItem('atheria_gridSizeInference');
            if (saved) {
                setGridSizeInference(parseInt(saved, 10));
            }
        };
        
        window.addEventListener('storage', handleStorageChange);
        return () => window.removeEventListener('storage', handleStorageChange);
    }, []);
    
    if (!activeExperiment) {
        return (
            <GlassPanel className="p-3">
                <div className="flex items-center gap-2">
                    <Info size={14} className="text-gray-500" />
                    <span className="text-xs text-gray-500">No hay experimento seleccionado</span>
                </div>
            </GlassPanel>
        );
    }
    
    const experiment = experimentsData?.find(exp => exp.name === activeExperiment);
    if (!experiment) {
        return (
            <GlassPanel className="p-3">
                <div className="flex items-center gap-2">
                    <Info size={14} className="text-gray-500" />
                    <span className="text-xs text-gray-500">Experimento no encontrado</span>
                </div>
            </GlassPanel>
        );
    }
    
    const config = experiment.config || {};
    const modelParams = config.MODEL_PARAMS || {};
    const trainingGridSize = config.GRID_SIZE_TRAINING || 64;
    const isScaled = trainingGridSize < gridSizeInference;
    
    return (
        <GlassPanel className="p-4">
            <div className="space-y-3">
                {/* Header */}
                <div className="flex items-center justify-between">
                    <div className="flex items-center gap-2 flex-wrap">
                        <Brain size={16} className="text-blue-400" />
                        <span className="text-sm font-bold text-blue-400">{activeExperiment}</span>
                        <span className={`px-2 py-0.5 rounded text-[10px] font-bold border flex items-center gap-1 ${
                            experiment.has_checkpoint 
                                ? 'bg-teal-500/20 text-teal-400 border-teal-500/50 shadow-glow-teal' 
                                : 'bg-pink-500/10 text-pink-400 border-pink-500/30'
                        }`}>
                            {experiment.has_checkpoint ? (
                                <>
                                    <span className="w-1.5 h-1.5 rounded-full bg-teal-400 animate-pulse" />
                                    <span>Checkpoint Disponible</span>
                                </>
                            ) : (
                                <>
                                    <span className="w-1.5 h-1.5 rounded-full bg-pink-400" />
                                    <span>Sin Checkpoint</span>
                                </>
                            )}
                        </span>
                        {compileStatus?.is_native && (
                            <span className="px-2 py-0.5 rounded text-[10px] font-bold border bg-blue-500/10 text-blue-400 border-blue-500/30">
                                ⚡ Nativo (C++)
                            </span>
                        )}
                        {compileStatus && !compileStatus.is_native && (
                            <span className="px-2 py-0.5 rounded text-[10px] font-bold border bg-gray-500/10 text-gray-400 border-gray-500/30">
                                🐍 Python
                            </span>
                        )}
                    </div>
                </div>
                
                <div className="h-px bg-white/10" />
                
                {/* Arquitectura */}
                <div className="space-y-1.5">
                    <div className="flex items-center gap-2">
                        <Settings size={12} className="text-gray-500" />
                        <span className="text-[10px] font-bold text-gray-500 uppercase tracking-wider">Arquitectura</span>
                    </div>
                    <div className="pl-4">
                        <span className="text-sm font-medium text-gray-200">{config.MODEL_ARCHITECTURE || 'N/A'}</span>
                    </div>
                </div>
                
                {/* Hiperparámetros */}
                <div className="space-y-1.5">
                    <span className="text-[10px] font-bold text-gray-500 uppercase tracking-wider">Hiperparámetros</span>
                    <div className="pl-4 grid grid-cols-2 gap-3">
                        <div>
                            <span className="text-[10px] text-gray-500 block">d_state</span>
                            <span className="text-sm font-medium text-gray-200">{modelParams.d_state || config.D_STATE || 'N/A'}</span>
                        </div>
                        <div>
                            <span className="text-[10px] text-gray-500 block">Hidden Channels</span>
                            <span className="text-sm font-medium text-gray-200">{modelParams.hidden_channels || config.HIDDEN_CHANNELS || 'N/A'}</span>
                        </div>
                        <div>
                            <span className="text-[10px] text-gray-500 block">Learning Rate</span>
                            <span className="text-sm font-medium text-gray-200">{config.LR_RATE_M || 'N/A'}</span>
                        </div>
                        <div>
                            <span className="text-[10px] text-gray-500 block">Grid Training</span>
                            <span className="text-sm font-medium text-gray-200">{trainingGridSize}x{trainingGridSize}</span>
                        </div>
                        <div>
                            <span className="text-[10px] text-gray-500 block">Grid Inferencia</span>
                            <span className={`text-sm font-medium ${isScaled ? 'text-pink-400' : 'text-blue-400'}`}>
                                {gridSizeInference}x{gridSizeInference}
                            </span>
                        </div>
                        <div>
                            <span className="text-[10px] text-gray-500 block">Gamma Decay</span>
                            <span className={`text-sm font-medium ${config.GAMMA_DECAY === 0 ? 'text-gray-400' : 'text-pink-400'}`}>
                                {config.GAMMA_DECAY || 0.01}
                                {config.GAMMA_DECAY === 0 ? ' (Cerrado)' : ' (Abierto)'}
                            </span>
                        </div>
                        <div>
                            <span className="text-[10px] text-gray-500 block">Inicialización</span>
                            <span className="text-sm font-medium text-cyan-400">{config.INITIAL_STATE_MODE_INFERENCE || 'complex_noise'}</span>
                        </div>
                    </div>
                    
                    {/* Indicador de Grid Scaling */}
                    {isScaled && (
                        <div className="col-span-2 mt-1 p-2 bg-pink-500/10 border border-pink-500/30 rounded flex items-start gap-2">
                            <AlertTriangle size={14} className="text-pink-400 shrink-0 mt-0.5" />
                            <div>
                                <div className="text-[10px] font-bold text-pink-400 uppercase mb-0.5">Grid Escalado</div>
                                <div className="text-[10px] text-pink-300">
                                    {trainingGridSize}x{trainingGridSize} (original) → {gridSizeInference}x{gridSizeInference} (inferencia)
                                </div>
                                <div className="text-[9px] text-gray-500 mt-1">
                                    El modelo replica el estado entrenado en un grid más grande
                                </div>
                            </div>
                        </div>
                    )}
                </div>
                
                {/* Información del Motor y Dispositivo */}
                <div className="space-y-1.5">
                    <span className="text-[10px] font-bold text-gray-500 uppercase tracking-wider">Motor y Dispositivo</span>
                    <div className="pl-4 grid grid-cols-2 gap-3">
                        <div>
                            <span className="text-[10px] text-gray-500 block">Motor</span>
                            {config.USE_NATIVE_ENGINE ? (
                                <span className="px-2 py-0.5 rounded text-[10px] font-bold border bg-teal-500/10 text-teal-400 border-teal-500/30 inline-block">
                                    ⚡ Nativo (C++)
                                </span>
                            ) : (
                                <span className="px-2 py-0.5 rounded text-[10px] font-bold border bg-teal-500/10 text-teal-300 border-teal-500/30 inline-block">
                                    🐍 Python
                                </span>
                            )}
                        </div>
                        <div>
                            <span className="text-[10px] text-gray-500 block">Dispositivo</span>
                            {(() => {
                                const device = config.TRAINING_DEVICE || 'cpu';
                                const isCuda = device.toLowerCase() === 'cuda';
                                return (
                                    <span className={`px-2 py-0.5 rounded text-[10px] font-bold border inline-block ${
                                        isCuda 
                                            ? 'bg-purple-500/10 text-purple-400 border-purple-500/30' 
                                            : 'bg-gray-500/10 text-gray-400 border-gray-500/30'
                                    }`}>
                                        {isCuda ? '🎮 CUDA (Gráfica)' : '💻 CPU'}
                                    </span>
                                );
                            })()}
                        </div>
                    </div>
                </div>
                
                {/* Entrenamiento */}
                {config.TOTAL_EPISODES && (
                    <div className="space-y-1.5">
                        <span className="text-[10px] font-bold text-gray-500 uppercase tracking-wider">Entrenamiento</span>
                        <div className="pl-4">
                            <span className="text-sm">
                                <span className="text-gray-500">Episodios: </span>
                                <span className="font-medium text-gray-200">{config.TOTAL_EPISODES}</span>
                            </span>
                        </div>
                    </div>
                )}
                
                {/* Transfer Learning */}
                {config.LOAD_FROM_EXPERIMENT && (
                    <div className="space-y-1.5">
                        <div className="flex items-center gap-2">
                            <ArrowRightLeft size={12} className="text-gray-500" />
                            <span className="text-[10px] font-bold text-gray-500 uppercase tracking-wider">Transfer Learning</span>
                        </div>
                        <div className="pl-4">
                            <span className="px-2 py-0.5 rounded text-[10px] font-medium border bg-blue-500/10 text-blue-400 border-blue-500/30">
                                {config.LOAD_FROM_EXPERIMENT}
                            </span>
                        </div>
                    </div>
                )}
            </div>
        </GlassPanel>
    );
}
