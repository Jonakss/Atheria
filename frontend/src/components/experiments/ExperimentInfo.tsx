// frontend/src/components/ExperimentInfo.tsx
import { Info, Brain, Settings, ArrowRightLeft } from 'lucide-react';
import { useWebSocket } from '../../hooks/useWebSocket';
import { GlassPanel } from '../../modules/Dashboard/components/GlassPanel';

export function ExperimentInfo() {
    const { activeExperiment, experimentsData, compileStatus } = useWebSocket();
    
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
    
    return (
        <GlassPanel className="p-4">
            <div className="space-y-3">
                {/* Header */}
                <div className="flex items-center justify-between">
                    <div className="flex items-center gap-2 flex-wrap">
                        <Brain size={16} className="text-blue-400" />
                        <span className="text-sm font-bold text-blue-400">{activeExperiment}</span>
                        <span className={`px-2 py-0.5 rounded text-[10px] font-bold border ${
                            experiment.has_checkpoint 
                                ? 'bg-emerald-500/10 text-emerald-400 border-emerald-500/30' 
                                : 'bg-amber-500/10 text-amber-400 border-amber-500/30'
                        }`}>
                            {experiment.has_checkpoint ? '✓ Entrenado' : '○ Sin entrenar'}
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
                            <span className="text-sm font-medium text-gray-200">{config.GRID_SIZE_TRAINING || 'N/A'}</span>
                        </div>
                        <div>
                            <span className="text-[10px] text-gray-500 block">Grid Inference</span>
                            <span className="text-sm font-medium text-blue-400">256</span>
                        </div>
                        <div>
                            <span className="text-[10px] text-gray-500 block">Gamma Decay</span>
                            <span className={`text-sm font-medium ${config.GAMMA_DECAY === 0 ? 'text-gray-400' : 'text-amber-400'}`}>
                                {config.GAMMA_DECAY || 0.01}
                                {config.GAMMA_DECAY === 0 ? ' (Cerrado)' : ' (Abierto)'}
                            </span>
                        </div>
                        <div>
                            <span className="text-[10px] text-gray-500 block">Inicialización</span>
                            <span className="text-sm font-medium text-cyan-400">{config.INITIAL_STATE_MODE_INFERENCE || 'complex_noise'}</span>
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
