import React, { useMemo, useState, useEffect } from 'react';
import { Play, Pause, RefreshCw, Eye, EyeOff, ChevronDown, ChevronUp, Clock, Save } from 'lucide-react';
import { GlassPanel } from './GlassPanel';
import { useWebSocket } from '../../../hooks/useWebSocket';

interface ToolbarProps {
  onToggleTimeline?: () => void;
  timelineOpen?: boolean;
}

import { useCallback } from 'react';

export const Toolbar: React.FC<ToolbarProps> = ({ onToggleTimeline, timelineOpen = false }) => {
  const { 
    inferenceStatus, 
    sendCommand, 
    simData, 
    connectionStatus, 
    activeExperiment, 
    experimentsData,
    liveFeedEnabled,
    setLiveFeedEnabled
  } = useWebSocket();
  const isPlaying = inferenceStatus === 'running';
  const isConnected = connectionStatus === 'connected';

  const [stepsInterval, setStepsInterval] = useState<number>(() => {
    const saved = localStorage.getItem('atheria_steps_interval');
    return saved ? parseInt(saved, 10) : 10;
  });
  const [showIntervalControl, setShowIntervalControl] = useState(false);

  useEffect(() => {
    localStorage.setItem('atheria_steps_interval', stepsInterval.toString());
    if (isConnected && !liveFeedEnabled) {
      sendCommand('simulation', 'set_steps_interval', { steps_interval: stepsInterval });
    }
  }, [stepsInterval, isConnected, liveFeedEnabled, sendCommand]);

  const hasActiveExperiment = useMemo(() => {
    if (!activeExperiment || !experimentsData) return false;
    const currentExperiment = experimentsData.find(exp => exp.name === activeExperiment);
    return currentExperiment?.has_checkpoint || false;
  }, [activeExperiment, experimentsData]);

  const canControlInference = useMemo(() => isConnected && (
    hasActiveExperiment || 
    inferenceStatus === 'running' || 
    (inferenceStatus === 'paused' && activeExperiment !== null)
  ), [isConnected, hasActiveExperiment, inferenceStatus, activeExperiment]);
  
  const currentStep = isConnected ? (simData?.step ?? simData?.simulation_info?.step ?? 0) : 0;
  const fps = isConnected ? (simData?.simulation_info?.fps ?? 0) : null;
  
  const particleCount = useMemo(() => {
    if (!isConnected || !simData?.map_data) return null;
    let count = 0;
    for (const row of simData.map_data) {
      if (Array.isArray(row)) {
        for (const val of row) {
          if (typeof val === 'number' && !isNaN(val) && val > 0.01) count++;
        }
      }
    }
    return count > 1000 ? `${(count / 1000).toFixed(1)}K` : count.toString();
  }, [simData?.map_data, isConnected]);

  const handlePlayPause = useCallback(() => {
    if (!canControlInference) return;
    const command = isPlaying ? 'pause' : 'play';
    sendCommand('inference', command);
  }, [canControlInference, isPlaying, sendCommand]);

  const handleReset = useCallback(() => {
    if (!canControlInference) return;
    sendCommand('inference', 'reset');
  }, [canControlInference, sendCommand]);

  const handleSaveSnapshot = useCallback(() => {
    if (!canControlInference) return;
    sendCommand('snapshot', 'save_snapshot');
  }, [canControlInference, sendCommand]);

  const handleToggleLiveFeed = useCallback(() => {
    if (!isConnected) return;
    setLiveFeedEnabled(!liveFeedEnabled);
  }, [isConnected, setLiveFeedEnabled, liveFeedEnabled]);

  return (
    <div className="absolute top-4 left-4 z-30 flex gap-2 pointer-events-none">
      {/* Botón Timeline */}
      {onToggleTimeline && (
        <GlassPanel className="pointer-events-auto">
          <button
            onClick={onToggleTimeline}
            className={`px-3 py-1.5 rounded text-xs font-bold flex items-center gap-2 transition-all border ${
              timelineOpen
                ? 'bg-teal-500/10 text-teal-400 border-teal-500/30 hover:bg-teal-500/20'
                : 'bg-white/5 text-gray-400 border-white/10 hover:bg-white/10'
            }`}
            title="Abrir/Cerrar Timeline Local"
          >
            <Clock size={12} />
            <span>TIMELINE</span>
          </button>
        </GlassPanel>
      )}
      {/* Panel Principal: Play/Pause + Reset + Paso Actual - según mockup */}
      <GlassPanel className="pointer-events-auto flex items-center p-1 gap-1">
        <button 
          onClick={handlePlayPause}
          disabled={!canControlInference}
          className={`px-4 py-1.5 rounded text-xs font-bold flex items-center gap-2 transition-all border ${
            !canControlInference
              ? 'bg-white/5 text-gray-600 border-white/5 cursor-not-allowed opacity-50'
              : isPlaying 
                ? 'bg-pink-500/10 text-pink-500 border-pink-500/30 hover:bg-pink-500/20' 
                : 'bg-teal-500/10 text-teal-400 border-teal-500/30 hover:bg-teal-500/20'
          }`}
          title={canControlInference ? (isPlaying ? 'Pausar simulación' : 'Iniciar simulación') : 'Necesitas un experimento con checkpoint activo'}
        >
          {isPlaying ? <Pause size={12} fill="currentColor" /> : <Play size={12} fill="currentColor" />}
          {isPlaying ? 'PAUSAR' : 'EJECUTAR'}
        </button>
        
        <button 
          onClick={handleReset}
          disabled={!canControlInference}
          className={`px-3 py-1.5 rounded text-xs font-bold flex items-center gap-1.5 transition-all border ${
            !canControlInference
              ? 'bg-white/5 text-gray-600 border-white/5 cursor-not-allowed opacity-50'
              : 'bg-white/5 text-gray-300 border-white/10 hover:bg-white/10'
          }`}
          title={canControlInference ? 'Reiniciar simulación' : 'Necesitas un experimento con checkpoint activo'}
        >
          <RefreshCw size={12} />
        </button>

        <button
          onClick={handleSaveSnapshot}
          disabled={!canControlInference}
          className={`px-3 py-1.5 rounded text-xs font-bold flex items-center gap-1.5 transition-all border ${
            !canControlInference
              ? 'bg-white/5 text-gray-600 border-white/5 cursor-not-allowed opacity-50'
              : 'bg-white/5 text-blue-400 border-white/10 hover:bg-white/10'
          }`}
          title={canControlInference ? 'Guardar Snapshot' : 'Necesitas una simulación activa para guardar'}
        >
          <Save size={12} />
        </button>
        
        <div className="w-px h-4 bg-white/10 mx-2" />
        
        <div 
          className="flex flex-col px-2 cursor-help"
          title={(() => {
            const initialStep = simData?.simulation_info?.initial_step ?? 0;
            const checkpointStep = simData?.simulation_info?.checkpoint_step ?? 0;
            const checkpointEpisode = simData?.simulation_info?.checkpoint_episode ?? 0;
            if (checkpointStep > 0) {
              return `Iniciado desde paso ${checkpointStep.toLocaleString()} (checkpoint episodio ${checkpointEpisode})`;
            }
            return initialStep > 0 ? `Iniciado desde paso ${initialStep.toLocaleString()}` : 'Paso inicial: 0';
          })()}
        >
          <span className="text-[8px] text-gray-500 uppercase font-bold">Paso Actual</span>
          <span className="text-xs font-mono text-gray-200">
            {(() => {
              const initialStep = simData?.simulation_info?.initial_step ?? 0;
              const totalSteps = currentStep || 0;
              
              // Solo calcular pasos relativos si hay datos válidos
              if (initialStep > 0 && totalSteps >= initialStep) {
                const relativeSteps = totalSteps - initialStep;
                return (
                  <>
                    <span>{totalSteps.toLocaleString()}</span>
                    <span className="text-gray-500 mx-1">-</span>
                    <span className="text-teal-400">{relativeSteps.toLocaleString()}</span>
                  </>
                );
              } else if (initialStep > 0 && totalSteps < initialStep) {
                // Si el step total es menor que el inicial, puede ser que aún no hay datos
                // Mostrar solo el total o indicar que está cargando
                if (totalSteps === 0) {
                  return <span className="text-gray-500">N/A</span>;
                }
                // Si hay un step pero es menor que el inicial, puede ser un estado intermedio
                return (
                  <>
                    <span>{totalSteps.toLocaleString()}</span>
                    <span className="text-gray-500 mx-1">-</span>
                    <span className="text-gray-500">cargando...</span>
                  </>
                );
              }
              // Sin step inicial, mostrar solo el total
              return totalSteps > 0 ? totalSteps.toLocaleString() : <span className="text-gray-500">N/A</span>;
            })()}
          </span>
        </div>
      </GlassPanel>

      {/* Panel Secundario: FPS + Partículas + Live Feed - según mockup */}
      <GlassPanel className="pointer-events-auto flex items-center px-3 py-1 gap-3">
        <div className="flex items-center gap-2">
          <span className="text-[10px] font-bold text-gray-500">FPS</span>
          <span className={`text-xs font-mono ${fps !== null && fps > 0 ? 'text-teal-400' : 'text-gray-600'}`}>
            {fps !== null && fps > 0 ? fps.toFixed(1) : 'N/A'}
          </span>
        </div>
        <div className="w-px h-3 bg-white/10" />
        <div className="flex items-center gap-2">
          <span className="text-[10px] font-bold text-gray-500">PARTÍCULAS</span>
          <span className={`text-xs font-mono ${particleCount !== null ? 'text-blue-400' : 'text-gray-600'}`}>
            {particleCount !== null ? particleCount : 'N/A'}
          </span>
        </div>
        <div className="w-px h-3 bg-white/10" />
        <div className="relative flex items-center gap-1">
          <button
            onClick={handleToggleLiveFeed}
            disabled={!isConnected || !canControlInference}
            className={`flex items-center gap-1.5 px-2 py-1 rounded text-[10px] font-bold border transition-all ${
              !isConnected || !canControlInference
                ? 'bg-white/5 text-gray-600 border-white/5 cursor-not-allowed opacity-50'
                : liveFeedEnabled
                  ? 'bg-blue-500/10 text-blue-400 border-blue-500/30 hover:bg-blue-500/20'
                  : 'bg-gray-800 text-gray-500 border-gray-700 hover:bg-gray-700'
            }`}
            title={
              liveFeedEnabled 
                ? 'Live Feed ON - Desactivar para acelerar simulación' 
                : stepsInterval === -1
                  ? 'Live Feed OFF - Modo FULLSPEED: máxima velocidad sin frames'
                  : stepsInterval === 0
                    ? 'Live Feed OFF - Modo MANUAL: actualizar con botón'
                    : `Live Feed OFF - Mostrando cada ${stepsInterval.toLocaleString()} pasos`
            }
          >
            {liveFeedEnabled ? <Eye size={12} /> : <EyeOff size={12} />}
            <span>{liveFeedEnabled ? 'LIVE' : 'OFF'}</span>
          </button>
          
          {!liveFeedEnabled && (
            <>
              <div className="flex items-center gap-0.5 px-1.5 py-0.5 bg-white/5 border border-white/10 rounded text-[9px] font-mono text-gray-400">
                {stepsInterval === -1 ? (
                  <>
                    <span className="text-red-400 font-bold">FULLSPEED</span>
                  </>
                ) : stepsInterval === 0 ? (
                  <>
                    <span className="text-amber-400 font-bold">MANUAL</span>
                  </>
                ) : (
                  <>
                    <span>cada</span>
                    <span className="text-blue-400 font-bold">{stepsInterval.toLocaleString()}</span>
                    <span>pasos</span>
                  </>
                )}
              </div>
              <button
                onClick={() => setShowIntervalControl(!showIntervalControl)}
                disabled={!isConnected || !canControlInference}
                className="p-0.5 text-gray-500 hover:text-gray-300 transition-colors disabled:opacity-50"
                title="Configurar intervalo de pasos"
              >
                {showIntervalControl ? <ChevronUp size={10} /> : <ChevronDown size={10} />}
              </button>
            </>
          )}
          
          {/* Control de intervalo (dropdown) */}
          {showIntervalControl && !liveFeedEnabled && (
            <div className="absolute top-full left-0 mt-1 z-50">
              <GlassPanel className="p-2 shadow-xl border border-white/20 min-w-[180px]">
                <div className="space-y-2">
                  <div className="text-[10px] font-bold text-gray-400 uppercase tracking-wider">
                    Intervalo de Pasos
                  </div>
                  <div className="flex items-center gap-2">
                    <label className="text-[10px] text-gray-500">Cada</label>
                    <input
                      type="number"
                      min={-1}
                      max={1000000}
                      step={1}
                      value={stepsInterval}
                      onChange={(e) => {
                        const val = parseInt(e.target.value, 10);
                        if (!isNaN(val) && val >= -1 && val <= 1000000) {
                          setStepsInterval(val);
                        }
                      }}
                      className="w-20 px-2 py-1 bg-white/5 border border-white/10 rounded text-xs font-mono text-gray-300 focus:outline-none focus:border-blue-500/50"
                    />
                    <span className="text-[10px] text-gray-500">pasos</span>
                  </div>
                  <div className="flex gap-1 flex-wrap">
                    {[-1, 0, 1, 5, 10, 25, 50, 100, 500, 1000, 5000, 10000].map(val => (
                      <button
                        key={val}
                        onClick={() => setStepsInterval(val)}
                        className={`px-2 py-0.5 text-[9px] font-mono rounded border transition-all ${
                          stepsInterval === val
                            ? 'bg-blue-500/20 text-blue-400 border-blue-500/30'
                            : 'bg-white/5 text-gray-500 border-white/10 hover:bg-white/10'
                        }`}
                      >
                        {val}
                      </button>
                    ))}
                  </div>
                  <div className="text-[9px] text-gray-600 pt-1 border-t border-white/10">
                    {stepsInterval === -1 
                      ? 'Modo fullspeed: simulación a máxima velocidad sin enviar frames'
                      : stepsInterval === 0 
                        ? 'Modo manual: frames solo con botón de actualización'
                        : `Mostrar frame cada ${stepsInterval.toLocaleString()} pasos cuando live feed está desactivado`}
                  </div>
                </div>
              </GlassPanel>
            </div>
          )}
        </div>
      </GlassPanel>
    </div>
  );
};
