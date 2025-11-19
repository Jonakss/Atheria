import React, { useMemo } from 'react';
import { Play, Pause } from 'lucide-react';
import { GlassPanel } from './GlassPanel';
import { useWebSocket } from '../../../hooks/useWebSocket';

export const Toolbar: React.FC = () => {
  const { inferenceStatus, sendCommand, simData } = useWebSocket();
  const isPlaying = inferenceStatus === 'running';
  
  const currentStep = simData?.step ?? simData?.simulation_info?.step ?? 0;
  
  // Calcular FPS aproximado (placeholder - se podría calcular desde timestamps)
  const fps = 118;
  
  // Calcular número de partículas (aproximado desde map_data)
  const particleCount = useMemo(() => {
    if (!simData?.map_data) return '0';
    
    let count = 0;
    for (const row of simData.map_data) {
      if (Array.isArray(row)) {
        for (const val of row) {
          if (typeof val === 'number' && !isNaN(val) && val > 0.01) count++;
        }
      }
    }
    return count > 1000 ? `${(count / 1000).toFixed(1)}K` : count.toString();
  }, [simData?.map_data]);

  const handlePlayPause = () => {
    const command = isPlaying ? 'pause' : 'play';
    sendCommand('inference', command);
  };

  return (
    <div className="absolute top-4 left-4 z-30 flex gap-2 pointer-events-none">
      <GlassPanel className="pointer-events-auto flex items-center p-1 gap-1">
        <button 
          onClick={handlePlayPause}
          className={`px-4 py-1.5 rounded text-xs font-bold flex items-center gap-2 transition-all border ${
            isPlaying 
              ? 'bg-amber-500/10 text-amber-500 border-amber-500/30 hover:bg-amber-500/20' 
              : 'bg-white/5 text-gray-200 border-white/10 hover:bg-white/10'
          }`}
        >
          {isPlaying ? <Pause size={12} fill="currentColor" /> : <Play size={12} fill="currentColor" />}
          {isPlaying ? 'PAUSAR' : 'EJECUTAR'}
        </button>
        
        <div className="w-px h-4 bg-white/10 mx-2" />
        
        <div className="flex flex-col px-2">
          <span className="text-[8px] text-gray-500 uppercase font-bold">Paso Actual</span>
          <span className="text-xs font-mono text-gray-200">{currentStep.toLocaleString()}</span>
        </div>
      </GlassPanel>

      <GlassPanel className="pointer-events-auto flex items-center px-3 py-1 gap-3">
        <div className="flex items-center gap-2">
          <span className="text-[10px] font-bold text-gray-500">FPS</span>
          <span className="text-xs font-mono text-emerald-400">{fps}</span>
        </div>
        <div className="w-px h-3 bg-white/10" />
        <div className="flex items-center gap-2">
          <span className="text-[10px] font-bold text-gray-500">PARTÍCULAS</span>
          <span className="text-xs font-mono text-blue-400">{particleCount}</span>
        </div>
      </GlassPanel>
    </div>
  );
};
