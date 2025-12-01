import { Activity, Waves, Zap } from 'lucide-react';
import React, { useMemo } from 'react';
import { useWebSocket } from '../../../hooks/useWebSocket';

interface ScientificMetricsProps {
  compact?: boolean;
}

/**
 * Scientific Metrics Display Component
 * 
 * Shows key physical metrics from the simulation:
 * - Energy (from field density)
 * - Entropy (complexity measure)
 * - Temperature (effective thermal energy)
 */
export const ScientificMetrics: React.FC<ScientificMetricsProps> = ({ compact = false }) => {
  const { simData } = useWebSocket();

  // Extract metrics from simulation_info
  const metrics = useMemo(() => {
    const epochMetrics = simData?.simulation_info?.epoch_metrics || {};
    const histData = simData?.hist_data || {};

    // Calculate energy from histogram if available
    // hist_data can be object with {mean, stddev} or object with {histogram: bin[]} 
    const energyValue = epochMetrics.energy !== undefined 
      ? epochMetrics.energy 
      : (typeof histData.mean === 'number' ? histData.mean : null);

    // Get entropy (clustering is related to entropy in QCA)
    const entropyValue = epochMetrics.clustering !== undefined
      ? epochMetrics.clustering
      : null;

    // Calculate temperature proxy from variance/stddev
    const temperatureValue = typeof histData.stddev === 'number'
      ? histData.stddev
      : (epochMetrics.symmetry !== undefined ? 1 - epochMetrics.symmetry : null);

    return { 
      energy: energyValue, 
      entropy: entropyValue, 
      temperature: temperatureValue 
    };
  }, [simData]);

  // Renderizado condicional basado en modo compacto o expandido
  if (compact) {
    return (
      <div className="flex items-center gap-4 text-xs font-mono">
        {metrics.energy !== null && (
          <div className="flex items-center gap-1.5" title="Energía Total (Hamiltoniano)">
            <span className="text-yellow-500">E:</span>
            <span className="text-yellow-100">{metrics.energy.toFixed(4)}</span>
          </div>
        )}
        {metrics.entropy !== null && (
          <div className="flex items-center gap-1.5" title="Entropía (Complejidad)">
            <span className="text-purple-500">S:</span>
            <span className="text-purple-100">{metrics.entropy.toFixed(4)}</span>
          </div>
        )}
        {metrics.temperature !== null && (
          <div className="flex items-center gap-1.5" title="Temperatura (Varianza)">
            <span className="text-red-500">T:</span>
            <span className="text-red-100">{metrics.temperature.toFixed(4)}</span>
          </div>
        )}
        {/* Nuevas métricas Harlow Limit - solo mostrar si existen */}
        {simData?.simulation_info?.epoch_metrics && 'fidelity' in simData.simulation_info.epoch_metrics && (
          <div className="flex items-center gap-1.5" title="Fidelidad Global (Harlow)">
            <span className="text-green-500">F:</span>
            <span className="text-green-100">{(simData.simulation_info.epoch_metrics as any).fidelity.toFixed(4)}</span>
          </div>
        )}
      </div>
    );
  }

  return (
    <div className="bg-black/40 backdrop-blur-sm border border-white/10 rounded-lg p-3 w-full">
      <div className="text-[10px] uppercase tracking-wider text-gray-500 mb-2 font-semibold">Métricas Científicas</div>
      <div className="grid grid-cols-2 gap-2">
        <div className="flex flex-col items-center gap-1 p-2 bg-yellow-500/5 border border-yellow-500/20 rounded">
          <div className="flex items-center gap-1">
            <Zap size={12} className="text-yellow-400" />
            <span className="text-[9px] font-bold text-gray-500 uppercase">Energy</span>
          </div>
          <span className="text-sm font-mono text-yellow-400">
            {metrics.energy !== null ? metrics.energy.toFixed(4) : 'N/A'}
          </span>
        </div>

        <div className="flex flex-col items-center gap-1 p-2 bg-purple-500/5 border border-purple-500/20 rounded">
          <div className="flex items-center gap-1">
            <Activity size={12} className="text-purple-400" />
            <span className="text-[9px] font-bold text-gray-500 uppercase">Entropy</span>
          </div>
          <span className="text-sm font-mono text-purple-400">
            {metrics.entropy !== null ? metrics.entropy.toFixed(4) : 'N/A'}
          </span>
        </div>

        <div className="flex flex-col items-center gap-1 p-2 bg-red-500/5 border border-red-500/20 rounded">
          <div className="flex items-center gap-1">
            <Waves size={12} className="text-red-400" />
            <span className="text-[9px] font-bold text-gray-500 uppercase">Temp</span>
          </div>
          <span className="text-sm font-mono text-red-400">
            {metrics.temperature !== null ? metrics.temperature.toFixed(4) : 'N/A'}
          </span>
        </div>

        {/* Harlow Limit Metrics */}
        {simData?.simulation_info?.epoch_metrics && 'fidelity' in simData.simulation_info.epoch_metrics && (
          <div className="flex flex-col items-center gap-1 p-2 bg-green-500/5 border border-green-500/20 rounded">
            <div className="flex items-center gap-1">
              <Activity size={12} className="text-green-400" />
              <span className="text-[9px] font-bold text-gray-500 uppercase">Fidelity</span>
            </div>
            <span className="text-sm font-mono text-green-400">
              {(simData.simulation_info.epoch_metrics as any).fidelity.toFixed(4)}
            </span>
          </div>
        )}
      </div>
    </div>
  );
};
