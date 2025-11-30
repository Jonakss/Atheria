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

  if (compact) {
    return (
      <div className="flex items-center gap-3 text-[10px] font-mono">
        {metrics.energy !== null && (
          <div className="flex items-center gap-1.5">
            <Zap size={10} className="text-yellow-400" />
            <span className="text-gray-500">E:</span>
            <span className="text-yellow-400">{metrics.energy.toFixed(3)}</span>
          </div>
        )}
        {metrics.entropy !== null && (
          <div className="flex items-center gap-1.5">
            <Activity size={10} className="text-purple-400" />
            <span className="text-gray-500">S:</span>
            <span className="text-purple-400">{metrics.entropy.toFixed(3)}</span>
          </div>
        )}
        {metrics.temperature !== null && (
          <div className="flex items-center gap-1.5">
            <Waves size={10} className="text-orange-400" />
            <span className="text-gray-500">T:</span>
            <span className="text-orange-400">{metrics.temperature.toFixed(3)}</span>
          </div>
        )}
      </div>
    );
  }

  return (
    <div className="grid grid-cols-3 gap-2">
      {/* Energy */}
      <div className="flex flex-col items-center gap-1 p-2 bg-yellow-500/5 border border-yellow-500/20 rounded">
        <div className="flex items-center gap-1">
          <Zap size={12} className="text-yellow-400" />
          <span className="text-[9px] font-bold text-gray-500 uppercase">Energy</span>
        </div>
        <span className="text-sm font-mono text-yellow-400">
          {metrics.energy !== null ? metrics.energy.toFixed(4) : 'N/A'}
        </span>
      </div>

      {/* Entropy */}
      <div className="flex flex-col items-center gap-1 p-2 bg-purple-500/5 border border-purple-500/20 rounded">
        <div className="flex items-center gap-1">
          <Activity size={12} className="text-purple-400" />
          <span className="text-[9px] font-bold text-gray-500 uppercase">Entropy</span>
        </div>
        <span className="text-sm font-mono text-purple-400">
          {metrics.entropy !== null ? metrics.entropy.toFixed(4) : 'N/A'}
        </span>
      </div>

      {/* Temperature */}
      <div className="flex flex-col items-center gap-1 p-2 bg-orange-500/5 border border-orange-500/20 rounded">
        <div className="flex items-center gap-1">
          <Waves size={12} className="text-orange-400" />
          <span className="text-[9px] font-bold text-gray-500 uppercase">Temp</span>
        </div>
        <span className="text-sm font-mono text-orange-400">
          {metrics.temperature !== null ? metrics.temperature.toFixed(4) : 'N/A'}
        </span>
      </div>
    </div>
  );
};
