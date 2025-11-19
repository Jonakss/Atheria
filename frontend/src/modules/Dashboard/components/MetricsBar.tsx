import React, { useMemo } from 'react';
import { MetricItem } from './MetricItem';
import { useWebSocket } from '../../../hooks/useWebSocket';

export const MetricsBar: React.FC = () => {
  const { simData, allLogs } = useWebSocket();
  
  // Calcular métricas desde simData
  const vacuumEnergy = simData?.map_data 
    ? (() => {
        let sum = 0;
        let count = 0;
        for (const row of simData.map_data) {
          for (const val of row) {
            if (typeof val === 'number' && !isNaN(val)) {
              sum += val;
              count++;
            }
          }
        }
        return count > 0 ? (sum / count * 0.0042).toFixed(4) : '0.0000';
      })()
    : '0.0000';

  const localEntropy = useMemo(() => {
    // Placeholder - calcular desde simData si es necesario
    return '12.50';
  }, [simData]);
  
  const ionqSymmetry = useMemo(() => {
    return simData?.map_data ? '0.98' : '0.00';
  }, [simData?.map_data]);
  
  const decayRate = '1.2e-4';

  // Filtrar logs recientes (últimos 2) - solo mensajes de texto
  const recentLogs = useMemo(() => {
    if (!allLogs || allLogs.length === 0) return [];
    
    // Filtrar solo logs de texto (string)
    const textLogs = allLogs.filter(log => typeof log === 'string' && log.trim().length > 0);
    return textLogs.slice(-2);
  }, [allLogs]);

  return (
    <div className="mt-auto z-30 border-t border-white/10 bg-[#050505]/95 backdrop-blur-sm p-4">
      <div className="max-w-6xl mx-auto grid grid-cols-5 gap-6">
        <MetricItem 
          label="Energía de Vacío" 
          value={vacuumEnergy} 
          unit="eV" 
          status="good" 
        />
        <MetricItem 
          label="Entropía Local" 
          value={localEntropy} 
          unit="bits" 
          status="neutral" 
        />
        <MetricItem 
          label="Simetría (IonQ)" 
          value={ionqSymmetry} 
          unit="idx" 
          status="good" 
        />
        <MetricItem 
          label="Decaimiento" 
          value={decayRate} 
          unit="rad/s" 
          status="warning" 
        />
        
        {/* Log Miniatura */}
        <div className="col-span-1 pl-4 border-l border-white/5 flex flex-col justify-center gap-1">
          {recentLogs.length > 0 ? (
            recentLogs.map((log, idx) => {
              const logStr = typeof log === 'string' ? log : JSON.stringify(log);
              const isError = logStr.toLowerCase().includes('error');
              const isInfo = logStr.toLowerCase().includes('nucleación') || logStr.toLowerCase().includes('optimiz');
              
              return (
                <div 
                  key={idx} 
                  className={`flex items-center gap-2 text-[10px] font-mono ${
                    isError ? 'text-red-500/80' : isInfo ? 'text-emerald-500/80' : 'text-blue-500/80'
                  }`}
                >
                  <span className={`w-1 h-1 rounded-full ${
                    isError ? 'bg-red-500' : isInfo ? 'bg-emerald-500' : 'bg-blue-500'
                  }`} />
                  {logStr.length > 50 ? logStr.substring(0, 50) + '...' : logStr}
                </div>
              );
            })
          ) : (
            <div className="flex items-center gap-2 text-[10px] font-mono text-gray-600">
              <span className="w-1 h-1 rounded-full bg-gray-600" />
              Esperando datos...
            </div>
          )}
        </div>
      </div>
    </div>
  );
};
