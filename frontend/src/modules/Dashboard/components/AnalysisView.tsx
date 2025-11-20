// frontend/src/modules/Dashboard/components/AnalysisView.tsx
import React from 'react';
import { Activity, TrendingUp, TrendingDown, BarChart3 } from 'lucide-react';
import { useWebSocket } from '../../../hooks/useWebSocket';
import { GlassPanel } from './GlassPanel';

export const AnalysisView: React.FC = () => {
  const { simData, trainingProgress, connectionStatus } = useWebSocket();
  const isConnected = connectionStatus === 'connected';
  
  // Calcular estadísticas desde simData
  const stats = React.useMemo(() => {
    if (!isConnected || !simData?.map_data) {
      return {
        densityMean: 0,
        densityStd: 0,
        densityMin: 0,
        densityMax: 0,
        energyTotal: 0,
        entropy: 0
      };
    }
    
    const flatData: number[] = [];
    for (const row of simData.map_data) {
      if (Array.isArray(row)) {
        for (const val of row) {
          if (typeof val === 'number' && !isNaN(val)) {
            flatData.push(val);
          }
        }
      }
    }
    
    if (flatData.length === 0) {
      return {
        densityMean: 0,
        densityStd: 0,
        densityMin: 0,
        densityMax: 0,
        energyTotal: 0,
        entropy: 0
      };
    }
    
    const mean = flatData.reduce((a, b) => a + b, 0) / flatData.length;
    const variance = flatData.reduce((acc, val) => acc + Math.pow(val - mean, 2), 0) / flatData.length;
    const std = Math.sqrt(variance);
    const min = Math.min(...flatData);
    const max = Math.max(...flatData);
    const totalEnergy = flatData.reduce((a, b) => a + Math.abs(b), 0);
    
    // Entropía aproximada (log base 2 del número de estados no cero)
    const nonZeroCount = flatData.filter(v => Math.abs(v) > 0.01).length;
    const entropy = nonZeroCount > 0 ? Math.log2(nonZeroCount / flatData.length) * -1 : 0;
    
    return {
      densityMean: mean,
      densityStd: std,
      densityMin: min,
      densityMax: max,
      energyTotal: totalEnergy,
      entropy
    };
  }, [simData?.map_data, isConnected]);

  if (!isConnected) {
    return (
      <div className="absolute inset-0 z-0 bg-gradient-to-b from-[#050505] to-black flex items-center justify-center">
        <div className="text-gray-600 text-sm">Conectando al servidor...</div>
      </div>
    );
  }

  return (
    <div className="absolute inset-0 z-0 bg-gradient-to-b from-[#050505] to-black overflow-auto custom-scrollbar">
      <div className="p-6 max-w-7xl mx-auto space-y-6">
        {/* Header */}
        <div className="flex items-center gap-3 mb-6">
          <Activity size={20} className="text-blue-400" />
          <h2 className="text-lg font-bold text-gray-200">Análisis de Simulación</h2>
        </div>

        {/* Métricas Principales */}
        <div className="grid grid-cols-3 gap-4">
          <GlassPanel className="p-4">
            <div className="flex items-center gap-2 mb-2">
              <BarChart3 size={16} className="text-emerald-400" />
              <span className="text-[10px] font-bold text-gray-400 uppercase tracking-wider">Densidad Media</span>
            </div>
            <div className="text-2xl font-mono font-bold text-emerald-400">
              {stats.densityMean.toFixed(6)}
            </div>
            <div className="text-[10px] text-gray-600 mt-1">± {stats.densityStd.toFixed(6)}</div>
          </GlassPanel>

          <GlassPanel className="p-4">
            <div className="flex items-center gap-2 mb-2">
              <TrendingUp size={16} className="text-blue-400" />
              <span className="text-[10px] font-bold text-gray-400 uppercase tracking-wider">Energía Total</span>
            </div>
            <div className="text-2xl font-mono font-bold text-blue-400">
              {stats.energyTotal.toFixed(4)}
            </div>
            <div className="text-[10px] text-gray-600 mt-1">Acumulada en grid</div>
          </GlassPanel>

          <GlassPanel className="p-4">
            <div className="flex items-center gap-2 mb-2">
              <TrendingDown size={16} className="text-amber-400" />
              <span className="text-[10px] font-bold text-gray-400 uppercase tracking-wider">Entropía</span>
            </div>
            <div className="text-2xl font-mono font-bold text-amber-400">
              {stats.entropy.toFixed(4)}
            </div>
            <div className="text-[10px] text-gray-600 mt-1">Bits (complejidad)</div>
          </GlassPanel>
        </div>

        {/* Estadísticas Detalladas */}
        <GlassPanel className="p-4">
          <div className="text-[10px] font-bold text-gray-400 uppercase tracking-wider mb-4">Estadísticas de Densidad</div>
          <div className="grid grid-cols-2 gap-4">
            <div>
              <div className="text-[10px] text-gray-500 mb-1">Mínimo</div>
              <div className="text-sm font-mono text-gray-300">{stats.densityMin.toFixed(6)}</div>
            </div>
            <div>
              <div className="text-[10px] text-gray-500 mb-1">Máximo</div>
              <div className="text-sm font-mono text-gray-300">{stats.densityMax.toFixed(6)}</div>
            </div>
            <div>
              <div className="text-[10px] text-gray-500 mb-1">Desviación Estándar</div>
              <div className="text-sm font-mono text-gray-300">{stats.densityStd.toFixed(6)}</div>
            </div>
            <div>
              <div className="text-[10px] text-gray-500 mb-1">Paso Actual</div>
              <div className="text-sm font-mono text-gray-300">{simData?.step || simData?.simulation_info?.step || 0}</div>
            </div>
          </div>
        </GlassPanel>

        {/* Placeholder para gráficos futuros */}
        <GlassPanel className="p-8">
          <div className="text-center text-gray-600 text-sm">
            Gráficos de análisis temporal - Próximamente
          </div>
        </GlassPanel>
      </div>
    </div>
  );
};

