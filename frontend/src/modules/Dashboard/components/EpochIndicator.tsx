import { Telescope } from 'lucide-react';
import React, { useMemo } from 'react';
import { useWebSocket } from '../../../hooks/useWebSocket';

/**
 * Componente que muestra la época cosmológica actual detectada por el EpochDetector
 * Incluye barra de progreso de "Evolución del Universo" según métricas físicas
 */
export const EpochIndicator: React.FC<{
  isCollapsed?: boolean;
  onToggleCollapse?: () => void;
}> = ({ isCollapsed = false, onToggleCollapse }) => {
  const { simData, connectionStatus } = useWebSocket();
  const isConnected = connectionStatus === 'connected';

  // Definición de épocas (sincronizado con backend epoch_detector.py)
  const EPOCHS = {
    0: { name: "Vacío Inestable", subtitle: "Big Bang", color: "text-gray-400", bgColor: "bg-gray-500/20", progress: 0 },
    1: { name: "Era Cuántica", subtitle: "Sopa de Probabilidad", color: "text-pink-400", bgColor: "bg-pink-500/20", progress: 20 },
    2: { name: "Era de Partículas", subtitle: "Cristalización Simétrica", color: "text-blue-400", bgColor: "bg-blue-500/20", progress: 40 },
    3: { name: "Era Química", subtitle: "Polímeros y Movimiento", color: "text-purple-400", bgColor: "bg-purple-500/20", progress: 60 },
    4: { name: "Era Gravitacional", subtitle: "Acreción de Materia", color: "text-teal-400", bgColor: "bg-teal-500/20", progress: 80 },
    5: { name: "Era Biológica", subtitle: "A-Life / Homeostasis", color: "text-green-400", bgColor: "bg-green-500/20", progress: 100 }
  };

  // Obtener época actual desde simData
  const currentEpoch = simData?.simulation_info?.epoch ?? 0;
  const epochMetrics = simData?.simulation_info?.epoch_metrics;

  // Obtener configuración de la época actual
  const epochConfig = EPOCHS[currentEpoch as keyof typeof EPOCHS] || EPOCHS[0];

  // Formatear métricas para tooltip
  const metricsFormatted = useMemo(() => {
    if (!epochMetrics) return null;
    
    return {
      energy: typeof epochMetrics.energy === 'number' ? epochMetrics.energy.toFixed(4) : 'N/A',
      clustering: typeof epochMetrics.clustering === 'number' ? epochMetrics.clustering.toFixed(4) : 'N/A',
      symmetry: typeof epochMetrics.symmetry === 'number' ? epochMetrics.symmetry.toFixed(4) : 'N/A'
    };
  }, [epochMetrics]);

  return (
    <div 
      className={`relative border border-white/10 rounded bg-dark-990/50 backdrop-blur-sm transition-all duration-300 ${
        isCollapsed ? 'p-2' : 'p-3'
      }`}
      title={metricsFormatted ? `Energía: ${metricsFormatted.energy} | Clustering: ${metricsFormatted.clustering} | Simetría: ${metricsFormatted.symmetry}` : 'Evolución del Universo'}
    >
      {/* Header */}
      <div className="flex items-center justify-between mb-2">
        <div className="flex items-center gap-2">
          <Telescope size={12} className="text-gray-400" />
          <span className="text-[10px] font-mono font-bold text-gray-400 uppercase tracking-wide">
            EVOLUCIÓN UNIVERSAL
          </span>
        </div>
        {onToggleCollapse && (
          <button
            onClick={onToggleCollapse}
            className="text-gray-500 hover:text-gray-300 transition-colors"
            title={isCollapsed ? "Expandir" : "Colapsar"}
          >
            <span className="text-[10px]">{isCollapsed ? '▼' : '▲'}</span>
          </button>
        )}
      </div>

      {/* Contenido principal */}
      {!isCollapsed && (
        <div className="space-y-2">
          {/* Época actual */}
          <div className={`p-2 rounded ${epochConfig.bgColor} border border-white/10`}>
            <div className="flex justify-between items-start">
              <div>
                <div className={`text-sm font-bold ${epochConfig.color}`}>
                  {currentEpoch}. {epochConfig.name}
                </div>
                <div className="text-[9px] text-gray-500 font-mono mt-0.5">
                  {epochConfig.subtitle}
                </div>
              </div>
              {isConnected && (
                <div className={`text-xs font-mono ${epochConfig.color}`}>
                  {epochConfig.progress}%
                </div>
              )}
            </div>
          </div>

          {/* Barra de progreso de evolución */}
          <div className="space-y-1">
            <div className="flex justify-between text-[9px] text-gray-500 font-mono">
              <span>Progreso Evolución</span>
              <span>{currentEpoch}/{Object.keys(EPOCHS).length - 1}</span>
            </div>
            <div className="w-full h-1.5 bg-dark-950 rounded-full overflow-hidden border border-white/10">
              <div
                className={`h-full ${epochConfig.color.replace('text-', 'bg-')} transition-all duration-500`}
                style={{ width: `${epochConfig.progress}%` }}
              />
            </div>
          </div>

          {/* Métricas de época (solo si están disponibles) */}
          {metricsFormatted && (
            <div className="grid grid-cols-3 gap-1 pt-2 border-t border-white/10">
              <div className="text-center">
                <div className="text-[8px] text-gray-500 font-mono uppercase">Energía</div>
                <div className="text-[10px] text-teal-400 font-mono font-bold">{metricsFormatted.energy}</div>
              </div>
              <div className="text-center">
                <div className="text-[8px] text-gray-500 font-mono uppercase">Clustering</div>
                <div className="text-[10px] text-pink-400 font-mono font-bold">{metricsFormatted.clustering}</div>
              </div>
              <div className="text-center">
                <div className="text-[8px] text-gray-500 font-mono uppercase">Simetría</div>
                <div className="text-[10px] text-purple-400 font-mono font-bold">{metricsFormatted.symmetry}</div>
              </div>
            </div>
          )}
        </div>
      )}

      {/* Vista colapsada - Solo mostrar época y progreso */}
      {isCollapsed && (
        <div className="space-y-1">
          <div className={`text-[10px] font-mono font-bold ${epochConfig.color} truncate`}>
            {currentEpoch}. {epochConfig.name}
          </div>
          <div className="w-full h-1 bg-dark-950 rounded-full overflow-hidden">
            <div
              className={`h-full ${epochConfig.color.replace('text-', 'bg-')}`}
              style={{ width: `${epochConfig.progress}%` }}
            />
          </div>
        </div>
      )}

      {/* Indicador de conexión */}
      {!isConnected && (
        <div className="absolute inset-0 bg-dark-990/80 backdrop-blur-sm rounded flex items-center justify-center">
          <span className="text-[10px] text-gray-600 font-mono">Detectando...</span>
        </div>
      )}
    </div>
  );
};
