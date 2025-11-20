import React, { useMemo, useState } from 'react';
import { FieldWidget } from './FieldWidget';
import { useWebSocket } from '../../../hooks/useWebSocket';
import { ChevronUp, ChevronDown } from 'lucide-react';

export const MetricsBar: React.FC = () => {
  const { simData, allLogs, sendCommand, connectionStatus } = useWebSocket();
  const isConnected = connectionStatus === 'connected';
  const [expanded, setExpanded] = useState(false);
  const [collapsedWidgets, setCollapsedWidgets] = useState<Set<string>>(new Set()); // Widgets individuales colapsados
  
  // Calcular métricas reales desde simData
  // Usar JSON.stringify para crear una dependencia estable de map_data
  const mapDataString = useMemo(() => {
    if (!simData?.map_data) return null;
    try {
      return JSON.stringify(simData.map_data);
    } catch {
      return null;
    }
  }, [simData?.map_data]);
  
  const vacuumEnergy = useMemo(() => {
    if (!isConnected || !mapDataString) return 'N/A';
    
    let mapData: number[][];
    try {
      mapData = JSON.parse(mapDataString);
    } catch {
      return 'N/A';
    }
    
    // Energía de vacío = promedio de |psi|² multiplicado por factor de conversión
    let sum = 0;
    let count = 0;
    for (const row of mapData) {
      if (Array.isArray(row)) {
        for (const val of row) {
          if (typeof val === 'number' && !isNaN(val) && isFinite(val)) {
            sum += Math.abs(val);
            count++;
          }
        }
      }
    }
    if (count === 0) return 'N/A';
    // Convertir a unidades de energía de vacío (factor de conversión empírico)
    const avgEnergy = sum / count;
    const vacuumEnergyValue = avgEnergy * 0.0042; // Factor de conversión
    return vacuumEnergyValue.toFixed(4);
  }, [mapDataString, isConnected]);

  // Entropía Local = Entropía de Shannon calculada desde map_data
  const localEntropy = useMemo(() => {
    if (!isConnected || !mapDataString) return 'N/A';
    
    let mapData: number[][];
    try {
      mapData = JSON.parse(mapDataString);
    } catch {
      return 'N/A';
    }
    
    // Calcular entropía de Shannon: H = -Σ p_i * log2(p_i)
    // Donde p_i es la probabilidad normalizada de cada valor
    const flatData: number[] = [];
    for (const row of mapData) {
      if (Array.isArray(row)) {
        for (const val of row) {
          if (typeof val === 'number' && !isNaN(val) && isFinite(val)) {
            flatData.push(Math.abs(val));
          }
        }
      }
    }
    
    if (flatData.length === 0) return 'N/A';
    
    // Normalizar valores a probabilidades
    const total = flatData.reduce((sum, val) => sum + val, 0);
    if (total === 0) return '0.0000';
    
    // Calcular entropía (simplificado: usar bins)
    const bins = 32;
    const binCounts = new Array(bins).fill(0);
    const minVal = Math.min(...flatData);
    const maxVal = Math.max(...flatData);
    const range = maxVal - minVal || 1;
    
    for (const val of flatData) {
      const bin = Math.floor(((val - minVal) / range) * bins);
      const binIndex = Math.min(bins - 1, Math.max(0, bin));
      binCounts[binIndex]++;
    }
    
    // Calcular entropía de Shannon
    let entropy = 0;
    for (const count of binCounts) {
      if (count > 0) {
        const prob = count / flatData.length;
        entropy -= prob * Math.log2(prob);
      }
    }
    
    return entropy.toFixed(4);
  }, [mapDataString, isConnected]);
  
  // Simetría IONQ = Simetría espacial calculada desde map_data
  const ionqSymmetry = useMemo(() => {
    if (!isConnected || !mapDataString) return 'N/A';
    
    let mapData: number[][];
    try {
      mapData = JSON.parse(mapDataString);
    } catch {
      return 'N/A';
    }
    
    const height = mapData.length;
    const width = mapData[0]?.length || 0;
    
    if (height === 0 || width === 0) return 'N/A';
    
    // Calcular simetría horizontal (reflexión sobre eje vertical)
    let horizontalSymmetry = 0;
    let horizontalCount = 0;
    for (let y = 0; y < height; y++) {
      for (let x = 0; x < Math.floor(width / 2); x++) {
        const leftVal = mapData[y]?.[x] || 0;
        const rightVal = mapData[y]?.[width - 1 - x] || 0;
        if (typeof leftVal === 'number' && typeof rightVal === 'number') {
          const diff = Math.abs(leftVal - rightVal);
          const avg = (Math.abs(leftVal) + Math.abs(rightVal)) / 2 || 1;
          horizontalSymmetry += 1 - (diff / avg);
          horizontalCount++;
        }
      }
    }
    
    // Calcular simetría vertical (reflexión sobre eje horizontal)
    let verticalSymmetry = 0;
    let verticalCount = 0;
    for (let y = 0; y < Math.floor(height / 2); y++) {
      for (let x = 0; x < width; x++) {
        const topVal = mapData[y]?.[x] || 0;
        const bottomVal = mapData[height - 1 - y]?.[x] || 0;
        if (typeof topVal === 'number' && typeof bottomVal === 'number') {
          const diff = Math.abs(topVal - bottomVal);
          const avg = (Math.abs(topVal) + Math.abs(bottomVal)) / 2 || 1;
          verticalSymmetry += 1 - (diff / avg);
          verticalCount++;
        }
      }
    }
    
    // Promedio de ambas simetrías
    const totalSymmetry = 
      (horizontalCount > 0 ? horizontalSymmetry / horizontalCount : 0) +
      (verticalCount > 0 ? verticalSymmetry / verticalCount : 0);
    const avgSymmetry = totalSymmetry / 2;
    
    return Math.max(0, Math.min(1, avgSymmetry)).toFixed(4);
  }, [mapDataString, isConnected]);
  
  // Decaimiento = Gamma decay rate del sistema (de simData o config)
  const decayRate = useMemo(() => {
    if (!isConnected) return 'N/A';
    
    // Intentar obtener gamma_decay de simData.simulation_info o usar valor por defecto
    const gammaDecay = simData?.simulation_info?.gamma_decay || 0.01;
    // Convertir a rad/s (factor de conversión)
    const decayValue = gammaDecay * 0.012; // Factor de conversión
    return decayValue < 0.001 ? decayValue.toExponential(1) : decayValue.toFixed(4);
  }, [simData?.simulation_info, isConnected]);

  // Filtrar logs recientes (últimos 2) - solo mensajes de texto
  const recentLogs = useMemo(() => {
    if (!allLogs || allLogs.length === 0) return [];
    
    // Filtrar solo logs de texto (string)
    const textLogs = allLogs.filter(log => typeof log === 'string' && log.trim().length > 0);
    return textLogs.slice(-2);
  }, [allLogs]);

  return (
    <div className={`mt-auto z-30 border-t border-white/10 bg-[#050505]/95 backdrop-blur-sm transition-all duration-300 ease-in-out relative ${
      expanded ? 'p-4' : 'p-2'
    }`}>
      <div className="max-w-6xl mx-auto relative">
        {/* Botón de expansión - Separador vertical discreto */}
        <div
          onMouseEnter={() => setExpanded(true)}
          onMouseLeave={() => setExpanded(false)}
          className="absolute left-1/2 -translate-x-1/2 top-0 h-full w-1 hover:w-2 bg-white/5 hover:bg-white/20 transition-all duration-300 cursor-ew-resize z-50 group"
          title={expanded ? "Contraer paneles (hover para expandir)" : "Expandir paneles (hover para contraer)"}
        >
          <div className="absolute left-1/2 top-1/2 -translate-x-1/2 -translate-y-1/2 opacity-0 group-hover:opacity-100 transition-opacity pointer-events-none">
            {expanded ? (
              <ChevronDown size={10} className="text-gray-400" />
            ) : (
              <ChevronUp size={10} className="text-gray-400" />
            )}
          </div>
        </div>
        
        <div className={`grid transition-all duration-300 ease-in-out ${
          expanded 
            ? 'grid-cols-5 gap-4' 
            : 'grid-cols-5 gap-2'
        }`}>
          {/* Widget de Energía de Vacío - con visualización de campo de densidad */}
          <FieldWidget
            label="ENERGÍA DE VACÍO"
            value={vacuumEnergy}
            unit="EV"
            status="good"
            fieldData={simData?.map_data} // Datos del campo de densidad
            fieldType="energy"
            isCollapsed={collapsedWidgets.has('vacuum_energy')}
            onToggleCollapse={() => {
              const newCollapsed = new Set(collapsedWidgets);
              if (newCollapsed.has('vacuum_energy')) {
                newCollapsed.delete('vacuum_energy');
              } else {
                newCollapsed.add('vacuum_energy');
              }
              setCollapsedWidgets(newCollapsed);
            }}
          />
          
          {/* Widget de Entropía Local - con visualización de campo de entropía */}
          <FieldWidget
            label="ENTROPÍA LOCAL"
            value={localEntropy}
            unit="BITS"
            status="neutral"
            fieldData={simData?.map_data} // Usar map_data para calcular distribución
            fieldType="density"
            isCollapsed={collapsedWidgets.has('local_entropy')}
            onToggleCollapse={() => {
              const newCollapsed = new Set(collapsedWidgets);
              if (newCollapsed.has('local_entropy')) {
                newCollapsed.delete('local_entropy');
              } else {
                newCollapsed.add('local_entropy');
              }
              setCollapsedWidgets(newCollapsed);
            }}
          />
          
          {/* Widget de Simetría (IONQ) - con visualización de simetría espacial */}
          <FieldWidget
            label="SIMETRÍA (IONQ)"
            value={ionqSymmetry}
            unit="IDX"
            status="good"
            fieldData={simData?.map_data} // Usar map_data para visualizar simetría
            fieldType="phase"
            isCollapsed={collapsedWidgets.has('ionq_symmetry')}
            onToggleCollapse={() => {
              const newCollapsed = new Set(collapsedWidgets);
              if (newCollapsed.has('ionq_symmetry')) {
                newCollapsed.delete('ionq_symmetry');
              } else {
                newCollapsed.add('ionq_symmetry');
              }
              setCollapsedWidgets(newCollapsed);
            }}
          />
          
          {/* Widget de Decaimiento - con visualización temporal */}
          <FieldWidget
            label="DECAIMIENTO"
            value={decayRate}
            unit="RAD/S"
            status="warning"
            fieldData={simData?.flow_data?.magnitude} // Usar magnitud de flujo como proxy de decaimiento
            fieldType="flow"
            isCollapsed={collapsedWidgets.has('decay_rate')}
            onToggleCollapse={() => {
              const newCollapsed = new Set(collapsedWidgets);
              if (newCollapsed.has('decay_rate')) {
                newCollapsed.delete('decay_rate');
              } else {
                newCollapsed.add('decay_rate');
              }
              setCollapsedWidgets(newCollapsed);
            }}
          />
        
          {/* Log - Más grande cuando no está expandido */}
          <div className={`pl-4 border-l border-white/5 flex flex-col justify-center transition-all duration-300 ${
            expanded ? 'gap-1' : 'gap-2'
          }`}>
            {recentLogs.length > 0 ? (
              recentLogs.map((log, idx) => {
                const logStr = typeof log === 'string' ? log : JSON.stringify(log);
                const isError = logStr.toLowerCase().includes('error');
                const isInfo = logStr.toLowerCase().includes('nucleación') || logStr.toLowerCase().includes('optimiz');
                
                return (
                  <div 
                    key={idx} 
                    className={`flex items-center gap-2 font-mono ${
                      expanded ? 'text-[10px]' : 'text-xs'
                    } ${
                      isError ? 'text-red-500/80' : isInfo ? 'text-emerald-500/80' : 'text-blue-500/80'
                    }`}
                  >
                    <span className={`rounded-full shrink-0 ${
                      expanded ? 'w-1 h-1' : 'w-1.5 h-1.5'
                    } ${
                      isError ? 'bg-red-500' : isInfo ? 'bg-emerald-500' : 'bg-blue-500'
                    }`} />
                    <span className={expanded ? 'truncate' : ''}>
                      {expanded 
                        ? (logStr.length > 50 ? logStr.substring(0, 50) + '...' : logStr)
                        : (logStr.length > 80 ? logStr.substring(0, 80) + '...' : logStr)
                      }
                    </span>
                  </div>
                );
              })
            ) : (
              <div className={`flex items-center gap-2 font-mono text-gray-600 ${
                expanded ? 'text-[10px]' : 'text-xs'
              }`}>
                <span className={`rounded-full bg-gray-600 shrink-0 ${
                  expanded ? 'w-1 h-1' : 'w-1.5 h-1.5'
                }`} />
                <span>Esperando datos...</span>
              </div>
            )}
          </div>
        </div>
      </div>
    </div>
  );
};
