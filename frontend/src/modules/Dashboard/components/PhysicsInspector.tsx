import { AlertCircle, ChevronLeft, ChevronRight, Microscope, Zap } from 'lucide-react';
import React, { useState } from 'react';
import { useWebSocket } from '../../../hooks/useWebSocket';

interface PhysicsInspectorProps {
  isCollapsed?: boolean;
  onToggleCollapse?: () => void;
  viewerVersion?: 'v1' | 'v2';
  onViewerVersionChange?: (version: 'v1' | 'v2') => void;
  selectedLayer?: number;
  onLayerChange?: (layer: number) => void;
}

export const PhysicsInspector: React.FC<PhysicsInspectorProps> = ({ 
  isCollapsed = false, 
  onToggleCollapse,
  viewerVersion = 'v1',
  onViewerVersionChange,
  selectedLayer = 0,
  onLayerChange
}) => {
  const { sendCommand, allLogs } = useWebSocket();
  const [gammaDecay, setGammaDecay] = useState(0.015);
  const [thermalNoise, setThermalNoise] = useState(0.002);
  // Internal state fallback if prop not provided
  const [internalLayer, setInternalLayer] = useState(0);
  
  const activeLayer = onLayerChange ? selectedLayer : internalLayer;
  const handleLayerChange = (layer: number) => {
    if (onLayerChange) {
      onLayerChange(layer);
    } else {
      setInternalLayer(layer);
    }
  };

  const [internalCollapsed, setInternalCollapsed] = useState(false);
  
  // Usar prop externo si está disponible, sino usar estado interno
  const collapsed = onToggleCollapse !== undefined ? isCollapsed : internalCollapsed;
  const handleToggle = onToggleCollapse || (() => setInternalCollapsed(!internalCollapsed));

  const VIZ_MODES = [
    { id: 0, label: 'Densidad', desc: 'Amplitud |ψ|', icon: '📊', color: 'text-blue-400', border: 'border-blue-500/50' },
    { id: 1, label: 'Fase', desc: 'Argumento φ', icon: '🌈', color: 'text-purple-400', border: 'border-purple-500/50' },
    { id: 2, label: 'Energía', desc: 'Hamiltoniano', icon: '⚡', color: 'text-yellow-400', border: 'border-yellow-500/50' },
    { id: 3, label: 'Flujo', desc: 'Corriente J', icon: '🌊', color: 'text-cyan-400', border: 'border-cyan-500/50' },
    { id: 4, label: 'Chunks', desc: 'Meta-Estructura', icon: '🧩', color: 'text-green-400', border: 'border-green-500/50' }
  ];

  const handleGammaChange = (value: number) => {
    setGammaDecay(value);
    // Enviar configuración al backend (formato correcto: snake_case)
    sendCommand('inference', 'set_config', {
      gamma_decay: value
    });
  };

  const handleThermalNoiseChange = (value: number) => {
    setThermalNoise(value);
    // TODO: Thermal noise aún no está implementado en el backend
  };

  // Filtrar logs recientes (últimos 2)
  const recentLogs = allLogs?.slice(-2) || [];

  return (
    <aside className={`${collapsed ? 'w-10' : 'w-72'} border-l border-white/5 bg-dark-950/80 backdrop-blur-md flex flex-col z-40 shrink-0 flex-shrink-0 transition-all duration-300 ease-in-out overflow-hidden`} style={{ minWidth: collapsed ? '2.5rem' : '18rem', maxWidth: collapsed ? '2.5rem' : '18rem' }}>
      {/* Título Panel - Con botón de colapsar */}
      <div className="h-10 border-b border-white/5 flex items-center justify-between px-3 bg-dark-980/90 shrink-0">
        {!collapsed && (
          <span className="text-[10px] font-bold text-dark-400 uppercase tracking-widest">Inspector Físico</span>
        )}
        <button
          onClick={handleToggle}
          className="p-1.5 text-dark-500 hover:text-dark-300 transition-colors rounded hover:bg-white/5"
          title={collapsed ? 'Expandir Inspector' : 'Colapsar Inspector'}
        >
          {collapsed ? <ChevronLeft size={14} /> : <ChevronRight size={14} />}
        </button>
      </div>

      {!collapsed && (
      <div className="flex-1 overflow-y-auto p-4 space-y-6" style={{ scrollbarWidth: 'thin', scrollbarColor: 'rgba(255,255,255,0.1) transparent' }}>
        
        {/* Sección: Selector de Visualización (Prominente) */}
        <div className="space-y-3">
          <div className="grid grid-cols-2 gap-2">
            {VIZ_MODES.map((mode) => (
              <button
                key={mode.id}
                onClick={() => handleLayerChange(mode.id)}
                className={`flex flex-col p-2 rounded-lg border transition-all duration-200 ${
                  activeLayer === mode.id
                    ? `bg-white/10 ${mode.border} shadow-lg`
                    : 'bg-white/5 border-white/5 hover:bg-white/10 hover:border-white/10'
                }`}
              >
                <div className="flex items-center justify-between mb-1">
                  <span className="text-lg">{mode.icon}</span>
                  {activeLayer === mode.id && <div className={`w-1.5 h-1.5 rounded-full ${mode.color.replace('text-', 'bg-')} shadow-[0_0_5px_currentColor]`} />}
                </div>
                <span className={`text-xs font-bold ${activeLayer === mode.id ? mode.color : 'text-gray-300'}`}>
                  {mode.label}
                </span>
                <span className="text-[9px] text-gray-500 font-mono leading-tight">
                  {mode.desc}
                </span>
              </button>
            ))}
          </div>
        </div>

        {/* Sección: Parámetros Globales */}
        <div className="space-y-4 pt-2 border-t border-white/5">
          
          {/* Control Slider Refinado - Gamma */}
          <div className="space-y-1.5">
            <div className="flex justify-between text-[10px]">
              <span className="text-gray-400">Gamma (Disipación)</span>
              <span className="font-mono text-gray-200 bg-white/5 px-1.5 rounded">{gammaDecay.toFixed(3)}</span>
            </div>
            {/* Slider usando input range HTML5 nativo para mejor UX */}
            <input
              type="range"
              min="0"
              max="10.0"
              step="0.001"
              value={gammaDecay}
              onChange={(e) => handleGammaChange(parseFloat(e.target.value))}
              className="w-full h-1 bg-gray-800 rounded-full appearance-none cursor-pointer slider-thumb"
              style={{
                background: `linear-gradient(to right, rgb(107, 114, 128) 0%, rgb(107, 114, 128) ${(gammaDecay / 10.0) * 100}%, rgb(31, 41, 55) ${(gammaDecay / 10.0) * 100}%, rgb(31, 41, 55) 100%)`
              }}
            />
          </div>

          {/* Control Slider Refinado - Thermal Noise */}
          <div className="space-y-1.5">
            <div className="flex justify-between text-[10px]">
              <span className="text-gray-400">Ruido Térmico</span>
              <span className="font-mono text-pink-400/80 bg-pink-900/10 px-1.5 rounded border border-pink-500/10">{thermalNoise.toFixed(3)}</span>
            </div>
            {/* Slider usando input range HTML5 nativo para mejor UX */}
            <input
              type="range"
              min="0"
              max="0.01"
              step="0.0001"
              value={thermalNoise}
              onChange={(e) => handleThermalNoiseChange(parseFloat(e.target.value))}
              className="w-full h-1 bg-gray-800 rounded-full appearance-none cursor-pointer slider-thumb"
              style={{
                background: `linear-gradient(to right, rgb(217, 119, 6) 0%, rgb(217, 119, 6) ${(thermalNoise / 0.01) * 100}%, rgb(31, 41, 55) ${(thermalNoise / 0.01) * 100}%, rgb(31, 41, 55) 100%)`
              }}
            />
          </div>
        </div>

        {/* Sección: Motor de Simulación */}
        <div className="space-y-3 pt-4 border-t border-white/5">
          <div className="flex items-center gap-2 text-gray-200 text-xs font-bold uppercase tracking-wider mb-2">
            <Zap size={12} className="text-green-500" /> Motor de Simulación
          </div>
          
          <div className="grid grid-cols-2 gap-2">
            <button 
              className="flex flex-col items-center justify-center p-2 bg-white/5 hover:bg-white/10 border border-white/5 rounded text-xs text-gray-300 transition-all group"
              onClick={() => sendCommand('inference', 'switch_engine', { engine: 'python' })}
            >
              <span className="font-bold text-blue-400">Python</span>
              <span className="text-[9px] text-gray-500">Flexible</span>
            </button>
            <button 
              className="flex flex-col items-center justify-center p-2 bg-white/5 hover:bg-white/10 border border-white/5 rounded text-xs text-gray-300 transition-all group"
              onClick={() => sendCommand('inference', 'switch_engine', { engine: 'native' })}
            >
              <span className="font-bold text-orange-400">Nativo (C++)</span>
              <span className="text-[9px] text-gray-500">Alto Rendimiento</span>
            </button>
          </div>
        </div>


        {/* Sección: Versión del Visor (Holographic) */}
        <div className="space-y-3 pt-4 border-t border-white/5">
          <div className="flex items-center gap-2 text-gray-200 text-xs font-bold uppercase tracking-wider mb-2">
            <Microscope size={12} className="text-purple-500" /> Versión Holográfica
          </div>
          
          <div className="flex bg-black/20 p-1 rounded border border-white/5">
            <button 
              className={`flex-1 py-1.5 text-[10px] font-medium rounded transition-all ${
                viewerVersion === 'v1' 
                  ? 'bg-purple-500/20 text-purple-200 border border-purple-500/30' 
                  : 'text-gray-500 hover:text-gray-300'
              }`}
              onClick={() => onViewerVersionChange?.('v1')}
            >
              v1.0 (Poincaré)
            </button>
            <button 
              className={`flex-1 py-1.5 text-[10px] font-medium rounded transition-all ${
                viewerVersion === 'v2' 
                  ? 'bg-purple-500/20 text-purple-200 border border-purple-500/30' 
                  : 'text-gray-500 hover:text-gray-300'
              }`}
              onClick={() => onViewerVersionChange?.('v2')}
            >
              v2.0 (AdS/CFT)
            </button>
          </div>
        </div>
        {/* Sección: Inyección (Génesis) */}
        <div className="space-y-3 pt-4 border-t border-white/5">
          <div className="flex items-center gap-2 text-gray-200 text-xs font-bold uppercase tracking-wider mb-2">
            <Zap size={12} className="text-yellow-500" /> Inyección de Energía
          </div>
          
          <div className="grid grid-cols-1 gap-2">
            <button 
              className="flex items-center justify-between px-3 py-2 bg-white/5 hover:bg-white/10 border border-white/5 rounded text-xs text-gray-300 transition-all group"
              onClick={() => sendCommand('inference', 'inject_energy', { type: 'primordial_soup' })}
            >
              <span>Sopa Primordial</span>
              <ChevronRight size={12} className="text-gray-600 group-hover:text-white" />
            </button>
            <button 
              className="flex items-center justify-between px-3 py-2 bg-white/5 hover:bg-white/10 border border-white/5 rounded text-xs text-gray-300 transition-all group"
              onClick={() => sendCommand('inference', 'inject_energy', { type: 'dense_monolith' })}
            >
              <span>Monolito Denso</span>
              <ChevronRight size={12} className="text-gray-600 group-hover:text-white" />
            </button>
            <button 
              className="flex items-center justify-between px-3 py-2 bg-blue-500/10 hover:bg-blue-500/20 border border-blue-500/20 rounded text-xs text-blue-200 transition-all group"
              onClick={() => sendCommand('inference', 'inject_energy', { type: 'symmetric_seed' })}
            >
              <span className="flex items-center gap-2"><Microscope size={12}/> Semilla Simétrica</span>
              <ChevronRight size={12} className="text-blue-500" />
            </button>
          </div>
        </div>

        {/* Sección: Debug de Campos (Legacy - Removido por Selector Prominente) */}
        
        {/* Mensaje de Alerta (Contextual) - Si hay logs de error */}
        {recentLogs.some(log => typeof log === 'string' && log.toLowerCase().includes('error')) && (
          <div className="mt-4 p-3 rounded bg-red-500/5 border border-red-500/10 flex gap-2 items-start">
            <AlertCircle size={14} className="text-red-500 shrink-0 mt-0.5" />
            <div className="flex flex-col">
              <span className="text-[10px] font-bold text-red-400 uppercase">Inestabilidad Detectada</span>
              <span className="text-[10px] text-red-300/70 leading-tight">
                El vacío armónico muestra picos de energía anómalos en el cuadrante negativo.
              </span>
            </div>
          </div>
        )}

      </div>
      )}

      {/* Footer del Sidebar - Removido botón duplicado (está en MetricsBar) */}
    </aside>
  );
};
