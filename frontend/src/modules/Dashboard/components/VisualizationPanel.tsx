import { Activity, ChevronRight, Eye, Microscope, Zap } from 'lucide-react';
import React, { useState } from 'react';
import { useWebSocket } from '../../../hooks/useWebSocket';

interface VisualizationPanelProps {
  isCollapsed?: boolean;
  onToggleCollapse?: () => void;
  viewerVersion?: 'v1' | 'v2';
  onViewerVersionChange?: (version: 'v1' | 'v2') => void;
  selectedLayer?: number;
  onLayerChange?: (layer: number) => void;
}

export const VisualizationPanel: React.FC<VisualizationPanelProps> = ({
  isCollapsed = false,
  onToggleCollapse,
  viewerVersion = 'v1',
  onViewerVersionChange,
  selectedLayer = 0,
  onLayerChange
}) => {
  const { sendCommand } = useWebSocket();
  const [gammaDecay, setGammaDecay] = useState(0.015);
  const [thermalNoise, setThermalNoise] = useState(0.002);
  
  // Internal state fallback if prop not provided
  const [internalLayer, setInternalLayer] = useState(0);
  const [internalCollapsed, setInternalCollapsed] = useState(false);
  
  const activeLayer = onLayerChange ? selectedLayer : internalLayer;
  const collapsed = onToggleCollapse !== undefined ? isCollapsed : internalCollapsed;
  const handleToggle = onToggleCollapse || (() => setInternalCollapsed(!internalCollapsed));

  const handleLayerChange = (layer: number) => {
    if (onLayerChange) {
      onLayerChange(layer);
    } else {
      setInternalLayer(layer);
    }
  };

  const VIZ_MODES = [
    { id: 0, label: 'Densidad', desc: 'Amplitud |ψ|', icon: '📊', color: 'text-blue-400', border: 'border-blue-500/50' },
    { id: 1, label: 'Fase', desc: 'Argumento φ', icon: '🌈', color: 'text-purple-400', border: 'border-purple-500/50' },
    { id: 2, label: 'Energía', desc: 'Hamiltoniano', icon: '⚡', color: 'text-yellow-400', border: 'border-yellow-500/50' },
    { id: 3, label: 'Flujo', desc: 'Corriente J', icon: '🌊', color: 'text-cyan-400', border: 'border-cyan-500/50' },
    { id: 4, label: 'Chunks', desc: 'Meta-Estructura', icon: '🧩', color: 'text-green-400', border: 'border-green-500/50' }
  ];

  const handleGammaChange = (value: number) => {
    setGammaDecay(value);
    sendCommand('inference', 'set_config', { gamma_decay: value });
  };

  const handleThermalNoiseChange = (value: number) => {
    setThermalNoise(value);
    // TODO: Implement backend support
  };

  return (
    <aside className={`${collapsed ? 'w-10' : 'w-72'} border-l border-white/5 bg-dark-950/80 backdrop-blur-md flex flex-col z-40 shrink-0 flex-shrink-0 transition-all duration-300 ease-in-out overflow-hidden h-full`} style={{ minWidth: collapsed ? '2.5rem' : '18rem', maxWidth: collapsed ? '2.5rem' : '18rem' }}>
      
      {/* Header */}
      <div className="h-10 border-b border-white/5 flex items-center justify-between px-3 bg-dark-980/90 shrink-0">
        {!collapsed && (
          <span className="text-[10px] font-bold text-dark-400 uppercase tracking-widest flex items-center gap-2">
            <Eye size={12} /> Visualización
          </span>
        )}
        <button
          onClick={handleToggle}
          className="p-1.5 text-dark-500 hover:text-dark-300 transition-colors rounded hover:bg-white/5"
          title={collapsed ? 'Expandir Panel' : 'Colapsar Panel'}
        >
          {collapsed ? <ChevronRight size={14} /> : <ChevronRight size={14} className="rotate-180" />}
        </button>
      </div>

      {!collapsed && (
        <div className="flex-1 overflow-y-auto p-4 space-y-6" style={{ scrollbarWidth: 'thin', scrollbarColor: 'rgba(255,255,255,0.1) transparent' }}>
          
          {/* Visualization Modes */}
          <div className="space-y-3">
            <div className="flex items-center gap-2 text-gray-200 text-xs font-bold uppercase tracking-wider mb-2">
              <Activity size={12} className="text-blue-500" /> Modos de Vista
            </div>
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

          {/* Global Parameters */}
          <div className="space-y-4 pt-4 border-t border-white/5">
            <div className="flex items-center gap-2 text-gray-200 text-xs font-bold uppercase tracking-wider mb-2">
              <Activity size={12} className="text-yellow-500" /> Parámetros Globales
            </div>
            
            {/* Gamma Slider */}
            <div className="space-y-1.5">
              <div className="flex justify-between text-[10px]">
                <span className="text-gray-400">Gamma (Disipación)</span>
                <span className="font-mono text-gray-200 bg-white/5 px-1.5 rounded">{gammaDecay.toFixed(3)}</span>
              </div>
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

            {/* Thermal Noise Slider */}
            <div className="space-y-1.5">
              <div className="flex justify-between text-[10px]">
                <span className="text-gray-400">Ruido Térmico</span>
                <span className="font-mono text-pink-400/80 bg-pink-900/10 px-1.5 rounded border border-pink-500/10">{thermalNoise.toFixed(3)}</span>
              </div>
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

          {/* Engine Selection */}
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

          {/* Viewer Version */}
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

        </div>
      )}
    </aside>
  );
};
