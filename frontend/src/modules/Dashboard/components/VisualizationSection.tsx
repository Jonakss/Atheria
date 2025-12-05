import { Activity, Box, Eye, Globe, Layers, Maximize, Microscope, Minimize, Palette, Waves, Zap } from 'lucide-react';
import React, { useState } from 'react';
import { useWebSocket } from '../../../hooks/useWebSocket';
import { Tooltip } from '../../../components/ui/common/Tooltip';

interface VisualizationSectionProps {
  viewerVersion?: 'v1' | 'v2';
  onViewerVersionChange?: (version: 'v1' | 'v2') => void;
  selectedLayer?: number;
  onLayerChange?: (layer: number) => void;
  theaterMode?: boolean;
  onToggleTheaterMode?: (enabled: boolean) => void;
}

export const VisualizationSection: React.FC<VisualizationSectionProps> = ({
  viewerVersion = 'v1',
  onViewerVersionChange,
  selectedLayer = 0,
  onLayerChange,
  theaterMode = false,
  onToggleTheaterMode
}) => {
  const { sendCommand, selectedViz, setSelectedViz, roiInfo } = useWebSocket();
  const [gammaDecay, setGammaDecay] = useState(0.015);
  const [thermalNoise, setThermalNoise] = useState(0.002);
  const [pointSize, setPointSize] = useState(1.0);
  const [threshold, setThreshold] = useState(0.01); // Umbral más bajo para ver más estructura
  const [renderMode, setRenderMode] = useState<'points' | 'wireframe' | 'mesh'>('points');
  
  const [internalLayer, setInternalLayer] = useState(0);
  const activeLayer = onLayerChange ? selectedLayer : internalLayer;

  const handleModeChange = (mode: any) => {
    // Update local state (for UI highlighting)
    if (onLayerChange) {
      onLayerChange(mode.id);
    } else {
      setInternalLayer(mode.id);
    }

    // Trigger backend visualization change
    if (mode.vizType) {
        setSelectedViz(mode.vizType);
        sendCommand('inference', 'set_viz', { viz_type: mode.vizType });
    }
  };

  const VIZ_MODES = [
    { id: 0, vizType: 'density', label: 'Densidad', desc: 'Magnitud de la función de onda |ψ|². Muestra la probabilidad de presencia.', icon: <Activity size={18} />, color: 'text-blue-400', border: 'border-blue-500/50' },
    { id: 1, vizType: 'phase', label: 'Fase', desc: 'Argumento complejo φ. Mapea la rotación cuántica a color (0-2π).', icon: <Box size={18} />, color: 'text-purple-400', border: 'border-purple-500/50' },
    { id: 2, vizType: 'flow', label: 'Flujo', desc: 'Corriente de probabilidad J. Muestra la dirección del movimiento.', icon: <Waves size={18} />, color: 'text-cyan-400', border: 'border-cyan-500/50' },
    { id: 3, vizType: 'spectral', label: 'Espectral', desc: 'Transformada de Fourier (FFT). Muestra la distribución de energía en frecuencia.', icon: <Zap size={18} />, color: 'text-yellow-400', border: 'border-yellow-500/50' },
    { id: 4, vizType: 'fields', label: 'Campos', desc: 'Teoría de Campos. Visualización RGB de los canales (EM, Gravedad, Higgs).', icon: <Layers size={18} />, color: 'text-green-400', border: 'border-green-500/50' },
    { id: 5, vizType: 'poincare', label: 'Poincaré', desc: 'Esfera de Bloch/Poincaré proyectada. Mapea la polarización del estado.', icon: <Globe size={18} />, color: 'text-indigo-400', border: 'border-indigo-500/50' },
    { id: 6, vizType: 'phase_hsv', label: 'Fase HSV', desc: 'Mapeo HSV completo de amplitud (valor) y fase (hue).', icon: <Palette size={18} />, color: 'text-pink-400', border: 'border-pink-500/50' },
    { id: 7, vizType: 'holographic', label: 'Holográfico', desc: 'Renderizado volumétrico 3D del bulk AdS.', icon: <Box size={18} />, color: 'text-teal-400', border: 'border-teal-500/50' },
    { id: 8, vizType: 'poincare_3d', label: 'Poincaré 3D', desc: 'Visualización 3D de la esfera de Poincaré.', icon: <Globe size={18} />, color: 'text-rose-400', border: 'border-rose-500/50' },
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
    <div className="p-4 space-y-6">
      {/* Visualization Modes */}
      <div className="space-y-3">
        <div className="flex items-center gap-2 text-gray-200 text-xs font-bold uppercase tracking-wider mb-2">
          <Eye size={12} className="text-blue-500" /> Modos de Vista
        </div>
        <div className="grid grid-cols-3 gap-2">
          {VIZ_MODES.map((mode) => (
            <Tooltip key={mode.id} label={<div className="max-w-[200px] text-center"><p className="font-bold mb-1">{mode.label}</p><p className="text-gray-400 leading-tight">{mode.desc}</p></div>}>
                <button
                onClick={() => handleModeChange(mode)}
                className={`flex flex-col items-center justify-center p-3 rounded-lg border transition-all duration-200 h-20 ${
                    (activeLayer === mode.id || selectedViz === mode.vizType)
                    ? `bg-white/10 ${mode.border} shadow-lg`
                    : 'bg-white/5 border-white/5 hover:bg-white/10 hover:border-white/10'
                }`}
                >
                <div className={`mb-1 ${(activeLayer === mode.id || selectedViz === mode.vizType) ? mode.color : 'text-gray-400'}`}>
                    {mode.icon}
                </div>
                <span className={`text-[10px] font-bold ${(activeLayer === mode.id || selectedViz === mode.vizType) ? mode.color : 'text-gray-400'}`}>
                    {mode.label}
                </span>
                </button>
            </Tooltip>
          ))}
        </div>
      </div>

      {/* ROI Control */}
      <div className="space-y-3 pt-4 border-t border-white/5">
        <div className="flex items-center gap-2 text-gray-200 text-xs font-bold uppercase tracking-wider mb-2">
          <Maximize size={12} className="text-orange-500" /> Control de Vista
        </div>
        
        <button
          onClick={() => sendCommand('inference', 'set_roi_mode', { enabled: !roiInfo?.enabled })}
          className={`w-full flex items-center justify-between p-2 rounded border transition-all ${
            !roiInfo?.enabled
              ? 'bg-orange-500/20 border-orange-500/50 text-orange-200'
              : 'bg-white/5 border-white/10 text-gray-400 hover:bg-white/10'
          }`}
        >
          <div className="flex items-center gap-2">
            {!roiInfo?.enabled ? <Minimize size={14} /> : <Maximize size={14} />}
            <span className="text-xs font-medium">
              {!roiInfo?.enabled ? 'Vista Completa (See All)' : 'Vista Enfocada (ROI)'}
            </span>
          </div>
          {roiInfo && (
            <span className="text-[9px] font-mono opacity-70">
              {roiInfo.enabled 
                ? `${roiInfo.width}x${roiInfo.height}` 
                : 'Full Grid'}
            </span>
          )}
        </button>
        
        {roiInfo?.enabled && (
          <div className="text-[9px] text-gray-500 px-1">
            <p>Viendo región de interés optimizada.</p>
            <p>Ratio de reducción: {roiInfo.reduction_ratio?.toFixed(1)}x</p>
          </div>
        )}

        {/* Theater Mode Toggle */}
        <button
          onClick={() => onToggleTheaterMode?.(!theaterMode)}
          className={`w-full flex items-center justify-between p-2 rounded border transition-all mt-2 ${
            theaterMode
              ? 'bg-indigo-500/20 border-indigo-500/50 text-indigo-200'
              : 'bg-white/5 border-white/10 text-gray-400 hover:bg-white/10'
          }`}
        >
          <div className="flex items-center gap-2">
            {theaterMode ? <Minimize size={14} /> : <Maximize size={14} />}
            <span className="text-xs font-medium">
              {theaterMode ? 'Salir Modo Teatro' : 'Modo Teatro'}
            </span>
          </div>
        </button>
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

      {/* Holographic Viewer Controls (solo visible si está en modo 3D) */}
      {(selectedViz === 'holographic' || selectedViz === 'poincare_3d' || selectedViz === '3d') && (
        <div className="space-y-3 pt-4 border-t border-white/5">
          <div className="flex items-center gap-2 text-gray-200 text-xs font-bold uppercase tracking-wider mb-2">
            <Box size={12} className="text-pink-500" /> Renderizado 3D
          </div>
          
          {/* Render Mode */}
          <div className="space-y-1.5">
            <label className="text-[10px] text-gray-400 uppercase">Modo de Renderizado</label>
            <div className="grid grid-cols-3 gap-1 bg-black/20 p-1 rounded border border-white/5">
              <button 
                className={`py-1 text-[9px] font-medium rounded transition-all ${
                  renderMode === 'points' 
                    ? 'bg-pink-500/20 text-pink-200 border border-pink-500/30' 
                    : 'text-gray-500 hover:text-gray-300'
                }`}
                onClick={() => setRenderMode('points')}
              >
                Puntos
              </button>
              <button 
                className={`py-1 text-[9px] font-medium rounded transition-all ${
                  renderMode === 'wireframe' 
                    ? 'bg-pink-500/20 text-pink-200 border border-pink-500/30' 
                    : 'text-gray-500 hover:text-gray-300'
                }`}
                onClick={() => setRenderMode('wireframe')}
              >
                Wireframe
              </button>
              <button 
                className={`py-1 text-[9px] font-medium rounded transition-all ${
                  renderMode === 'mesh' 
                    ? 'bg-pink-500/20 text-pink-200 border border-pink-500/30' 
                    : 'text-gray-500 hover:text-gray-300'
                }`}
                onClick={() => setRenderMode('mesh')}
              >
                Mesh
              </button>
            </div>
          </div>

          {/* Point Size */}
          <div className="space-y-1.5">
            <div className="flex justify-between text-[10px]">
              <span className="text-gray-400">Tamaño de Punto</span>
              <span className="font-mono text-gray-200 bg-white/5 px-1.5 rounded">{pointSize.toFixed(1)}x</span>
            </div>
            <input
              type="range"
              min="0.5"
              max="5.0"
              step="0.1"
              value={pointSize}
              onChange={(e) => setPointSize(parseFloat(e.target.value))}
              className="w-full h-1 bg-gray-800 rounded-full appearance-none cursor-pointer slider-thumb"
              style={{
                background: `linear-gradient(to right, rgb(236, 72, 153) 0%, rgb(236, 72, 153) ${((pointSize - 0.5) / 4.5) * 100}%, rgb(31, 41, 55) ${((pointSize - 0.5) / 4.5) * 100}%, rgb(31, 41, 55) 100%)`
              }}
            />
          </div>

          {/* Threshold */}
          <div className="space-y-1.5">
            <div className="flex justify-between text-[10px]">
              <span className="text-gray-400">Umbral de Densidad</span>
              <span className="font-mono text-gray-200 bg-white/5 px-1.5 rounded">{threshold.toFixed(3)}</span>
            </div>
            <input
              type="range"
              min="0.001"
              max="0.5"
              step="0.001"
              value={threshold}
              onChange={(e) => setThreshold(parseFloat(e.target.value))}
              className="w-full h-1 bg-gray-800 rounded-full appearance-none cursor-pointer slider-thumb"
              style={{
                background: `linear-gradient(to right, rgb(14, 165, 233) 0%, rgb(14, 165, 233) ${(threshold / 0.5) * 100}%, rgb(31, 41, 55) ${(threshold / 0.5) * 100}%, rgb(31, 41, 55) 100%)`
              }}
            />
          </div>
        </div>
      )}
    </div>
  );
};
