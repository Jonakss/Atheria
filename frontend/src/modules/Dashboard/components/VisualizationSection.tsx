import { Activity, Box, Eye, Globe, Layers, Maximize, Minimize, Palette, Waves, Zap } from 'lucide-react';
import React, { useState } from 'react';
import { Tooltip } from '../../../components/ui/common/Tooltip';
import { useWebSocket } from '../../../hooks/useWebSocket';

interface VisualizationSectionProps {
  selectedLayer?: number;
  onLayerChange?: (layer: number) => void;
  theaterMode?: boolean;
  onToggleTheaterMode?: (enabled: boolean) => void;
  // Gateway Props
  binaryMode?: boolean;
  onToggleBinaryMode?: (enabled: boolean) => void;
  binaryThreshold?: number;
  onChangeBinaryThreshold?: (val: number) => void;
  binaryColor?: string;
  onChangeBinaryColor?: (val: string) => void;
  // 3D Visualizer Props
  pointSize?: number;
  onPointSizeChange?: (val: number) => void;
  densityThreshold?: number;
  onDensityThresholdChange?: (val: number) => void;
  renderMode?: 'points' | 'wireframe' | 'mesh';
  onRenderModeChange?: (val: 'points' | 'wireframe' | 'mesh') => void;
}

export const VisualizationSection: React.FC<VisualizationSectionProps> = ({
  selectedLayer = 0,
  onLayerChange,
  theaterMode = false,
  onToggleTheaterMode,
  binaryMode = false,
  onToggleBinaryMode,
  binaryThreshold = 0.5,
  onChangeBinaryThreshold,
  binaryColor = '#FFFFFF',
  onChangeBinaryColor,
  pointSize = 1.0,
  onPointSizeChange,
  densityThreshold = 0.01,
  onDensityThresholdChange,
  renderMode = 'points',
  onRenderModeChange
}) => {
  const { sendCommand, selectedViz, setSelectedViz, roiInfo } = useWebSocket();
  const [gammaDecay, setGammaDecay] = useState(0.015);
  const [thermalNoise, setThermalNoise] = useState(0.002);
  
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
    { id: 9, vizType: 'real', label: 'Parte Real', desc: 'Visualiza la parte real de la Función de Onda.', icon: <Activity size={18} />, color: 'text-emerald-400', border: 'border-emerald-500/50' },
    { id: 10, vizType: 'imag', label: 'Parte Imag', desc: 'Visualiza la parte imaginaria de la Función de Onda.', icon: <Activity size={18} />, color: 'text-fuchsia-400', border: 'border-fuchsia-500/50' },
    { id: 11, vizType: 'holographic_bulk', label: 'Bulk', desc: 'Holografía real 3D usando scale-space.', icon: <Layers size={18} />, color: 'text-orange-400', border: 'border-orange-500/50' },
  ];

  const handleGammaChange = (value: number) => {
    setGammaDecay(value);
    sendCommand('inference', 'set_config', { gamma_decay: value });
  };

  const handleThermalNoiseChange = (value: number) => {
    setThermalNoise(value);
    // TODO: Implement backend support
  };

  // Gateway Phase 2: Click-Out State
  const [clickOutEnabled, setClickOutEnabled] = useState(false);
  const [clickOutChance, setClickOutChance] = useState(0.01);

  const toggleClickOut = () => {
      const newState = !clickOutEnabled;
      setClickOutEnabled(newState);
      sendCommand('inference', 'apply_tool', { 
          action: 'set_click_out', 
          params: {
            enabled: newState, 
            chance: clickOutChance 
          }
      });
  };

  const handleClickOutChanceChange = (value: number) => {
      setClickOutChance(value);
      if (clickOutEnabled) {
          sendCommand('inference', 'apply_tool', { 
              action: 'set_click_out', 
              params: {
                enabled: true, 
                chance: value 
              }
          });
      }
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

      {/* Gateway Process: Observer Control */}
      <div className="space-y-3 pt-4 border-t border-white/5">
        <div className="flex items-center gap-2 text-gray-200 text-xs font-bold uppercase tracking-wider mb-2">
          <Eye size={12} className="text-purple-500" /> Control del Observador (Gateway)
        </div>
        
        {/* Binary Mode Toggle */}
        <button
          onClick={() => onToggleBinaryMode?.(!binaryMode)}
          className={`w-full flex items-center justify-between p-2 rounded border transition-all ${
            binaryMode
              ? 'bg-purple-500/20 border-purple-500/50 text-purple-200 shadow-[0_0_15px_rgba(168,85,247,0.3)]'
              : 'bg-white/5 border-white/10 text-gray-400 hover:bg-white/10'
          }`}
        >
          <div className="flex items-center gap-2">
            {binaryMode ? <Box size={14} /> : <Activity size={14} />}
            <div className="flex flex-col items-start">
               <span className="text-xs font-medium">
                 {binaryMode ? 'Modo Binario (Materia)' : 'Modo Holográfico (Onda)'}
               </span>
               <span className="text-[9px] opacity-70">
                 {binaryMode ? 'Colapso de Función de Onda' : 'Interferencia Continua'}
               </span>
            </div>
          </div>
          <div className={`w-2 h-2 rounded-full ${binaryMode ? 'bg-purple-400 animate-pulse' : 'bg-gray-600'}`} />
        </button>

        {/* Reality Threshold Slider - Only visible in Binary Mode */}
        {binaryMode && (
          <div className="space-y-1.5 animate-in fade-in slide-in-from-top-2 duration-300">
             <div className="flex items-center justify-between">
                  <span className="text-[10px] text-gray-400">Umbral de Realidad (Click-out)</span>
                  <span className="text-[10px] font-mono text-purple-300">{binaryThreshold?.toFixed(2)}</span>
             </div>
             <input 
                 type="range" 
                 min="0" max="1" step="0.01"
                 value={binaryThreshold}
                 onChange={(e) => onChangeBinaryThreshold?.(parseFloat(e.target.value))}
                 className="w-full accent-purple-500 h-1 bg-gray-700 rounded-lg appearance-none cursor-pointer"
             />
             
             {/* Add Color Picker */}
             <div className="flex items-center justify-between mt-2 pt-2 border-t border-white/5">
                  <span className="text-[10px] text-gray-400">Color de Materia</span>
                  <div className="flex items-center gap-2">
                      <span className="text-[10px] font-mono text-gray-500">{binaryColor}</span>
                      <input 
                         type="color" 
                         value={binaryColor}
                         onChange={(e) => onChangeBinaryColor?.(e.target.value)}
                         className="w-6 h-6 rounded border-none cursor-pointer bg-transparent"
                      />
                  </div>
             </div>
            <p className="text-[9px] text-gray-500 italic mt-1">
               &quot;La materia emerge cuando la energía supera el umbral del observador.&quot;
            </p>
          </div>
        )}

        {/* Click-Out Mechanism (Phase 2) */}
        <div className="pt-2 mt-2 border-t border-white/5">
            <button
            onClick={toggleClickOut}
            className={`w-full flex items-center justify-between p-2 rounded border transition-all ${
                clickOutEnabled
                ? 'bg-cyan-500/20 border-cyan-500/50 text-cyan-200'
                : 'bg-white/5 border-white/10 text-gray-400 hover:bg-white/10'
            }`}
            >
            <div className="flex items-center gap-2">
                <Zap size={14} className={clickOutEnabled ? "text-cyan-400" : "text-gray-600"} />
                <span className="text-xs font-medium">Mecánica Click-Out</span>
            </div>
            <div className={`w-2 h-2 rounded-full ${clickOutEnabled ? 'bg-cyan-400 animate-ping' : 'bg-gray-600'}`} />
            </button>

            {clickOutEnabled && (
            <div className="space-y-1.5 mt-2 animate-in fade-in slide-in-from-left-2 duration-300">
                <div className="flex justify-between text-[10px]">
                <span className="text-gray-400">Probabilidad de Túnel</span>
                <span className="font-mono text-cyan-200 bg-cyan-900/40 px-1.5 rounded border border-cyan-500/30">{(clickOutChance * 100).toFixed(1)}%</span>
                </div>
                <input
                type="range"
                min="0.0"
                max="0.1"
                step="0.001"
                value={clickOutChance}
                onChange={(e) => handleClickOutChanceChange(parseFloat(e.target.value))}
                className="w-full h-1 bg-gray-800 rounded-full appearance-none cursor-pointer slider-thumb"
                style={{
                    background: `linear-gradient(to right, rgb(6, 182, 212) 0%, rgb(6, 182, 212) ${(clickOutChance / 0.1) * 100}%, rgb(31, 41, 55) ${(clickOutChance / 0.1) * 100}%, rgb(31, 41, 55) 100%)`
                }}
                />
                <p className="text-[9px] text-gray-500 italic mt-1">
                 &quot;Resonancia no local a través del vacío.&quot;
                </p>
            </div>
            )}
        </div>
        
        {/* End Gateway Control */}
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
                onClick={() => onRenderModeChange?.('points')}
              >
                Puntos
              </button>
              <button 
                className={`py-1 text-[9px] font-medium rounded transition-all ${
                  renderMode === 'wireframe' 
                    ? 'bg-pink-500/20 text-pink-200 border border-pink-500/30' 
                    : 'text-gray-500 hover:text-gray-300'
                }`}
                onClick={() => onRenderModeChange?.('wireframe')}
              >
                Wireframe
              </button>
              <button 
                className={`py-1 text-[9px] font-medium rounded transition-all ${
                  renderMode === 'mesh' 
                    ? 'bg-pink-500/20 text-pink-200 border border-pink-500/30' 
                    : 'text-gray-500 hover:text-gray-300'
                }`}
                onClick={() => onRenderModeChange?.('mesh')}
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
              onChange={(e) => onPointSizeChange?.(parseFloat(e.target.value))}
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
              <span className="font-mono text-gray-200 bg-white/5 px-1.5 rounded">{densityThreshold.toFixed(3)}</span>
            </div>
            <input
              type="range"
              min="0.001"
              max="0.5"
              step="0.001"
              value={densityThreshold}
              onChange={(e) => onDensityThresholdChange?.(parseFloat(e.target.value))}
              className="w-full h-1 bg-gray-800 rounded-full appearance-none cursor-pointer slider-thumb"
              style={{
                background: `linear-gradient(to right, rgb(14, 165, 233) 0%, rgb(14, 165, 233) ${(densityThreshold / 0.5) * 100}%, rgb(31, 41, 55) ${(densityThreshold / 0.5) * 100}%, rgb(31, 41, 55) 100%)`
              }}
            />
          </div>
        </div>
      )}
    </div>
  );
};
