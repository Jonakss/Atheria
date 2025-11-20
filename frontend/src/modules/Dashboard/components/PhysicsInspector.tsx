import React, { useState } from 'react';
import { Settings, Zap, ChevronRight, MoreHorizontal, AlertCircle, Microscope, ChevronLeft } from 'lucide-react';
import { useWebSocket } from '../../../hooks/useWebSocket';

interface PhysicsInspectorProps {
  isCollapsed?: boolean;
  onToggleCollapse?: () => void;
}

export const PhysicsInspector: React.FC<PhysicsInspectorProps> = ({ 
  isCollapsed = false, 
  onToggleCollapse 
}) => {
  const { sendCommand, allLogs, connectionStatus } = useWebSocket();
  const [gammaDecay, setGammaDecay] = useState(0.015);
  const [thermalNoise, setThermalNoise] = useState(0.002);
  const [selectedLayer, setSelectedLayer] = useState(0);
  const [internalCollapsed, setInternalCollapsed] = useState(false);
  
  // Usar prop externo si está disponible, sino usar estado interno
  const collapsed = onToggleCollapse !== undefined ? isCollapsed : internalCollapsed;
  const handleToggle = onToggleCollapse || (() => setInternalCollapsed(!internalCollapsed));

  const layers = ['Densidad (Scalar)', 'Fase (Complex)', 'Flujo (Vector)', 'Chunks (Meta)'];

  const handleGammaChange = (value: number) => {
    setGammaDecay(value);
    // Enviar configuración al backend (formato correcto: snake_case)
    sendCommand('inference', 'set_config', {
      gamma_decay: value
    });
  };

  const handleThermalNoiseChange = (value: number) => {
    setThermalNoise(value);
    // Thermal noise podría ser parte de la configuración de ruido cuántico
    // Por ahora solo guardamos el estado local
    // TODO: Integrar con sistema de ruido cuántico si existe
  };

  // Filtrar logs recientes (últimos 2)
  const recentLogs = allLogs?.slice(-2) || [];

  return (
    <aside className={`${collapsed ? 'w-10' : 'w-72'} border-l border-white/10 bg-[#080808] flex flex-col z-40 shrink-0 flex-shrink-0 transition-all duration-300 ease-in-out overflow-hidden`} style={{ minWidth: collapsed ? '2.5rem' : '18rem', maxWidth: collapsed ? '2.5rem' : '18rem' }}>
      {/* Título Panel - Con botón de colapsar */}
      <div className="h-10 border-b border-white/5 flex items-center justify-between px-4 bg-[#0a0a0a] shrink-0">
        {!collapsed && (
          <span className="text-[10px] font-bold text-gray-400 uppercase tracking-widest">Inspector Físico</span>
        )}
        <button
          onClick={handleToggle}
          className="w-6 h-6 flex items-center justify-center text-gray-600 hover:text-gray-300 hover:bg-white/5 rounded transition-all"
          title={collapsed ? 'Expandir Inspector' : 'Colapsar Inspector'}
        >
          {collapsed ? <ChevronLeft size={14} /> : <ChevronRight size={14} />}
        </button>
      </div>

      {!collapsed && (
      <div className="flex-1 overflow-y-auto p-4 space-y-8" style={{ scrollbarWidth: 'thin', scrollbarColor: 'rgba(255,255,255,0.1) transparent' }}>
        
        {/* Sección: Parámetros Globales */}
        <div className="space-y-3">
          <div className="flex items-center gap-2 text-gray-200 text-xs font-bold uppercase tracking-wider mb-2">
            <Settings size={12} className="text-blue-500" /> Variables de Entorno
          </div>
          
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
              max="0.1"
              step="0.001"
              value={gammaDecay}
              onChange={(e) => handleGammaChange(parseFloat(e.target.value))}
              className="w-full h-1 bg-gray-800 rounded-full appearance-none cursor-pointer slider-thumb"
              style={{
                background: `linear-gradient(to right, rgb(107, 114, 128) 0%, rgb(107, 114, 128) ${(gammaDecay / 0.1) * 100}%, rgb(31, 41, 55) ${(gammaDecay / 0.1) * 100}%, rgb(31, 41, 55) 100%)`
              }}
            />
          </div>

          {/* Control Slider Refinado - Thermal Noise */}
          <div className="space-y-1.5">
            <div className="flex justify-between text-[10px]">
              <span className="text-gray-400">Ruido Térmico</span>
              <span className="font-mono text-amber-400/80 bg-amber-900/10 px-1.5 rounded border border-amber-500/10">{thermalNoise.toFixed(3)}</span>
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

        {/* Sección: Debug de Campos */}
        <div className="space-y-3 pt-4 border-t border-white/5">
          <span className="text-[10px] font-bold text-gray-500 uppercase tracking-wider">Visualización de Campos</span>
          <div className="space-y-1">
            {layers.map((layer, i) => (
              <label 
                key={layer} 
                className="flex items-center gap-2 cursor-pointer p-1.5 rounded hover:bg-white/5 transition-colors"
                onClick={() => setSelectedLayer(i)}
              >
                <div className={`w-3 h-3 rounded-sm border ${
                  i === selectedLayer 
                    ? 'bg-blue-500 border-blue-500' 
                    : 'border-gray-600 bg-transparent'
                } flex items-center justify-center`}>
                  {i === selectedLayer && <div className="w-1.5 h-1.5 bg-white rounded-[1px]" />}
                </div>
                <span className={`text-xs ${i === selectedLayer ? 'text-gray-200 font-medium' : 'text-gray-500'}`}>
                  {layer}
                </span>
              </label>
            ))}
          </div>
        </div>
        
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
