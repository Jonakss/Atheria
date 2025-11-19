import React, { useState } from 'react';
import { Settings, Zap, ChevronRight, MoreHorizontal, AlertCircle, Microscope } from 'lucide-react';
import { useWebSocket } from '../../../hooks/useWebSocket';

export const PhysicsInspector: React.FC = () => {
  const { sendCommand, simData, allLogs } = useWebSocket();
  const [gammaDecay, setGammaDecay] = useState(0.015);
  const [thermalNoise, setThermalNoise] = useState(0.002);
  const [selectedLayer, setSelectedLayer] = useState(0);

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
    <aside className="w-72 border-l border-white/10 bg-[#080808] flex flex-col z-40 shrink-0">
      {/* Título Panel */}
      <div className="h-10 border-b border-white/5 flex items-center justify-between px-4 bg-[#0a0a0a]">
        <span className="text-[10px] font-bold text-gray-400 uppercase tracking-widest">Inspector Físico</span>
        <MoreHorizontal size={14} className="text-gray-600 cursor-pointer hover:text-gray-400" />
      </div>

      <div className="flex-1 overflow-y-auto p-4 space-y-8 custom-scrollbar">
        
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
            <div className="h-1 w-full bg-gray-800 rounded-full overflow-hidden cursor-pointer hover:bg-gray-700 transition-colors">
              <div 
                className="h-full bg-gray-500 cursor-pointer"
                style={{ width: `${(gammaDecay / 0.1) * 100}%` }}
                onClick={(e) => {
                  const rect = e.currentTarget.parentElement!.getBoundingClientRect();
                  const x = e.clientX - rect.left;
                  const newValue = (x / rect.width) * 0.1;
                  handleGammaChange(Math.max(0, Math.min(0.1, newValue)));
                }}
              />
            </div>
          </div>

          {/* Control Slider Refinado - Thermal Noise */}
          <div className="space-y-1.5">
            <div className="flex justify-between text-[10px]">
              <span className="text-gray-400">Ruido Térmico</span>
              <span className="font-mono text-amber-400/80 bg-amber-900/10 px-1.5 rounded border border-amber-500/10">{thermalNoise.toFixed(3)}</span>
            </div>
            <div className="h-1 w-full bg-gray-800 rounded-full overflow-hidden cursor-pointer hover:bg-gray-700 transition-colors">
              <div 
                className="h-full bg-amber-600 cursor-pointer"
                style={{ width: `${(thermalNoise / 0.01) * 100}%` }}
                onClick={(e) => {
                  const rect = e.currentTarget.parentElement!.getBoundingClientRect();
                  const x = e.clientX - rect.left;
                  const newValue = (x / rect.width) * 0.01;
                  handleThermalNoiseChange(Math.max(0, Math.min(0.01, newValue)));
                }}
              />
            </div>
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

      {/* Footer del Sidebar */}
      <div className="p-3 border-t border-white/5 bg-[#080808]">
        <button 
          className="w-full py-2 bg-gray-800 hover:bg-gray-700 text-gray-300 text-xs font-bold rounded border border-white/5 transition-all flex items-center justify-center gap-2"
          onClick={() => sendCommand('simulation', 'capture_snapshot', {})}
        >
          <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
            <path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4" />
            <polyline points="17 8 12 3 7 8" />
            <line x1="12" y1="3" x2="12" y2="15" />
          </svg>
          CAPTURAR ESTADO
        </button>
      </div>
    </aside>
  );
};
