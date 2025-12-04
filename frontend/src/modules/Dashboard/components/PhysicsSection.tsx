import { AlertCircle, ChevronRight, Microscope, Zap, Clock } from 'lucide-react';
import React, { useState } from 'react';
import { useWebSocket } from '../../../hooks/useWebSocket';
import { ScientificMetrics } from './ScientificMetrics';
import { QuantumToolbox } from '../../../components/QuantumToolbox';
import { QuantumControls } from './QuantumControls';

interface PhysicsSectionProps {
  // Section doesn't handle collapse, parent does
}

export const PhysicsSection: React.FC<PhysicsSectionProps> = () => {
  const { sendCommand, allLogs } = useWebSocket();
  const [showQuantumControls, setShowQuantumControls] = useState(false);

  // Filtrar logs recientes (últimos 2)
  const recentLogs = allLogs?.slice(-2) || [];

  const handleOpenQuantum = () => {
    setShowQuantumControls(true);
  };

  return (
    <div className="p-4 space-y-6">
      {/* Quantum Controls Modal */}
      <QuantumControls
        isOpen={showQuantumControls}
        onClose={() => {
            setShowQuantumControls(false);
        }}
      />

      {/* Métricas Científicas (Full Display) */}
      <div className="space-y-3">
        <ScientificMetrics compact={false} />
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

      {/* Quantum Toolbox */}
      <QuantumToolbox />

      {/* Quantum Time Warp Trigger */}
      <div className="pt-2 border-t border-white/5">
         <button
            onClick={handleOpenQuantum}
            className="w-full flex items-center justify-between px-3 py-2 bg-indigo-900/20 hover:bg-indigo-900/40 border border-indigo-500/20 rounded text-xs text-indigo-300 transition-all group"
         >
            <div className="flex items-center gap-2">
                <Clock size={12} className="text-indigo-400" />
                <span className="font-bold">Quantum Fast Forward</span>
            </div>
            <ChevronRight size={12} className="text-indigo-500 group-hover:translate-x-0.5 transition-transform" />
         </button>
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
  );
};
