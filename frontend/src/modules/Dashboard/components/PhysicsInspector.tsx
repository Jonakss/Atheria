import { AlertCircle, ChevronLeft, ChevronRight, Microscope, Zap } from 'lucide-react';
import React, { useState } from 'react';
import { useWebSocket } from '../../../hooks/useWebSocket';

interface PhysicsInspectorProps {
  isCollapsed?: boolean;
  onToggleCollapse?: () => void;
}

export const PhysicsInspector: React.FC<PhysicsInspectorProps> = ({ 
  isCollapsed = false, 
  onToggleCollapse
}) => {
  const { sendCommand, allLogs } = useWebSocket();
  // Internal state fallback if prop not provided
  const [internalCollapsed, setInternalCollapsed] = useState(false);
  
  // Usar prop externo si está disponible, sino usar estado interno
  const collapsed = onToggleCollapse !== undefined ? isCollapsed : internalCollapsed;
  const handleToggle = onToggleCollapse || (() => setInternalCollapsed(!internalCollapsed));

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
        
        {/* Sección: Inyección (Génesis) */}
        <div className="space-y-3">
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

    </aside>
  );
};
