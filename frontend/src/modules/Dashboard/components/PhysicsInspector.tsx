import { AlertCircle, ChevronLeft, ChevronRight, Cube, Microscope, Zap } from 'lucide-react';
import React, { useEffect, useState } from 'react';
import { HolographicVolumeViewer } from '../../../components/visualization/HolographicVolumeViewer';
import { useWebSocket } from '../../../hooks/useWebSocket';

interface PhysicsInspectorProps {
  isCollapsed?: boolean;
  onToggleCollapse?: () => void;
}

export const PhysicsInspector: React.FC<PhysicsInspectorProps> = ({ 
  isCollapsed = false, 
  onToggleCollapse
}) => {
  const { sendCommand, allLogs, lastMessage, compileStatus } = useWebSocket();
  
  // Internal state fallback if prop not provided
  const [internalCollapsed, setInternalCollapsed] = useState(false);
  
  // Holographic 3D View state
  const [show3DView, setShow3DView] = useState(false);
  const [volumeData, setVolumeData] = useState<{
    data: number[];
    depth: number;
    height: number;
    width: number;
  }>({ data: [], depth: 0, height: 0, width: 0 });
  
  // Usar prop externo si está disponible, sino usar estado interno
  const collapsed = onToggleCollapse !== undefined ? isCollapsed : internalCollapsed;
  const handleToggle = onToggleCollapse || (() => setInternalCollapsed(!internalCollapsed));

  // Filtrar logs recientes (últimos 2)
  const recentLogs = allLogs?.slice(-2) || [];
  
  // Detectar si el motor soporta visualización 3D
  const engineType = compileStatus?.engine_type || '';
  const supports3D = ['HOLOGRAPHIC', 'CARTESIAN', 'POLAR', 'HARMONIC', 'LATTICE'].includes(engineType.toUpperCase());

  // Escuchar mensajes de volumen 3D
  useEffect(() => {
    if (lastMessage?.type === 'bulk_volume_data' || 
        lastMessage?.type === 'holographic_projection_data') {
      setVolumeData({
        data: lastMessage.payload.volume_data,
        depth: lastMessage.payload.depth,
        height: lastMessage.payload.height,
        width: lastMessage.payload.width
      });
    }
  }, [lastMessage]);

  // Solicitar datos 3D cuando se activa la vista
  useEffect(() => {
    if (show3DView && supports3D) {
      if (engineType.toUpperCase() === 'HOLOGRAPHIC') {
        // HolographicEngine: usar bulk físico
        sendCommand('inference', 'get_bulk_volume', {});
      } else {
        // Otros motores: usar proyección genérica
        sendCommand('inference', 'get_holographic_projection', { 
          depth: 8,
          use_phase: false 
        });
      }
    }
  }, [show3DView, engineType, supports3D]);

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
        
        {/* Vista Holográfica 3D */}
        {supports3D && (
          <div className="space-y-3">
            <div className="flex items-center justify-between">
              <div className="flex items-center gap-2 text-gray-200 text-xs font-bold uppercase tracking-wider">
                <Cube size={12} className="text-purple-400" /> Vista 3D
              </div>
              <button
                onClick={() => setShow3DView(!show3DView)}
                className={`px-2 py-1 text-[10px] font-bold rounded transition-all ${
                  show3DView 
                    ? 'bg-purple-500/20 text-purple-300 border border-purple-500/30' 
                    : 'bg-white/5 text-gray-500 border border-white/10 hover:bg-white/10'
                }`}
              >
                {show3DView ? 'Ocultar' : 'Mostrar'}
              </button>
            </div>
            
            {show3DView && volumeData.data.length > 0 && (
              <div className="bg-black/50 rounded border border-white/10 overflow-hidden">
                <HolographicVolumeViewer
                  volumeData={volumeData.data}
                  depth={volumeData.depth}
                  width={volumeData.width}
                  height={volumeData.height}
                  threshold={0.01}
                />
                <div className="px-2 py-1 bg-dark-900/50 border-t border-white/10">
                  <p className="text-[9px] text-gray-500">
                    {engineType.toUpperCase() === 'HOLOGRAPHIC' 
                      ? '🔮 Bulk Físico (AdS/CFT)' 
                      : '🔮 Proyección Holográfica (Scale-Space)'}
                    {' • '}{volumeData.depth}×{volumeData.height}×{volumeData.width}
                  </p>
                </div>
              </div>
            )}
          </div>
        )}
        
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

