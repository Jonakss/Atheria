// frontend/src/modules/Dashboard/components/HistoryView.tsx
import React from 'react';
import { Database, Clock, Download, Trash2 } from 'lucide-react';
import { useWebSocket } from '../../../hooks/useWebSocket';
import { GlassPanel } from './GlassPanel';

export const HistoryView: React.FC = () => {
  const { experimentsData, connectionStatus } = useWebSocket();
  const isConnected = connectionStatus === 'connected';

  if (!isConnected) {
    return (
      <div className="absolute inset-0 z-0 bg-gradient-to-b from-[#050505] to-black flex items-center justify-center">
        <div className="text-gray-600 text-sm">Conectando al servidor...</div>
      </div>
    );
  }

  return (
    <div className="absolute inset-0 z-0 bg-gradient-to-b from-[#050505] to-black overflow-auto custom-scrollbar">
      <div className="p-6 max-w-7xl mx-auto space-y-6">
        {/* Header */}
        <div className="flex items-center justify-between mb-6">
          <div className="flex items-center gap-3">
            <Database size={20} className="text-blue-400" />
            <h2 className="text-lg font-bold text-gray-200">Historial de Experimentos</h2>
          </div>
          <button className="flex items-center gap-2 px-3 py-2 bg-white/5 hover:bg-white/10 border border-white/10 text-gray-300 text-xs font-bold rounded transition-all">
            <Download size={14} />
            Exportar
          </button>
        </div>

        {/* Lista de Experimentos */}
        {experimentsData && experimentsData.length > 0 ? (
          <div className="space-y-3">
            {experimentsData.map((exp) => (
              <GlassPanel key={exp.name} className="p-4 hover:bg-white/5 transition-colors">
                <div className="flex items-start justify-between">
                  <div className="flex-1">
                    <div className="flex items-center gap-3 mb-2">
                      <span className="text-sm font-bold text-gray-200">{exp.name}</span>
                      <span className={`px-2 py-0.5 rounded text-[10px] font-bold ${
                        exp.has_checkpoint 
                          ? 'bg-emerald-500/10 text-emerald-400 border border-emerald-500/20' 
                          : 'bg-gray-500/10 text-gray-400 border border-gray-500/20'
                      }`}>
                        {exp.has_checkpoint ? '✓ Entrenado' : '○ Sin entrenar'}
                      </span>
                    </div>
                    <div className="grid grid-cols-3 gap-4 text-xs text-gray-400 mt-3">
                      <div>
                        <div className="text-[10px] text-gray-500 uppercase mb-1">Arquitectura</div>
                        <div className="font-mono text-gray-300">{exp.model_architecture || 'N/A'}</div>
                      </div>
                      <div>
                        <div className="text-[10px] text-gray-500 uppercase mb-1">Grid Size</div>
                        <div className="font-mono text-gray-300">{exp.grid_size_training || 'N/A'}</div>
                      </div>
                      <div>
                        <div className="text-[10px] text-gray-500 uppercase mb-1">Episodios</div>
                        <div className="font-mono text-gray-300">{exp.total_episodes || 0}</div>
                      </div>
                    </div>
                    {exp.created_at && (
                      <div className="flex items-center gap-2 mt-3 text-[10px] text-gray-600">
                        <Clock size={12} />
                        <span>Creado: {new Date(exp.created_at).toLocaleString()}</span>
                      </div>
                    )}
                  </div>
                  <div className="flex items-center gap-2 ml-4">
                    <button 
                      className="p-2 text-gray-400 hover:text-gray-200 hover:bg-white/5 rounded transition-all"
                      title="Exportar"
                    >
                      <Download size={14} />
                    </button>
                    <button 
                      className="p-2 text-gray-400 hover:text-red-400 hover:bg-red-500/10 rounded transition-all"
                      title="Eliminar"
                    >
                      <Trash2 size={14} />
                    </button>
                  </div>
                </div>
              </GlassPanel>
            ))}
          </div>
        ) : (
          <GlassPanel className="p-8">
            <div className="text-center text-gray-600 text-sm">
              No hay experimentos guardados aún
            </div>
          </GlassPanel>
        )}
      </div>
    </div>
  );
};

