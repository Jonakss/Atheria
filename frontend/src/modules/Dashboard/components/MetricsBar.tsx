import { ChevronDown, ChevronUp } from 'lucide-react';
import React, { useEffect, useRef, useState } from 'react';
import { useWebSocket } from '../../../hooks/useWebSocket';
import { HistoryControls } from '../../History/HistoryControls';

export const MetricsBar: React.FC = () => {
  const { allLogs } = useWebSocket();
  const [expanded, setExpanded] = useState(false);
  
  // Filtrar logs recientes (últimos 2) - solo mensajes de texto
  // Usar useRef para mantener una referencia estable y evitar re-renders infinitos
  const logsRef = useRef<string[]>([]);
  const logsLengthRef = useRef<number>(0);
  
  useEffect(() => {
    if (allLogs && allLogs.length !== logsLengthRef.current) {
      logsLengthRef.current = allLogs.length;
      // Filtrar solo logs de texto (string)
      const textLogs = allLogs.filter(log => typeof log === 'string' && log.trim().length > 0);
      logsRef.current = textLogs.slice(-2);
    }
  }, [allLogs]);
  
  const recentLogs = logsRef.current;

  return (
    <div className={`mt-auto z-30 border-t border-white/10 bg-[#050505]/95 backdrop-blur-sm transition-all duration-300 ease-in-out relative ${
      expanded ? 'p-4' : 'p-2'
    }`}>
      <div className="max-w-6xl mx-auto relative flex items-center justify-between gap-4">
        
        {/* Controls Section (Replaces Metrics) */}
        <div className="flex-1">
            <HistoryControls />
        </div>

        {/* Log Section */}
        <div className={`w-1/3 pl-4 border-l border-white/5 flex flex-col justify-center transition-all duration-300 ${
            expanded ? 'gap-1' : 'gap-2'
        }`}>
            {recentLogs.length > 0 ? (
              recentLogs.map((log, idx) => {
                const logStr = typeof log === 'string' ? log : JSON.stringify(log);
                const isError = logStr.toLowerCase().includes('error');
                const isInfo = logStr.toLowerCase().includes('nucleación') || logStr.toLowerCase().includes('optimiz');
                
                return (
                  <div 
                    key={idx} 
                    className={`flex items-center gap-2 font-mono ${
                      expanded ? 'text-[10px]' : 'text-xs'
                    } ${
                      isError ? 'text-red-500/80' : isInfo ? 'text-teal-500/80' : 'text-blue-500/80'
                    }`}
                  >
                    <span className={`rounded-full shrink-0 ${
                      expanded ? 'w-1 h-1' : 'w-1.5 h-1.5'
                    } ${
                      isError ? 'bg-red-500' : isInfo ? 'bg-teal-500' : 'bg-blue-500'
                    }`} />
                    <span className={expanded ? 'truncate' : ''}>
                      {expanded 
                        ? (logStr.length > 50 ? logStr.substring(0, 50) + '...' : logStr)
                        : (logStr.length > 80 ? logStr.substring(0, 80) + '...' : logStr)
                      }
                    </span>
                  </div>
                );
              })
            ) : (
              <div className={`flex items-center gap-2 font-mono text-gray-600 ${
                expanded ? 'text-[10px]' : 'text-xs'
              }`}>
                <span className={`rounded-full bg-gray-600 shrink-0 ${
                  expanded ? 'w-1 h-1' : 'w-1.5 h-1.5'
                }`} />
                <span>Esperando datos...</span>
              </div>
            )}
        </div>

        {/* Botón de expansión (Opcional, mantenido por si el usuario quiere expandir logs) */}
        <div
          onMouseEnter={() => setExpanded(true)}
          onMouseLeave={() => setExpanded(false)}
          className="absolute right-0 top-0 h-full w-4 bg-transparent cursor-pointer z-50 flex items-center justify-center opacity-50 hover:opacity-100"
          title={expanded ? "Contraer" : "Expandir"}
        >
             {expanded ? <ChevronDown size={12} className="text-gray-500" /> : <ChevronUp size={12} className="text-gray-500" />}
        </div>
      </div>
    </div>
  );
};
