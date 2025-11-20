// frontend/src/modules/Dashboard/components/LogsView.tsx
import React, { useRef, useEffect, useState } from 'react';
import { Terminal, AlertCircle, Info, AlertTriangle, X, Download, Send, ChevronRight } from 'lucide-react';
import { useWebSocket } from '../../../hooks/useWebSocket';
import { GlassPanel } from './GlassPanel';

export const LogsView: React.FC = () => {
  const { allLogs, connectionStatus, sendCommand } = useWebSocket();
  const isConnected = connectionStatus === 'connected';
  const scrollRef = useRef<HTMLDivElement>(null);
  const inputRef = useRef<HTMLInputElement>(null);
  const [autoScroll, setAutoScroll] = React.useState(true);
  const [filter, setFilter] = React.useState<'all' | 'info' | 'warning' | 'error'>('all');
  const [commandInput, setCommandInput] = useState('');
  const [commandHistory, setCommandHistory] = useState<string[]>([]);
  const [historyIndex, setHistoryIndex] = useState(-1);

  // Auto-scroll cuando hay nuevos logs
  useEffect(() => {
    if (autoScroll && scrollRef.current) {
      scrollRef.current.scrollTop = scrollRef.current.scrollHeight;
    }
  }, [allLogs, autoScroll]);

  // Manejar envío de comandos
  const handleSendCommand = () => {
    if (!commandInput.trim() || !isConnected) return;
    
    try {
      // Parsear comando manual (formato: scope.command {args})
      // Ejemplo: inference.play {}
      // Ejemplo: inference.inject_energy {"type": "primordial_soup"}
      const parts = commandInput.trim().split(/\s+/);
      if (parts.length < 2) {
        alert('⚠️ Formato incorrecto. Usa: scope.command {args}\nEjemplo: inference.play {}');
        return;
      }
      
      const [scope, command] = parts[0].split('.');
      if (!scope || !command) {
        alert('⚠️ Formato incorrecto. Usa: scope.command {args}\nEjemplo: inference.play {}');
        return;
      }
      
      // Parsear args (JSON)
      let args = {};
      if (parts.length > 1) {
        const argsStr = parts.slice(1).join(' ');
        if (argsStr.trim()) {
          try {
            args = JSON.parse(argsStr);
          } catch (e) {
            alert(`⚠️ Error parseando argumentos JSON: ${e}`);
            return;
          }
        }
      }
      
      // Enviar comando
      sendCommand(scope, command, args);
      
      // Agregar a historial
      const newHistory = [...commandHistory, commandInput.trim()];
      setCommandHistory(newHistory.slice(-50)); // Mantener últimos 50 comandos
      setHistoryIndex(-1);
      setCommandInput('');
      
      // Scroll al final para ver el comando enviado
      setTimeout(() => {
        if (scrollRef.current) {
          scrollRef.current.scrollTop = scrollRef.current.scrollHeight;
        }
      }, 100);
    } catch (error: any) {
      alert(`⚠️ Error al enviar comando: ${error.message}`);
    }
  };

  // Manejar teclas especiales
  const handleKeyDown = (e: React.KeyboardEvent<HTMLInputElement>) => {
    if (e.key === 'Enter') {
      e.preventDefault();
      handleSendCommand();
    } else if (e.key === 'ArrowUp') {
      e.preventDefault();
      if (commandHistory.length > 0) {
        const newIndex = historyIndex === -1 
          ? commandHistory.length - 1 
          : Math.max(0, historyIndex - 1);
        setHistoryIndex(newIndex);
        setCommandInput(commandHistory[newIndex]);
      }
    } else if (e.key === 'ArrowDown') {
      e.preventDefault();
      if (historyIndex >= 0) {
        const newIndex = historyIndex + 1;
        if (newIndex >= commandHistory.length) {
          setHistoryIndex(-1);
          setCommandInput('');
        } else {
          setHistoryIndex(newIndex);
          setCommandInput(commandHistory[newIndex]);
        }
      }
    }
  };

  // Filtrar logs según el tipo seleccionado
  const filteredLogs = React.useMemo(() => {
    if (!allLogs) return [];
    
    let logs = allLogs.filter(log => typeof log === 'string' && log.trim().length > 0);
    
    if (filter !== 'all') {
      logs = logs.filter(log => {
        const logLower = log.toLowerCase();
        if (filter === 'error') {
          return logLower.includes('error') || logLower.includes('[error]');
        } else if (filter === 'warning') {
          return logLower.includes('warning') || logLower.includes('[warning]');
        } else if (filter === 'info') {
          return !logLower.includes('error') && !logLower.includes('warning');
        }
        return true;
      });
    }
    
    return logs;
  }, [allLogs, filter]);

  const getLogIcon = (log: string) => {
    const logLower = log.toLowerCase();
    if (logLower.includes('error') || logLower.includes('[error]')) {
      return <AlertCircle size={12} className="text-red-400" />;
    } else if (logLower.includes('warning') || logLower.includes('[warning]')) {
      return <AlertTriangle size={12} className="text-amber-400" />;
    }
    return <Info size={12} className="text-blue-400" />;
  };

  const getLogColor = (log: string) => {
    const logLower = log.toLowerCase();
    if (logLower.includes('error') || logLower.includes('[error]')) {
      return 'text-red-400';
    } else if (logLower.includes('warning') || logLower.includes('[warning]')) {
      return 'text-amber-400';
    } else if (logLower.includes('success') || logLower.includes('[success]')) {
      return 'text-emerald-400';
    }
    return 'text-gray-300';
  };

  if (!isConnected) {
    return (
      <div className="absolute inset-0 z-0 bg-gradient-to-b from-[#050505] to-black flex items-center justify-center">
        <div className="text-gray-600 text-sm">Conectando al servidor...</div>
      </div>
    );
  }

  return (
    <div className="absolute inset-0 z-0 bg-gradient-to-b from-[#050505] to-black flex flex-col overflow-hidden">
      {/* Header con controles */}
      <div className="flex items-center justify-between p-4 border-b border-white/10 shrink-0">
        <div className="flex items-center gap-3">
          <Terminal size={18} className="text-blue-400" />
          <h2 className="text-sm font-bold text-gray-200 uppercase tracking-wider">Consola de Logs</h2>
        </div>
        <div className="flex items-center gap-2">
          {/* Filtros */}
          <div className="flex items-center gap-1 bg-white/5 rounded border border-white/10 p-1">
            {(['all', 'info', 'warning', 'error'] as const).map((filterType) => (
              <button
                key={filterType}
                onClick={() => setFilter(filterType)}
                className={`px-3 py-1 rounded text-[10px] font-bold transition-all ${
                  filter === filterType
                    ? 'bg-blue-500/20 text-blue-300 border border-blue-500/30'
                    : 'text-gray-500 hover:text-gray-300'
                }`}
              >
                {filterType === 'all' ? 'Todos' : filterType === 'info' ? 'Info' : filterType === 'warning' ? 'Warning' : 'Error'}
              </button>
            ))}
          </div>
          <button
            onClick={() => setAutoScroll(!autoScroll)}
            className={`px-3 py-1.5 rounded text-[10px] font-bold border transition-all ${
              autoScroll
                ? 'bg-blue-500/10 text-blue-400 border-blue-500/30'
                : 'bg-white/5 text-gray-400 border-white/10'
            }`}
            title={autoScroll ? 'Desactivar auto-scroll' : 'Activar auto-scroll'}
          >
            {autoScroll ? 'Auto' : 'Manual'}
          </button>
          <button
            className="px-3 py-1.5 bg-white/5 hover:bg-white/10 border border-white/10 text-gray-300 text-[10px] font-bold rounded transition-all flex items-center gap-2"
            title="Exportar logs"
          >
            <Download size={12} />
            Exportar
          </button>
        </div>
      </div>

      {/* Logs Container */}
      <div 
        ref={scrollRef}
        className="flex-1 overflow-y-auto custom-scrollbar p-4 space-y-1 font-mono"
        style={{ fontSize: '11px', lineHeight: '1.5' }}
      >
        {filteredLogs.length > 0 ? (
          filteredLogs.map((log, idx) => {
            const logStr = typeof log === 'string' ? log : JSON.stringify(log);
            return (
              <div
                key={idx}
                className={`flex items-start gap-2 p-2 rounded hover:bg-white/5 transition-colors ${getLogColor(logStr)}`}
              >
                <div className="shrink-0 mt-0.5">
                  {getLogIcon(logStr)}
                </div>
                <div className="flex-1 break-words">
                  <span className="text-[10px] text-gray-600 mr-2">[{new Date().toLocaleTimeString()}]</span>
                  {logStr}
                </div>
              </div>
            );
          })
        ) : (
          <div className="flex items-center justify-center h-full text-gray-600 text-sm">
            {filter === 'all' ? 'No hay logs disponibles' : `No hay logs de tipo "${filter}"`}
          </div>
        )}
      </div>

      {/* Consola de Comandos */}
      <div className="border-t border-white/10 bg-[#0a0a0a] p-3 shrink-0">
        <div className="flex items-center gap-2">
          <ChevronRight size={14} className="text-blue-400 shrink-0" />
          <input
            ref={inputRef}
            type="text"
            value={commandInput}
            onChange={(e) => setCommandInput(e.target.value)}
            onKeyDown={handleKeyDown}
            disabled={!isConnected}
            placeholder={isConnected ? "Escribe un comando (ej: inference.play {})" : "Conecta al servidor primero..."}
            className="flex-1 px-3 py-2 bg-white/5 border border-white/10 rounded text-xs text-gray-200 font-mono focus:outline-none focus:border-blue-500/50 disabled:opacity-50 disabled:cursor-not-allowed"
          />
          <button
            onClick={handleSendCommand}
            disabled={!isConnected || !commandInput.trim()}
            className={`px-4 py-2 rounded text-xs font-bold border transition-all flex items-center gap-2 ${
              isConnected && commandInput.trim()
                ? 'bg-blue-500/10 text-blue-400 border-blue-500/30 hover:bg-blue-500/20'
                : 'bg-white/5 text-gray-600 border-white/10 cursor-not-allowed opacity-50'
            }`}
          >
            <Send size={12} />
            Enviar
          </button>
        </div>
        <div className="text-[9px] text-gray-600 mt-2 font-mono">
          <span className="text-gray-500">Formato:</span> scope.command {`{args}`} | 
          <span className="text-gray-500 ml-1">Ejemplos:</span> inference.play {`{}`} | inference.inject_energy {`{"type": "primordial_soup"}`} | simulation.set_viz {`{"viz_type": "density"}`}
        </div>
      </div>
    </div>
  );
};

