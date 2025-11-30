import { ChevronLeft, ChevronRight, Terminal } from 'lucide-react';
import React, { useEffect, useRef, useState } from 'react';
import { useWebSocket } from '../../../hooks/useWebSocket';
import { HistoryControls } from '../../History/HistoryControls';
import { ScientificMetrics } from './ScientificMetrics';

export const MetricsBar: React.FC = () => {
  const { allLogs } = useWebSocket();
  const [viewMode, setViewMode] = useState<'controls' | 'logs'>('controls');

  // Use a ref for logs to avoid unnecessary re-renders if logs are updated frequently
  // but we only want to display the last few.
  // Actually, for a "full log mode", we might want to show more than 2 lines.
  // But let's start by respecting the current "recent logs" logic for the summary,
  // and maybe show more in 'logs' mode if we had a full log viewer component.
  // For now, I'll stick to the existing log logic but allow the container to expand.
  
  const logsRef = useRef<string[]>([]);
  const logsLengthRef = useRef<number>(0);
  
  useEffect(() => {
    if (allLogs && allLogs.length !== logsLengthRef.current) {
      logsLengthRef.current = allLogs.length;
      // Filter only text logs
      const textLogs = allLogs.filter(log => typeof log === 'string' && log.trim().length > 0);
      // Keep a larger buffer for the expanded view, or just the recent ones?
      // The user asked for "logs collapsible", implying they want to see logs.
      // Let's show more logs in expanded mode (e.g., last 5-10) and fewer in compact.
      logsRef.current = textLogs.slice(-10);
    }
  }, [allLogs]);
  
  const recentLogs = logsRef.current;
  const displayLogs = viewMode === 'logs' ? recentLogs : recentLogs.slice(-2);

  return (
    <div className="mt-auto z-30 border-t border-white/10 bg-[#050505]/95 backdrop-blur-sm p-2 transition-all duration-300">
      <div className="max-w-7xl mx-auto flex items-stretch gap-4">
        
        {/* Controls Section */}
        <div className={`transition-all duration-300 ease-in-out ${ viewMode === 'controls' ? 'flex-1' : 'w-auto'
        }`}>
            <HistoryControls mode={viewMode === 'controls' ? 'full' : 'compact'} />
        </div>

        {/* Scientific Metrics - Compact Display */}
        {viewMode === 'controls' && (
          <div className="flex items-center px-4 border-l border-white/5">
            <ScientificMetrics compact={true} />
          </div>
        )}

        {/* Separator / Toggle Button */}
        <div className="flex items-center">
            <button
                onClick={() => setViewMode(prev => prev === 'controls' ? 'logs' : 'controls')}
                className="p-1.5 rounded-full hover:bg-white/10 text-gray-500 hover:text-white transition-colors border border-transparent hover:border-white/10"
                title={viewMode === 'controls' ? 'Show Logs' : 'Show Controls'}
            >
                {viewMode === 'controls' ? <ChevronLeft size={16} /> : <ChevronRight size={16} />}
            </button>
        </div>

        {/* Log Section */}
        <div className={`transition-all duration-300 ease-in-out flex flex-col justify-center overflow-hidden relative ${
            viewMode === 'logs' ? 'flex-1 opacity-100' : 'w-0 opacity-0'
        }`}>
             {/* Log Content */}
             <div className="max-h-32 overflow-y-auto pr-2 custom-scrollbar flex flex-col justify-center">
                {displayLogs.length > 0 ? (
                displayLogs.map((log, idx) => {
                    const logStr = typeof log === 'string' ? log : JSON.stringify(log);
                    const isError = logStr.toLowerCase().includes('error');
                    const isInfo = logStr.toLowerCase().includes('nucleación') || logStr.toLowerCase().includes('optimiz');

                    return (
                    <div
                        key={idx}
                        className={`flex items-start gap-2 font-mono text-[10px] py-0.5 ${
                        isError ? 'text-red-500/90' : isInfo ? 'text-teal-500/90' : 'text-blue-500/90'
                        }`}
                    >
                        <span className={`mt-1 rounded-full shrink-0 w-1 h-1 ${
                        isError ? 'bg-red-500' : isInfo ? 'bg-teal-500' : 'bg-blue-500'
                        }`} />
                        <span className="break-all leading-tight">
                        {logStr}
                        </span>
                    </div>
                    );
                })
                ) : (
                <div className="flex items-center gap-2 font-mono text-gray-600 text-[10px]">
                    <Terminal size={12} />
                    <span>Esperando datos del sistema...</span>
                </div>
                )}
             </div>
        </div>
      </div>
    </div>
  );
};
