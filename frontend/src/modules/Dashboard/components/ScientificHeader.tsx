import React from 'react';
import { Settings } from 'lucide-react';
import { EpochBadge } from './EpochBadge';

interface ScientificHeaderProps {
  currentEpoch: number;
}

export const ScientificHeader: React.FC<ScientificHeaderProps> = ({ currentEpoch }) => {
  return (
    <header className="h-12 border-b border-white/10 bg-[#050505] flex items-center justify-between px-4 z-50 shrink-0">
      <div className="flex items-center gap-6">
        {/* Identidad */}
        <div className="flex items-center gap-3 group cursor-pointer opacity-90 hover:opacity-100 transition-opacity">
          <div className="relative w-6 h-6 flex items-center justify-center border border-white/10 rounded bg-white/5">
            <svg 
              width="14" 
              height="14" 
              viewBox="0 0 24 24" 
              fill="none" 
              stroke="currentColor" 
              strokeWidth="2"
              className="text-gray-300"
            >
              <circle cx="12" cy="12" r="10" />
              <circle cx="12" cy="12" r="4" />
            </svg>
          </div>
          <div className="flex flex-col leading-none">
            <span className="text-xs font-bold text-gray-200 tracking-wide">
              ATHERIA<span className="text-blue-500">_LAB</span>
            </span>
            <span className="text-[8px] text-gray-600 font-mono uppercase mt-0.5">Ver. 4.0.2-RC</span>
          </div>
        </div>

        {/* Separador Vertical */}
        <div className="h-4 w-px bg-white/10" />

        {/* Timeline de Épocas (Compacto) */}
        <div className="flex items-center gap-0.5">
          {[0, 1, 2, 3, 4, 5].map(e => (
            <EpochBadge key={e} era={e} current={currentEpoch} />
          ))}
        </div>
      </div>

      <div className="flex items-center gap-4">
        {/* Indicador de Estado del Sistema */}
        <div className="flex items-center gap-2 px-2 py-1 bg-white/5 rounded border border-white/5">
          <div className="w-1.5 h-1.5 rounded-full bg-emerald-500 animate-pulse shadow-[0_0_8px_rgba(16,185,129,0.4)]" />
          <span className="text-[10px] font-mono text-emerald-500 tracking-wide">SPARSE_ENGINE::READY</span>
        </div>
        
        <div className="h-4 w-px bg-white/10" />

        {/* Perfil / Config */}
        <button className="p-1.5 text-gray-500 hover:text-gray-300 transition-colors">
          <Settings size={16} />
        </button>
        <div className="w-6 h-6 rounded bg-gray-800 border border-gray-700 flex items-center justify-center text-[10px] text-gray-400 font-bold">JS</div>
      </div>
    </header>
  );
};
