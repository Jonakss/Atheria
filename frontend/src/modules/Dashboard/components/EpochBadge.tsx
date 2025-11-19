import React from 'react';

interface EpochBadgeProps {
  era: number;
  current: number;
}

const EPOCH_LABELS = ["VACÍO", "CUÁNTICA", "PARTÍCULAS", "QUÍMICA", "GRAVEDAD", "BIOLOGÍA"];

export const EpochBadge: React.FC<EpochBadgeProps> = ({ era, current }) => {
  const isActive = era === current;
  const isPast = era < current;
  
  return (
    <div className={`flex items-center gap-2 px-3 py-1 rounded border text-[10px] font-mono font-medium tracking-wider transition-all ${
      isActive 
        ? 'bg-blue-500/10 border-blue-500/40 text-blue-400' 
        : isPast 
          ? 'bg-white/5 border-white/5 text-gray-600' 
          : 'bg-transparent border-transparent text-gray-800'
    }`}>
      <div className={`w-1 h-1 rounded-full ${
        isActive 
          ? 'bg-blue-400 shadow-[0_0_5px_cyan]' 
          : isPast 
            ? 'bg-gray-600' 
            : 'bg-gray-800'
      }`} />
      {EPOCH_LABELS[era]}
    </div>
  );
};
