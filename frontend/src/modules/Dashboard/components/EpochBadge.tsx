import React, { useState } from 'react';
import { EPOCH_CONFIGS, EPOCH_LABELS } from './epochConfig';


interface EpochBadgeProps {
  era: number;
  current: number;
  onClick?: () => void;
}

/**
 * EpochBadge: Badge de estado temporal según Design System.
 * Incluye tooltip con información y configuración de época.
 * 
 * Design System Spec:
 * - Estilo: text-[10px] font-mono font-medium tracking-wider px-3 py-1 rounded border
 * - Activo: bg-blue-500/10 border-blue-500/40 text-blue-400
 * - Inactivo: bg-white/5 border-white/5 text-gray-600
 * - Gap: gap-2 (según mockup)
 */
export const EpochBadge: React.FC<EpochBadgeProps> = ({ era, current, onClick }) => {
  const [showTooltip, setShowTooltip] = useState(false);
  const isActive = era === current;
  const isPast = era < current;
  const config = EPOCH_CONFIGS[era];
  
  const colorClasses = {
    purple: { active: 'bg-purple-500/10 border-purple-500/40 text-purple-400', dot: 'bg-purple-400' },
    blue: { active: 'bg-blue-500/10 border-blue-500/40 text-blue-400', dot: 'bg-blue-400' },
    teal: { active: 'bg-teal-500/10 border-teal-500/40 text-teal-400', dot: 'bg-teal-400' },
    pink: { active: 'bg-pink-500/10 border-pink-500/40 text-pink-400', dot: 'bg-pink-400' },
    green: { active: 'bg-green-500/10 border-green-500/40 text-green-400', dot: 'bg-green-400' }
  };
  
  const activeColorClass = colorClasses[config.color as keyof typeof colorClasses] || colorClasses.blue;
  
  return (
    <div 
      className="relative"
      onMouseEnter={() => setShowTooltip(true)}
      onMouseLeave={() => setShowTooltip(false)}
    >
      <div 
        onClick={onClick}
        className={`flex items-center gap-2 px-3 py-1 rounded border text-[10px] font-mono font-medium tracking-wider transition-all cursor-pointer hover:scale-105 ${
          isActive 
            ? activeColorClass.active
            : isPast 
              ? 'bg-white/5 border-white/5 text-gray-600 hover:bg-white/10' 
              : 'bg-transparent border-transparent text-gray-800 hover:bg-white/5'
        }`}
      >
        <div className={`w-1 h-1 rounded-full transition-all ${
          isActive 
            ? `${activeColorClass.dot} shadow-[0_0_5px_currentColor]` 
            : isPast 
              ? 'bg-gray-600' 
              : 'bg-gray-800'
        }`} />
        <span>{EPOCH_LABELS[era]}</span>
      </div>
      
      {/* Tooltip con información de la época */}
      {showTooltip && (
        <div className="absolute top-full left-1/2 -translate-x-1/2 mt-2 z-50 pointer-events-none">
          <div className="bg-[#0a0a0a] border border-white/20 rounded-lg p-3 shadow-xl min-w-[200px]">
            <div className="text-xs font-bold text-gray-200 mb-1.5">
              {config.label}
            </div>
            <div className="text-[10px] text-gray-400 mb-2 leading-relaxed">
              {config.description}
            </div>
            <div className="space-y-1 pt-2 border-t border-white/10">
              <div className="flex items-center justify-between text-[10px]">
                <span className="text-gray-500">Gamma Decay:</span>
                <span className="text-teal-400 font-mono">{config.gammaDecay.toFixed(3)}</span>
              </div>
              <div className="flex items-center justify-between text-[10px]">
                <span className="text-gray-500">Visualización:</span>
                <span className="text-blue-400 font-mono uppercase">{config.vizType}</span>
              </div>
            </div>
            {!isActive && (
              <div className="mt-2 pt-2 border-t border-white/10 text-[9px] text-gray-600">
                Click para aplicar esta configuración
              </div>
            )}
          </div>
        </div>
      )}
    </div>
  );
};

