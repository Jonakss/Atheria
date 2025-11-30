import React from 'react';

interface ColorScaleLegendProps {
  mode: string;
  minValue: number;
  maxValue: number;
  position?: 'top-right' | 'bottom-right';
  gradient?: string;
}

export const ColorScaleLegend: React.FC<ColorScaleLegendProps> = ({
  mode, 
  minValue, 
  maxValue, 
  position = 'bottom-right',
  gradient = 'linear-gradient(to top, #000, #0ff, #fff)'
}) => {
  return (
    <div className={`absolute ${position === 'top-right' ? 'top-4 right-4' : 'bottom-4 right-4'} 
                     bg-black/80 backdrop-blur-sm border border-white/10 rounded-lg p-3 z-30 pointer-events-none transition-all duration-300`}>
      <div className="text-[9px] font-mono text-gray-400 uppercase mb-2 tracking-wider border-b border-white/5 pb-1">
        {mode}
      </div>
      <div className="flex items-center gap-2">
        {/* Gradient bar */}
        <div className="w-3 h-24 rounded-sm relative border border-white/10"
             style={{ background: gradient }}>
        </div>
        
        {/* Value labels */}
        <div className="flex flex-col justify-between h-24 text-[9px] font-mono text-gray-300">
          <span>{maxValue.toFixed(2)}</span>
          <span className="text-gray-500">{((maxValue + minValue) / 2).toFixed(2)}</span>
          <span>{minValue.toFixed(2)}</span>
        </div>
      </div>
    </div>
  );
};
