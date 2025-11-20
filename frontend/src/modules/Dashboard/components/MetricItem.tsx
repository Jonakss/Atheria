import React from 'react';

interface MetricItemProps {
  label: string;
  value: string;
  unit?: string;
  status?: "neutral" | "good" | "warning" | "critical";
}

/**
 * MetricItem: Visualizador de datos científicos según Design System.
 * Siempre usa borde izquierdo (border-l-2 border-white/5).
 * 
 * Design System Spec:
 * - Label: text-[10px] uppercase tracking-widest font-semibold text-gray-500
 * - Value: text-lg font-mono font-medium text-gray-100 (o status color)
 * - Unit: text-[10px] text-gray-600 font-mono uppercase
 * - Border: border-l-2 border-white/5
 * - Padding: pl-3 py-1
 * 
 * Status Colors:
 * - good: text-emerald-400
 * - warning: text-amber-400
 * - critical: text-rose-400
 * - neutral: text-gray-100 (no text-gray-300)
 */
export const MetricItem: React.FC<MetricItemProps> = ({ 
  label, 
  value, 
  unit, 
  status = "neutral" 
}) => {
  const statusColor = 
    status === "good" ? "text-emerald-400" :
    status === "warning" ? "text-amber-400" :
    status === "critical" ? "text-rose-400" : "text-gray-100"; // Design System: text-gray-100 para neutral

  return (
    <div className="flex flex-col border-l-2 border-white/5 pl-3 py-1">
      <span className="text-[10px] uppercase tracking-widest font-semibold text-gray-500 mb-1">
        {label}
      </span>
          <div className="flex items-baseline gap-1.5">
                <span className={`text-lg font-mono font-medium ${value === 'N/A' ? 'text-gray-600' : statusColor}`}>
                  {value}
                </span>
                {unit && value !== 'N/A' && <span className="text-[10px] text-gray-600 font-mono uppercase">{unit}</span>}
              </div>
    </div>
  );
};
