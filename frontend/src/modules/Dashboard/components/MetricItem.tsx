import React from 'react';

interface MetricItemProps {
  label: string;
  value: string;
  unit?: string;
  status?: "neutral" | "good" | "warning" | "critical";
}

export const MetricItem: React.FC<MetricItemProps> = ({ 
  label, 
  value, 
  unit, 
  status = "neutral" 
}) => {
  const statusColor = 
    status === "good" ? "text-emerald-400" :
    status === "warning" ? "text-amber-400" :
    status === "critical" ? "text-rose-400" : "text-gray-300";

  return (
    <div className="flex flex-col border-l-2 border-white/5 pl-3 py-1">
      <span className="text-[10px] uppercase tracking-widest font-semibold text-gray-500 mb-1">
        {label}
      </span>
      <div className="flex items-baseline gap-1.5">
        <span className={`text-lg font-mono font-medium ${statusColor}`}>{value}</span>
        {unit && <span className="text-[10px] text-gray-600 font-mono uppercase">{unit}</span>}
      </div>
    </div>
  );
};
