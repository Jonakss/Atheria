import React, { useMemo } from 'react';
import { MetricItem } from './MetricItem';

interface FieldWidgetProps {
  label: string;
  value: string;
  unit?: string;
  status?: "neutral" | "good" | "warning" | "critical";
  fieldData?: number[] | number[][]; // Datos del campo para visualización
  fieldType?: 'density' | 'flow' | 'phase' | 'energy'; // Tipo de campo
  isCollapsed: boolean;
  onToggleCollapse?: () => void;
}

/**
 * FieldWidget: Widget colapsable con visualización de campo cuántico.
 * 
 * Estados:
 * - Colapsado: Solo muestra el nombre verticalmente
 * - Expandido: Muestra valor numérico + mini visualización del campo
 */
export const FieldWidget: React.FC<FieldWidgetProps> = ({ 
  label, 
  value, 
  unit, 
  status = "neutral",
  fieldData,
  fieldType = 'density',
  isCollapsed,
  onToggleCollapse
}) => {
  // Estabilizar referencia de fieldData usando JSON.stringify para evitar re-renders infinitos
  const fieldDataString = useMemo(() => {
    if (!fieldData) return null;
    try {
      return JSON.stringify(fieldData);
    } catch {
      return null;
    }
  }, [fieldData]);

  // Preparar datos para mini visualización (muestrear para rendimiento)
  const visualizationData = useMemo(() => {
    if (!fieldDataString || isCollapsed) return null;
    
    // Parsear datos estables
    let parsedData: number[] | number[][];
    try {
      parsedData = JSON.parse(fieldDataString);
    } catch {
      return null;
    }
    
    // Si es array 2D, aplanar
    let flatData: number[] = [];
    if (Array.isArray(parsedData)) {
      if (Array.isArray(parsedData[0])) {
        // Array 2D: aplanar
        for (const row of parsedData) {
          if (Array.isArray(row)) {
            const filteredRow = row.filter((v: any) => typeof v === 'number' && !isNaN(v)) as number[];
            flatData = [...flatData, ...filteredRow];
          }
        }
      } else {
        // Array 1D
        flatData = (parsedData as number[]).filter((v: any) => typeof v === 'number' && !isNaN(v));
      }
    }
    
    if (flatData.length === 0) return null;
    
    // Muestrear para mini visualización (máximo 64 puntos para rendimiento)
    const sampleSize = Math.min(64, flatData.length);
    const step = Math.max(1, Math.floor(flatData.length / sampleSize));
    const sampled = [];
    for (let i = 0; i < flatData.length; i += step) {
      sampled.push(flatData[i]);
      if (sampled.length >= sampleSize) break;
    }
    
    // Normalizar a [0, 1]
    const min = Math.min(...sampled);
    const max = Math.max(...sampled);
    const range = max - min || 1;
    return sampled.map(v => (v - min) / range);
  }, [fieldDataString, isCollapsed]);

  // Calcular estadísticas para gráfico
  const stats = useMemo(() => {
    if (!visualizationData || visualizationData.length === 0) return null;
    return {
      min: Math.min(...visualizationData),
      max: Math.max(...visualizationData),
      avg: visualizationData.reduce((a, b) => a + b, 0) / visualizationData.length,
      current: visualizationData[visualizationData.length - 1]
    };
  }, [visualizationData]);

  if (isCollapsed) {
    // Solo mostrar nombre verticalmente
    return (
      <div 
        onClick={onToggleCollapse}
        className="flex flex-col items-center justify-center py-1 cursor-pointer hover:bg-white/5 transition-colors rounded group"
        title={`Click para expandir: ${label}`}
      >
        <span className="text-[8px] uppercase tracking-wider font-semibold text-gray-600 transform -rotate-90 whitespace-nowrap group-hover:text-gray-400 transition-colors">
          {label}
        </span>
      </div>
    );
  }

  // Expandido: mostrar valor + mini visualización
  return (
    <div 
      onClick={onToggleCollapse}
      className="cursor-pointer hover:bg-white/5 transition-colors rounded p-2"
      title={`Click para colapsar: ${label}`}
    >
      <MetricItem 
        label={label}
        value={value}
        unit={unit}
        status={status}
      />
      
      {/* Mini visualización del campo */}
      {visualizationData && visualizationData.length > 0 && (
        <div className="mt-2 h-12 w-full relative overflow-hidden rounded border border-white/5 bg-white/5">
          {/* Gráfico de línea para densidad/energía */}
          {fieldType === 'density' || fieldType === 'energy' ? (
            <svg className="absolute inset-0 w-full h-full" preserveAspectRatio="none" viewBox={`0 0 ${visualizationData.length} 1`}>
              <polyline
                points={visualizationData.map((v, i) => `${i},${1 - v}`).join(' ')}
                fill="none"
                stroke="currentColor"
                strokeWidth="0.5"
                className={status === "good" ? "text-emerald-400" : status === "warning" ? "text-amber-400" : "text-blue-400"}
              />
              {/* Línea promedio */}
              {stats && (
                <line
                  x1="0"
                  y1={1 - stats.avg}
                  x2={visualizationData.length}
                  y2={1 - stats.avg}
                  stroke="currentColor"
                  strokeWidth="0.2"
                  strokeDasharray="2,2"
                  className="text-gray-500 opacity-50"
                />
              )}
            </svg>
          ) : fieldType === 'flow' ? (
            // Mini visualización de flujo (gradiente)
            <svg className="absolute inset-0 w-full h-full" preserveAspectRatio="none" viewBox={`0 0 ${visualizationData.length} 1`}>
              {visualizationData.map((v, i) => (
                <rect
                  key={i}
                  x={i}
                  y={0}
                  width={1}
                  height={1}
                  fill={`rgba(59, 130, 246, ${v})`}
                />
              ))}
            </svg>
          ) : fieldType === 'phase' ? (
            // Mini visualización de fase (color ciclico)
            <svg className="absolute inset-0 w-full h-full" preserveAspectRatio="none" viewBox={`0 0 ${visualizationData.length} 1`}>
              {visualizationData.map((v, i) => {
                const hue = (v * 360) % 360;
                return (
                  <rect
                    key={i}
                    x={i}
                    y={0}
                    width={1}
                    height={1}
                    fill={`hsl(${hue}, 70%, 50%)`}
                  />
                );
              })}
            </svg>
          ) : null}
          
          {/* Estadísticas en la esquina */}
          {stats && (
            <div className="absolute bottom-0 right-0 p-1 bg-black/50 rounded-tl text-[8px] font-mono text-gray-400">
              {stats.current.toFixed(2)}
            </div>
          )}
        </div>
      )}
    </div>
  );
};

