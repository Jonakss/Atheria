import React from 'react';

interface GlassPanelProps {
  children: React.ReactNode;
  className?: string;
  title?: string;
}

/**
 * GlassPanel: Contenedor estándar del Design System.
 * Implementa el efecto "glass" con backdrop blur según especificaciones.
 * 
 * Design System Spec:
 * - Background: #0a0a0a con 90% opacidad (bg-[#0a0a0a]/90)
 * - Backdrop Blur: backdrop-blur-md
 * - Border: border-white/10 (medio)
 * - Shadow: shadow-lg
 * - Radius: rounded-lg
 */
export const GlassPanel: React.FC<GlassPanelProps> = ({ children, className = "", title }) => (
  <div className={`bg-[#0a0a0a]/90 backdrop-blur-md border border-white/10 shadow-lg rounded-lg ${className}`}>
    {title && (
      <div className="border-b border-white/10 px-4 py-3">
        <h3 className="text-sm font-medium text-slate-200">{title}</h3>
      </div>
    )}
    {children}
  </div>
);
