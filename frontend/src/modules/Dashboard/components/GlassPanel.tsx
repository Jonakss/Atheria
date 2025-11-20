import React from 'react';

interface GlassPanelProps {
  children: React.ReactNode;
  className?: string;
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
export const GlassPanel: React.FC<GlassPanelProps> = ({ children, className = "" }) => (
  <div className={`bg-[#0a0a0a]/90 backdrop-blur-md border border-white/10 shadow-lg rounded-lg ${className}`}>
    {children}
  </div>
);
