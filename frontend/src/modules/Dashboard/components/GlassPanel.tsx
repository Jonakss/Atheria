import React from 'react';

interface GlassPanelProps {
  children: React.ReactNode;
  className?: string;
}

export const GlassPanel: React.FC<GlassPanelProps> = ({ children, className = "" }) => (
  <div className={`bg-[#0a0a0a]/90 backdrop-blur-md border border-white/10 shadow-lg rounded-lg ${className}`}>
    {children}
  </div>
);
