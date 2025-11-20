import React from 'react';
import { LucideIcon } from 'lucide-react';

interface NavButtonProps {
  icon: LucideIcon;
  label: string;
  isActive?: boolean;
  onClick?: () => void;
}

/**
 * NavButton: Botón de navegación para sidebar según Design System.
 * 
 * Design System Spec:
 * - Normal: text-gray-600 hover:text-gray-300 hover:bg-white/5
 * - Activo: bg-blue-500/10 text-blue-400 + Indicador de borde izquierdo (border-l-2 border-blue-500)
 * - Tamaño: w-8 h-8
 * - Indicador activo: w-0.5 h-4 bg-blue-500 rounded-r (borde izquierdo)
 */
export const NavButton: React.FC<NavButtonProps> = ({ 
  icon: Icon, 
  label, 
  isActive = false,
  onClick 
}) => {
  return (
    <button 
      onClick={onClick}
      className={`w-8 h-8 rounded flex items-center justify-center transition-all relative group ${
        isActive 
          ? 'bg-blue-500/10 text-blue-400' 
          : 'text-gray-600 hover:text-gray-300 hover:bg-white/5'
      }`}
      title={label}
    >
      <Icon size={16} strokeWidth={2} />
      {/* Indicador Activo (borde izquierdo) */}
      {isActive && (
        <div className="absolute left-0 top-1/2 -translate-y-1/2 w-0.5 h-4 bg-blue-500 rounded-r" />
      )}
    </button>
  );
};

