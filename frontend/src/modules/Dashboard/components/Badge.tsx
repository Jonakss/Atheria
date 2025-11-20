import React from 'react';

interface BadgeProps {
  children: React.ReactNode;
  color?: 'blue' | 'green' | 'orange' | 'red' | 'gray' | 'yellow';
  variant?: 'filled' | 'light' | 'outline';
  size?: 'xs' | 'sm' | 'md' | 'lg';
  leftSection?: React.ReactNode;
  className?: string;
}

/**
 * Badge: Componente de badge según Design System.
 */
export const Badge: React.FC<BadgeProps> = ({ 
  children, 
  color = 'gray',
  variant = 'filled',
  size = 'sm',
  leftSection,
  className = ''
}) => {
  const sizeClasses = {
    xs: 'text-[9px] px-1.5 py-0.5',
    sm: 'text-[10px] px-2 py-1',
    md: 'text-xs px-2.5 py-1.5',
    lg: 'text-sm px-3 py-2'
  };

  const colorClasses = {
    blue: {
      filled: 'bg-blue-500/20 text-blue-400 border-blue-500/30',
      light: 'bg-blue-500/10 text-blue-400 border-blue-500/20',
      outline: 'bg-transparent text-blue-400 border-blue-500/30'
    },
    green: {
      filled: 'bg-emerald-500/20 text-emerald-400 border-emerald-500/30',
      light: 'bg-emerald-500/10 text-emerald-400 border-emerald-500/20',
      outline: 'bg-transparent text-emerald-400 border-emerald-500/30'
    },
    orange: {
      filled: 'bg-amber-500/20 text-amber-400 border-amber-500/30',
      light: 'bg-amber-500/10 text-amber-400 border-amber-500/20',
      outline: 'bg-transparent text-amber-400 border-amber-500/30'
    },
    red: {
      filled: 'bg-red-500/20 text-red-400 border-red-500/30',
      light: 'bg-red-500/10 text-red-400 border-red-500/20',
      outline: 'bg-transparent text-red-400 border-red-500/30'
    },
    gray: {
      filled: 'bg-white/10 text-gray-300 border-white/20',
      light: 'bg-white/5 text-gray-400 border-white/10',
      outline: 'bg-transparent text-gray-400 border-white/20'
    },
    yellow: {
      filled: 'bg-yellow-500/20 text-yellow-400 border-yellow-500/30',
      light: 'bg-yellow-500/10 text-yellow-400 border-yellow-500/20',
      outline: 'bg-transparent text-yellow-400 border-yellow-500/30'
    }
  };

  return (
    <span 
      className={`
        inline-flex items-center gap-1 rounded border font-bold
        ${sizeClasses[size]}
        ${colorClasses[color][variant]}
        ${className}
      `}
    >
      {leftSection && <span>{leftSection}</span>}
      {children}
    </span>
  );
};

