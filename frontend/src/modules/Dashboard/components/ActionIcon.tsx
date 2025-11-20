import React from 'react';

interface ActionIconProps {
  children: React.ReactNode;
  className?: string;
  onClick?: () => void;
  variant?: 'filled' | 'light' | 'outline' | 'subtle' | 'transparent';
  color?: 'blue' | 'gray' | 'red' | 'green' | 'amber';
  size?: 'xs' | 'sm' | 'md' | 'lg';
  disabled?: boolean;
  title?: string;
}

/**
 * ActionIcon: Botón semántico de icono (sin texto).
 * Equivalente a Mantine ActionIcon, usado para acciones con iconos.
 */
export const ActionIcon: React.FC<ActionIconProps> = ({ 
  children, 
  className = '',
  onClick,
  variant = 'subtle',
  color = 'gray',
  size = 'md',
  disabled = false,
  title
}) => {
  const sizeClass = {
    xs: 'w-6 h-6',
    sm: 'w-7 h-7',
    md: 'w-8 h-8',
    lg: 'w-10 h-10'
  }[size];

  const variantClass = {
    filled: {
      blue: 'bg-blue-500 hover:bg-blue-600 text-white',
      gray: 'bg-gray-500 hover:bg-gray-600 text-white',
      red: 'bg-red-500 hover:bg-red-600 text-white',
      green: 'bg-emerald-500 hover:bg-emerald-600 text-white',
      amber: 'bg-amber-500 hover:bg-amber-600 text-white'
    },
    light: {
      blue: 'bg-blue-500/10 hover:bg-blue-500/20 text-blue-400',
      gray: 'bg-white/5 hover:bg-white/10 text-gray-300',
      red: 'bg-red-500/10 hover:bg-red-500/20 text-red-400',
      green: 'bg-emerald-500/10 hover:bg-emerald-500/20 text-emerald-400',
      amber: 'bg-amber-500/10 hover:bg-amber-500/20 text-amber-400'
    },
    outline: {
      blue: 'border border-blue-500 hover:bg-blue-500/10 text-blue-400',
      gray: 'border border-white/20 hover:bg-white/5 text-gray-300',
      red: 'border border-red-500 hover:bg-red-500/10 text-red-400',
      green: 'border border-emerald-500 hover:bg-emerald-500/10 text-emerald-400',
      amber: 'border border-amber-500 hover:bg-amber-500/10 text-amber-400'
    },
    subtle: {
      blue: 'hover:bg-blue-500/10 text-blue-400',
      gray: 'hover:bg-white/5 text-gray-400',
      red: 'hover:bg-red-500/10 text-red-400',
      green: 'hover:bg-emerald-500/10 text-emerald-400',
      amber: 'hover:bg-amber-500/10 text-amber-400'
    },
    transparent: {
      blue: 'text-blue-400',
      gray: 'text-gray-400',
      red: 'text-red-400',
      green: 'text-emerald-400',
      amber: 'text-amber-400'
    }
  }[variant][color];

  return (
    <button
      onClick={onClick}
      disabled={disabled}
      title={title}
      className={`
        ${sizeClass}
        ${variantClass}
        flex items-center justify-center
        rounded
        transition-all
        disabled:opacity-50 disabled:cursor-not-allowed
        ${className}
      `}
    >
      {children}
    </button>
  );
};

