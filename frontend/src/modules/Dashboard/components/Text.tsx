import React from 'react';

interface TextProps {
  children: React.ReactNode;
  className?: string;
  size?: 'xs' | 'sm' | 'md' | 'lg' | 'xl';
  weight?: 'normal' | 'medium' | 'bold';
  color?: 'default' | 'dimmed' | 'muted';
  align?: 'left' | 'center' | 'right';
  style?: React.CSSProperties;
}

/**
 * Text: Componente semántico de texto.
 * Equivalente a Mantine Text, usado para texto estilizado.
 */
export const Text: React.FC<TextProps> = ({ 
  children, 
  className = '',
  size = 'sm',
  weight = 'normal',
  color = 'default',
  align = 'left',
  style
}) => {
  const sizeClass = {
    xs: 'text-xs',
    sm: 'text-sm',
    md: 'text-base',
    lg: 'text-lg',
    xl: 'text-xl'
  }[size];

  const weightClass = {
    normal: 'font-normal',
    medium: 'font-medium',
    bold: 'font-bold'
  }[weight];

  const colorClass = {
    default: 'text-gray-200',
    dimmed: 'text-gray-500',
    muted: 'text-gray-400'
  }[color];

  const alignClass = {
    left: 'text-left',
    center: 'text-center',
    right: 'text-right'
  }[align];

  return (
    <span className={`${sizeClass} ${weightClass} ${colorClass} ${alignClass} ${className}`} style={style}>
      {children}
    </span>
  );
};

