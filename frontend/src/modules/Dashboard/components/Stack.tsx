import React from 'react';

interface StackProps {
  children: React.ReactNode;
  className?: string;
  gap?: number | string;
  align?: 'start' | 'center' | 'end' | 'stretch';
  justify?: 'start' | 'center' | 'end' | 'space-between' | 'space-around';
}

/**
 * Stack: Contenedor semántico en columna (flex-col).
 * Equivalente a Mantine Stack, usado para apilar elementos verticalmente.
 */
export const Stack: React.FC<StackProps> = ({ 
  children, 
  className = '', 
  gap = 4,
  align = 'stretch',
  justify = 'start'
}) => {
  const gapClass = typeof gap === 'number' 
    ? (gap === 1 ? 'gap-1' : gap === 2 ? 'gap-2' : gap === 3 ? 'gap-3' : gap === 4 ? 'gap-4' : `gap-[${gap}px]`)
    : `gap-[${gap}]`;
  const alignClass = {
    start: 'items-start',
    center: 'items-center',
    end: 'items-end',
    stretch: 'items-stretch'
  }[align];
  const justifyClass = {
    start: 'justify-start',
    center: 'justify-center',
    end: 'justify-end',
    'space-between': 'justify-between',
    'space-around': 'justify-around'
  }[justify];

  return (
    <div className={`flex flex-col ${gapClass} ${alignClass} ${justifyClass} ${className}`}>
      {children}
    </div>
  );
};

