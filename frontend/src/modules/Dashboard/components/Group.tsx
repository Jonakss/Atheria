import React from 'react';

interface GroupProps {
  children: React.ReactNode;
  className?: string;
  gap?: number | string;
  align?: 'start' | 'center' | 'end' | 'stretch';
  justify?: 'start' | 'center' | 'end' | 'space-between' | 'space-around';
  wrap?: boolean;
}

/**
 * Group: Contenedor semántico en fila (flex-row).
 * Equivalente a Mantine Group, usado para agrupar elementos horizontalmente.
 */
export const Group: React.FC<GroupProps> = ({ 
  children, 
  className = '', 
  gap = 4,
  align = 'center',
  justify = 'start',
  wrap = false
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
  const wrapClass = wrap ? 'flex-wrap' : 'flex-nowrap';

  return (
    <div className={`flex flex-row ${gapClass} ${alignClass} ${justifyClass} ${wrapClass} ${className}`}>
      {children}
    </div>
  );
};

