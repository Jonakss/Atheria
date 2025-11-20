import React from 'react';

interface BoxProps extends React.HTMLAttributes<HTMLDivElement> {
  children: React.ReactNode;
  className?: string;
  style?: React.CSSProperties;
}

/**
 * Box: Contenedor semántico genérico.
 * Equivalente a <div> pero con significado semántico.
 */
export const Box: React.FC<BoxProps> = ({ children, className = '', style, ...props }) => (
  <div className={className} style={style} {...props}>
    {children}
  </div>
);

