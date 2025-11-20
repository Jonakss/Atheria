import React from 'react';

interface ContainerProps {
  children: React.ReactNode;
  className?: string;
  size?: 'xs' | 'sm' | 'md' | 'lg' | 'xl' | 'fluid';
}

/**
 * Container: Contenedor semántico con ancho controlado.
 * Equivalente a Mantine Container, usado para centrar contenido con max-width.
 */
export const Container: React.FC<ContainerProps> = ({ 
  children, 
  className = '',
  size = 'md'
}) => {
  const sizeClass = {
    xs: 'max-w-xs',
    sm: 'max-w-sm',
    md: 'max-w-md',
    lg: 'max-w-lg',
    xl: 'max-w-xl',
    fluid: 'max-w-full'
  }[size];

  return (
    <div className={`mx-auto ${sizeClass} ${className}`}>
      {children}
    </div>
  );
};

