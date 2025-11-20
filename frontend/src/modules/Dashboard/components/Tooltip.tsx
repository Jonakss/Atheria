import React, { useState, useRef, useEffect } from 'react';

interface TooltipProps {
  label: string;
  children: React.ReactNode;
  position?: 'top' | 'bottom' | 'left' | 'right';
  className?: string;
}

/**
 * Tooltip: Componente semántico de tooltip.
 * Equivalente a Mantine Tooltip, usado para mostrar información adicional.
 */
export const Tooltip: React.FC<TooltipProps> = ({ 
  label,
  children,
  position = 'top',
  className = ''
}) => {
  const [isVisible, setIsVisible] = useState(false);
  const wrapperRef = useRef<HTMLDivElement>(null);
  const tooltipRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    if (!isVisible) return;

    const updatePosition = () => {
      if (!wrapperRef.current || !tooltipRef.current) return;

      const wrapperRect = wrapperRef.current.getBoundingClientRect();
      const tooltipRect = tooltipRef.current.getBoundingClientRect();

      let top = 0;
      let left = 0;

      switch (position) {
        case 'top':
          top = wrapperRect.top - tooltipRect.height - 8;
          left = wrapperRect.left + wrapperRect.width / 2 - tooltipRect.width / 2;
          break;
        case 'bottom':
          top = wrapperRect.bottom + 8;
          left = wrapperRect.left + wrapperRect.width / 2 - tooltipRect.width / 2;
          break;
        case 'left':
          top = wrapperRect.top + wrapperRect.height / 2 - tooltipRect.height / 2;
          left = wrapperRect.left - tooltipRect.width - 8;
          break;
        case 'right':
          top = wrapperRect.top + wrapperRect.height / 2 - tooltipRect.height / 2;
          left = wrapperRect.right + 8;
          break;
      }

      tooltipRef.current.style.top = `${top}px`;
      tooltipRef.current.style.left = `${left}px`;
    };

    updatePosition();
    window.addEventListener('scroll', updatePosition, true);
    window.addEventListener('resize', updatePosition);

    return () => {
      window.removeEventListener('scroll', updatePosition, true);
      window.removeEventListener('resize', updatePosition);
    };
  }, [isVisible, position]);

  return (
    <div
      ref={wrapperRef}
      className={`relative inline-block ${className}`}
      onMouseEnter={() => setIsVisible(true)}
      onMouseLeave={() => setIsVisible(false)}
    >
      {children}
      {isVisible && (
        <div
          ref={tooltipRef}
          className="fixed z-50 px-2 py-1 bg-gray-900 text-white text-xs rounded shadow-lg pointer-events-none whitespace-nowrap"
          style={{ top: 0, left: 0 }}
        >
          {label}
        </div>
      )}
    </div>
  );
};

