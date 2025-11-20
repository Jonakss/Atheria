import React from 'react';
import { X } from 'lucide-react';
import { GlassPanel } from './GlassPanel';

interface ModalProps {
  opened: boolean;
  onClose: () => void;
  title?: React.ReactNode;
  children: React.ReactNode;
  size?: 'sm' | 'md' | 'lg' | 'xl' | 'full';
  closeOnClickOutside?: boolean;
}

/**
 * Modal: Componente modal según Design System.
 * Implementa overlay oscuro con panel glass en el centro.
 */
export const Modal: React.FC<ModalProps> = ({ 
  opened, 
  onClose, 
  title, 
  children,
  size = 'md',
  closeOnClickOutside = true
}) => {
  if (!opened) return null;

  const sizeClasses = {
    sm: 'max-w-md',
    md: 'max-w-2xl',
    lg: 'max-w-4xl',
    xl: 'max-w-6xl',
    full: 'max-w-[95vw] max-h-[95vh]'
  };

  const handleOverlayClick = (e: React.MouseEvent<HTMLDivElement>) => {
    if (closeOnClickOutside && e.target === e.currentTarget) {
      onClose();
    }
  };

  return (
    <div 
      className="fixed inset-0 z-50 flex items-center justify-center bg-black/80 backdrop-blur-sm"
      onClick={handleOverlayClick}
    >
      <GlassPanel className={`w-full ${sizeClasses[size]} max-h-[90vh] flex flex-col overflow-hidden`}>
        {/* Header */}
        {title && (
          <div className="flex items-center justify-between p-4 border-b border-white/10 shrink-0">
            <div className="flex items-center gap-2 text-sm font-bold text-gray-200">
              {title}
            </div>
            <button
              onClick={onClose}
              className="p-1.5 text-gray-400 hover:text-gray-200 hover:bg-white/5 rounded transition-colors"
              title="Cerrar"
            >
              <X size={16} />
            </button>
          </div>
        )}
        
        {/* Content */}
        <div className="flex-1 overflow-y-auto custom-scrollbar p-4">
          {children}
        </div>
      </GlassPanel>
    </div>
  );
};

