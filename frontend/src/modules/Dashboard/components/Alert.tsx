import React from 'react';
import { Info, CheckCircle, XCircle, AlertTriangle } from 'lucide-react';

interface AlertProps {
  children: React.ReactNode;
  icon?: React.ReactNode;
  color?: 'blue' | 'green' | 'orange' | 'red' | 'yellow';
  variant?: 'light' | 'filled' | 'outline';
  title?: React.ReactNode;
  withCloseButton?: boolean;
  onClose?: () => void;
  className?: string;
}

const defaultIcons = {
  blue: Info,
  green: CheckCircle,
  orange: AlertTriangle,
  red: XCircle,
  yellow: AlertTriangle
};

/**
 * Alert: Componente de alerta según Design System.
 */
export const Alert: React.FC<AlertProps> = ({ 
  children, 
  icon,
  color = 'blue',
  variant = 'light',
  title,
  withCloseButton = false,
  onClose,
  className = ''
}) => {
  const DefaultIcon = defaultIcons[color];
  const displayIcon = icon || <DefaultIcon size={16} />;

  const colorClasses = {
    blue: {
      light: 'bg-blue-500/10 border-blue-500/30 text-blue-400',
      filled: 'bg-blue-500/20 border-blue-500/40 text-blue-300',
      outline: 'bg-transparent border-blue-500/50 text-blue-400'
    },
    green: {
      light: 'bg-emerald-500/10 border-emerald-500/30 text-emerald-400',
      filled: 'bg-emerald-500/20 border-emerald-500/40 text-emerald-300',
      outline: 'bg-transparent border-emerald-500/50 text-emerald-400'
    },
    orange: {
      light: 'bg-amber-500/10 border-amber-500/30 text-amber-400',
      filled: 'bg-amber-500/20 border-amber-500/40 text-amber-300',
      outline: 'bg-transparent border-amber-500/50 text-amber-400'
    },
    red: {
      light: 'bg-red-500/10 border-red-500/30 text-red-400',
      filled: 'bg-red-500/20 border-red-500/40 text-red-300',
      outline: 'bg-transparent border-red-500/50 text-red-400'
    },
    yellow: {
      light: 'bg-yellow-500/10 border-yellow-500/30 text-yellow-400',
      filled: 'bg-yellow-500/20 border-yellow-500/40 text-yellow-300',
      outline: 'bg-transparent border-yellow-500/50 text-yellow-400'
    }
  };

  return (
    <div className={`rounded-lg border p-3 ${colorClasses[color][variant]} ${className}`}>
      <div className="flex items-start gap-3">
        <div className="shrink-0 mt-0.5">
          {displayIcon}
        </div>
        <div className="flex-1 min-w-0">
          {title && (
            <div className="text-xs font-bold mb-1">
              {title}
            </div>
          )}
          <div className="text-xs">
            {children}
          </div>
        </div>
        {withCloseButton && onClose && (
          <button
            onClick={onClose}
            className="shrink-0 p-1 hover:bg-white/5 rounded transition-colors"
          >
            <XCircle size={14} />
          </button>
        )}
      </div>
    </div>
  );
};

