import React from 'react';

interface SwitchProps {
  label?: string;
  description?: string;
  checked: boolean;
  onChange: (event: React.ChangeEvent<HTMLInputElement>) => void;
  disabled?: boolean;
  size?: 'xs' | 'sm' | 'md';
  className?: string;
}

/**
 * Switch: Componente semántico de interruptor (toggle).
 * Equivalente a Mantine Switch, usado para activar/desactivar opciones.
 */
export const Switch: React.FC<SwitchProps> = ({ 
  label,
  description,
  checked,
  onChange,
  disabled = false,
  size = 'md',
  className = ''
}) => {
  const sizeClass = {
    xs: 'w-8 h-4',
    sm: 'w-10 h-5',
    md: 'w-12 h-6'
  }[size];

  const thumbSizeClass = {
    xs: 'w-3 h-3',
    sm: 'w-4 h-4',
    md: 'w-5 h-5'
  }[size];

  return (
    <label className={`flex items-center gap-2 cursor-pointer group ${disabled ? 'opacity-50 cursor-not-allowed' : ''} ${className}`}>
      <div className={`relative ${sizeClass} rounded-full transition-colors ${checked ? 'bg-blue-500' : 'bg-gray-700'} ${disabled ? '' : 'group-hover:bg-opacity-80'}`}>
        <input
          type="checkbox"
          checked={checked}
          onChange={onChange}
          disabled={disabled}
          className="sr-only"
        />
        <div
          className={`absolute ${thumbSizeClass} bg-white rounded-full shadow-md transform transition-transform ${
            checked ? `translate-x-full ${size === 'xs' ? '-translate-x-[calc(100%+4px)]' : size === 'sm' ? '-translate-x-[calc(100%+4px)]' : '-translate-x-[calc(100%+8px)]'}` : 'translate-x-0.5'
          } top-1/2 -translate-y-1/2`}
        />
      </div>
      {(label || description) && (
        <div className="flex flex-col">
          {label && <span className="text-xs text-gray-300 font-medium">{label}</span>}
          {description && <span className="text-[10px] text-gray-500">{description}</span>}
        </div>
      )}
    </label>
  );
};

