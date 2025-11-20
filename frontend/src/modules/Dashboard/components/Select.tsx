import React from 'react';

interface SelectOption {
  value: string;
  label: string;
}

interface SelectProps {
  value: string;
  onChange: (value: string | null) => void;
  data: SelectOption[];
  disabled?: boolean;
  size?: 'xs' | 'sm' | 'md';
  placeholder?: string;
  className?: string;
}

/**
 * Select: Componente semántico de selector (dropdown).
 * Equivalente a Mantine Select, usado para seleccionar de una lista.
 */
export const Select: React.FC<SelectProps> = ({ 
  value,
  onChange,
  data,
  disabled = false,
  size = 'md',
  placeholder = 'Seleccionar...',
  className = ''
}) => {
  const sizeClass = {
    xs: 'px-2 py-1 text-xs',
    sm: 'px-3 py-1.5 text-sm',
    md: 'px-3 py-2 text-base'
  }[size];

  return (
    <select
      value={value}
      onChange={(e) => onChange(e.target.value || null)}
      disabled={disabled}
      className={`
        ${sizeClass}
        w-full
        bg-white/5 border border-white/10 rounded
        text-gray-300
        focus:outline-none focus:border-blue-500/50 focus:ring-1 focus:ring-blue-500/50
        disabled:opacity-50 disabled:cursor-not-allowed
        appearance-none cursor-pointer
        ${className}
      `}
      style={{
        backgroundColor: 'rgba(255, 255, 255, 0.05)',
        color: 'rgb(209, 213, 219)',
        borderColor: 'rgba(255, 255, 255, 0.1)'
      }}
    >
      {!value && <option value="" style={{ backgroundColor: '#0a0a0a', color: 'rgb(209, 213, 219)' }}>{placeholder}</option>}
      {data.map((option) => (
        <option 
          key={option.value} 
          value={option.value} 
          style={{ backgroundColor: '#0a0a0a', color: 'rgb(209, 213, 219)' }}
        >
          {option.label}
        </option>
      ))}
    </select>
  );
};

