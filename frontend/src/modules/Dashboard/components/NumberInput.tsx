import React from 'react';

interface NumberInputProps {
  value: number;
  onChange: (value: number | undefined) => void;
  min?: number;
  max?: number;
  step?: number;
  size?: 'xs' | 'sm' | 'md';
  style?: React.CSSProperties;
  className?: string;
  disabled?: boolean;
  placeholder?: string;
}

/**
 * NumberInput: Componente de input numérico según Design System.
 */
export const NumberInput: React.FC<NumberInputProps> = ({
  value,
  onChange,
  min,
  max,
  step = 1,
  size = 'sm',
  style,
  className = '',
  disabled = false,
  placeholder
}) => {
  const sizeClasses = {
    xs: 'text-[10px] px-2 py-1',
    sm: 'text-xs px-3 py-1.5',
    md: 'text-sm px-3 py-2'
  };

  const handleChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const numValue = e.target.value === '' ? undefined : Number(e.target.value);
    onChange(numValue);
  };

  return (
    <input
      type="number"
      value={value ?? ''}
      onChange={handleChange}
      min={min}
      max={max}
      step={step}
      disabled={disabled}
      placeholder={placeholder}
      className={`
        ${sizeClasses[size]}
        bg-white/5 border border-white/10 rounded text-gray-300
        placeholder-gray-600 focus:outline-none focus:border-blue-500/50
        disabled:opacity-50 disabled:cursor-not-allowed font-mono
        ${className}
      `}
      style={style}
    />
  );
};

