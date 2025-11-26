import React from 'react';
import { cn } from '../../../utils/cn';

interface NumberInputProps extends Omit<React.InputHTMLAttributes<HTMLInputElement>, 'onChange'> {
  label?: string;
  error?: string;
  description?: string;
  onChange?: (value: number) => void;
}

export const NumberInput = React.forwardRef<HTMLInputElement, NumberInputProps>(
  ({ className, label, error, description, onChange, ...props }, ref) => {
    return (
      <div className="w-full space-y-1">
        {label && (
          <label className="text-[10px] font-medium text-slate-400 uppercase tracking-wider">
            {label}
          </label>
        )}
        <input
          type="number"
          ref={ref}
          onChange={(e) => onChange?.(Number(e.target.value))}
          className={cn(
            'flex h-8 w-full rounded-md border border-white/10 bg-dark-950 px-3 py-1 text-xs text-slate-200 placeholder:text-slate-500 focus:outline-none focus:ring-1 focus:ring-teal-500 disabled:cursor-not-allowed disabled:opacity-50 font-mono',
            error && 'border-red-500 focus:ring-red-500',
            className
          )}
          {...props}
        />
        {description && <p className="text-[10px] text-slate-500">{description}</p>}
        {error && <p className="text-[10px] font-medium text-red-500">{error}</p>}
      </div>
    );
  }
);

NumberInput.displayName = 'NumberInput';
