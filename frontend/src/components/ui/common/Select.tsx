import { ChevronDown } from 'lucide-react';
import React from 'react';
import { cn } from '../../../utils/cn';

interface SelectProps extends React.SelectHTMLAttributes<HTMLSelectElement> {
  label?: string;
  error?: string;
  description?: string;
  options?: { value: string | number; label: string }[];
}

export const Select = React.forwardRef<HTMLSelectElement, SelectProps>(
  function Select({ className, label, error, description, options, children, ...props }, ref) {
    return (
      <div className="w-full space-y-1">
        {label && (
          <label className="text-[10px] font-medium text-slate-400 uppercase tracking-wider">
            {label}
          </label>
        )}
        <div className="relative">
          <select
            ref={ref}
            className={cn(
              'flex h-8 w-full items-center justify-between rounded-md border border-white/10 bg-dark-950 px-3 py-1 text-xs text-slate-200 placeholder:text-slate-500 focus:outline-none focus:ring-1 focus:ring-teal-500 disabled:cursor-not-allowed disabled:opacity-50 appearance-none',
              error && 'border-red-500 focus:ring-red-500',
              className
            )}
            {...props}
          >
            {options
              ? options.map((opt) => (
                  <option key={opt.value} value={opt.value}>
                    {opt.label}
                  </option>
                ))
              : children
            }
          </select>
          <ChevronDown className="absolute right-2 top-2.5 h-3 w-3 opacity-50 pointer-events-none" />
        </div>
        {description && <p className="text-[10px] text-slate-500">{description}</p>}
        {error && <p className="text-[10px] font-medium text-red-500">{error}</p>}
      </div>
    );
  }
);
