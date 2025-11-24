import React from 'react';
import { cn } from '../../../utils/cn';

interface ActionIconProps extends React.ButtonHTMLAttributes<HTMLButtonElement> {
  variant?: 'default' | 'outline' | 'ghost' | 'filled';
  size?: 'xs' | 'sm' | 'md' | 'lg';
  active?: boolean;
}

export const ActionIcon = React.forwardRef<HTMLButtonElement, ActionIconProps>(
  ({ className, variant = 'ghost', size = 'md', active, children, ...props }, ref) => {
    const variants = {
      default: 'bg-dark-800 text-slate-200 hover:bg-dark-700 border border-white/5',
      outline: 'border border-slate-700 bg-transparent hover:bg-slate-800 text-slate-200',
      ghost: 'hover:bg-white/5 text-slate-400 hover:text-slate-200',
      filled: 'bg-teal-500 text-white hover:bg-teal-600 shadow-sm shadow-teal-500/20',
    };

    const sizes = {
      xs: 'h-6 w-6 p-0.5',
      sm: 'h-8 w-8 p-1',
      md: 'h-9 w-9 p-1.5',
      lg: 'h-10 w-10 p-2',
    };

    return (
      <button
        ref={ref}
        className={cn(
          'inline-flex items-center justify-center rounded-md transition-colors focus-visible:outline-none focus-visible:ring-1 focus-visible:ring-teal-500 disabled:pointer-events-none disabled:opacity-50',
          variants[variant],
          sizes[size],
          active && 'bg-teal-500/10 text-teal-400 border-teal-500/20',
          className
        )}
        {...props}
      >
        {children}
      </button>
    );
  }
);
