import React from 'react';
import { cn } from '../../../utils/cn';
import { Loader2 } from 'lucide-react';

interface ButtonProps extends React.ButtonHTMLAttributes<HTMLButtonElement> {
  variant?: 'primary' | 'secondary' | 'ghost' | 'danger' | 'outline';
  size?: 'xs' | 'sm' | 'md' | 'lg' | 'icon';
  loading?: boolean;
}

export const Button = React.forwardRef<HTMLButtonElement, ButtonProps>(
  ({ className, variant = 'primary', size = 'sm', loading, children, disabled, ...props }, ref) => {
    const variants = {
      primary: 'bg-teal-500 hover:bg-teal-600 text-white shadow-sm shadow-teal-500/20',
      secondary: 'bg-white/5 hover:bg-white/10 text-slate-200 border border-white/10',
      ghost: 'hover:bg-white/5 text-slate-300 hover:text-white',
      danger: 'bg-red-500/10 hover:bg-red-500/20 text-red-400 border border-red-500/20',
      outline: 'border border-slate-600 text-slate-300 hover:border-slate-500 hover:text-white bg-transparent',
    };

    const sizes = {
      xs: 'h-6 px-2 text-[10px]',
      sm: 'h-8 px-3 text-xs',
      md: 'h-10 px-4 text-sm',
      lg: 'h-12 px-6 text-base',
      icon: 'h-8 w-8 p-0 flex items-center justify-center',
    };

    return (
      <button
        ref={ref}
        className={cn(
          'inline-flex items-center justify-center rounded-md font-medium transition-colors focus-visible:outline-none focus-visible:ring-1 focus-visible:ring-teal-500 disabled:pointer-events-none disabled:opacity-50',
          variants[variant],
          sizes[size],
          className
        )}
        disabled={disabled || loading}
        {...props}
      >
        {loading && <Loader2 className="mr-2 h-3 w-3 animate-spin" />}
        {children}
      </button>
    );
  }
);
