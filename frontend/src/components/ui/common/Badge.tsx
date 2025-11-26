import React from 'react';
import { cn } from '../../../utils/cn';

interface BadgeProps extends React.HTMLAttributes<HTMLDivElement> {
  variant?: 'default' | 'outline' | 'secondary' | 'accent';
  color?: 'teal' | 'blue' | 'pink' | 'amber' | 'emerald' | 'red' | 'gray';
  size?: 'xs' | 'sm';
}

export const Badge = React.forwardRef<HTMLDivElement, BadgeProps>(
  ({ className, color = 'teal', size = 'xs', ...props }, ref) => {
    // Maps for colors in different variants
    const colorStyles = {
      teal: 'bg-teal-500/10 text-teal-400 border-teal-500/20',
      blue: 'bg-blue-500/10 text-blue-400 border-blue-500/20',
      pink: 'bg-pink-500/10 text-pink-400 border-pink-500/20',
      amber: 'bg-amber-500/10 text-amber-400 border-amber-500/20',
      emerald: 'bg-emerald-500/10 text-emerald-400 border-emerald-500/20',
      red: 'bg-red-500/10 text-red-400 border-red-500/20',
      gray: 'bg-slate-500/10 text-slate-400 border-slate-500/20',
    };

    const sizes = {
      xs: 'text-[10px] px-1.5 py-0.5 h-5',
      sm: 'text-xs px-2.5 py-0.5 h-6',
    };

    return (
      <div
        ref={ref}
        className={cn(
          'inline-flex items-center rounded-full border font-semibold transition-colors focus:outline-none focus:ring-2 focus:ring-ring focus:ring-offset-2',
          colorStyles[color],
          sizes[size],
          className
        )}
        {...props}
      />
    );
  }
);

Badge.displayName = 'Badge';
