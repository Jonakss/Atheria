import React from 'react';
import { cn } from '../../../utils/cn';

interface PaperProps extends React.HTMLAttributes<HTMLDivElement> {
  withBorder?: boolean;
}

export const Paper = React.forwardRef<HTMLDivElement, PaperProps>(
  function Paper({ className, withBorder = true, ...props }, ref) {
    return (
      <div
        ref={ref}
        className={cn(
          'bg-dark-900/40 backdrop-blur-sm rounded-lg text-slate-200',
          withBorder && 'border border-white/5',
          className
        )}
        {...props}
      />
    );
  }
);
