import React from 'react';
import { cn } from '../../../utils/cn';

interface SliderProps extends Omit<React.InputHTMLAttributes<HTMLInputElement>, 'type'> {
  label?: string;
  min?: number;
  max?: number;
  step?: number;
  value?: number;
  onChange?: (e: React.ChangeEvent<HTMLInputElement>) => void;
  formatTooltip?: (value: number) => string;
}

export const Slider = React.forwardRef<HTMLInputElement, SliderProps>(
  ({ className, label, min = 0, max = 100, step = 1, value, onChange, formatTooltip, ...props }, ref) => {
    // Calculate percentage for background gradient
    const percent = value !== undefined
      ? ((Number(value) - min) / (max - min)) * 100
      : 0;

    return (
      <div className={cn("w-full space-y-1.5", className)}>
        {label && (
          <div className="flex justify-between text-[10px]">
            <span className="text-slate-400 font-medium uppercase tracking-wider">{label}</span>
            <span className="font-mono text-slate-200 bg-white/5 px-1.5 rounded">
              {formatTooltip ? formatTooltip(Number(value)) : value}
            </span>
          </div>
        )}
        <div className="relative h-4 flex items-center">
          <input
            type="range"
            ref={ref}
            min={min}
            max={max}
            step={step}
            value={value}
            onChange={onChange}
            className="w-full h-1.5 bg-dark-800 rounded-full appearance-none cursor-pointer slider-thumb focus:outline-none focus:ring-1 focus:ring-teal-500/50"
            style={{
              backgroundImage: `linear-gradient(to right, rgb(20, 184, 166) 0%, rgb(20, 184, 166) ${percent}%, rgb(30, 41, 59) ${percent}%, rgb(30, 41, 59) 100%)`
            }}
            {...props}
          />
        </div>
      </div>
    );
  }
);
