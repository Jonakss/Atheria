import { Check, ChevronRight } from 'lucide-react';
import React, { ReactElement } from 'react';

interface StepperProps {
  active: number;
  onStepClick?: (step: number) => void;
  children: ReactElement<StepProps>[];
  orientation?: 'horizontal' | 'vertical';
}

interface StepProps {
  label: React.ReactNode;
  description?: React.ReactNode;
  icon?: React.ReactNode;
  children: React.ReactNode;
}

/**
 * Stepper: Componente de pasos según Design System.
 */
export const Stepper: React.FC<StepperProps> = ({ 
  active, 
  onStepClick,
  children,
  orientation = 'horizontal'
}) => {
  const steps = React.Children.toArray(children).filter(
    (child): child is ReactElement<StepProps> =>
      React.isValidElement(child) && child.type === Step
  );

  const activeStep = steps[active];
  const isCompleted = active >= steps.length;

  if (orientation === 'vertical') {
    return (
      <div className="flex flex-col gap-4">
        {steps.map((step, index) => {
          const stepProps = step.props;
          const isActive = index === active;
          const isPast = index < active;


          return (
            <div key={index} className="flex gap-4">
              <div className="flex flex-col items-center">
                <button
                  onClick={() => onStepClick?.(index)}
                  disabled={!onStepClick}
                  className={`
                    w-8 h-8 rounded-full border-2 flex items-center justify-center font-bold text-xs transition-colors
                    ${isPast 
                      ? 'bg-emerald-500/20 text-emerald-400 border-emerald-500/30' 
                      : isActive
                        ? 'bg-blue-500/20 text-blue-400 border-blue-500/30'
                        : 'bg-white/5 text-gray-500 border-white/10'
                    }
                    ${onStepClick ? 'cursor-pointer hover:bg-white/10' : 'cursor-default'}
                  `}
                >
                  {isPast ? <Check size={14} /> : index + 1}
                </button>
                {index < steps.length - 1 && (
                  <div className={`w-0.5 h-full min-h-8 my-2 ${
                    isPast ? 'bg-emerald-500/30' : 'bg-white/10'
                  }`} />
                )}
              </div>
              <div className="flex-1 pb-8">
                <div className="space-y-2">
                  <div>
                    <div className="text-xs font-bold text-gray-300 mb-1">
                      {stepProps.label}
                    </div>
                    {stepProps.description && (
                      <div className="text-[10px] text-gray-500">
                        {stepProps.description}
                      </div>
                    )}
                  </div>
                  {isActive && (
                    <div className="mt-4">
                      {stepProps.children}
                    </div>
                  )}
                </div>
              </div>
            </div>
          );
        })}
      </div>
    );
  }

  return (
    <div className="flex flex-col">
      {/* Stepper Header */}
      <div className="flex items-center justify-between mb-6">
        {steps.map((step, index) => {
          const stepProps = step.props;
          const isActive = index === active;
          const isPast = index < active;


          return (
            <React.Fragment key={index}>
              <div className="flex items-center">
                <button
                  onClick={() => onStepClick?.(index)}
                  disabled={!onStepClick}
                  className={`
                    flex flex-col items-center gap-1 min-w-0 flex-1 transition-colors
                    ${onStepClick ? 'cursor-pointer hover:opacity-80' : 'cursor-default'}
                  `}
                >
                  <div className={`
                    w-8 h-8 rounded-full border-2 flex items-center justify-center font-bold text-xs transition-colors
                    ${isPast 
                      ? 'bg-emerald-500/20 text-emerald-400 border-emerald-500/30' 
                      : isActive
                        ? 'bg-blue-500/20 text-blue-400 border-blue-500/30'
                        : 'bg-white/5 text-gray-500 border-white/10'
                    }
                  `}>
                    {isPast ? <Check size={14} /> : index + 1}
                  </div>
                  <div className="text-center">
                    <div className={`text-[10px] font-bold ${
                      isActive ? 'text-blue-400' : isPast ? 'text-emerald-400' : 'text-gray-500'
                    }`}>
                      {stepProps.label}
                    </div>
                    {stepProps.description && (
                      <div className="text-[9px] text-gray-600 mt-0.5">
                        {stepProps.description}
                      </div>
                    )}
                  </div>
                </button>
              </div>
              {index < steps.length - 1 && (
                <ChevronRight size={16} className={`mx-2 text-gray-600 ${
                  isPast ? 'text-emerald-500/50' : ''
                }`} />
              )}
            </React.Fragment>
          );
        })}
      </div>

      {/* Active Step Content */}
      {!isCompleted && activeStep && (
        <div className="mt-4">
          {activeStep.props.children}
        </div>
      )}
    </div>
  );
};

export const Step: React.FC<StepProps> = ({ children }) => {
  return <>{children}</>;
};

/**
 * Completed: Componente para mostrar cuando se completa el stepper.
 */
export const StepperCompleted: React.FC<{ children: React.ReactNode }> = ({ children }) => {
  return <>{children}</>;
};

