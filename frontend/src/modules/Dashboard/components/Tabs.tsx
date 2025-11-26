import React, { ReactElement, useState } from 'react';

interface TabsProps {
  defaultValue?: string;
  value?: string;
  onTabChange?: (value: string) => void;
  children: ReactElement<TabPanelProps>[] | ReactElement<TabPanelProps>;
}

interface TabListProps {
  children: ReactElement<TabProps>[];
}

interface TabProps {
  value: string;
  label: React.ReactNode;
  icon?: React.ReactNode;
  rightSection?: React.ReactNode; // Para badges, etc.
}

interface TabPanelProps {
  value: string;
  children: React.ReactNode;
}

/**
 * Tabs: Componente de pestañas según Design System.
 */
export const Tabs: React.FC<TabsProps> = ({ 
  defaultValue, 
  value: controlledValue, 
  onTabChange, 
  children 
}) => {
  const [internalValue, setInternalValue] = useState(defaultValue || '');
  const activeValue = controlledValue ?? internalValue;

  const handleTabChange = (newValue: string) => {
    if (!controlledValue) {
      setInternalValue(newValue);
    }
    onTabChange?.(newValue);
  };

  // Extraer TabList y TabPanels
  const tabList = React.Children.toArray(children).find(
    (child): child is ReactElement<TabListProps> => 
      React.isValidElement(child) && child.type === TabList
  );

  const tabPanels = React.Children.toArray(children).filter(
    (child): child is ReactElement<TabPanelProps> =>
      React.isValidElement(child) && child.type === TabPanel
  );

  return (
    <div className="flex flex-col h-full">
      {tabList && (
        <div className="shrink-0">
          {React.cloneElement(tabList as ReactElement<TabListProps & { activeValue?: string; onTabChange?: (value: string) => void }>, { activeValue, onTabChange: handleTabChange })}
        </div>
      )}
      <div className="flex-1 overflow-hidden">
        {tabPanels.find(panel => panel.props.value === activeValue) || null}
      </div>
    </div>
  );
};

export const TabList: React.FC<TabListProps & { activeValue?: string; onTabChange?: (value: string) => void }> = ({ 
  children, 
  activeValue, 
  onTabChange 
}) => {
  return (
    <div className="flex gap-1 border-b border-white/10 mb-4">
      {React.Children.map(children, (child) => {
        if (React.isValidElement<TabProps>(child)) {
          return React.cloneElement(child as ReactElement<TabProps & { isActive?: boolean; onClick?: () => void }>, {
            isActive: child.props.value === activeValue,
            onClick: () => onTabChange?.(child.props.value)
          });
        }
        return child;
      })}
    </div>
  );
};

export const Tab: React.FC<TabProps & { isActive?: boolean; onClick?: () => void }> = ({ 
  label, 
  icon, 
  rightSection,
  isActive,
  onClick
}) => {
  return (
    <button
      onClick={onClick}
      className={`
        flex items-center gap-2 px-4 py-2 text-xs font-bold uppercase tracking-wider
        border-b-2 transition-colors
        ${isActive
          ? 'text-blue-400 border-blue-500/50 bg-blue-500/5'
          : 'text-gray-500 border-transparent hover:text-gray-300 hover:border-white/10'
        }
      `}
    >
      {icon && <span>{icon}</span>}
      <span>{label}</span>
      {rightSection && <span className="ml-1">{rightSection}</span>}
    </button>
  );
};

export const TabPanel: React.FC<TabPanelProps> = ({ children }) => {
  return <div>{children}</div>;
};

