import { ChevronLeft, ChevronRight, Eye, Microscope } from 'lucide-react';
import React, { useState } from 'react';

interface RightDrawerProps {
  isCollapsed?: boolean;
  onToggleCollapse?: () => void;
  activeTab?: 'visualization' | 'physics';
  onTabChange?: (tab: 'visualization' | 'physics') => void;
  children?: React.ReactNode;
}

type TabType = 'visualization' | 'physics';

export const RightDrawer: React.FC<RightDrawerProps> = ({
  isCollapsed = false,
  onToggleCollapse,
  activeTab: externalActiveTab,
  onTabChange,
  children
}) => {
  const [internalActiveTab, setInternalActiveTab] = useState<TabType>('visualization');
  const [internalCollapsed, setInternalCollapsed] = useState(false);

  // Use props if provided, otherwise use internal state
  const collapsed = onToggleCollapse !== undefined ? isCollapsed : internalCollapsed;
  const handleToggle = onToggleCollapse || (() => setInternalCollapsed(!internalCollapsed));
  
  const activeTab = externalActiveTab || internalActiveTab;
  const handleTabChange = (tab: TabType) => {
    if (onTabChange) {
      onTabChange(tab);
    } else {
      setInternalActiveTab(tab);
    }
  };

  const tabs = [
    { id: 'visualization' as TabType, label: 'Visualización', icon: Eye },
    { id: 'physics' as TabType, label: 'Física', icon: Microscope }
  ];

  return (
    <aside 
      className={`${
        collapsed ? 'w-12' : 'w-80'
      } border-l border-white/5 bg-dark-950/80 backdrop-blur-md flex flex-col z-40 shrink-0 transition-all duration-300 ease-in-out overflow-hidden`}
      style={{ 
        minWidth: collapsed ? '3rem' : '20rem', 
        maxWidth: collapsed ? '3rem' : '20rem' 
      }}
    >
      {/* Header with collapse button */}
      <div className="h-10 border-b border-white/5 flex items-center justify-between px-3 bg-dark-980/90 shrink-0">
        {!collapsed && (
          <span className="text-[10px] font-bold text-dark-400 uppercase tracking-widest">
            Panel de Control
          </span>
        )}
        <button
          onClick={handleToggle}
          className="p-1.5 text-dark-500 hover:text-dark-300 transition-colors rounded hover:bg-white/5"
          title={collapsed ? 'Expandir Panel' : 'Colapsar Panel'}
        >
          {collapsed ? <ChevronLeft size={14} /> : <ChevronRight size={14} />}
        </button>
      </div>

      {collapsed ? (
        // Icon-only mode
        <div className="flex flex-col gap-2 p-2">
          {tabs.map((tab) => {
            const Icon = tab.icon;
            return (
              <button
                key={tab.id}
                onClick={() => {
                handleTabChange(tab.id);
                handleToggle(); // Auto-expand when clicking icon
              }}
                className={`p-2 rounded transition-colors ${
                  activeTab === tab.id
                    ? 'bg-teal-500/20 text-teal-400'
                    : 'text-dark-500 hover:text-dark-300 hover:bg-white/5'
                }`}
                title={tab.label}
              >
                <Icon size={16} />
              </button>
            );
          })}
        </div>
      ) : (
        // Expanded mode with tabs
        <>
          {/* Tab navigation */}
          <div className="flex border-b border-white/5 bg-dark-970/50">
            {tabs.map((tab) => {
              const Icon = tab.icon;
              return (
                <button
                  key={tab.id}
                  onClick={() => handleTabChange(tab.id)}
                  className={`flex-1 flex items-center justify-center gap-2 px-4 py-2 text-xs font-medium transition-colors ${
                    activeTab === tab.id
                      ? 'text-teal-400 border-b-2 border-teal-400 bg-teal-500/5'
                      : 'text-dark-500 hover:text-dark-300 hover:bg-white/5'
                  }`}
                >
                  <Icon size={14} />
                  <span>{tab.label}</span>
                </button>
              );
            })}
          </div>

          {/* Tab content */}
          <div className="flex-1 overflow-y-auto" style={{ scrollbarWidth: 'thin', scrollbarColor: 'rgba(255,255,255,0.1) transparent' }}>
            {children}
          </div>
        </>
      )}
    </aside>
  );
};
