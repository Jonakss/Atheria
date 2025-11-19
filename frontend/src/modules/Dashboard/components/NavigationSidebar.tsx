import React from 'react';
import { Activity, Database, Terminal, Microscope } from 'lucide-react';

type TabType = 'lab' | 'analysis' | 'history' | 'logs';

interface NavigationSidebarProps {
  activeTab: TabType;
  onTabChange: (tab: TabType) => void;
}

const NAV_ITEMS = [
  { id: 'lab' as TabType, icon: Microscope, label: 'Lab' },
  { id: 'analysis' as TabType, icon: Activity, label: 'Analytics' },
  { id: 'history' as TabType, icon: Database, label: 'Data' },
  { id: 'logs' as TabType, icon: Terminal, label: 'Console' },
];

export const NavigationSidebar: React.FC<NavigationSidebarProps> = ({ activeTab, onTabChange }) => {
  return (
    <aside className="w-12 border-r border-white/5 bg-[#050505] flex flex-col items-center py-3 gap-2 z-40 shrink-0">
      {NAV_ITEMS.map((item) => (
        <button 
          key={item.id}
          onClick={() => onTabChange(item.id)}
          className={`w-8 h-8 rounded flex items-center justify-center transition-all relative group ${
            activeTab === item.id 
              ? 'bg-blue-500/10 text-blue-400' 
              : 'text-gray-600 hover:text-gray-300 hover:bg-white/5'
          }`}
          title={item.label}
        >
          <item.icon size={16} strokeWidth={2} />
          {/* Indicador Activo */}
          {activeTab === item.id && (
            <div className="absolute left-0 top-1/2 -translate-y-1/2 w-0.5 h-4 bg-blue-500 rounded-r" />
          )}
        </button>
      ))}
    </aside>
  );
};
