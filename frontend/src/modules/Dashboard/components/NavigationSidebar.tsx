import React, { useState } from 'react';
import { Activity, Database, Terminal, Microscope, FlaskConical, Brain, BarChart3 } from 'lucide-react';
import { NavButton } from './NavButton';

type TabType = 'lab' | 'analysis' | 'history' | 'logs';
type LabSection = 'inference' | 'training' | 'analysis';

interface NavigationSidebarProps {
  activeTab: TabType;
  onTabChange: (tab: TabType) => void;
  labPanelOpen?: boolean;
  activeLabSection?: LabSection;
  onLabSectionChange?: (section: LabSection) => void;
}

const NAV_ITEMS = [
  { id: 'lab' as TabType, icon: Microscope, label: 'Lab', hasSubmenu: true },
  { id: 'analysis' as TabType, icon: Activity, label: 'Analytics' },
  { id: 'history' as TabType, icon: Database, label: 'Data' },
  { id: 'logs' as TabType, icon: Terminal, label: 'Console' },
];

// Sub-menús para Lab
const LAB_SUBMENU_ITEMS = [
  { id: 'inference' as LabSection, icon: FlaskConical, label: 'Inferencia', color: 'blue' },
  { id: 'training' as LabSection, icon: Brain, label: 'Entrenamiento', color: 'emerald' },
  { id: 'analysis' as LabSection, icon: BarChart3, label: 'Análisis', color: 'amber' },
];

/**
 * NavigationSidebar: Sidebar de navegación con sub-menús deslizables.
 * 
 * Design System Spec:
 * - Ancho: w-12 (48px) o w-16 (64px) - usando w-12 según mockup
 * - Background: bg-[#050505] (Surface)
 * - Border: border-r border-white/5 (Sutil)
 * - Layout: flex flex-col items-center py-3 gap-2
 * - Sub-menús se deslizan hacia abajo cuando el menú principal está activo
 */
export const NavigationSidebar: React.FC<NavigationSidebarProps> = ({ 
  activeTab, 
  onTabChange, 
  labPanelOpen = false,
  activeLabSection = 'inference',
  onLabSectionChange
}) => {
  const isLabActive = activeTab === 'lab' && labPanelOpen;
  
  // Colores accent para cada sub-sección de Lab
  const accentColors = {
    inference: {
      bg: 'bg-blue-500/20',
      text: 'text-blue-400',
      border: 'border-blue-500/50',
      bar: 'bg-blue-500',
      glow: 'shadow-[0_0_8px_rgba(59,130,246,0.4)]'
    },
    training: {
      bg: 'bg-emerald-500/20',
      text: 'text-emerald-400',
      border: 'border-emerald-500/50',
      bar: 'bg-emerald-500',
      glow: 'shadow-[0_0_8px_rgba(16,185,129,0.4)]'
    },
    analysis: {
      bg: 'bg-amber-500/20',
      text: 'text-amber-400',
      border: 'border-amber-500/50',
      bar: 'bg-amber-500',
      glow: 'shadow-[0_0_8px_rgba(251,191,36,0.4)]'
    }
  };

  // Calcular el índice del item "lab" para insertar sub-menús después
  const labIndex = NAV_ITEMS.findIndex(item => item.id === 'lab');

  return (
    <aside className="w-12 border-r border-white/5 bg-dark-990 flex flex-col items-center py-3 gap-2 z-40 shrink-0 relative">
      {NAV_ITEMS.map((item, index) => {
        const isActive = activeTab === item.id;
        const showSubmenu = item.id === 'lab' && isLabActive;
        
        return (
          <React.Fragment key={item.id}>
            {/* Botón principal del menú */}
            <NavButton
              icon={item.icon}
              label={item.label}
              isActive={isActive}
              onClick={() => onTabChange(item.id)}
            />
            
            {/* Sub-menús de Lab - Se deslizan hacia abajo cuando Lab está activo */}
            {item.id === 'lab' && showSubmenu && (
              <div 
                className="w-full flex flex-col gap-2 transition-all duration-300 ease-in-out"
                style={{ 
                  maxHeight: '200px',
                  opacity: 1,
                  transform: 'translateY(0)'
                }}
              >
                {LAB_SUBMENU_ITEMS.map((subItem) => {
                  const Icon = subItem.icon;
                  const isSubActive = activeLabSection === subItem.id;
                  const colors = accentColors[subItem.id as keyof typeof accentColors];
                  
                  return (
                    <div key={subItem.id} className="relative flex flex-col items-center w-full">
                      <button
                        onClick={() => onLabSectionChange?.(subItem.id)}
                        className={`w-10 h-10 rounded-lg flex items-center justify-center transition-all relative group border ml-1 ${
                          isSubActive 
                            ? `${colors.bg} ${colors.text} ${colors.border} ${colors.glow}`
                            : 'border-transparent text-gray-600 hover:text-gray-300 hover:bg-white/5'
                        }`}
                        title={subItem.label}
                      >
                        <Icon size={18} strokeWidth={2.5} />
                      </button>
                      
                      {/* Indicador de punto cuando está activo */}
                      {isSubActive && (
                        <div className={`absolute left-1/2 -translate-x-1/2 -bottom-1 w-1.5 h-1.5 rounded-full ${colors.bar} ${colors.glow} z-10`} />
                      )}
                    </div>
                  );
                })}
              </div>
            )}
          </React.Fragment>
        );
      })}
    </aside>
  );
};
