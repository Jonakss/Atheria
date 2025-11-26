import React from 'react';
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
 * - Ancho: w-12 (48px)
 * - Background: bg-[#050505] (Surface)
 * - Border: border-r border-white/5 (Sutil)
 * - Layout: flex flex-col items-center py-3 gap-2
 * - Sub-menús se deslizan hacia abajo cuando el menú principal está activo
 */
export const NavigationSidebar: React.FC<NavigationSidebarProps> = ({ 
  activeTab, 
  onTabChange, 
  activeLabSection = 'inference',
  onLabSectionChange
}) => {
  
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

  return (
    <aside className="w-12 border-r border-white/5 bg-dark-990 flex flex-col items-center py-3 gap-2 z-40 shrink-0 relative">
      {NAV_ITEMS.map((item) => {
        const isActive = activeTab === item.id;
        // Mostrar submenú si es Lab y la pestaña está activa
        const showSubmenu = item.id === 'lab' && isActive;
        
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
            {item.id === 'lab' && (
              <div 
                className="w-full flex flex-col gap-2 overflow-hidden transition-all duration-300 ease-in-out"
                style={{ 
                  maxHeight: showSubmenu ? '200px' : '0px',
                  opacity: showSubmenu ? 1 : 0,
                  transform: showSubmenu ? 'translateY(0)' : 'translateY(-10px)',
                  marginBottom: showSubmenu ? '0.5rem' : '0px'
                }}
              >
                {LAB_SUBMENU_ITEMS.map((subItem) => {
                  const Icon = subItem.icon;
                  const isSubActive = activeLabSection === subItem.id;
                  const colors = accentColors[subItem.id as keyof typeof accentColors];
                  
                  return (
                    <div key={subItem.id} className="relative flex flex-col items-center w-full group">
                      <button
                        onClick={(e) => {
                          e.stopPropagation(); // Evitar que burbujee si es necesario
                          onLabSectionChange?.(subItem.id);
                        }}
                        className={`w-9 h-9 rounded-lg flex items-center justify-center transition-all relative border ${
                          isSubActive 
                            ? `${colors.bg} ${colors.text} ${colors.border} ${colors.glow}`
                            : 'border-transparent text-gray-500 hover:text-gray-300 hover:bg-white/5'
                        }`}
                        title={subItem.label}
                      >
                        <Icon size={16} strokeWidth={2.5} />
                      </button>
                      
                      {/* Indicador de punto cuando está activo */}
                      {isSubActive && (
                        <div className={`absolute left-0 top-1/2 -translate-y-1/2 w-1 h-3 rounded-r-full ${colors.bar} ${colors.glow} z-10`} />
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
