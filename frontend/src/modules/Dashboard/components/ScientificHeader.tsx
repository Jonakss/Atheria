import React, { useState } from 'react';
import { Settings, Aperture } from 'lucide-react';
import { EpochBadge } from './EpochBadge';
import { SettingsPanel } from './SettingsPanel';
import { useWebSocket } from '../../../hooks/useWebSocket';
import { getFormattedVersion } from '../../../utils/version';

interface ScientificHeaderProps {
  currentEpoch: number;
  onEpochChange?: (epoch: number) => void;
}

export const ScientificHeader: React.FC<ScientificHeaderProps> = ({ currentEpoch, onEpochChange }) => {
  const [settingsOpen, setSettingsOpen] = useState(false);
  const { connectionStatus, compileStatus } = useWebSocket();
  
  // Determinar estado del sistema según connectionStatus y compileStatus
  const getSystemStatus = () => {
    if (connectionStatus === 'connected' && compileStatus?.is_native) {
      return { 
        text: 'SPARSE_ENGINE::READY', 
        dotColor: 'bg-emerald-500',
        textColor: 'text-emerald-500',
        pulse: true,
        shadow: 'shadow-[0_0_8px_rgba(16,185,129,0.4)]'
      };
    } else if (connectionStatus === 'connected' && compileStatus?.is_compiled) {
      return { 
        text: 'PYTORCH_ENGINE::READY', 
        dotColor: 'bg-blue-500',
        textColor: 'text-blue-500',
        pulse: true,
        shadow: 'shadow-[0_0_8px_rgba(59,130,246,0.4)]'
      };
    } else if (connectionStatus === 'connected') {
      return { 
        text: 'ENGINE::CONNECTED', 
        dotColor: 'bg-blue-400',
        textColor: 'text-blue-400',
        pulse: false,
        shadow: ''
      };
    } else if (connectionStatus === 'connecting') {
      return { 
        text: 'ENGINE::CONNECTING', 
        dotColor: 'bg-amber-400',
        textColor: 'text-amber-400',
        pulse: true,
        shadow: 'shadow-[0_0_8px_rgba(251,191,36,0.4)]'
      };
    } else {
      return { 
        text: 'ENGINE::DISCONNECTED', 
        dotColor: 'bg-gray-500',
        textColor: 'text-gray-500',
        pulse: false,
        shadow: ''
      };
    }
  };
  
  const systemStatus = getSystemStatus();

  return (
    <header className="h-12 border-b border-white/10 bg-[#050505] flex items-center justify-between px-4 z-50 shrink-0">
      <div className="flex items-center gap-6">
        {/* Identidad */}
        <div className="flex items-center gap-3 group cursor-pointer opacity-90 hover:opacity-100 transition-opacity">
          <div className="relative w-6 h-6 flex items-center justify-center border border-white/10 rounded bg-white/5">
            <Aperture size={14} className="text-gray-300" />
          </div>
          <div className="flex flex-col leading-none">
            <span className="text-xs font-bold text-gray-200 tracking-wide">
              ATHERIA<span className="text-blue-500">_LAB</span>
            </span>
            <span className="text-[8px] text-gray-600 font-mono uppercase mt-0.5">{getFormattedVersion()}</span>
          </div>
        </div>

        {/* Separador Vertical */}
        <div className="h-4 w-px bg-white/10" />

        {/* Timeline de Épocas (Tabs interactivos) - Design System: gap-0.5 según mockup */}
        <div className="flex items-center gap-0.5">
          {[0, 1, 2, 3, 4, 5].map(e => (
            <EpochBadge 
              key={e}
              era={e} 
              current={currentEpoch}
              onClick={() => onEpochChange?.(e)}
            />
          ))}
        </div>
      </div>

      <div className="flex items-center gap-4">
        {/* Indicador de Estado del Sistema - Conectado a datos reales */}
        <div className="flex items-center gap-2 px-2 py-1 bg-white/5 rounded border border-white/5">
          <div className={`w-1.5 h-1.5 rounded-full ${systemStatus.dotColor} ${
            systemStatus.pulse ? `animate-pulse ${systemStatus.shadow}` : ''
          }`} />
          <span className={`text-[10px] font-mono ${systemStatus.textColor} tracking-wide`}>
            {systemStatus.text}
          </span>
        </div>
        
        <div className="h-4 w-px bg-white/10" />

        {/* Perfil / Config */}
        <button 
          onClick={() => setSettingsOpen(true)}
          className="p-1.5 text-gray-500 hover:text-gray-300 transition-colors" 
          title="Configuración Global"
        >
          <Settings size={16} />
        </button>
        <div className="w-6 h-6 rounded bg-gray-800 border border-gray-700 flex items-center justify-center text-[10px] text-gray-400 font-bold">JS</div>
      </div>
      
      {/* Panel de Configuración */}
      <SettingsPanel isOpen={settingsOpen} onClose={() => setSettingsOpen(false)} />
    </header>
  );
};
