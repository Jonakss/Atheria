import React, { useState } from 'react';
import { Settings, Aperture, Power, Plug, ChevronRight, ChevronLeft } from 'lucide-react';
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
  const [epochsExpanded, setEpochsExpanded] = useState(false); // Badges colapsados por defecto
  const { connectionStatus, compileStatus, connect, disconnect } = useWebSocket();
  
  const handleConnectDisconnect = () => {
    if (connectionStatus === 'connected') {
      disconnect();
    } else {
      connect();
    }
  };
  
  // Determinar estado del sistema según connectionStatus y compileStatus
  const getSystemStatus = () => {
    if (connectionStatus === 'connected') {
      // Debug: Log compileStatus para verificar qué se recibe
      if (compileStatus) {
        console.log('🔍 ScientificHeader - compileStatus:', compileStatus);
      }
      
      // Mostrar dispositivo (CPU/CUDA) y tipo de motor
      const device = compileStatus?.device_str?.toUpperCase() || 'CPU'; // CORREGIDO: usar device_str
      const isNative = compileStatus?.is_native || false;
      const isCompiled = compileStatus?.is_compiled || false;
      
      console.log(`🔍 ScientificHeader - device=${device}, isNative=${isNative}, isCompiled=${isCompiled}`);
      
      let text = device;
      let dotColor = device === 'CUDA' ? 'bg-emerald-500' : 'bg-blue-500';
      let textColor = device === 'CUDA' ? 'text-emerald-500' : 'text-blue-500';
      let pulse = false;
      let shadow = '';
      
      if (isNative) {
        text = `${device} (SPARSE)`;
        dotColor = device === 'CUDA' ? 'bg-emerald-500' : 'bg-emerald-400';
        textColor = device === 'CUDA' ? 'text-emerald-500' : 'text-emerald-400';
        pulse = true;
        shadow = 'shadow-[0_0_8px_rgba(16,185,129,0.4)]';
      } else if (isCompiled) {
        text = `${device} (COMPILED)`;
        dotColor = 'bg-blue-500';
        textColor = 'text-blue-500';
        pulse = true;
        shadow = 'shadow-[0_0_8px_rgba(59,130,246,0.4)]';
      }
      
      return { 
        text, 
        dotColor,
        textColor,
        pulse,
        shadow
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
            <span className="text-[8px] text-gray-600 font-mono uppercase mt-0.5" title={`Frontend: ${getFormattedVersion()}`}>{getFormattedVersion()}</span>
          </div>
        </div>

        {/* Separador Vertical */}
        <div className="h-4 w-px bg-white/10" />

        {/* Timeline de Épocas (Tabs interactivos) - Colapsable */}
        <div 
          className="relative flex items-center"
          onMouseEnter={() => setEpochsExpanded(true)}
          onMouseLeave={() => setEpochsExpanded(false)}
        >
          {/* Barra fina cuando está colapsado */}
          {!epochsExpanded && (
            <button
              onClick={() => setEpochsExpanded(true)}
              onMouseEnter={() => setEpochsExpanded(true)}
              className="w-1 h-8 bg-white/10 hover:w-2 hover:bg-white/30 rounded-full transition-all duration-300 cursor-pointer group relative"
              title="Expandir épocas - Click o hover para mostrar"
            >
              {/* Indicador de época activa en la barra */}
              <div 
                className="absolute left-1/2 top-1/2 -translate-x-1/2 -translate-y-1/2 w-2 h-2 rounded-full transition-all"
                style={{
                  top: `${(currentEpoch / 5) * 100}%`,
                  backgroundColor: 'rgba(59, 130, 246, 0.6)',
                  boxShadow: '0 0 4px rgba(59, 130, 246, 0.8)'
                }}
              />
              <div className="w-full h-full bg-gradient-to-b from-blue-500/20 via-purple-500/20 to-green-500/20 rounded-full opacity-0 group-hover:opacity-100 transition-opacity" />
            </button>
          )}
          
          {/* Badges de época cuando está expandido */}
          {epochsExpanded && (
            <div className="flex items-center gap-0.5 animate-in slide-in-from-left-2 duration-200">
              {[0, 1, 2, 3, 4, 5].map(e => (
                <EpochBadge 
                  key={e}
                  era={e} 
                  current={currentEpoch}
                  onClick={() => onEpochChange?.(e)}
                />
              ))}
              {/* Botón para colapsar */}
              <button
                onClick={(e) => {
                  e.stopPropagation();
                  setEpochsExpanded(false);
                }}
                className="ml-1 p-1 text-gray-500 hover:text-gray-300 transition-colors"
                title="Colapsar épocas"
              >
                <ChevronLeft size={12} />
              </button>
            </div>
          )}
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
        
        {/* Botón Conectar/Desconectar */}
        <button
          onClick={handleConnectDisconnect}
          className={`w-7 h-7 rounded flex items-center justify-center transition-all relative group border ${
            connectionStatus === 'connected'
              ? 'bg-emerald-500/10 border-emerald-500/50 text-emerald-400 hover:bg-emerald-500/20'
              : 'bg-gray-800/50 border-gray-700 text-gray-400 hover:bg-gray-700 hover:text-gray-300'
          }`}
          title={connectionStatus === 'connected' ? 'Desconectar' : 'Conectar'}
        >
          {connectionStatus === 'connected' ? (
            <Power size={14} strokeWidth={2.5} className="text-emerald-400" />
          ) : (
            <Plug size={14} strokeWidth={2.5} className="text-gray-400" />
          )}
          {connectionStatus === 'connected' && (
            <div className="absolute inset-0 rounded bg-emerald-500/20 animate-pulse" />
          )}
        </button>
      </div>
      
      {/* Panel de Configuración */}
      <SettingsPanel isOpen={settingsOpen} onClose={() => setSettingsOpen(false)} />
    </header>
  );
};
