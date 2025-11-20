import React, { useState, useMemo, useCallback } from 'react';
import { Settings, Aperture, Power, Plug, ChevronRight, ChevronLeft, ChevronDown, Cpu, Gauge, RefreshCcw } from 'lucide-react';
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
  const [engineDropdownOpen, setEngineDropdownOpen] = useState(false);
  const { connectionStatus, compileStatus, connect, disconnect, sendCommand, simData } = useWebSocket();
  
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
      // Determinar tipo de motor
      const isNative = compileStatus?.is_native || false;
      const isCompiled = compileStatus?.is_compiled || false;
      const device = compileStatus?.device_str?.toUpperCase() || 'CPU';
      
      let engineType = 'PYTHON';
      let dotColor = 'bg-teal-400';
      let textColor = 'text-teal-300';
      let pulse = false;
      let shadow = '';
      
      if (isNative) {
        engineType = 'NATIVE';
        dotColor = device === 'CUDA' ? 'bg-teal-500' : 'bg-teal-400';
        textColor = device === 'CUDA' ? 'text-teal-400' : 'text-teal-300';
        pulse = true;
        shadow = 'shadow-glow-teal';
      } else if (isCompiled) {
        engineType = 'COMPILED';
        dotColor = 'bg-teal-500';
        textColor = 'text-teal-400';
        pulse = true;
        shadow = 'shadow-glow-teal';
      } else {
        dotColor = 'bg-teal-400';
        textColor = 'text-teal-300';
      }
      
      return {
        engineText: `Engine::${engineType}`,
        statusText: 'Status::CONNECTED',
        dotColor,
        textColor,
        pulse,
        shadow
      };
    } else if (connectionStatus === 'connecting') {
      return {
        engineText: 'Engine::CONNECTING',
        statusText: 'Status::CONNECTING',
        dotColor: 'bg-amber-400',
        textColor: 'text-amber-400',
        pulse: true,
        shadow: 'shadow-[0_0_8px_rgba(251,191,36,0.4)]'
      };
    } else {
      return {
        engineText: 'Engine::DISCONNECTED',
        statusText: 'Status::DISCONNECTED',
        dotColor: 'bg-gray-500',
        textColor: 'text-gray-500',
        pulse: false,
        shadow: ''
      };
    }
  };
  
  const systemStatus = getSystemStatus();

  // Obtener información del engine y FPS
  const engineInfo = useMemo(() => {
    if (connectionStatus !== 'connected') {
      return {
        type: 'disconnected' as const,
        label: 'DESCONECTADO',
        color: 'text-gray-500',
        bgColor: 'bg-gray-500/10',
        borderColor: 'border-gray-500/20',
        dotColor: 'bg-gray-500',
        device: 'N/A',
        version: null as string | null
      };
    }
    
    const deviceLabel = compileStatus?.device_str?.toUpperCase() || 'N/A';
    
    if (compileStatus?.is_native) {
      return {
        type: 'native' as const,
        label: 'NATIVO (C++)',
        color: 'text-emerald-400',
        bgColor: 'bg-emerald-500/10',
        borderColor: 'border-emerald-500/30',
        dotColor: 'bg-emerald-500',
        device: deviceLabel,
        version: compileStatus?.native_version || compileStatus?.wrapper_version || null
      };
    } else if (compileStatus?.is_compiled) {
      return {
        type: 'compiled' as const,
        label: 'COMPILADO (PyTorch)',
        color: 'text-teal-400',
        bgColor: 'bg-teal-500/10',
        borderColor: 'border-teal-500/30',
        dotColor: 'bg-teal-500',
        device: deviceLabel,
        version: null
      };
    } else {
      return {
        type: 'python' as const,
        label: 'PYTHON',
        color: 'text-amber-400',
        bgColor: 'bg-amber-500/10',
        borderColor: 'border-amber-500/30',
        dotColor: 'bg-amber-500',
        device: deviceLabel,
        version: compileStatus?.python_version || null
      };
    }
  }, [compileStatus, connectionStatus]);

  const fps = simData?.simulation_info?.fps ?? 0;
  
  // Determinar color del FPS según velocidad
  const fpsColor = useMemo(() => {
    if (fps === 0) return 'text-gray-500';
    if (fps < 5) return 'text-red-400';
    if (fps < 15) return 'text-amber-400';
    if (fps < 30) return 'text-yellow-400';
    return 'text-emerald-400';
  }, [fps]);

  // Función para cambiar de motor
  const handleSwitchEngine = useCallback((targetEngine: 'native' | 'python') => {
    console.log(`🔄 handleSwitchEngine llamado con: ${targetEngine}`);
    console.log(`🔍 Estado: connectionStatus=${connectionStatus}, model_name=${compileStatus?.model_name}`);
    
    if (connectionStatus !== 'connected') {
      console.warn('⚠️ No se puede cambiar de motor: no hay conexión');
      setEngineDropdownOpen(false);
      return;
    }
    
    if (!compileStatus?.model_name || compileStatus.model_name === 'None') {
      console.warn('⚠️ No se puede cambiar de motor: no hay modelo cargado');
      setEngineDropdownOpen(false);
      return;
    }
    
    const currentIsNative = compileStatus.is_native || false;
    const targetIsNative = targetEngine === 'native';
    
    // Solo cambiar si es diferente al actual
    if (currentIsNative === targetIsNative) {
      console.info(`ℹ️ Ya estás usando el motor ${targetEngine}`);
      setEngineDropdownOpen(false);
      return;
    }
    
    console.log(`✅ Enviando comando switch_engine: ${currentIsNative ? 'Native' : 'Python'} → ${targetEngine}`);
    try {
      sendCommand('inference', 'switch_engine', { engine: targetEngine });
      console.log(`✅ Comando enviado exitosamente`);
    } catch (error) {
      console.error(`❌ Error enviando comando:`, error);
    }
    setEngineDropdownOpen(false);
  }, [connectionStatus, compileStatus, sendCommand]);

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
              ATHERIA<span className="text-teal-400">_LAB</span>
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
                  backgroundColor: 'rgba(20, 184, 166, 0.6)',
                  boxShadow: '0 0 4px rgba(20, 184, 166, 0.8)'
                }}
              />
              <div className="w-full h-full bg-gradient-to-b from-teal-500/20 via-pink-500/20 to-teal-400/20 rounded-full opacity-0 group-hover:opacity-100 transition-opacity" />
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
        {/* Indicador de Estado del Sistema - Formato: Engine::<engine> Status::CONNECTED */}
        <div className="flex items-center gap-3">
          {/* Badge del Engine (clickeable para cambiar) */}
          <div className="relative">
            <button
              onClick={() => connectionStatus === 'connected' && compileStatus?.model_name && compileStatus.model_name !== 'None' && setEngineDropdownOpen(!engineDropdownOpen)}
              className={`flex items-center gap-2 px-2 py-1 bg-white/5 rounded border ${engineInfo.borderColor} transition-all ${
                connectionStatus === 'connected' && compileStatus?.model_name && compileStatus.model_name !== 'None' 
                  ? 'hover:bg-white/10 cursor-pointer' 
                  : 'cursor-default'
              }`}
              title={connectionStatus === 'connected' && compileStatus?.model_name && compileStatus.model_name !== 'None' ? 'Cambiar motor de simulación' : 'Motor de simulación'}
            >
              <div className={`w-1.5 h-1.5 rounded-full ${engineInfo.dotColor} ${
                connectionStatus === 'connected' && compileStatus?.is_native ? 'animate-pulse' : ''
              }`} />
              <span className={`text-[10px] font-mono ${engineInfo.color} tracking-wide`}>
                {systemStatus.engineText}
              </span>
              {connectionStatus === 'connected' && compileStatus?.model_name && compileStatus.model_name !== 'None' && (
                <ChevronDown size={10} className={`text-gray-500 transition-transform ${engineDropdownOpen ? 'rotate-180' : ''}`} />
              )}
            </button>
          
            {/* Dropdown para cambiar motor */}
            {engineDropdownOpen && connectionStatus === 'connected' && compileStatus?.model_name && compileStatus.model_name !== 'None' && (
              <div className="absolute top-full left-0 mt-1 z-50 min-w-[220px]">
                <div className="bg-[#0a0a0a] border border-white/20 rounded shadow-xl p-2 space-y-1">
                  <div className="text-[10px] font-bold text-gray-400 uppercase tracking-wider px-2 py-1">
                    Cambiar Motor
                  </div>
                  
                  {/* Opción: Motor Nativo (C++) */}
                  {(!compileStatus?.is_native) && (
                    <button
                      onClick={() => handleSwitchEngine('native')}
                      className="w-full flex items-center justify-between px-3 py-2 bg-teal-500/10 hover:bg-teal-500/20 border border-teal-500/30 rounded text-[10px] text-teal-400 transition-all"
                    >
                      <div className="flex items-center gap-2">
                        <Cpu size={12} className="text-teal-400" />
                        <span className="font-mono">Native (C++)</span>
                      </div>
                      {compileStatus?.native_version && (
                        <span className="text-[9px] text-gray-500 font-mono">v{compileStatus.native_version}</span>
                      )}
                    </button>
                  )}
                  
                  {/* Opción: Motor Python */}
                  {(compileStatus?.is_native) && (
                    <button
                      onClick={() => handleSwitchEngine('python')}
                      className="w-full flex items-center justify-between px-3 py-2 bg-teal-500/10 hover:bg-teal-500/20 border border-teal-500/30 rounded text-[10px] text-teal-400 transition-all"
                    >
                      <div className="flex items-center gap-2">
                        <Cpu size={12} className="text-teal-400" />
                        <span className="font-mono">Python</span>
                      </div>
                      {compileStatus?.python_version && (
                        <span className="text-[9px] text-gray-500 font-mono">v{compileStatus.python_version}</span>
                      )}
                    </button>
                  )}
                  
                  {/* Información de versión y device */}
                  <div className="px-3 py-2 text-[9px] text-gray-500 font-mono border-t border-white/10 pt-2 space-y-0.5">
                    {engineInfo.version && (
                      <div>Versión: {engineInfo.version}</div>
                    )}
                    {engineInfo.device && engineInfo.device !== 'N/A' && (
                      <div>Device: {engineInfo.device}</div>
                    )}
                  </div>
                </div>
              </div>
            )}
          </div>
          
          {/* Status Badge */}
          <div className="flex items-center gap-2 px-2 py-1 bg-white/5 rounded border border-white/5">
            <div className={`w-1.5 h-1.5 rounded-full ${systemStatus.dotColor} ${
              systemStatus.pulse ? `animate-pulse ${systemStatus.shadow}` : ''
            }`} />
            <span className={`text-[10px] font-mono ${systemStatus.textColor} tracking-wide`}>
              {systemStatus.statusText}
            </span>
          </div>
        </div>
        
        {/* FPS y Velocidad */}
        {connectionStatus === 'connected' && (
          <div className="flex items-center gap-2 px-2 py-1 bg-white/5 rounded border border-white/5">
            <Gauge size={12} className="text-gray-400" />
            <span className={`text-[10px] font-mono ${fpsColor} font-semibold`}>
              {fps > 0 ? `${fps.toFixed(1)} FPS` : 'N/A'}
            </span>
          </div>
        )}
        
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
      
      {/* Cerrar dropdown al hacer click fuera */}
      {engineDropdownOpen && (
        <div 
          className="fixed inset-0 z-40" 
          onClick={() => setEngineDropdownOpen(false)}
        />
      )}
    </header>
  );
};
