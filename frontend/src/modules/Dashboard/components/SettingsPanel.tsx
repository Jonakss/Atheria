// frontend/src/modules/Dashboard/components/SettingsPanel.tsx
import React, { useState } from 'react';
import { X, Save, RotateCcw, Power } from 'lucide-react';
import { useWebSocket } from '../../../hooks/useWebSocket';
import { GlassPanel } from './GlassPanel';

interface SettingsPanelProps {
  isOpen: boolean;
  onClose: () => void;
}

export const SettingsPanel: React.FC<SettingsPanelProps> = ({ isOpen, onClose }) => {
  const { sendCommand, connectionStatus, compileStatus, serverConfig, updateServerConfig, connect, activeExperiment, experimentsData, simData } = useWebSocket();
  const isConnected = connectionStatus === 'connected';
  
  // Estados para configuraciones del servidor
  const [serverHost, setServerHost] = useState(serverConfig?.host || 'localhost');
  const [serverPort, setServerPort] = useState(serverConfig?.port || 8000);
  const [serverProtocol, setServerProtocol] = useState<'ws' | 'wss'>(serverConfig?.protocol || 'ws');
  const [serverPath, setServerPath] = useState(serverConfig?.path || '/ws');
  const [serverConfigChanged, setServerConfigChanged] = useState(false);
  
  // Estados para configuraciones globales
  const [compressionEnabled, setCompressionEnabled] = useState(true);
  const [roiEnabled, setRoiEnabled] = useState(true);
  const [autoRotate, setAutoRotate] = useState(false);
  const [frameRate, setFrameRate] = useState(30);
  const [maxFramesPerSecond, setMaxFramesPerSecond] = useState(60);
  const [logLevel, setLogLevel] = useState<'info' | 'warning' | 'error'>('info');
  
  // Obtener información del experimento activo (solo para mostrar)
  const currentExperiment = activeExperiment 
    ? experimentsData?.find(exp => exp.name === activeExperiment) 
    : null;
  
  // Sincronizar estados del servidor cuando serverConfig cambia
  React.useEffect(() => {
    if (serverConfig) {
      setServerHost(serverConfig.host);
      setServerPort(serverConfig.port);
      setServerProtocol(serverConfig.protocol);
      setServerPath(serverConfig.path || '/ws');
      setServerConfigChanged(false);
    }
  }, [serverConfig]);
  
  // Detectar cambios en la configuración del servidor
  React.useEffect(() => {
    const changed = serverConfig && (
      serverHost !== serverConfig.host ||
      serverPort !== serverConfig.port ||
      serverProtocol !== serverConfig.protocol ||
      serverPath !== (serverConfig.path || '/ws')
    );
    setServerConfigChanged(changed || false);
  }, [serverHost, serverPort, serverProtocol, serverPath, serverConfig]);
  
  
  const handleSaveSettings = () => {
    if (!isConnected) {
      alert('⚠️ No hay conexión con el servidor.');
      return;
    }
    
    // Enviar configuraciones al backend
    sendCommand('simulation', 'set_global_config', {
      compression_enabled: compressionEnabled,
      roi_enabled: roiEnabled,
      auto_rotate: autoRotate,
      frame_rate: frameRate,
      max_frames_per_second: maxFramesPerSecond,
      log_level: logLevel
    });
    
    // Guardar en localStorage para persistencia
    localStorage.setItem('atheria_global_config', JSON.stringify({
      compressionEnabled,
      roiEnabled,
      autoRotate,
      frameRate,
      maxFramesPerSecond,
      logLevel
    }));
    
    alert('✅ Configuraciones guardadas correctamente.');
  };
  
  const handleSaveServerConfig = () => {
    updateServerConfig({
      host: serverHost,
      port: serverPort,
      protocol: serverProtocol,
      path: serverPath
    });
    
    // Reconectar con la nueva configuración
    setTimeout(() => {
      connect();
      alert('✅ Configuración del servidor guardada. Reconectando...');
    }, 100);
  };
  
  const handleResetSettings = () => {
    if (!confirm('¿Restablecer todas las configuraciones a sus valores por defecto?')) {
      return;
    }
    
    setCompressionEnabled(true);
    setRoiEnabled(true);
    setAutoRotate(false);
    setFrameRate(30);
    setMaxFramesPerSecond(60);
    setLogLevel('info');
    // Resetear configuración del servidor
    if (updateServerConfig) {
      updateServerConfig({
        host: 'localhost',
        port: 8000,
        protocol: 'ws',
        path: '/ws'
      });
    }
    
    localStorage.removeItem('atheria_global_config');
  };
  
  const handleShutdown = () => {
    if (!isConnected) {
      alert('⚠️ No hay conexión con el servidor.');
      return;
    }
    
    if (!confirm('⚠️ ¿Estás seguro de que quieres apagar el servidor?\n\nEsto cerrará todas las conexiones y detendrá la simulación.')) {
      return;
    }
    
    // Enviar comando de shutdown con confirmación
    sendCommand('server', 'shutdown', { confirm: true });
  };
  
  // Cargar configuraciones desde localStorage al montar
  React.useEffect(() => {
    const saved = localStorage.getItem('atheria_global_config');
    if (saved) {
      try {
        const config = JSON.parse(saved);
        setCompressionEnabled(config.compressionEnabled ?? true);
        setRoiEnabled(config.roiEnabled ?? true);
        setAutoRotate(config.autoRotate ?? false);
        setFrameRate(config.frameRate ?? 30);
        setMaxFramesPerSecond(config.maxFramesPerSecond ?? 60);
        setLogLevel(config.logLevel ?? 'info');
      } catch (e) {
        console.error('Error cargando configuraciones:', e);
      }
    }
  }, []);

  if (!isOpen) return null;

  return (
    <div className="fixed inset-0 z-[200] flex items-center justify-center pointer-events-none">
      {/* Backdrop */}
      <div 
        className="absolute inset-0 bg-black/80 backdrop-blur-sm pointer-events-auto"
        onClick={onClose}
      />
      
      {/* Panel de Configuración */}
      <GlassPanel className="relative w-[600px] max-h-[80vh] bg-[#0a0a0a] border-white/20 pointer-events-auto flex flex-col overflow-hidden">
        {/* Header */}
        <div className="flex items-center justify-between p-4 border-b border-white/10 shrink-0">
          <h2 className="text-sm font-bold text-gray-200 uppercase tracking-wider">Configuración Global</h2>
          <button
            onClick={onClose}
            className="p-1.5 text-gray-400 hover:text-gray-200 transition-colors rounded hover:bg-white/5"
          >
            <X size={18} />
          </button>
        </div>
        
        {/* Content */}
        <div className="flex-1 overflow-y-auto custom-scrollbar p-4 space-y-6">
          {/* Sección: Configuración del Servidor */}
          <div className="space-y-3">
            <div className="text-[10px] font-bold text-gray-400 uppercase tracking-wider">Conexión del Servidor</div>
            
            <div className="space-y-3">
              <div className="space-y-1.5">
                <label className="block text-xs text-gray-300 font-medium">
                  Host del Servidor
                </label>
                <input
                  type="text"
                  value={serverHost}
                  onChange={(e) => {
                    setServerHost(e.target.value);
                  }}
                  disabled={isConnected}
                  placeholder="localhost"
                  className="w-full px-3 py-1.5 bg-white/5 border border-white/10 rounded text-xs text-gray-300 focus:outline-none focus:border-blue-500/50 disabled:opacity-50 disabled:cursor-not-allowed font-mono"
                />
                <div className="text-[10px] text-gray-600">Dirección IP o dominio del servidor</div>
              </div>
              
              <div className="grid grid-cols-2 gap-3">
                <div className="space-y-1.5">
                  <label className="block text-xs text-gray-300 font-medium">
                    Puerto
                  </label>
                  <input
                    type="number"
                    value={serverPort}
                    onChange={(e) => {
                      const port = Number(e.target.value);
                      if (port > 0 && port <= 65535) {
                        setServerPort(port);
                      }
                    }}
                    disabled={isConnected}
                    min={1}
                    max={65535}
                    placeholder="8000"
                    className="w-full px-3 py-1.5 bg-white/5 border border-white/10 rounded text-xs text-gray-300 focus:outline-none focus:border-blue-500/50 disabled:opacity-50 disabled:cursor-not-allowed font-mono"
                  />
                </div>
                
                <div className="space-y-1.5">
                  <label className="block text-xs text-gray-300 font-medium">
                    Protocolo
                  </label>
                  <select
                    value={serverProtocol}
                    onChange={(e) => setServerProtocol(e.target.value as 'ws' | 'wss')}
                    disabled={isConnected}
                    className="w-full px-3 py-1.5 bg-white/5 border border-white/10 rounded text-xs text-gray-300 focus:outline-none focus:border-blue-500/50 disabled:opacity-50 disabled:cursor-not-allowed"
                  >
                    <option value="ws">WS (WebSocket)</option>
                    <option value="wss">WSS (Secure WebSocket)</option>
                  </select>
                </div>
              </div>
              
              <div className="space-y-1.5">
                <label className="block text-xs text-gray-300 font-medium">
                  Ruta WebSocket
                </label>
                <input
                  type="text"
                  value={serverPath}
                  onChange={(e) => setServerPath(e.target.value)}
                  disabled={isConnected}
                  placeholder="/ws"
                  className="w-full px-3 py-1.5 bg-white/5 border border-white/10 rounded text-xs text-gray-300 focus:outline-none focus:border-blue-500/50 disabled:opacity-50 disabled:cursor-not-allowed font-mono"
                />
                <div className="text-[10px] text-gray-600">Ruta del endpoint WebSocket (ej: /ws)</div>
              </div>
              
              {serverConfigChanged && !isConnected && (
                <button
                  onClick={handleSaveServerConfig}
                  className="w-full px-3 py-2 bg-blue-500/10 hover:bg-blue-500/20 border border-blue-500/30 text-blue-300 text-xs font-bold rounded transition-all flex items-center justify-center gap-2"
                >
                  <Save size={14} />
                  Guardar y Reconectar
                </button>
              )}
              
              <div className="p-2 bg-white/5 rounded border border-white/10">
                <div className="text-[10px] text-gray-500 font-mono">
                  URL: {serverProtocol}://{serverHost}:{serverPort}{serverPath}
                </div>
              </div>
            </div>
          </div>
          
          {/* Separador */}
          <div className="border-t border-white/5" />
          
          {/* Sección: Rendimiento */}
          <div className="space-y-3">
            <div className="text-[10px] font-bold text-gray-400 uppercase tracking-wider">Rendimiento</div>
            
            <div className="space-y-2">
              <label className="flex items-center justify-between cursor-pointer group">
                <div className="flex flex-col">
                  <span className="text-xs text-gray-300 font-medium">Compresión de Datos</span>
                  <span className="text-[10px] text-gray-600">Habilitar compresión LZ4 para transferencia WebSocket</span>
                </div>
                <input
                  type="checkbox"
                  checked={compressionEnabled}
                  onChange={(e) => setCompressionEnabled(e.target.checked)}
                  disabled={!isConnected}
                  className="w-4 h-4 rounded border-white/20 bg-white/5 checked:bg-blue-500 checked:border-blue-500 focus:ring-2 focus:ring-blue-500/50 disabled:opacity-50 cursor-pointer"
                />
              </label>
              
              <label className="flex items-center justify-between cursor-pointer group">
                <div className="flex flex-col">
                  <span className="text-xs text-gray-300 font-medium">Region of Interest (ROI)</span>
                  <span className="text-[10px] text-gray-600">Enviar solo la región visible para optimizar ancho de banda</span>
                </div>
                <input
                  type="checkbox"
                  checked={roiEnabled}
                  onChange={(e) => setRoiEnabled(e.target.checked)}
                  disabled={!isConnected}
                  className="w-4 h-4 rounded border-white/20 bg-white/5 checked:bg-blue-500 checked:border-blue-500 focus:ring-2 focus:ring-blue-500/50 disabled:opacity-50 cursor-pointer"
                />
              </label>
              
              <div className="space-y-1.5">
                <label className="block text-xs text-gray-300 font-medium">
                  Frame Rate (FPS)
                </label>
                <input
                  type="number"
                  value={frameRate}
                  onChange={(e) => setFrameRate(Math.max(1, Math.min(120, Number(e.target.value) || 30)))}
                  min={1}
                  max={120}
                  disabled={!isConnected}
                  className="w-full px-3 py-1.5 bg-white/5 border border-white/10 rounded text-xs text-gray-300 focus:outline-none focus:border-blue-500/50 disabled:opacity-50 font-mono"
                />
                <div className="text-[10px] text-gray-600">Frames por segundo objetivo para visualización</div>
              </div>
              
              <div className="space-y-1.5">
                <label className="block text-xs text-gray-300 font-medium">
                  Máximo FPS
                </label>
                <input
                  type="number"
                  value={maxFramesPerSecond}
                  onChange={(e) => setMaxFramesPerSecond(Math.max(1, Math.min(240, Number(e.target.value) || 60)))}
                  min={1}
                  max={240}
                  disabled={!isConnected}
                  className="w-full px-3 py-1.5 bg-white/5 border border-white/10 rounded text-xs text-gray-300 focus:outline-none focus:border-blue-500/50 disabled:opacity-50 font-mono"
                />
                <div className="text-[10px] text-gray-600">Límite máximo de frames por segundo</div>
              </div>
            </div>
          </div>
          
          
          {/* Sección: Visualización */}
          <div className="space-y-3 pt-4 border-t border-white/5">
            <div className="text-[10px] font-bold text-gray-400 uppercase tracking-wider">Visualización</div>
            
            <label className="flex items-center justify-between cursor-pointer group">
              <div className="flex flex-col">
                <span className="text-xs text-gray-300 font-medium">Auto-Rotar Vista 3D</span>
                <span className="text-[10px] text-gray-600">Rotación automática en HolographicViewer</span>
              </div>
              <input
                type="checkbox"
                checked={autoRotate}
                onChange={(e) => setAutoRotate(e.target.checked)}
                className="w-4 h-4 rounded border-white/20 bg-white/5 checked:bg-blue-500 checked:border-blue-500 focus:ring-2 focus:ring-blue-500/50 cursor-pointer"
              />
            </label>
          </div>
          
          {/* Sección: Logging */}
          <div className="space-y-3 pt-4 border-t border-white/5">
            <div className="text-[10px] font-bold text-gray-400 uppercase tracking-wider">Logging</div>
            
            <div className="space-y-1.5">
              <label className="block text-xs text-gray-300 font-medium">
                Nivel de Log
              </label>
              <select
                value={logLevel}
                onChange={(e) => setLogLevel(e.target.value as 'info' | 'warning' | 'error')}
                disabled={!isConnected}
                className="w-full px-3 py-1.5 bg-white/5 border border-white/10 rounded text-xs text-gray-300 focus:outline-none focus:border-blue-500/50 disabled:opacity-50"
              >
                <option value="info">Info (Todos los mensajes)</option>
                <option value="warning">Warning (Solo advertencias y errores)</option>
                <option value="error">Error (Solo errores)</option>
              </select>
              <div className="text-[10px] text-gray-600">Nivel de verbosidad para los logs del sistema</div>
            </div>
          </div>
          
          {/* Sección: Sistema */}
          <div className="space-y-3 pt-4 border-t border-white/5">
            <div className="text-[10px] font-bold text-gray-400 uppercase tracking-wider">Sistema</div>
            
            <div className="p-3 bg-white/5 rounded border border-white/10 space-y-2">
              <div className="flex items-center justify-between text-xs">
                <span className="text-gray-400">Estado de Conexión</span>
                <span className={`font-mono font-medium ${
                  isConnected ? 'text-teal-400' : 'text-gray-600'
                }`}>
                  {connectionStatus === 'connected' ? 'CONECTADO' : connectionStatus === 'connecting' ? 'CONECTANDO...' : 'DESCONECTADO'}
                </span>
              </div>
              
              {compileStatus && (
                <>
                  <div className="flex items-center justify-between text-xs">
                    <span className="text-gray-400">Motor</span>
                    <span className={`font-mono font-medium ${
                      compileStatus.is_native ? 'text-teal-400' : 
                      compileStatus.is_compiled ? 'text-blue-400' : 'text-gray-600'
                    }`}>
                      {compileStatus.is_native ? 'NATIVO (C++)' : 
                       compileStatus.is_compiled ? 'COMPILADO (PyTorch)' : 'PYTHON'}
                    </span>
                  </div>
                </>
              )}
            </div>
          </div>
          
          {/* Sección: Control del Servidor */}
          <div className="space-y-3 pt-4 border-t border-white/5">
            <div className="text-[10px] font-bold text-gray-400 uppercase tracking-wider">Control del Servidor</div>
            
            <div className="p-3 bg-red-500/5 border border-red-500/20 rounded">
              <div className="text-xs text-gray-300 mb-2">
                Apagar el servidor cerrará todas las conexiones y detendrá la simulación.
              </div>
              <button
                onClick={handleShutdown}
                disabled={!isConnected}
                className="w-full px-3 py-2 bg-red-500/10 hover:bg-red-500/20 border border-red-500/30 text-red-300 text-xs font-bold rounded transition-all flex items-center justify-center gap-2 disabled:opacity-50 disabled:cursor-not-allowed"
              >
                <Power size={14} />
                Apagar Servidor
              </button>
            </div>
          </div>
        </div>
        
        {/* Footer */}
        <div className="flex items-center justify-between p-4 border-t border-white/10 shrink-0 gap-3">
          <button
            onClick={handleResetSettings}
            disabled={!isConnected}
            className="flex items-center gap-2 px-3 py-2 bg-white/5 hover:bg-white/10 border border-white/10 text-gray-300 text-xs font-bold rounded transition-all disabled:opacity-50 disabled:cursor-not-allowed"
          >
            <RotateCcw size={14} />
            Restablecer
          </button>
          <div className="flex gap-2">
            <button
              onClick={onClose}
              className="px-4 py-2 bg-white/5 hover:bg-white/10 border border-white/10 text-gray-300 text-xs font-bold rounded transition-all"
            >
              Cancelar
            </button>
            <button
              onClick={handleSaveSettings}
              disabled={!isConnected}
              className="flex items-center gap-2 px-4 py-2 bg-blue-500/10 hover:bg-blue-500/20 border border-blue-500/30 text-blue-300 text-xs font-bold rounded transition-all disabled:opacity-50 disabled:cursor-not-allowed"
            >
              <Save size={14} />
              Guardar
            </button>
          </div>
        </div>
      </GlassPanel>
    </div>
  );
};

