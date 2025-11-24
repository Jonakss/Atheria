import React, { useState, useMemo, useEffect } from 'react';
import { ScientificHeader } from '../components/ScientificHeader';
import { NavigationSidebar } from '../components/NavigationSidebar';
import { PhysicsInspector } from '../components/PhysicsInspector';
import { MetricsBar } from '../components/MetricsBar';
import { Toolbar } from '../components/Toolbar';
import { AnalysisView } from '../components/AnalysisView';
import { HistoryView } from '../components/HistoryView';
import { LogsView } from '../components/LogsView';
import { PanZoomCanvas } from '../../../components/ui/PanZoomCanvas';
import { TimelineViewer } from '../../../components/ui/TimelineViewer';
import HolographicViewer from '../../../components/visualization/HolographicViewer';
import { LabSider } from '../../../components/ui/LabSider';
import { useWebSocket } from '../../../hooks/useWebSocket';
import { EPOCH_CONFIGS } from '../components/EpochBadge';

type TabType = 'lab' | 'analysis' | 'history' | 'logs';
type LabSection = 'inference' | 'training' | 'analysis';

export const DashboardLayout: React.FC = () => {
  const [activeTab, setActiveTab] = useState<TabType>('lab');
  const [labPanelOpen, setLabPanelOpen] = useState(true); // Panel de laboratorio visible por defecto
  const [activeLabSection, setActiveLabSection] = useState<LabSection>('inference'); // Sub-sección activa de Lab
  const [physicsInspectorCollapsed, setPhysicsInspectorCollapsed] = useState(false); // Inspector físico colapsado
  const [timelineOpen, setTimelineOpen] = useState(false); // Timeline viewer abierto/cerrado
  const [selectedTimelineFrame, setSelectedTimelineFrame] = useState<{
    step: number;
    timestamp: string;
    map_data: number[][];
  } | null>(null);
  const { simData, selectedViz, connectionStatus, sendCommand, setSelectedViz, inferenceStatus } = useWebSocket();
  
  // Si la simulación está corriendo, deseleccionar frame del timeline para mostrar datos en vivo
  useEffect(() => {
    if (inferenceStatus === 'running' && selectedTimelineFrame !== null) {
      setSelectedTimelineFrame(null);
    }
  }, [inferenceStatus, selectedTimelineFrame]);
  
  // Obtener época detectada del backend
  const detectedEpoch = useMemo(() => simData?.simulation_info?.epoch ?? 2, [simData?.simulation_info?.epoch]);
  const [currentEpoch, setCurrentEpoch] = useState(detectedEpoch);
  
  // Sincronizar época cuando cambie desde el backend
  useEffect(() => {
    if (detectedEpoch !== undefined && detectedEpoch !== currentEpoch) {
      setCurrentEpoch(detectedEpoch);
    }
  }, [detectedEpoch, currentEpoch]);
  
  // Handler para cambio de época - aplicar configuración automáticamente
  const handleEpochChange = (epoch: number) => {
    setCurrentEpoch(epoch);
    const config = EPOCH_CONFIGS[epoch];
    
    if (config && connectionStatus === 'connected') {
      // Cambiar gamma decay
      sendCommand('inference', 'set_config', {
        gamma_decay: config.gammaDecay
      });
      
      // Cambiar tipo de visualización
      if (config.vizType && config.vizType !== selectedViz) {
        setSelectedViz(config.vizType);
        sendCommand('simulation', 'set_viz', { viz_type: config.vizType });
      }
    }
  };

  // Obtener datos del mapa para renderizado
  const mapData = simData?.map_data;
  const gridWidth = mapData?.[0]?.length || 0;
  const gridHeight = mapData?.length || 0;

  // Convertir map_data a array plano para HolographicViewer si es necesario
  const flatMapData = useMemo(() => {
    if (!mapData || mapData.length === 0) return [];
    
    const flat: number[] = [];
    for (let y = 0; y < mapData.length; y++) {
      const row = mapData[y];
      if (Array.isArray(row)) {
        for (let x = 0; x < row.length; x++) {
          const val = row[x];
          if (typeof val === 'number' && !isNaN(val)) {
            flat.push(val);
          } else {
            flat.push(0);
          }
        }
      }
    }
    return flat;
  }, [mapData]);

  // Determinar qué visualización mostrar según el tab activo
  const renderContentView = () => {
    switch (activeTab) {
      case 'lab':
        // Vista principal: 2D canvas o 3D holographic según selectedViz
        if (selectedViz === 'holographic' || selectedViz === '3d') {
          return (
            <div className="absolute inset-0 z-0 bg-gradient-deep-space overflow-hidden">
              {flatMapData.length > 0 && gridWidth > 0 && gridHeight > 0 ? (
                <HolographicViewer 
                  data={flatMapData}
                  width={gridWidth}
                  height={gridHeight}
                />
              ) : (
                <div className="flex items-center justify-center h-full text-dark-300 text-sm">
                  Esperando datos de simulación...
                </div>
              )}
              {/* Design System: text-[9px] font-mono text-gray-700 según mockup */}
              <div className="absolute bottom-4 right-4 text-[9px] font-mono text-dark-500 pointer-events-none text-right">
                VIEWPORT: ORTHOGRAPHIC<br/>
                RENDER: WEBGL2 / HIGH_PRECISION
              </div>
            </div>
          );
        } else {
          // Vista 2D con PanZoomCanvas
          return (
            <div className="absolute inset-0 z-0 bg-gradient-deep-space overflow-hidden">
              {connectionStatus === 'connected' ? (
                <>
                  <PanZoomCanvas historyFrame={selectedTimelineFrame} />
                  {/* Timeline Viewer - Panel flotante */}
                  {timelineOpen && (
                    <div className="absolute bottom-20 left-4 z-50 w-80 max-h-96 overflow-y-auto">
                      <TimelineViewer
                        onFrameSelect={(frame) => {
                          if (frame) {
                            setSelectedTimelineFrame({
                              step: frame.step,
                              timestamp: new Date(frame.timestamp).toISOString(),
                              map_data: frame.map_data,
                            });
                          } else {
                            setSelectedTimelineFrame(null);
                          }
                        }}
                      />
                    </div>
                  )}
                </>
              ) : (
                <div className="flex items-center justify-center h-full text-gray-300 text-sm">
                  Conectando al servidor...
                </div>
              )}
            </div>
          );
        }
      
      case 'analysis':
        return <AnalysisView />;
      
      case 'history':
        return <HistoryView />;
      
      case 'logs':
        return <LogsView />;
      
      default:
        return null;
    }
  };

  return (
    <div className="h-screen bg-dark-bg text-dark-200 font-sans selection:bg-teal-500/30 overflow-hidden flex flex-col">
      
              {/* Header: Barra de Comando Técnica */}
              <ScientificHeader currentEpoch={currentEpoch} onEpochChange={handleEpochChange} />

      {/* Contenedor Principal */}
      <div className="flex-1 flex overflow-hidden">
        
        {/* Sidebar de Navegación (Iconos Minimalistas) - Siempre visible */}
        <NavigationSidebar 
          activeTab={activeTab} 
          onTabChange={(tab) => {
            setActiveTab(tab);
            // Si se selecciona 'lab', abrir/cerrar panel de laboratorio si ya estaba activo, o abrirlo si no
            if (tab === 'lab') {
               // Si ya estábamos en lab, toggles. Si no, lo abre.
               if (activeTab === 'lab') {
                   setLabPanelOpen(!labPanelOpen);
               } else {
                   setLabPanelOpen(true);
               }
            } else {
              setLabPanelOpen(false);
            }
          }}
          labPanelOpen={labPanelOpen}
          activeLabSection={activeLabSection}
          onLabSectionChange={(section) => {
              setActiveLabSection(section);
              if (!labPanelOpen) setLabPanelOpen(true);
          }}
        />

        {/* Panel de Laboratorio (Experimentos/Entrenamiento) - Controlado por NavigationSidebar */}
        {labPanelOpen && activeTab === 'lab' && (
          <aside className="flex-col border-r border-white/5 bg-dark-980/40 backdrop-blur-md z-40 shrink-0 transition-all duration-300 w-[380px] flex overflow-hidden relative">
            <LabSider 
              activeSection={activeLabSection} 
              onClose={() => setLabPanelOpen(false)}
            />
          </aside>
        )}

        {/* Área de Trabajo (Viewport + Paneles Flotantes) */}
        {/* Design System: flex-1 (Ocupa todo el espacio restante), relative para overlays */}
        <main className="flex-1 relative bg-dark-990 flex flex-col overflow-hidden">
          
          {/* Barra de Herramientas Superior (Flotante) */}
          <Toolbar onToggleTimeline={() => setTimelineOpen(prev => !prev)} timelineOpen={timelineOpen} />

          {/* Viewport (Fondo) - Design System: bg-[#050505] a black según mockup */}
          {renderContentView()}

          {/* Panel Inferior (Métricas Críticas) - Design System: bg-[#050505]/95 backdrop-blur-sm */}
          <MetricsBar />
        </main>

        {/* Panel Lateral Derecho (Inspector y Controles) - Colapsible */}
        {/* Design System: w-72 (288px) o w-80 (320px) - usando w-72 según mockup */}
        <PhysicsInspector 
          isCollapsed={physicsInspectorCollapsed}
          onToggleCollapse={() => setPhysicsInspectorCollapsed(!physicsInspectorCollapsed)}
        />
      </div>
    </div>
  );
};
