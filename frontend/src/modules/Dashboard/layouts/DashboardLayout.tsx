import React, { useEffect, useMemo, useState } from 'react';
import { ColorScaleLegend } from '../../../components/ui/ColorScaleLegend';
import { LabSider } from '../../../components/ui/LabSider';
import { PanZoomCanvas } from '../../../components/ui/PanZoomCanvas';
import { TimelineViewer } from '../../../components/ui/TimelineViewer';
import HolographicViewer from '../../../components/visualization/HolographicViewer';
import HolographicViewer2 from '../../../components/visualization/HolographicViewer2';
import { useWebSocket } from '../../../hooks/useWebSocket';
import { AnalysisView } from '../components/AnalysisView';
import { HistoryView } from '../components/HistoryView';
import { LogsView } from '../components/LogsView';
import { MetricsBar } from '../components/MetricsBar';
import { NavigationSidebar } from '../components/NavigationSidebar';
import { PhysicsInspector } from '../components/PhysicsInspector';
import { ScientificHeader } from '../components/ScientificHeader';
import { TrainingView } from '../components/TrainingView';
import { VisualizationPanel } from '../components/VisualizationPanel';

type TabType = 'lab' | 'analysis' | 'history' | 'logs';
type LabSection = 'inference' | 'training' | 'analysis';

// Helper to get legend props based on layer
const getLegendProps = (layer: number) => {
  switch (layer) {
    case 0: return { mode: 'DENSITY', min: 0, max: 1.0, gradient: 'linear-gradient(to top, #000, #00f, #0ff, #fff)' };
    case 1: return { mode: 'PHASE', min: -3.14, max: 3.14, gradient: 'linear-gradient(to top, #f00, #ff0, #0f0, #0ff, #00f, #f0f, #f00)' };
    case 2: return { mode: 'ENERGY', min: 0, max: 1.0, gradient: 'linear-gradient(to top, #000, #f00, #ff0, #fff)' };
    case 3: return { mode: 'FLOW', min: 0, max: 1.0, gradient: 'linear-gradient(to top, #000, #0ff, #fff)' };
    case 4: return { mode: 'CHUNKS', min: 0, max: 1, gradient: 'linear-gradient(to top, #000, #0f0, #fff)' };
    default: return { mode: 'UNKNOWN', min: 0, max: 1, gradient: 'linear-gradient(to top, #000, #fff)' };
  }
};

export const DashboardLayout: React.FC = () => {
  const [activeTab, setActiveTab] = useState<TabType>('lab');
  const [labPanelOpen, setLabPanelOpen] = useState(true);
  const [activeLabSection, setActiveLabSection] = useState<LabSection>('inference');
  const [physicsInspectorCollapsed, setPhysicsInspectorCollapsed] = useState(false);
  const [visualizationPanelCollapsed, setVisualizationPanelCollapsed] = useState(false);
  const [viewerVersion, setViewerVersion] = useState<'v1' | 'v2'>('v1');
  const [selectedLayer, setSelectedLayer] = useState(0); // 0: Density, 1: Phase, 2: Energy, 3: Flow
  const [selectedTimelineFrame, setSelectedTimelineFrame] = useState<{
    step: number;
    timestamp: string;
    map_data: number[][];
  } | null>(null);
  const { simData, selectedViz, connectionStatus, inferenceStatus } = useWebSocket();

  useEffect(() => {
    if (inferenceStatus === 'running' && selectedTimelineFrame !== null) {
      setSelectedTimelineFrame(null);
    }
  }, [inferenceStatus, selectedTimelineFrame]);


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
        // Si estamos en la sección de entrenamiento, mostrar la vista de entrenamiento
        if (activeLabSection === 'training') {
          return <TrainingView />;
        }

        // Vista principal: 2D canvas o 3D holographic según selectedViz
        if (selectedViz === 'holographic' || selectedViz === '3d') {
          return (
            <div className="absolute inset-0 z-0 bg-gradient-deep-space overflow-hidden">
              {flatMapData.length > 0 && gridWidth > 0 && gridHeight > 0 ? (
                viewerVersion === 'v1' ? (
                  <HolographicViewer 
                    data={flatMapData}
                    width={gridWidth}
                    height={gridHeight}
                    vizType={selectedViz}
                  />
                ) : (
                  <HolographicViewer2 
                    data={flatMapData}
                    width={gridWidth}
                    height={gridHeight}
                    vizType={selectedViz}
                  />
                )
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
                  {/* TEMPORALMENTE DESHABILITADO - Revisar performance issues */}
                  {false && (
                      <TimelineViewer
                        className="absolute bottom-0 left-0 right-0 z-30"
                        onFrameSelect={(frame) => {
                          if (frame) {
                            setSelectedTimelineFrame({
                              ...frame,
                              timestamp: String(frame.timestamp),
                            });
                          } else {
                            setSelectedTimelineFrame(null);
                          }
                        }}
                      />
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
              <ScientificHeader />

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
          
          {/* Viewport (Fondo) - Design System: bg-[#050505] a black según mockup */}
          {renderContentView()}
          
          {/* Color Scale Legend - Only visible in holographic/3d modes */}
          {(selectedViz === 'holographic' || selectedViz === '3d') && (
            <ColorScaleLegend 
              mode={getLegendProps(selectedLayer).mode}
              minValue={getLegendProps(selectedLayer).min}
              maxValue={getLegendProps(selectedLayer).max}
              gradient={getLegendProps(selectedLayer).gradient}
            />
          )}

          {/* Panel Inferior (Controles + Logs) - Design System: bg-[#050505]/95 backdrop-blur-sm */}
          <MetricsBar />
        </main>

        {/* Panel Lateral Derecho (Inspector y Controles) - Colapsible */}
        {/* Design System: w-72 (288px) o w-80 (320px) - usando w-72 según mockup */}
        {/* Panel Lateral Derecho (Inspector y Controles) - Colapsible */}
        {/* Design System: w-72 (288px) o w-80 (320px) - usando w-72 según mockup */}
        {/* Panel Lateral Derecho (Visualización) - Colapsible */}
        <VisualizationPanel 
          isCollapsed={visualizationPanelCollapsed}
          onToggleCollapse={() => setVisualizationPanelCollapsed(!visualizationPanelCollapsed)}
          viewerVersion={viewerVersion}
          onViewerVersionChange={setViewerVersion}
          selectedLayer={selectedLayer}
          onLayerChange={setSelectedLayer}
        />

        {/* Panel Lateral Derecho (Inspector de Física) - Colapsible */}
        <PhysicsInspector 
          isCollapsed={physicsInspectorCollapsed}
          onToggleCollapse={() => setPhysicsInspectorCollapsed(!physicsInspectorCollapsed)}
        />
      </div>
    </div>
  );
};
