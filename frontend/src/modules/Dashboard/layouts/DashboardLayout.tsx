import { Minimize } from 'lucide-react';
import React, { useEffect, useMemo, useState } from 'react';
import { ColorScaleLegend } from '../../../components/ui/ColorScaleLegend';
import { LabSider } from '../../../components/ui/LabSider';
import { PanZoomCanvas } from '../../../components/ui/PanZoomCanvas';
import HolographicViewer from '../../../components/visualization/HolographicViewer';
import HolographicViewer2 from '../../../components/visualization/HolographicViewer2';
import { useWebSocket } from '../../../hooks/useWebSocket';
import PhaseSpaceViewer from '../../PhaseSpaceViewer/PhaseSpaceViewer';
import { AnalysisView } from '../components/AnalysisView';
import { HistoryView } from '../components/HistoryView';
import { LogsView } from '../components/LogsView';
import { MetricsBar } from '../components/MetricsBar';
import { NavigationSidebar } from '../components/NavigationSidebar';
import { PhysicsSection } from '../components/PhysicsSection';
import { RightDrawer } from '../components/RightDrawer';
import { ScientificHeader } from '../components/ScientificHeader';
import { TrainingView } from '../components/TrainingView';
import { VisualizationSection } from '../components/VisualizationSection';

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
  // Initialize collapsed by default for better mobile UX and cleaner desktop start
  const [rightDrawerCollapsed, setRightDrawerCollapsed] = useState(true);
  const [activeDrawerTab, setActiveDrawerTab] = useState<'visualization' | 'physics'>('visualization');
  const [viewerVersion, setViewerVersion] = useState<'v1' | 'v2'>('v1');

  // Inicializar estado responsivo y listener
  useEffect(() => {
    const handleResize = () => {
      if (window.innerWidth < 768 && !rightDrawerCollapsed) {
        setRightDrawerCollapsed(true);
      } else if (window.innerWidth >= 768 && rightDrawerCollapsed) {
          // Optional: Auto-expand on desktop? Maybe not to respect user choice.
          // setRightDrawerCollapsed(false);
      }
    };

    window.addEventListener('resize', handleResize);
    return () => window.removeEventListener('resize', handleResize);
  }, [rightDrawerCollapsed]);
  const [selectedLayer, setSelectedLayer] = useState(0); // 0: Density, 1: Phase, 2: Energy, 3: Flow
  const [selectedTimelineFrame, setSelectedTimelineFrame] = useState<{
    step: number;
    timestamp: string;
    map_data: number[][];
  } | null>(null);
  const [theaterMode, setTheaterMode] = useState(false);
  const { simData, selectedViz, connectionStatus, inferenceStatus } = useWebSocket();

  // Efecto para Modo Teatro: Colapsar paneles cuando se activa
  useEffect(() => {
    if (theaterMode) {
      setLabPanelOpen(false);
      setRightDrawerCollapsed(true);
    } else {
      // Al salir, restaurar paneles (opcional, o dejar como estaban antes? Por ahora restauramos defaults)
      setLabPanelOpen(true);
      setRightDrawerCollapsed(false);
    }
  }, [theaterMode]);

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
        if (selectedViz === 'holographic' || selectedViz === '3d' || selectedViz === 'poincare_3d') {
          return (
            <div className="absolute inset-0 z-0 bg-gradient-deep-space overflow-hidden">
              {flatMapData.length > 0 && gridWidth > 0 && gridHeight > 0 ? (
                viewerVersion === 'v1' ? (
                  <HolographicViewer 
                    data={flatMapData}
                    width={gridWidth}
                    height={gridHeight}
                    vizType={selectedViz}
                    threshold={0.01}
                  />
                ) : (
                  <HolographicViewer2 
                    data={flatMapData}
                    width={gridWidth}
                    height={gridHeight}
                    vizType={selectedViz}
                    threshold={0.01}
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
        } else if (selectedViz === 'phase_space') {
            return (
                <div className="absolute inset-0 z-0 bg-slate-900 overflow-hidden p-4">
                    <PhaseSpaceViewer />
                </div>
            );
        } else {
          // Vista 2D con PanZoomCanvas
          return (
            <div className="absolute inset-0 z-0 bg-gradient-deep-space overflow-hidden">
              {connectionStatus === 'connected' ? (
                <>
                  <PanZoomCanvas 
                    historyFrame={selectedTimelineFrame} 
                    theaterMode={theaterMode}
                  />
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
              <ScientificHeader
                onToggleMobileMenu={() => setRightDrawerCollapsed(!rightDrawerCollapsed)}
              />

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
          <aside className="
            flex-col border-r border-white/5 bg-dark-980/95 backdrop-blur-md z-50 shrink-0 transition-all duration-300 flex overflow-hidden
            fixed inset-0 w-full h-full md:relative md:w-[380px] md:h-auto
          ">
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

          {/* Floating Exit Theater Mode Button */}
          {theaterMode && (
            <button
              onClick={() => setTheaterMode(false)}
              className="absolute top-4 right-4 z-50 bg-black/50 hover:bg-black/80 text-white px-4 py-2 rounded-full backdrop-blur-md border border-white/10 transition-all flex items-center gap-2 shadow-xl"
            >
              <Minimize size={16} />
              <span className="text-xs font-bold">Salir Modo Teatro</span>
            </button>
          )}
        </main>

        {/* Panel Lateral Derecho (Unified Drawer) - Colapsible */}
        <RightDrawer 
          isCollapsed={rightDrawerCollapsed}
          onToggleCollapse={() => setRightDrawerCollapsed(!rightDrawerCollapsed)}
          activeTab={activeDrawerTab}
          onTabChange={setActiveDrawerTab}
        >
          {activeDrawerTab === 'visualization' ? (
            <VisualizationSection
              viewerVersion={viewerVersion}
              onViewerVersionChange={setViewerVersion}
              selectedLayer={selectedLayer}
              onLayerChange={setSelectedLayer}
              theaterMode={theaterMode}
              onToggleTheaterMode={setTheaterMode}
            />
          ) : (
            <PhysicsSection />
          )}
        </RightDrawer>
      </div>
    </div>
  );
};
