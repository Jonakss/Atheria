import React, { useState, useMemo } from 'react';
import { ScientificHeader } from '../components/ScientificHeader';
import { NavigationSidebar } from '../components/NavigationSidebar';
import { PhysicsInspector } from '../components/PhysicsInspector';
import { MetricsBar } from '../components/MetricsBar';
import { Toolbar } from '../components/Toolbar';
import { AnalysisView } from '../components/AnalysisView';
import { HistoryView } from '../components/HistoryView';
import { LogsView } from '../components/LogsView';
import { PanZoomCanvas } from '../../../components/ui/PanZoomCanvas';
import HolographicViewer from '../../../components/visualization/HolographicViewer';
import { LabSider } from '../../../components/ui/LabSider';
import { useWebSocket } from '../../../hooks/useWebSocket';

type TabType = 'lab' | 'analysis' | 'history' | 'logs';

export const DashboardLayout: React.FC = () => {
  const [activeTab, setActiveTab] = useState<TabType>('lab');
  const [currentEpoch, setCurrentEpoch] = useState(2); // Era de Partículas - PARTÍCULAS está activa
  const [labPanelOpen, setLabPanelOpen] = useState(true); // Panel de laboratorio visible por defecto
  const { simData, selectedViz, connectionStatus } = useWebSocket();

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
            <div className="absolute inset-0 z-0 bg-gradient-to-b from-[#050505] to-black overflow-hidden">
              {flatMapData.length > 0 && gridWidth > 0 && gridHeight > 0 ? (
                <HolographicViewer 
                  data={flatMapData}
                  width={gridWidth}
                  height={gridHeight}
                />
              ) : (
                <div className="flex items-center justify-center h-full text-gray-300 text-sm">
                  Esperando datos de simulación...
                </div>
              )}
              {/* Design System: text-[9px] font-mono text-gray-700 según mockup */}
              <div className="absolute bottom-4 right-4 text-[9px] font-mono text-gray-700 pointer-events-none text-right">
                VIEWPORT: ORTHOGRAPHIC<br/>
                RENDER: WEBGL2 / HIGH_PRECISION
              </div>
            </div>
          );
        } else {
          // Vista 2D con PanZoomCanvas
          return (
            <div className="absolute inset-0 z-0 bg-gradient-to-b from-[#050505] to-black overflow-hidden">
              {connectionStatus === 'connected' ? (
                <PanZoomCanvas />
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
    <div className="h-screen bg-[#020202] text-gray-300 font-sans selection:bg-blue-500/30 overflow-hidden flex flex-col">
      
      {/* Header: Barra de Comando Técnica */}
      <ScientificHeader currentEpoch={currentEpoch} onEpochChange={setCurrentEpoch} />

      {/* Contenedor Principal */}
      <div className="flex-1 flex overflow-hidden">
        
        {/* Sidebar de Navegación (Iconos Minimalistas) - Siempre visible */}
        <NavigationSidebar 
          activeTab={activeTab} 
          onTabChange={(tab) => {
            setActiveTab(tab);
            // Si se selecciona 'lab', abrir/cerrar panel de laboratorio
            if (tab === 'lab') {
              setLabPanelOpen(!labPanelOpen);
            } else {
              setLabPanelOpen(false);
            }
          }}
          labPanelOpen={labPanelOpen}
        />

        {/* Panel de Laboratorio (Experimentos/Entrenamiento) - Visible por defecto */}
        {/* Integrado directamente sin wrapper adicional */}
        {labPanelOpen && (
          <div className="w-[380px] border-r border-white/10 bg-[#080808] flex flex-col z-50 shrink-0 overflow-hidden">
            <LabSider />
          </div>
        )}

        {/* Área de Trabajo (Viewport + Paneles Flotantes) */}
        {/* Design System: flex-1 (Ocupa todo el espacio restante), relative para overlays */}
        <main className="flex-1 relative bg-black flex flex-col overflow-hidden">
          
          {/* Barra de Herramientas Superior (Flotante) */}
          <Toolbar />

          {/* Viewport (Fondo) - Design System: bg-[#050505] a black según mockup */}
          {renderContentView()}

          {/* Panel Inferior (Métricas Críticas) - Design System: bg-[#050505]/95 backdrop-blur-sm */}
          <MetricsBar />
        </main>

        {/* Panel Lateral Derecho (Inspector y Controles) */}
        {/* Design System: w-72 (288px) o w-80 (320px) - usando w-72 según mockup */}
        <PhysicsInspector />
      </div>
    </div>
  );
};
