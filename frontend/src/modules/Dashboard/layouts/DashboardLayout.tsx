import React, { useState, useMemo } from 'react';
import { ScientificHeader } from '../components/ScientificHeader';
import { NavigationSidebar } from '../components/NavigationSidebar';
import { PhysicsInspector } from '../components/PhysicsInspector';
import { MetricsBar } from '../components/MetricsBar';
import { Toolbar } from '../components/Toolbar';
import { PanZoomCanvas } from '../../../components/ui/PanZoomCanvas';
import HolographicViewer from '../../../components/visualization/HolographicViewer';
import { useWebSocket } from '../../../hooks/useWebSocket';

type TabType = 'lab' | 'analysis' | 'history' | 'logs';

export const DashboardLayout: React.FC = () => {
  const [activeTab, setActiveTab] = useState<TabType>('lab');
  const [currentEpoch, setCurrentEpoch] = useState(2); // Era de Partículas
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
                <div className="flex items-center justify-center h-full text-gray-600 text-sm">
                  Esperando datos de simulación...
                </div>
              )}
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
                <div className="flex items-center justify-center h-full text-gray-600 text-sm">
                  Conectando al servidor...
                </div>
              )}
            </div>
          );
        }
      
      case 'analysis':
        return (
          <div className="absolute inset-0 z-0 bg-gradient-to-b from-[#050505] to-black flex items-center justify-center">
            <div className="text-gray-600 text-sm">Análisis - Próximamente</div>
          </div>
        );
      
      case 'history':
        return (
          <div className="absolute inset-0 z-0 bg-gradient-to-b from-[#050505] to-black flex items-center justify-center">
            <div className="text-gray-600 text-sm">Historial - Próximamente</div>
          </div>
        );
      
      case 'logs':
        return (
          <div className="absolute inset-0 z-0 bg-gradient-to-b from-[#050505] to-black flex items-center justify-center">
            <div className="text-gray-600 text-sm">Logs - Próximamente</div>
          </div>
        );
      
      default:
        return null;
    }
  };

  return (
    <div className="h-screen bg-[#020202] text-gray-300 font-sans selection:bg-blue-500/30 overflow-hidden flex flex-col">
      
      {/* Header: Barra de Comando Técnica */}
      <ScientificHeader currentEpoch={currentEpoch} />

      {/* Contenedor Principal */}
      <div className="flex-1 flex overflow-hidden">
        
        {/* Sidebar de Navegación (Iconos Minimalistas) */}
        <NavigationSidebar activeTab={activeTab} onTabChange={setActiveTab} />

        {/* Área de Trabajo (Viewport + Paneles Flotantes) */}
        <main className="flex-1 relative bg-black flex flex-col overflow-hidden">
          
          {/* Barra de Herramientas Superior (Flotante) */}
          <Toolbar />

          {/* Viewport (Fondo) */}
          {renderContentView()}

          {/* Panel Inferior (Métricas Críticas) */}
          <MetricsBar />
        </main>

        {/* Panel Lateral Derecho (Inspector y Controles) */}
        <PhysicsInspector />
      </div>
    </div>
  );
};
