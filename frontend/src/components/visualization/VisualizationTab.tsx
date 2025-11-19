// frontend/src/components/VisualizationTab.tsx
import { useState, useEffect } from 'react';
import { Box, Stack, Collapse, Button, Switch, Group, Text } from '@mantine/core';
import { PanZoomCanvas } from '../ui/PanZoomCanvas';
import { HistogramPanel } from './HistogramPanel';
import { TrainingDashboard } from '../training/TrainingDashboard';
import { PhaseAttractorViewer } from './PhaseAttractorViewer';
import { FlowViewer } from './FlowViewer';
import { History3DViewer } from './History3DViewer';
import { Complex3DViewer } from './Complex3DViewer';
import { Poincare3DViewer } from './Poincare3DViewer';
import { TimelineControl } from '../controls/TimelineControl';
import { useWebSocket } from '../../hooks/useWebSocket';
import { IconChevronDown, IconChevronUp, IconLivePhoto } from '@tabler/icons-react';

interface HistoryFrame {
    step: number;
    timestamp: string;
    map_data: number[][];
}

export function VisualizationTab() {
    const { selectedViz, simData, sendCommand, liveFeedEnabled, setLiveFeedEnabled, connectionStatus } = useWebSocket();
    const [historyFrames, setHistoryFrames] = useState<HistoryFrame[]>([]);
    const [currentFrameIndex, setCurrentFrameIndex] = useState(-1); // -1 = usar tiempo real
    const [isPlaying, setIsPlaying] = useState(false);
    const [playInterval, setPlayInterval] = useState<ReturnType<typeof setInterval> | null>(null);
    const [playbackSpeed, setPlaybackSpeed] = useState(10); // FPS
    const [toolsExpanded, setToolsExpanded] = useState(false);
    
    // Escuchar cuando se carga un historial desde archivo
    useEffect(() => {
        const handleHistoryFileLoaded = (event: CustomEvent) => {
            const { frames } = event.detail;
            if (frames && Array.isArray(frames)) {
                setHistoryFrames(frames);
                setCurrentFrameIndex(0);
            }
        };
        
        window.addEventListener('history_file_loaded', handleHistoryFileLoaded as EventListener);
        return () => {
            window.removeEventListener('history_file_loaded', handleHistoryFileLoaded as EventListener);
        };
    }, []);
    
    // Acumular frames del historial en tiempo real (solo si no estamos en modo reproducción)
    useEffect(() => {
        if (currentFrameIndex >= 0) return; // No acumular si estamos reproduciendo
        
        if (simData?.map_data && simData?.step !== undefined) {
            setHistoryFrames(prev => {
                // Evitar duplicados
                const exists = prev.some(f => f.step === simData.step);
                if (exists) return prev;
                
                const newFrame: HistoryFrame = {
                    step: simData.step || 0,
                    timestamp: new Date().toISOString(),
                    map_data: simData.map_data as number[][]
                };
                
                // Mantener solo los últimos 1000 frames
                const updated = [...prev, newFrame].slice(-1000);
                return updated;
            });
        }
    }, [simData?.step, simData?.map_data, currentFrameIndex]);
    
    // Reproducción automática
    useEffect(() => {
        if (isPlaying && historyFrames.length > 0 && currentFrameIndex >= 0) {
            const interval = setInterval(() => {
                setCurrentFrameIndex(prev => {
                    const next = prev + 1;
                    if (next >= historyFrames.length) {
                        setIsPlaying(false);
                        return prev;
                    }
                    return next;
                });
            }, 1000 / playbackSpeed); // Convertir FPS a ms
            setPlayInterval(interval);
            return () => clearInterval(interval);
        } else if (playInterval) {
            clearInterval(playInterval);
            setPlayInterval(null);
        }
    }, [isPlaying, historyFrames.length, currentFrameIndex, playbackSpeed]);
    
    const handlePlayPause = () => {
        if (historyFrames.length === 0) return;
        if (currentFrameIndex < 0) {
            // Si estamos en tiempo real, empezar desde el último frame
            setCurrentFrameIndex(historyFrames.length - 1);
        }
        setIsPlaying(!isPlaying);
    };
    
    const handleFrameChange = (index: number) => {
        setCurrentFrameIndex(index);
        setIsPlaying(false); // Pausar al cambiar manualmente
    };
    
    const handleClear = () => {
        setHistoryFrames([]);
        setCurrentFrameIndex(-1);
        setIsPlaying(false);
    };
    
    // Determinar qué frame mostrar
    const frameToShow = currentFrameIndex >= 0 && historyFrames.length > 0 
        ? historyFrames[currentFrameIndex] 
        : null;
    
    return (
        <Box style={{ flex: 1, position: 'relative', height: '100vh', minHeight: '800px', width: '100%', display: 'flex', flexDirection: 'column' }}>
            {/* Control de Timeline / Live Feed Toggle */}
            <Box style={{ padding: 'var(--mantine-spacing-sm)', borderBottom: '1px solid var(--mantine-color-dark-4)' }}>
                <Group justify="space-between" mb="xs">
                    <Group gap="xs">
                        <IconLivePhoto size={18} color={liveFeedEnabled ? 'green' : 'gray'} />
                        <Text size="sm" fw={500}>Modo Live Feed</Text>
                        <Switch
                            checked={liveFeedEnabled}
                            disabled={connectionStatus !== 'connected'}
                            onChange={(e) => {
                                const enabled = e.currentTarget.checked;
                                setLiveFeedEnabled(enabled);
                                
                                if (enabled) {
                                    // Volver a tiempo real cuando se activa live feed
                                    setCurrentFrameIndex(-1);
                                    setIsPlaying(false);
                                }
                            }}
                            label={liveFeedEnabled ? 'Activo' : 'Inactivo'}
                            color="green"
                        />
                        {!liveFeedEnabled && (
                            <Text size="xs" c="yellow" fw={500}>
                                ⚠️ Solo estado (sin datos de visualización)
                            </Text>
                        )}
                    </Group>
                    {!liveFeedEnabled && (
                        <Text size="xs" c="dimmed">
                            Modo Historial: Navegando frame {currentFrameIndex >= 0 ? currentFrameIndex + 1 : 0} de {historyFrames.length}
                        </Text>
                    )}
                </Group>
                {!liveFeedEnabled && (
                    <TimelineControl
                        frames={historyFrames}
                        currentFrameIndex={currentFrameIndex}
                        onFrameChange={handleFrameChange}
                        onPlayPause={handlePlayPause}
                        isPlaying={isPlaying}
                        onClear={handleClear}
                        playbackSpeed={playbackSpeed}
                        onPlaybackSpeedChange={setPlaybackSpeed}
                    />
                )}
                {liveFeedEnabled && (
                    <Group gap="xs" mt="xs">
                        <Text size="xs" c="dimmed">
                            Frame en tiempo real: {simData?.step || 0} | 
                            Velocidad: {playbackSpeed} FPS
                        </Text>
                        <Button
                            variant="light"
                            size="xs"
                            onClick={handleClear}
                        >
                            Limpiar Historial
                        </Button>
                    </Group>
                )}
            </Box>
            
            {/* Área principal de visualización - más grande */}
            <Box style={{ flex: 1, position: 'relative', minHeight: '600px', height: '70vh' }}>
                {selectedViz === 'history_3d' ? (
                    <History3DViewer />
                ) : selectedViz === 'complex_3d' ? (
                    <Complex3DViewer />
                ) : selectedViz === 'poincare_3d' ? (
                    <Poincare3DViewer />
                ) : (
                    <PanZoomCanvas historyFrame={liveFeedEnabled ? null : frameToShow} />
                )}
            </Box>
            
            {/* Herramientas rápidas colapsibles abajo */}
            <Box 
                style={{ 
                    borderTop: '1px solid var(--mantine-color-dark-4)', 
                    backgroundColor: 'var(--mantine-color-dark-8)',
                    position: 'relative',
                    zIndex: 10
                }}
            >
                <Button
                    variant="subtle"
                    fullWidth
                    leftSection={toolsExpanded ? <IconChevronUp size={16} /> : <IconChevronDown size={16} />}
                    onClick={() => setToolsExpanded(!toolsExpanded)}
                    style={{ borderRadius: 0 }}
                    size="sm"
                >
                    {toolsExpanded ? 'Ocultar' : 'Mostrar'} Herramientas Rápidas
                </Button>
                <Collapse in={toolsExpanded} transitionDuration={200}>
                    <Box 
                        p="md" 
                        style={{ 
                            maxHeight: '400px', 
                            overflowY: 'auto',
                            overflowX: 'hidden'
                        }}
                    >
                        <Stack gap="md">
                            {selectedViz !== 'histogram' && <HistogramPanel />}
                            {selectedViz !== 'phase_attractor' && <PhaseAttractorViewer />}
                            {selectedViz !== 'flow' && selectedViz !== 'poincare' && <FlowViewer />}
                        </Stack>
                    </Box>
                </Collapse>
            </Box>
            
            {/* Dashboard de entrenamiento siempre visible */}
            <TrainingDashboard />
        </Box>
    );
}

