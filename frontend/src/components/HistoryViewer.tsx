// frontend/src/components/HistoryViewer.tsx
import { useState, useEffect } from 'react';
import { 
    Paper, Stack, Group, Text, Badge, Button, 
    Table, ScrollArea, Tooltip, ActionIcon, Divider,
    Card, Alert, Select, Box, Slider, Title, Center
} from '@mantine/core';
import { useWebSocket } from '../hooks/useWebSocket';
import { 
    IconDownload, IconTrash, IconInfoCircle, IconFile,
    IconClock, IconPlayerPlay, IconPlayerPause, IconRefresh,
    IconDatabase, IconTimeline, IconChartBar
} from '@tabler/icons-react';

interface HistoryFile {
    filename: string;
    filepath: string;
    frames: number;
    created_at?: string;
    min_step?: number;
    max_step?: number;
}

interface HistoryFrame {
    step: number;
    timestamp: string;
    map_data: number[][];
    hist_data?: Record<string, any>;
}

export function HistoryViewer() {
    const { sendCommand, simData } = useWebSocket();
    const [historyFiles, setHistoryFiles] = useState<HistoryFile[]>([]);
    const [selectedFile, setSelectedFile] = useState<string | null>(null);
    const [currentFrame, setCurrentFrame] = useState<HistoryFrame | null>(null);
    const [frameIndex, setFrameIndex] = useState(0);
    const [frames, setFrames] = useState<HistoryFrame[]>([]);
    const [loading, setLoading] = useState(false);
    const [playing, setPlaying] = useState(false);
    const [playInterval, setPlayInterval] = useState<ReturnType<typeof setInterval> | null>(null);

    useEffect(() => {
        loadHistoryFiles();
        
        // Escuchar eventos de WebSocket
        const handleHistoryFilesList = (event: CustomEvent) => {
            setHistoryFiles(event.detail);
            setLoading(false);
        };
        
        const handleHistoryFileLoaded = (event: CustomEvent) => {
            const { frames, metadata } = event.detail;
            setFrames(frames);
            setFrameIndex(0);
            setLoading(false);
        };
        
        window.addEventListener('history_files_list', handleHistoryFilesList as EventListener);
        window.addEventListener('history_file_loaded', handleHistoryFileLoaded as EventListener);
        
        return () => {
            window.removeEventListener('history_files_list', handleHistoryFilesList as EventListener);
            window.removeEventListener('history_file_loaded', handleHistoryFileLoaded as EventListener);
        };
    }, []);

    useEffect(() => {
        if (selectedFile) {
            loadHistoryFile(selectedFile);
        }
    }, [selectedFile]);

    useEffect(() => {
        if (frames.length > 0 && frameIndex >= 0 && frameIndex < frames.length) {
            setCurrentFrame(frames[frameIndex]);
        }
    }, [frames, frameIndex]);

    useEffect(() => {
        if (playing && frames.length > 0) {
            const interval = setInterval(() => {
                setFrameIndex(prev => {
                    const next = prev + 1;
                    if (next >= frames.length) {
                        setPlaying(false);
                        return prev;
                    }
                    return next;
                });
            }, 100); // 10 FPS
            setPlayInterval(interval);
            return () => clearInterval(interval);
        } else if (playInterval) {
            clearInterval(playInterval);
            setPlayInterval(null);
        }
    }, [playing, frames.length]);

    const loadHistoryFiles = () => {
        setLoading(true);
        sendCommand('simulation', 'list_history_files', {});
    };

    const loadHistoryFile = (filename: string) => {
        setLoading(true);
        sendCommand('simulation', 'load_history_file', { filename });
    };

    const handlePlayPause = () => {
        if (playing) {
            setPlaying(false);
        } else {
            if (frameIndex >= frames.length - 1) {
                setFrameIndex(0);
            }
            setPlaying(true);
        }
    };

    const handleFrameChange = (newIndex: number) => {
        setFrameIndex(Math.max(0, Math.min(newIndex, frames.length - 1)));
        setPlaying(false);
    };

    const formatDate = (dateString?: string) => {
        if (!dateString) return 'N/A';
        return new Date(dateString).toLocaleString();
    };

    // Calcular estadísticas del archivo cargado
    const fileStats = frames.length > 0 && currentFrame ? {
        totalFrames: frames.length,
        currentStep: currentFrame.step,
        // Usar reduce en lugar de spread operator para evitar stack overflow con arrays grandes
        minStep: frames.reduce((min, f) => Math.min(min, f.step), Infinity),
        maxStep: frames.reduce((max, f) => Math.max(max, f.step), -Infinity),
        gridSize: `${currentFrame.map_data?.[0]?.length || 0}x${currentFrame.map_data?.length || 0}`,
        timeSpan: frames.length > 1 ? frames[frames.length - 1].step - frames[0].step : 0
    } : null;

    return (
        <Stack gap="lg" p="md">
            {/* Header con explicación */}
            <Paper p="md" withBorder style={{ backgroundColor: 'var(--mantine-color-dark-6)' }}>
                <Stack gap="sm">
                    <Group gap="xs">
                        <IconDatabase size={24} />
                        <Title order={4}>Análisis de Historia de Simulación</Title>
                    </Group>
                    <Text size="sm" c="dimmed">
                        Esta pestaña te permite <strong>cargar y analizar</strong> simulaciones guardadas anteriormente. 
                        Las historias se guardan desde los <strong>Controles Avanzados</strong> durante una simulación activa.
                        Aquí puedes:
                    </Text>
                    <Box pl="md">
                        <Text size="xs" c="dimmed" component="ul" style={{ margin: 0, paddingLeft: '1rem' }}>
                            <li>Ver todas las simulaciones guardadas</li>
                            <li>Cargar una historia completa y reproducirla frame por frame</li>
                            <li>Analizar la evolución temporal de los datos</li>
                            <li>Exportar frames específicos para análisis posterior</li>
                        </Text>
                    </Box>
                </Stack>
            </Paper>

            <Group justify="space-between">
                <Group gap="xs">
                    <IconFile size={20} />
                    <Text size="lg" fw={600}>Archivos de Historia Guardados</Text>
                </Group>
                <Button
                    variant="light"
                    size="sm"
                    leftSection={<IconRefresh size={16} />}
                    onClick={loadHistoryFiles}
                    loading={loading}
                >
                    Actualizar Lista
                </Button>
            </Group>

            <Divider />

            {/* Lista de archivos guardados */}
            <Paper p="md" withBorder>
                <Text size="sm" fw={500} mb="md">
                    <IconDatabase size={16} style={{ verticalAlign: 'middle', marginRight: 8 }} />
                    Historial Guardado
                </Text>
                {historyFiles.length === 0 ? (
                    <Alert icon={<IconInfoCircle size={16} />} color="blue" title="No hay historias guardadas">
                        <Text size="sm">
                            No se han encontrado archivos de historia. Para guardar una simulación:
                        </Text>
                        <Text size="xs" c="dimmed" mt="xs" component="ol" pl="md">
                            <li>Inicia una simulación cargando un modelo</li>
                            <li>Ve a <strong>Controles Avanzados</strong></li>
                            <li>Activa "Habilitar Historia" y ejecuta la simulación</li>
                            <li>Cuando termines, haz clic en "Guardar Historia"</li>
                        </Text>
                    </Alert>
                ) : (
                    <ScrollArea h={250}>
                        <Table highlightOnHover>
                            <Table.Thead>
                                <Table.Tr>
                                    <Table.Th>Archivo</Table.Th>
                                    <Table.Th>Frames</Table.Th>
                                    <Table.Th>Rango de Steps</Table.Th>
                                    <Table.Th>Fecha Creación</Table.Th>
                                    <Table.Th>Acciones</Table.Th>
                                </Table.Tr>
                            </Table.Thead>
                            <Table.Tbody>
                                {historyFiles.map((file) => (
                                    <Table.Tr 
                                        key={file.filename}
                                        style={{ 
                                            cursor: 'pointer',
                                            backgroundColor: selectedFile === file.filename 
                                                ? 'var(--mantine-color-blue-9)' 
                                                : 'transparent'
                                        }}
                                        onClick={() => setSelectedFile(file.filename)}
                                    >
                                        <Table.Td>
                                            <Group gap="xs">
                                                <IconFile size={16} />
                                                <Text size="sm" fw={selectedFile === file.filename ? 600 : 400}>
                                                    {file.filename}
                                                </Text>
                                            </Group>
                                        </Table.Td>
                                        <Table.Td>
                                            <Badge size="sm" variant="light" color="blue">
                                                {file.frames} frames
                                            </Badge>
                                        </Table.Td>
                                        <Table.Td>
                                            {file.min_step !== undefined && file.max_step !== undefined ? (
                                                <Text size="xs" c="dimmed">
                                                    Step {file.min_step} → {file.max_step}
                                                    {file.max_step - file.min_step > 0 && (
                                                        <Badge size="xs" variant="subtle" ml="xs">
                                                            {file.max_step - file.min_step} steps
                                                        </Badge>
                                                    )}
                                                </Text>
                                            ) : (
                                                <Text size="xs" c="dimmed">N/A</Text>
                                            )}
                                        </Table.Td>
                                        <Table.Td>
                                            <Group gap="xs">
                                                <IconClock size={14} />
                                                <Text size="xs" c="dimmed">
                                                    {formatDate(file.created_at)}
                                                </Text>
                                            </Group>
                                        </Table.Td>
                                        <Table.Td>
                                            <ActionIcon
                                                variant="light"
                                                color="blue"
                                                onClick={(e) => {
                                                    e.stopPropagation();
                                                    loadHistoryFile(file.filename);
                                                }}
                                                loading={loading && selectedFile === file.filename}
                                            >
                                                <IconDownload size={16} />
                                            </ActionIcon>
                                        </Table.Td>
                                    </Table.Tr>
                                ))}
                            </Table.Tbody>
                        </Table>
                    </ScrollArea>
                )}
            </Paper>

            {/* Reproductor y visualizador de frames */}
            {frames.length > 0 && (
                <Paper p="md" withBorder>
                    <Stack gap="md">
                        {/* Header del reproductor */}
                        <Group justify="space-between">
                            <Group gap="xs">
                                <IconTimeline size={20} />
                                <div>
                                    <Text size="sm" fw={500}>
                                        Reproduciendo: <strong>{selectedFile}</strong>
                                    </Text>
                                    <Text size="xs" c="dimmed">
                                        {fileStats && `${fileStats.totalFrames} frames cargados | Steps ${fileStats.minStep} - ${fileStats.maxStep}`}
                                    </Text>
                                </div>
                            </Group>
                            <Badge size="lg" variant="light" color="blue">
                                Frame {frameIndex + 1} / {frames.length}
                            </Badge>
                        </Group>

                        <Divider />

                        {/* Estadísticas del frame actual */}
                        {fileStats && (
                            <Group grow>
                                <Paper p="sm" withBorder style={{ backgroundColor: 'var(--mantine-color-dark-7)' }}>
                                    <Stack gap={4} align="center">
                                        <Text size="xs" c="dimmed">Step Actual</Text>
                                        <Text size="xl" fw={700}>{fileStats.currentStep}</Text>
                                    </Stack>
                                </Paper>
                                <Paper p="sm" withBorder style={{ backgroundColor: 'var(--mantine-color-dark-7)' }}>
                                    <Stack gap={4} align="center">
                                        <Text size="xs" c="dimmed">Tamaño de Grid</Text>
                                        <Text size="xl" fw={700}>{fileStats.gridSize}</Text>
                                    </Stack>
                                </Paper>
                                <Paper p="sm" withBorder style={{ backgroundColor: 'var(--mantine-color-dark-7)' }}>
                                    <Stack gap={4} align="center">
                                        <Text size="xs" c="dimmed">Span Temporal</Text>
                                        <Text size="xl" fw={700}>{fileStats.timeSpan} steps</Text>
                                    </Stack>
                                </Paper>
                            </Group>
                        )}

                        {/* Controles de reproducción */}
                        <Group grow>
                            <Button
                                variant="light"
                                leftSection={playing ? <IconPlayerPause size={16} /> : <IconPlayerPlay size={16} />}
                                onClick={handlePlayPause}
                                disabled={frames.length === 0}
                            >
                                {playing ? 'Pausar' : 'Reproducir'}
                            </Button>
                            <Button
                                variant="light"
                                onClick={() => {
                                    setFrameIndex(0);
                                    setPlaying(false);
                                }}
                                disabled={frameIndex === 0}
                            >
                                Inicio
                            </Button>
                            <Button
                                variant="light"
                                onClick={() => {
                                    setFrameIndex(frames.length - 1);
                                    setPlaying(false);
                                }}
                                disabled={frameIndex === frames.length - 1}
                            >
                                Final
                            </Button>
                        </Group>

                        {/* Slider de navegación */}
                        <Box>
                            <Group justify="space-between" mb="xs">
                                <Text size="xs" c="dimmed">
                                    Step: {currentFrame?.step || 0}
                                </Text>
                                <Text size="xs" c="dimmed">
                                    <IconClock size={12} style={{ verticalAlign: 'middle', marginRight: 4 }} />
                                    {currentFrame?.timestamp ? formatDate(currentFrame.timestamp) : 'N/A'}
                                </Text>
                            </Group>
                            <Slider
                                value={frameIndex}
                                onChange={handleFrameChange}
                                min={0}
                                max={frames.length - 1}
                                step={1}
                                marks={[
                                    { value: 0, label: '0' },
                                    { value: Math.floor(frames.length / 2), label: String(Math.floor(frames.length / 2)) },
                                    { value: frames.length - 1, label: String(frames.length - 1) }
                                ]}
                                label={(value) => `Frame ${value + 1}`}
                            />
                        </Box>

                        {/* Información del frame */}
                        {currentFrame && (
                            <Card withBorder p="sm" style={{ backgroundColor: 'var(--mantine-color-dark-7)' }}>
                                <Group gap="xs" mb="xs">
                                    <IconChartBar size={16} />
                                    <Text size="xs" fw={500}>Datos del Frame Actual</Text>
                                </Group>
                                <Group gap="md">
                                    <Badge variant="light" size="lg">Step: {currentFrame.step}</Badge>
                                    <Badge variant="light" size="lg">
                                        Grid: {currentFrame.map_data?.[0]?.length || 0}x{currentFrame.map_data?.length || 0}
                                    </Badge>
                                    {currentFrame.hist_data && (
                                        <Badge variant="light" size="lg">
                                            Histogramas: {Object.keys(currentFrame.hist_data).length}
                                        </Badge>
                                    )}
                                </Group>
                            </Card>
                        )}

                        <Alert color="blue" icon={<IconInfoCircle size={16} />} title="Nota">
                            <Text size="xs">
                                Los frames cargados están disponibles para visualización en la pestaña <strong>Visualización</strong>. 
                                Cambia a esa pestaña para ver la evolución temporal de los datos.
                            </Text>
                        </Alert>
                    </Stack>
                </Paper>
            )}
        </Stack>
    );
}
