// frontend/src/components/AdvancedControls.tsx
import { useState } from 'react';
import { 
    Collapse, Paper, Stack, Group, Text, NumberInput, 
    Switch, Slider, Button, Select, Divider, Badge, Tooltip, Box
} from '@mantine/core';
import { useDisclosure } from '@mantine/hooks';
import { useWebSocket } from '../hooks/useWebSocket';
import { 
    IconSettings, IconChevronDown, IconChevronUp,
    IconPlayerPlay, IconPlayerPause, IconRefresh,
    IconZoomIn, IconZoomOut, IconArrowsMaximize,
    IconCamera
} from '@tabler/icons-react';

export function AdvancedControls() {
    const { sendCommand, simData, inferenceStatus, selectedViz, setSelectedViz, snapshotCount } = useWebSocket();
    const [opened, { toggle }] = useDisclosure(false);
    
    // Estados para controles de simulación
    const [simulationSpeed, setSimulationSpeed] = useState(1.0);
    const [gridSize, setGridSize] = useState(256);
    
    // Estados para controles de visualización
    const [zoomLevel, setZoomLevel] = useState(1.0);
    const [showPoincare, setShowPoincare] = useState(true);
    const [showHistogram, setShowHistogram] = useState(true);
    
    // Estados para controles de entrenamiento
    const [autoSave, setAutoSave] = useState(true);
    const [saveInterval, setSaveInterval] = useState(50);

    const handleResetZoom = () => {
        setZoomLevel(1.0);
        // Enviar comando para resetear zoom (si el backend lo soporta)
    };

    const handleSpeedChange = (value: number) => {
        setSimulationSpeed(value);
        // Enviar comando para cambiar velocidad de simulación
        sendCommand('simulation', 'set_speed', { speed: value });
    };

    const handleGridSizeChange = (value: number) => {
        setGridSize(value);
        sendCommand('simulation', 'set_grid_size', { grid_size: value });
    };

    return (
        <Paper p="md" withBorder>
            <Group justify="space-between" mb="md">
                <Group>
                    <IconSettings size={20} />
                    <Text fw={700}>Controles Avanzados</Text>
                </Group>
                <Button
                    variant="subtle"
                    size="xs"
                    leftSection={opened ? <IconChevronUp size={16} /> : <IconChevronDown size={16} />}
                    onClick={toggle}
                >
                    {opened ? 'Ocultar' : 'Mostrar'}
                </Button>
            </Group>

            <Collapse in={opened}>
                <Stack gap="md">
                    {/* Controles de Simulación */}
                    <div>
                        <Text size="sm" fw={500} mb="xs">Simulación</Text>
                        <Stack gap="sm">
                            <div>
                                <Group justify="space-between" mb="xs">
                                    <Text size="xs" c="dimmed">Velocidad de Simulación</Text>
                                    <Badge variant="light">{simulationSpeed.toFixed(1)}x</Badge>
                                </Group>
                                <Slider
                                    value={simulationSpeed}
                                    onChange={handleSpeedChange}
                                    min={0.1}
                                    max={5.0}
                                    step={0.1}
                                    marks={[
                                        { value: 0.5, label: '0.5x' },
                                        { value: 1.0, label: '1x' },
                                        { value: 2.0, label: '2x' },
                                        { value: 5.0, label: '5x' }
                                    ]}
                                />
                            </div>
                            
                            <Group grow>
                                <Tooltip label="Pausar/Reanudar simulación">
                                    <Button
                                        leftSection={inferenceStatus === 'running' ? <IconPlayerPause size={16} /> : <IconPlayerPlay size={16} />}
                                        color={inferenceStatus === 'running' ? 'yellow' : 'green'}
                                        onClick={() => {
                                            const command = inferenceStatus === 'running' ? 'pause' : 'play';
                                            sendCommand('inference', command);
                                        }}
                                        disabled={!simData}
                                    >
                                        {inferenceStatus === 'running' ? 'Pausar' : 'Iniciar'}
                                    </Button>
                                </Tooltip>
                                <Tooltip label="Reiniciar simulación al estado inicial">
                                    <Button
                                        leftSection={<IconRefresh size={16} />}
                                        variant="default"
                                        onClick={() => sendCommand('inference', 'reset')}
                                        disabled={!simData}
                                    >
                                        Reiniciar
                                    </Button>
                                </Tooltip>
                            </Group>
                        </Stack>
                    </div>

                    <Divider />

                    {/* Controles de Visualización */}
                    <div>
                        <Text size="sm" fw={500} mb="xs">Visualización</Text>
                        <Stack gap="sm">
                            <Select
                                label="Tipo de Visualización"
                                data={[
                                    { value: 'density', label: 'Densidad' },
                                    { value: 'phase', label: 'Fase' },
                                    { value: 'poincare', label: 'Poincaré' }
                                ]}
                                value={selectedViz}
                                onChange={(value) => value && setSelectedViz(value)}
                            />
                            
                            <div>
                                <Group justify="space-between" mb="xs">
                                    <Text size="xs" c="dimmed">Nivel de Zoom</Text>
                                    <Badge variant="light">{zoomLevel.toFixed(1)}x</Badge>
                                </Group>
                                <Group>
                                    <Slider
                                        value={zoomLevel}
                                        onChange={setZoomLevel}
                                        min={0.1}
                                        max={5.0}
                                        step={0.1}
                                        style={{ flex: 1 }}
                                    />
                                    <Button
                                        size="xs"
                                        variant="subtle"
                                        onClick={handleResetZoom}
                                    >
                                        Reset
                                    </Button>
                                </Group>
                            </div>

                            <Group>
                                <Switch
                                    label="Mostrar Poincaré"
                                    checked={showPoincare}
                                    onChange={(e) => setShowPoincare(e.currentTarget.checked)}
                                />
                                <Switch
                                    label="Mostrar Histograma"
                                    checked={showHistogram}
                                    onChange={(e) => setShowHistogram(e.currentTarget.checked)}
                                />
                            </Group>
                        </Stack>
                    </div>

                    <Divider />

                    {/* Controles de Entrenamiento */}
                    <div>
                        <Text size="sm" fw={500} mb="xs">Entrenamiento</Text>
                        <Stack gap="sm">
                            <Switch
                                label="Guardado Automático"
                                description="Guardar checkpoints automáticamente durante el entrenamiento"
                                checked={autoSave}
                                onChange={(e) => setAutoSave(e.currentTarget.checked)}
                            />
                            
                            {autoSave && (
                                <NumberInput
                                    label="Intervalo de Guardado (episodios)"
                                    description="Cada cuántos episodios se guarda un checkpoint"
                                    value={saveInterval}
                                    onChange={(val) => setSaveInterval(Number(val) || 50)}
                                    min={10}
                                    max={1000}
                                    step={10}
                                />
                            )}
                        </Stack>
                    </div>

                    <Divider />

                    {/* Controles de Análisis t-SNE */}
                    <div>
                        <Text size="sm" fw={500} mb="xs">Análisis t-SNE</Text>
                        <Stack gap="sm">
                            <Group justify="space-between" align="flex-end">
                                <NumberInput
                                    label="Intervalo de Snapshots"
                                    description="Cada cuántos pasos capturar un snapshot automáticamente"
                                    defaultValue={500}
                                    min={1}
                                    max={1000}
                                    step={10}
                                    style={{ flex: 1 }}
                                    onChange={(val) => {
                                        const interval = Number(val) || 500;
                                        sendCommand('simulation', 'set_snapshot_interval', { interval });
                                    }}
                                />
                                <Tooltip label="Capturar snapshot manual del estado actual">
                                    <Button
                                        leftSection={<IconCamera size={16} />}
                                        variant="light"
                                        color="blue"
                                        onClick={() => sendCommand('simulation', 'capture_snapshot', {})}
                                        disabled={!simData}
                                    >
                                        Capturar
                                    </Button>
                                </Tooltip>
                            </Group>
                            {snapshotCount > 0 && (
                                <Badge color="blue" variant="light" size="lg">
                                    {snapshotCount} snapshots capturados
                                </Badge>
                            )}
                        </Stack>
                    </div>

                    <Divider />

                    {/* Controles de Historia */}
                    <div>
                        <Text size="sm" fw={500} mb="xs">Historia de Simulación</Text>
                        <Stack gap="sm">
                            <Switch
                                label="Guardar historia"
                                description="Guarda frames para análisis posterior (puede consumir memoria)"
                                onChange={(e) => {
                                    sendCommand('simulation', 'enable_history', { enabled: e.currentTarget.checked });
                                }}
                            />
                            <Group grow>
                                <Button
                                    variant="light"
                                    size="sm"
                                    onClick={() => sendCommand('simulation', 'save_history', {})}
                                    disabled={!simData}
                                >
                                    Guardar Historia
                                </Button>
                                <Button
                                    variant="light"
                                    size="sm"
                                    color="red"
                                    onClick={() => sendCommand('simulation', 'clear_history', {})}
                                >
                                    Limpiar
                                </Button>
                            </Group>
                        </Stack>
                    </div>

                    <Divider />

                    {/* Controles de Sistema */}
                    <div>
                        <Text size="sm" fw={500} mb="xs">Sistema</Text>
                        <Stack gap="sm">
                            <NumberInput
                                label="Tamaño de Grid"
                                description="Tamaño de la cuadrícula de simulación"
                                value={gridSize}
                                onChange={(val) => handleGridSizeChange(Number(val) || 256)}
                                min={64}
                                max={512}
                                step={64}
                            />
                            
                            <Group grow>
                                <Button
                                    variant="light"
                                    size="sm"
                                    onClick={() => sendCommand('system', 'clear_cache', {})}
                                >
                                    Limpiar Cache
                                </Button>
                                <Button
                                    variant="light"
                                    size="sm"
                                    onClick={() => sendCommand('system', 'refresh_experiments', {})}
                                >
                                    Actualizar Lista
                                </Button>
                            </Group>
                        </Stack>
                    </div>
                </Stack>
            </Collapse>
        </Paper>
    );
}

