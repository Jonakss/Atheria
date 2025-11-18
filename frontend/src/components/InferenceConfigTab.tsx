// frontend/src/components/InferenceConfigTab.tsx
import { useState, useEffect } from 'react';
import {
    Paper, Stack, Group, Text, NumberInput, Select, Button, Alert, Divider,
    Badge, Tooltip, Switch, ActionIcon, Card, Title, Box
} from '@mantine/core';
import {
    IconSettings, IconInfoCircle, IconCheck, IconX, IconRefresh,
    IconBrain, IconGrid, IconWave, IconAtom
} from '@tabler/icons-react';
import { useWebSocket } from '../hooks/useWebSocket';

interface InferenceConfig {
    grid_size: number;
    initial_state_mode: string;
    gamma_decay: number;
    live_feed_enabled: boolean;
    data_compression_enabled: boolean;
    downsample_factor: number;
}

export function InferenceConfigTab() {
    const { sendCommand, activeExperiment, simData } = useWebSocket();
    
    // Estados de configuración de inferencia
    const [gridSize, setGridSize] = useState(256);
    const [initialStateMode, setInitialStateMode] = useState('complex_noise');
    const [gammaDecay, setGammaDecay] = useState(0.01);
    
    // Estados de optimización
    const [liveFeedEnabled, setLiveFeedEnabled] = useState(true);
    const [compressionEnabled, setCompressionEnabled] = useState(true);
    const [downsampleFactor, setDownsampleFactor] = useState(1);
    
    // Estado de ROI
    const [roiEnabled, setRoiEnabled] = useState(false);
    const [roiX, setRoiX] = useState(0);
    const [roiY, setRoiY] = useState(0);
    const [roiWidth, setRoiWidth] = useState(256);
    const [roiHeight, setRoiHeight] = useState(256);
    
    const [hasChanges, setHasChanges] = useState(false);
    const [isApplying, setIsApplying] = useState(false);

    // Cargar configuración actual cuando cambia el experimento activo
    useEffect(() => {
        if (activeExperiment) {
            // Los valores se cargarán desde el servidor o usar defaults
            setHasChanges(false);
        }
    }, [activeExperiment]);

    const handleApplyConfig = async () => {
        setIsApplying(true);
        try {
            // Aplicar configuración de inferencia
            if (hasChanges) {
                // Parámetros que requieren recargar el experimento
                const needsReload = gridSize !== 256 || initialStateMode !== 'complex_noise' || gammaDecay !== 0.01;
                
                if (needsReload) {
                    sendCommand('inference', 'set_config', {
                        grid_size: gridSize,
                        initial_state_mode: initialStateMode,
                        gamma_decay: gammaDecay
                    });
                }
                
                // Parámetros que se pueden cambiar en caliente
                sendCommand('simulation', 'set_live_feed', { enabled: liveFeedEnabled });
                sendCommand('simulation', 'set_compression', { enabled: compressionEnabled });
                sendCommand('simulation', 'set_downsample', { factor: downsampleFactor });
                
                // Configurar ROI
                if (roiEnabled) {
                    sendCommand('simulation', 'set_roi', {
                        x: roiX,
                        y: roiY,
                        width: roiWidth,
                        height: roiHeight,
                        enabled: true
                    });
                } else {
                    sendCommand('simulation', 'set_roi', { enabled: false });
                }
            }
            
            setHasChanges(false);
        } catch (error) {
            console.error('Error aplicando configuración:', error);
        } finally {
            setIsApplying(false);
        }
    };

    const handleResetToDefaults = () => {
        setGridSize(256);
        setInitialStateMode('complex_noise');
        setGammaDecay(0.01);
        setLiveFeedEnabled(true);
        setCompressionEnabled(true);
        setDownsampleFactor(1);
        setRoiEnabled(false);
        setHasChanges(true);
    };

    // Detectar cambios
    useEffect(() => {
        // Marcar cambios cuando se modifica cualquier parámetro
        setHasChanges(true);
    }, [gridSize, initialStateMode, gammaDecay, liveFeedEnabled, compressionEnabled, downsampleFactor, roiEnabled, roiX, roiY, roiWidth, roiHeight]);

    const currentGridSize = simData?.map_data?.length || gridSize;

    return (
        <Stack gap="md" p="md">
            <Group justify="space-between" align="center">
                <Group gap="xs">
                    <IconSettings size={20} />
                    <Title order={4}>Configuración de Inferencia</Title>
                </Group>
                {hasChanges && (
                    <Badge color="orange" variant="light">Cambios pendientes</Badge>
                )}
            </Group>

            <Alert icon={<IconInfoCircle size={16} />} color="blue" variant="light">
                <Text size="sm">
                    Configura los parámetros de inferencia y simulación. 
                    {!activeExperiment && ' Carga un experimento primero para aplicar la configuración.'}
                </Text>
            </Alert>

            {!activeExperiment && (
                <Alert icon={<IconX size={16} />} color="yellow" variant="light">
                    <Text size="sm">No hay experimento cargado. Carga un experimento desde el panel lateral para configurar la inferencia.</Text>
                </Alert>
            )}

            <Divider label="Parámetros de Simulación" labelPosition="center" />

            <Card withBorder p="md">
                <Stack gap="md">
                    <NumberInput
                        label="Tamaño de Grid"
                        description={`Tamaño de la cuadrícula para inferencia (actual: ${currentGridSize}x${currentGridSize})`}
                        value={gridSize}
                        onChange={(val) => setGridSize(Number(val) || 256)}
                        min={32}
                        max={1024}
                        step={32}
                        leftSection={<IconGrid size={16} />}
                        disabled={!activeExperiment}
                    />
                    <Tooltip label="Nota: Cambiar el tamaño de grid requiere recargar el experimento">
                        <Text size="xs" c="dimmed">⚠️ Requiere recargar el experimento para aplicar</Text>
                    </Tooltip>

                    <Select
                        label="Modo de Inicialización"
                        description="Cómo se inicializa el estado cuántico al comenzar la simulación"
                        value={initialStateMode}
                        onChange={(val) => setInitialStateMode(val || 'complex_noise')}
                        data={[
                            { value: 'complex_noise', label: 'Ruido Complejo (default, más estable)' },
                            { value: 'random', label: 'Aleatorio Normalizado (más variado)' },
                            { value: 'zeros', label: 'Ceros (requiere activación externa)' }
                        ]}
                        leftSection={<IconAtom size={16} />}
                        disabled={!activeExperiment}
                    />
                    <Tooltip label="Nota: Cambiar el modo de inicialización requiere recargar el experimento">
                        <Text size="xs" c="dimmed">⚠️ Requiere recargar el experimento para aplicar</Text>
                    </Tooltip>

                    <NumberInput
                        label="Gamma Decay (Lindbladian)"
                        description="Término de decaimiento para sistemas abiertos (0.0 = cerrado, >0 = abierto)"
                        value={gammaDecay}
                        onChange={(val) => setGammaDecay(Number(val) || 0.01)}
                        precision={4}
                        step={0.001}
                        min={0}
                        max={1}
                        leftSection={<IconWave size={16} />}
                        disabled={!activeExperiment}
                    />
                    <Tooltip label="Nota: Cambiar Gamma Decay requiere recargar el experimento">
                        <Text size="xs" c="dimmed">⚠️ Requiere recargar el experimento para aplicar</Text>
                    </Tooltip>
                </Stack>
            </Card>

            <Divider label="Optimización de Rendimiento" labelPosition="center" />

            <Card withBorder p="md">
                <Stack gap="md">
                    <Switch
                        label="Live Feed"
                        description="Enviar datos de visualización en tiempo real. Desactivar para ahorrar recursos."
                        checked={liveFeedEnabled}
                        onChange={(e) => setLiveFeedEnabled(e.currentTarget.checked)}
                        disabled={!activeExperiment}
                    />

                    <Switch
                        label="Compresión de Datos"
                        description="Comprimir datos antes de enviar (reduce tamaño, puede aumentar CPU)"
                        checked={compressionEnabled}
                        onChange={(e) => setCompressionEnabled(e.currentTarget.checked)}
                        disabled={!activeExperiment}
                    />

                    <NumberInput
                        label="Factor de Downsampling"
                        description="Reducir resolución antes de enviar (1 = sin reducción, 2 = mitad, etc.)"
                        value={downsampleFactor}
                        onChange={(val) => setDownsampleFactor(Number(val) || 1)}
                        min={1}
                        max={8}
                        step={1}
                        disabled={!activeExperiment}
                    />
                </Stack>
            </Card>

            <Divider label="Region of Interest (ROI)" labelPosition="center" />

            <Card withBorder p="md">
                <Stack gap="md">
                    <Switch
                        label="Habilitar ROI"
                        description="Visualizar solo una región específica del grid (reduce transferencia de datos)"
                        checked={roiEnabled}
                        onChange={(e) => setRoiEnabled(e.currentTarget.checked)}
                        disabled={!activeExperiment}
                    />

                    {roiEnabled && (
                        <Group grow>
                            <NumberInput
                                label="X"
                                value={roiX}
                                onChange={(val) => setRoiX(Number(val) || 0)}
                                min={0}
                                max={currentGridSize - 1}
                                disabled={!activeExperiment}
                            />
                            <NumberInput
                                label="Y"
                                value={roiY}
                                onChange={(val) => setRoiY(Number(val) || 0)}
                                min={0}
                                max={currentGridSize - 1}
                                disabled={!activeExperiment}
                            />
                            <NumberInput
                                label="Ancho"
                                value={roiWidth}
                                onChange={(val) => setRoiWidth(Number(val) || 256)}
                                min={32}
                                max={currentGridSize}
                                disabled={!activeExperiment}
                            />
                            <NumberInput
                                label="Alto"
                                value={roiHeight}
                                onChange={(val) => setRoiHeight(Number(val) || 256)}
                                min={32}
                                max={currentGridSize}
                                disabled={!activeExperiment}
                            />
                        </Group>
                    )}
                </Stack>
            </Card>

            <Group justify="flex-end" mt="md">
                <Button
                    variant="outline"
                    leftSection={<IconRefresh size={16} />}
                    onClick={handleResetToDefaults}
                    disabled={!activeExperiment}
                >
                    Restaurar Defaults
                </Button>
                <Button
                    leftSection={<IconCheck size={16} />}
                    onClick={handleApplyConfig}
                    loading={isApplying}
                    disabled={!activeExperiment || !hasChanges}
                >
                    Aplicar Configuración
                </Button>
            </Group>

            {activeExperiment && (
                <Alert icon={<IconBrain size={16} />} color="green" variant="light">
                    <Text size="sm">
                        <strong>Experimento activo:</strong> {activeExperiment}
                    </Text>
                    <Text size="xs" c="dimmed" mt="xs">
                        Algunos parámetros (grid size, initialization mode, gamma decay) requieren recargar el experimento para aplicar cambios.
                    </Text>
                </Alert>
            )}
        </Stack>
    );
}

