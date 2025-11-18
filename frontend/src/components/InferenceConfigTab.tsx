// frontend/src/components/InferenceConfigTab.tsx
import { useState, useEffect } from 'react';
import {
    Paper, Stack, Group, Text, NumberInput, Select, Button, Alert, Divider,
    Badge, Tooltip, Switch, ActionIcon, Card, Title, Box, TextInput
} from '@mantine/core';
import {
    IconSettings, IconInfoCircle, IconCheck, IconX, IconRefresh,
    IconBrain, IconBox, IconTrendingDown, IconAtom, IconServer, IconArrowsMaximize
} from '@tabler/icons-react';
import { useWebSocket } from '../hooks/useWebSocket';
import { resetServerConfig } from '../utils/serverConfig';

interface InferenceConfig {
    grid_size: number;
    initial_state_mode: string;
    gamma_decay: number;
    live_feed_enabled: boolean;
    data_compression_enabled: boolean;
    downsample_factor: number;
}

export function InferenceConfigTab() {
    const { sendCommand, activeExperiment, simData, serverConfig, updateServerConfig, connectionStatus, connect } = useWebSocket();
    
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
    
    // Estados para configuración del servidor (editables)
    const [serverHost, setServerHost] = useState(serverConfig.host);
    const [serverPort, setServerPort] = useState(serverConfig.port);
    const [serverProtocol, setServerProtocol] = useState<'ws' | 'wss'>(serverConfig.protocol);
    const [serverConfigChanged, setServerConfigChanged] = useState(false);
    
    // Sincronizar estados locales con la configuración cuando cambia
    useEffect(() => {
        setServerHost(serverConfig.host);
        setServerPort(serverConfig.port);
        setServerProtocol(serverConfig.protocol);
        setServerConfigChanged(false);
    }, [serverConfig]);
    
    // Información del experimento activo para escalado
    const { experimentsData } = useWebSocket();
    const currentExperiment = activeExperiment 
        ? experimentsData?.find(exp => exp.name === activeExperiment) 
        : null;
    const trainingGridSize = currentExperiment?.config?.GRID_SIZE_TRAINING || 64;

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

    // Detectar cambios en la configuración del servidor
    useEffect(() => {
        const changed = serverHost !== serverConfig.host || 
                       serverPort !== serverConfig.port || 
                       serverProtocol !== serverConfig.protocol;
        setServerConfigChanged(changed);
    }, [serverHost, serverPort, serverProtocol, serverConfig]);

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

            {/* Sección de Configuración del Servidor */}
            <Card withBorder p="md">
                <Stack gap="sm">
                    <Group justify="space-between" align="center">
                        <Group gap="xs">
                            <IconServer size={18} />
                            <Text fw={600} size="sm">Configuración del Servidor</Text>
                        </Group>
                        <Badge 
                            color={connectionStatus === 'connected' ? 'green' : connectionStatus === 'connecting' ? 'yellow' : 'red'}
                            variant="light"
                        >
                            {connectionStatus === 'connected' ? 'Conectado' : 
                             connectionStatus === 'connecting' ? 'Conectando...' : 
                             'Desconectado'}
                        </Badge>
                    </Group>
                    
                    <Group grow>
                        <Select
                            label="Protocolo"
                            value={serverProtocol}
                            onChange={(val) => val && setServerProtocol(val as 'ws' | 'wss')}
                            data={[
                                { value: 'ws', label: 'WS (No seguro)' },
                                { value: 'wss', label: 'WSS (Seguro/TLS)' }
                            ]}
                            description="ws para desarrollo local, wss para producción"
                        />
                        <TextInput
                            label="Host"
                            value={serverHost}
                            onChange={(e) => setServerHost(e.target.value)}
                            placeholder="localhost o IP del servidor"
                            description="Dirección del servidor"
                        />
                        <NumberInput
                            label="Puerto"
                            value={serverPort}
                            onChange={(val) => setServerPort(Number(val) || 8000)}
                            min={1}
                            max={65535}
                            description="Puerto del servidor WebSocket"
                        />
                    </Group>
                    
                    <Group>
                        <Text size="xs" c="dimmed">
                            URL actual: <strong>{serverProtocol}://{serverHost}:{serverPort}/ws</strong>
                        </Text>
                    </Group>
                    
                    <Group>
                        <Button
                            onClick={() => {
                                updateServerConfig({
                                    host: serverHost,
                                    port: serverPort,
                                    protocol: serverProtocol
                                });
                                setServerConfigChanged(false);
                                setTimeout(() => connect(), 500);
                            }}
                            disabled={!serverConfigChanged}
                            leftSection={<IconCheck size={16} />}
                            color="blue"
                        >
                            Aplicar y Reconectar
                        </Button>
                        <Button
                            onClick={() => {
                                resetServerConfig();
                                const defaultConfig = { host: 'localhost', port: 8000, protocol: 'ws' as const };
                                updateServerConfig(defaultConfig);
                                setServerHost(defaultConfig.host);
                                setServerPort(defaultConfig.port);
                                setServerProtocol(defaultConfig.protocol);
                                setServerConfigChanged(false);
                                setTimeout(() => connect(), 500);
                            }}
                            variant="light"
                            leftSection={<IconRefresh size={16} />}
                        >
                            Restaurar por Defecto
                        </Button>
                    </Group>
                    
                    <Alert icon={<IconInfoCircle size={16} />} color="blue" variant="light" p="xs">
                        <Text size="xs">
                            La configuración se guarda automáticamente en el navegador. 
                            Útil para conectar a servidores remotos (ej: Lightning AI).
                        </Text>
                    </Alert>
                </Stack>
            </Card>

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

            <Divider label="Escalado del Mundo (Reutilización de Entrenamiento)" labelPosition="center" />

            <Alert icon={<IconArrowsMaximize size={16} />} color="blue" variant="light">
                <Text size="sm" mb="xs">
                    <strong>Escalado de Inferencia:</strong> Los modelos convolucionales (UNet, etc.) pueden inferir en grids más grandes que los de entrenamiento.
                </Text>
                <Text size="xs" c="dimmed">
                    Entrenas en un grid pequeño (ej: 64x64) para eficiencia, y luego escalas a grids grandes (ej: 256x256, 512x512) para visualización detallada.
                </Text>
            </Alert>

            <Card withBorder p="md">
                <Stack gap="md">
                    {currentExperiment && (
                        <Paper p="xs" withBorder style={{ backgroundColor: 'var(--mantine-color-blue-0)' }}>
                            <Group justify="space-between">
                                <Box>
                                    <Text size="xs" fw={600}>Grid de Entrenamiento:</Text>
                                    <Text size="sm" c="blue">{trainingGridSize}x{trainingGridSize}</Text>
                                </Box>
                                <Box>
                                    <Text size="xs" fw={600}>Grid de Inferencia:</Text>
                                    <Text size="sm" c="green">{gridSize}x{gridSize}</Text>
                                </Box>
                                <Box>
                                    <Text size="xs" fw={600}>Factor de Escala:</Text>
                                    <Text size="sm" c="orange">{(gridSize / trainingGridSize).toFixed(2)}x</Text>
                                </Box>
                            </Group>
                        </Paper>
                    )}
                    
                    <NumberInput
                        label="Tamaño de Grid de Inferencia"
                        description={
                            currentExperiment 
                                ? `Escalar desde ${trainingGridSize}x${trainingGridSize} (entrenamiento) a ${gridSize}x${gridSize} (inferencia)`
                                : `Tamaño de la cuadrícula para inferencia (actual: ${currentGridSize}x${currentGridSize})`
                        }
                        value={gridSize}
                        onChange={(val) => setGridSize(Number(val) || 256)}
                        min={32}
                        max={1024}
                        step={32}
                        leftSection={<IconBox size={16} />}
                        disabled={!activeExperiment}
                    />
                    
                    {currentExperiment && gridSize !== trainingGridSize && (
                        <Alert icon={<IconInfoCircle size={16} />} color="green" variant="light">
                            <Text size="xs">
                                ✅ El modelo entrenado en {trainingGridSize}x{trainingGridSize} se escalará automáticamente a {gridSize}x{gridSize}.
                                Esto permite reutilizar el entrenamiento en grids más grandes sin reentrenar.
                            </Text>
                        </Alert>
                    )}
                    
                    {currentExperiment && gridSize === trainingGridSize && (
                        <Alert icon={<IconInfoCircle size={16} />} color="yellow" variant="light">
                            <Text size="xs">
                                ℹ️ Grid de inferencia igual al de entrenamiento. Puedes aumentar el tamaño para ver más detalle.
                            </Text>
                        </Alert>
                    )}
                    
                    <Tooltip label="Nota: Cambiar el tamaño de grid requiere recargar el experimento">
                        <Text size="xs" c="dimmed">⚠️ Requiere recargar el experimento para aplicar</Text>
                    </Tooltip>
                </Stack>
            </Card>

            <Divider label="Parámetros de Simulación" labelPosition="center" />

            <Card withBorder p="md">
                <Stack gap="md">
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
                        leftSection={<IconTrendingDown size={16} />}
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

            <Divider label="Configuración del Servidor" labelPosition="center" />

            <Card withBorder p="md">
                <Stack gap="md">
                    <Group>
                        <IconServer size={18} />
                        <Text size="sm" fw={600}>Información del Servidor</Text>
                    </Group>
                    
                    <Group grow>
                        <Box>
                            <Text size="xs" c="dimmed" mb={4}>Host</Text>
                            <Text size="sm" fw={500}>{serverHost}</Text>
                        </Box>
                        <Box>
                            <Text size="xs" c="dimmed" mb={4}>Puerto</Text>
                            <Text size="sm" fw={500}>{serverPort}</Text>
                        </Box>
                    </Group>
                    
                    <Alert icon={<IconInfoCircle size={16} />} color="gray" variant="light">
                        <Text size="xs">
                            La configuración del servidor (host, puerto) se establece en <code>src/config.py</code>.
                            Para cambiar estos valores, edita el archivo de configuración y reinicia el servidor.
                        </Text>
                    </Alert>
                    
                    <Paper p="xs" withBorder style={{ backgroundColor: 'var(--mantine-color-dark-6)' }}>
                        <Text size="xs" fw={600} mb="xs">Variables de Configuración:</Text>
                        <Stack gap={4}>
                            <Text size="xs" c="dimmed">
                                <strong>LAB_SERVER_HOST:</strong> {serverHost}
                            </Text>
                            <Text size="xs" c="dimmed">
                                <strong>LAB_SERVER_PORT:</strong> {serverPort}
                            </Text>
                            <Text size="xs" c="dimmed">
                                <strong>GRID_SIZE_TRAINING:</strong> {trainingGridSize} (del experimento activo)
                            </Text>
                            <Text size="xs" c="dimmed">
                                <strong>GRID_SIZE_INFERENCE:</strong> {gridSize} (configurado)
                            </Text>
                        </Stack>
                    </Paper>
                </Stack>
            </Card>

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

