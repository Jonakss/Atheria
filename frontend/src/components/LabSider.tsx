// frontend/src/components/LabSider.tsx
import { useState } from 'react';
import { Box, Button, NavLink, ScrollArea, Select, Stack, Text, Group, NumberInput, Progress, Collapse, Divider, Badge, Tooltip, Alert, Paper } from '@mantine/core';
import { IconPlayerPlay, IconPlayerPause, IconRefresh, IconUpload, IconPlug, IconCheck, IconX, IconAlertCircle, IconInfoCircle, IconTransfer } from '@tabler/icons-react';
import { useWebSocket } from '../hooks/useWebSocket';
import { modelOptions, vizOptions } from '../utils/vizOptions';
import { AdvancedControls } from './AdvancedControls';
import { ExperimentManager } from './ExperimentManager';
import { CheckpointManager } from './CheckpointManager';
import { ExperimentInfo } from './ExperimentInfo';
import classes from './LabSider.module.css';

export function LabSider() {
    const { 
        sendCommand, experimentsData, trainingStatus, trainingProgress, 
        inferenceStatus, connectionStatus, connect, selectedViz, setSelectedViz, simData,
        activeExperiment, setActiveExperiment
    } = useWebSocket();
    
    // Estados para los inputs de entrenamiento
    const [selectedModel, setSelectedModel] = useState<string | null>('UNET');
    const [learningRate, setLearningRate] = useState(0.0001);
    const [gridSize, setGridSize] = useState(64);
    const [qcaSteps, setQcaSteps] = useState(16);
    const [dState, setDState] = useState(8);
    const [hiddenChannels, setHiddenChannels] = useState(32);
    const [episodesToAdd, setEpisodesToAdd] = useState(100);
    const [transferFromExperiment, setTransferFromExperiment] = useState<string | null>(null);
    const [gammaDecay, setGammaDecay] = useState(0.01);  // Término Lindbladian (decaimiento)
    const [initialStateMode, setInitialStateMode] = useState('complex_noise');  // Modo de inicialización del estado

    const handleCreateExperiment = () => {
        if (!selectedModel) return;
        const expName = `${selectedModel}-d${dState}-h${hiddenChannels}-g${gridSize}-lr${learningRate.toExponential(0)}`;
        const args: Record<string, any> = { 
            EXPERIMENT_NAME: expName, 
            MODEL_ARCHITECTURE: selectedModel, 
            LR_RATE_M: learningRate,
            GRID_SIZE_TRAINING: gridSize, 
            QCA_STEPS_TRAINING: qcaSteps,
            TOTAL_EPISODES: episodesToAdd,
            MODEL_PARAMS: { d_state: dState, hidden_channels: hiddenChannels, alpha: 0.9, beta: 0.85 },
            GAMMA_DECAY: gammaDecay,  // Término Lindbladian: presión evolutiva hacia metabolismo (0.0 = cerrado, >0 = abierto)
            INITIAL_STATE_MODE_INFERENCE: initialStateMode  // Modo de inicialización del estado cuántico
        };
        
        // Si se seleccionó un experimento base, agregar transfer learning
        if (transferFromExperiment) {
            args.LOAD_FROM_EXPERIMENT = transferFromExperiment;
        }
        
        sendCommand('experiment', 'create', args);
    };

    const handleContinueExperiment = () => {
        if (activeExperiment) sendCommand('experiment', 'continue', { 
            EXPERIMENT_NAME: activeExperiment,
            EPISODES_TO_ADD: episodesToAdd
        });
    };

    const handleLoadExperiment = () => { 
        if (activeExperiment) {
            // Verificar que el experimento tenga checkpoint antes de cargar
            const exp = experimentsData?.find(e => e.name === activeExperiment);
            if (exp && !exp.has_checkpoint) {
                // No hacer nada, el botón ya está deshabilitado
                return;
            }
            sendCommand('inference', 'load_experiment', { experiment_name: activeExperiment }); 
        }
    };
    const handleResetSimulation = () => sendCommand('inference', 'reset');
    const togglePlayPause = () => {
        const command = inferenceStatus === 'running' ? 'pause' : 'play';
        sendCommand('inference', command);
    };

    const handleVizChange = (value: string | null) => {
        if (value) {
            setSelectedViz(value);
            sendCommand('simulation', 'set_viz', { viz_type: value });
        }
    };

    const progressPercent = trainingProgress ? (trainingProgress.current_episode / trainingProgress.total_episodes) * 100 : 0;
    
    // Encontrar el experimento activo desde experimentsData
    const currentExperiment = activeExperiment 
        ? experimentsData?.find(exp => exp.name === activeExperiment) 
        : null;

    return (
        <Box className={classes.sider}>
            <Box className={classes.header}>
                <Text size="lg" fw={700}>Laboratorio Aetheria</Text>
                <Button 
                    onClick={connect} 
                    size="xs" 
                    variant="outline" 
                    leftSection={<IconPlug size={14}/>} 
                    loading={connectionStatus === 'connecting'}
                    color={
                        connectionStatus === 'connected' ? 'green' :
                        connectionStatus === 'server_unavailable' ? 'red' :
                        'gray'
                    }
                >
                    {connectionStatus === 'connected' ? 'Conectado' : 
                     connectionStatus === 'server_unavailable' ? 'Servidor no disponible' :
                     connectionStatus === 'connecting' ? 'Conectando...' : 'Conectar'}
                </Button>
            </Box>

            <ScrollArea style={{ flex: 1, marginTop: 'var(--mantine-spacing-md)' }}>
                <Stack gap="xl">
                    <Collapse in={trainingStatus === 'running'}>
                        <Stack gap="xs">
                            <Text size="xs" fw={700} className={classes.sectionTitle}>PROGRESO DEL ENTRENAMIENTO</Text>
                            <Progress value={progressPercent} animated />
                            {trainingProgress && (
                                <Text size="xs" c="dimmed">
                                    Episodio {trainingProgress.current_episode}/{trainingProgress.total_episodes} | 
                                    Pérdida: {trainingProgress.avg_loss.toFixed(6)}
                                </Text>
                            )}
                            <Button color="red" variant="outline" size="xs" onClick={() => sendCommand('experiment', 'stop')}>
                                Detener Entrenamiento
                            </Button>
                        </Stack>
                    </Collapse>

                    <Stack gap="sm">
                        <Text size="xs" fw={700} className={classes.sectionTitle}>INFERENCIA</Text>
                        <Tooltip 
                            label={!simData && inferenceStatus !== 'running' ? 
                                   "Carga un modelo entrenado primero para iniciar la simulación" : 
                                   inferenceStatus === 'running' ? "Pausar simulación" : "Iniciar simulación"}
                            position="top"
                        >
                            <Group grow>
                                <Button 
                                    onClick={togglePlayPause} 
                                    leftSection={inferenceStatus === 'running' ? <IconPlayerPause size={16} /> : <IconPlayerPlay size={16} />} 
                                    color={inferenceStatus === 'running' ? 'yellow' : 'green'}
                                    disabled={!simData && inferenceStatus !== 'running'}
                                >
                                    {inferenceStatus === 'running' ? 'Pausar' : 'Iniciar'}
                                </Button>
                                <Button 
                                    leftSection={<IconRefresh size={14} />} 
                                    variant="default" 
                                    onClick={handleResetSimulation}
                                    disabled={!simData}
                                >
                                    Reiniciar
                                </Button>
                            </Group>
                        </Tooltip>
                        <Select 
                            label="Mapa de Visualización" 
                            data={vizOptions} 
                            value={selectedViz} 
                            onChange={handleVizChange} 
                        />
                    </Stack>

                    <Stack gap="sm">
                        <Text size="xs" fw={700} className={classes.sectionTitle}>EXPERIMENTO ACTIVO</Text>
                        
                        {/* Información del experimento activo */}
                        <ExperimentInfo />
                        
                        <Tooltip 
                            label={!activeExperiment ? "Selecciona un experimento primero" : 
                                   !currentExperiment?.has_checkpoint ? 
                                   "Este experimento no tiene checkpoints. Entrénalo primero." :
                                   inferenceStatus === 'running' ? "Detén la simulación primero para cargar otro modelo" :
                                   "Cargar modelo para inferencia"}
                            position="top"
                        >
                            <Button 
                                leftSection={<IconUpload size={14} />} 
                                variant="default" 
                                onClick={handleLoadExperiment} 
                                disabled={!activeExperiment || !currentExperiment?.has_checkpoint || inferenceStatus === 'running'}
                                color={currentExperiment?.has_checkpoint && inferenceStatus !== 'running' ? "blue" : "gray"}
                                fullWidth
                            >
                                Cargar Modelo para Inferencia
                            </Button>
                        </Tooltip>
                        
                        <Divider />
                        
                        <Text size="xs" fw={700} className={classes.sectionTitle}>GESTIÓN DE EXPERIMENTOS</Text>
                        
                        {/* Gestor de Experimentos Completo */}
                        <ExperimentManager />
                        
                        {/* Gestor de Checkpoints y Notas */}
                        <CheckpointManager />
                    </Stack>

                    <Stack gap="sm">
                        <Text size="xs" fw={700} className={classes.sectionTitle}>ENTRENAMIENTO</Text>
                        {activeExperiment && (
                            <Box p="xs" style={{ backgroundColor: 'var(--mantine-color-dark-6)', borderRadius: 'var(--mantine-radius-sm)' }}>
                                <Text size="xs" c="dimmed" mb={4}>Experimento activo:</Text>
                                <Text size="sm" fw={500}>{activeExperiment}</Text>
                            </Box>
                        )}
                        <NumberInput 
                            label="Episodios a Añadir/Entrenar" 
                            value={episodesToAdd} 
                            onChange={(val) => setEpisodesToAdd(Number(val) || 0)} 
                            min={1} 
                            step={100}
                            description={activeExperiment 
                                ? `Si el experimento tiene ${currentExperiment?.config?.TOTAL_EPISODES || 0} episodios, añadir ${episodesToAdd} más llegará a ${(currentExperiment?.config?.TOTAL_EPISODES || 0) + episodesToAdd} totales`
                                : `Número de episodios para entrenar. Si continúas un experimento, se añadirán a los existentes.`}
                        />
                        <Tooltip 
                            label={!activeExperiment ? "Selecciona un experimento primero" : 
                                   trainingStatus === 'running' ? "El entrenamiento ya está en curso" : 
                                   "Continuar entrenamiento del experimento seleccionado"}
                            position="top"
                        >
                            <Button 
                                leftSection={<IconPlayerPlay size={14} />} 
                                variant="default" 
                                onClick={handleContinueExperiment} 
                                disabled={!activeExperiment || trainingStatus === 'running'}
                                loading={trainingStatus === 'running'}
                            >
                                {trainingStatus === 'running' ? 'Entrenando...' : 'Continuar Entrenamiento'}
                            </Button>
                        </Tooltip>
                        <Divider label="O crear uno nuevo" labelPosition="center" my="sm" />
                        
                        {/* Selector de Transfer Learning con validación */}
                        <Box>
                            <Select 
                                label="Entrenamiento Progresivo (Opcional)" 
                                placeholder="Seleccionar experimento base"
                                data={experimentsData?.filter(exp => {
                                    // Solo mostrar experimentos que tienen checkpoint
                                    if (!exp.has_checkpoint) return false;
                                    // Evitar seleccionar el mismo experimento (si ya existe)
                                    return true;
                                }).map(exp => ({
                                    value: exp.name,
                                    label: `${exp.name} (${exp.config.MODEL_ARCHITECTURE})`
                                })) || []}
                                value={transferFromExperiment}
                                onChange={(value) => {
                                    // Validar que no sea circular
                                    if (value && experimentsData) {
                                        const selectedExp = experimentsData.find(e => e.name === value);
                                        if (selectedExp?.config?.LOAD_FROM_EXPERIMENT) {
                                            // Verificar cadena de dependencias
                                            let current = selectedExp.config.LOAD_FROM_EXPERIMENT;
                                            const chain = [value];
                                            while (current) {
                                                if (chain.includes(current)) {
                                                    alert(`⚠️ Dependencia circular detectada. No se puede usar '${value}' como base.`);
                                                    return;
                                                }
                                                chain.push(current);
                                                const exp = experimentsData.find(e => e.name === current);
                                                current = exp?.config?.LOAD_FROM_EXPERIMENT;
                                            }
                                        }
                                    }
                                    setTransferFromExperiment(value);
                                }}
                                clearable
                                leftSection={<IconTransfer size={16} />}
                                description="Carga pesos de un experimento entrenado para acelerar el entrenamiento."
                            />
                            {transferFromExperiment && (
                                <Alert 
                                    icon={<IconInfoCircle size={16} />} 
                                    color="blue" 
                                    variant="light" 
                                    mt="xs" 
                                    p="xs"
                                >
                                    <Text size="xs">
                                        Transfer desde: <strong>{transferFromExperiment}</strong>
                                    </Text>
                                    {experimentsData?.find(e => e.name === transferFromExperiment)?.config?.LOAD_FROM_EXPERIMENT && (
                                        <Text size="xs" c="dimmed" mt={4}>
                                            (Este experimento también usa transfer learning)
                                        </Text>
                                    )}
                                </Alert>
                            )}
                        </Box>
                        
                        <Select label="Arquitectura del Modelo" data={modelOptions} value={selectedModel} onChange={setSelectedModel} />
                        <Group grow>
                            <NumberInput label="Grid Size" value={gridSize} onChange={(val) => setGridSize(Number(val) || 0)} min={16} step={16} />
                            <NumberInput label="QCA Steps" value={qcaSteps} onChange={(val) => setQcaSteps(Number(val) || 0)} min={1} />
                        </Group>
                        <Group grow>
                            <NumberInput label="d_state" value={dState} onChange={(val) => setDState(Number(val) || 0)} min={2} />
                            <NumberInput label="Hidden Ch." value={hiddenChannels} onChange={(val) => setHiddenChannels(Number(val) || 0)} min={4} />
                        </Group>
                        <NumberInput 
                            label="Learning Rate" 
                            value={learningRate} 
                            onChange={(val) => setLearningRate(Number(val) || 0)} 
                            precision={5} 
                            step={0.00001} 
                            format="decimal" 
                        />
                        <NumberInput 
                            label="Gamma Decay (Lindbladian)" 
                            description="Término de decaimiento para sistemas abiertos (0.0 = cerrado, >0 = abierto)"
                            value={gammaDecay} 
                            onChange={(val) => setGammaDecay(Number(val) || 0)} 
                            precision={4} 
                            step={0.001} 
                            min={0} 
                            max={1}
                            format="decimal"
                        />
                        <Select
                            label="Modo de Inicialización"
                            description="Cómo se inicializa el estado cuántico en inferencia"
                            value={initialStateMode}
                            onChange={(val) => setInitialStateMode(val || 'complex_noise')}
                            data={[
                                { value: 'complex_noise', label: 'Ruido Complejo (default, más estable)' },
                                { value: 'random', label: 'Aleatorio Normalizado (más variado)' },
                                { value: 'zeros', label: 'Ceros (requiere activación externa)' }
                            ]}
                        />
                        <Button onClick={handleCreateExperiment} loading={trainingStatus === 'running'} disabled={!selectedModel}>
                            {transferFromExperiment ? 'Crear con Transfer Learning' : 'Crear Nuevo Experimento'}
                        </Button>
                    </Stack>
                </Stack>
            </ScrollArea>
        </Box>
    );
}
