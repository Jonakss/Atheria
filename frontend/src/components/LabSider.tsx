// frontend/src/components/LabSider.tsx
import { useState } from 'react';
import { Box, Button, NavLink, ScrollArea, Select, Stack, Text, Group, NumberInput, Progress, Collapse, Divider } from '@mantine/core';
import { IconPlayerPlay, IconPlayerPause, IconRefresh, IconUpload, IconPlug } from '@tabler/icons-react';
import { useWebSocket } from '../hooks/useWebSocket';
import { modelOptions, vizOptions } from '../utils/vizOptions';
import classes from './LabSider.module.css';

export function LabSider() {
    const { 
        sendCommand, experimentsData, trainingStatus, trainingProgress, 
        inferenceStatus, connectionStatus, connect, selectedViz, setSelectedViz 
    } = useWebSocket();
    
    const [activeExperiment, setActiveExperiment] = useState<string | null>(null);
    
    // Estados para los inputs de entrenamiento
    const [selectedModel, setSelectedModel] = useState<string | null>('UNET');
    const [learningRate, setLearningRate] = useState(0.0001);
    const [gridSize, setGridSize] = useState(64);
    const [qcaSteps, setQcaSteps] = useState(16);
    const [dState, setDState] = useState(8);
    const [hiddenChannels, setHiddenChannels] = useState(32);
    const [episodesToAdd, setEpisodesToAdd] = useState(100);

    const handleCreateExperiment = () => {
        if (!selectedModel) return;
        const expName = `${selectedModel}-d${dState}-h${hiddenChannels}-g${gridSize}-lr${learningRate.toExponential(0)}`;
        sendCommand('experiment', 'create', { 
            EXPERIMENT_NAME: expName, MODEL_ARCHITECTURE: selectedModel, LR_RATE_M: learningRate,
            GRID_SIZE_TRAINING: gridSize, QCA_STEPS_TRAINING: qcaSteps,
            TOTAL_EPISODES: episodesToAdd,
            MODEL_PARAMS: { d_state: dState, hidden_channels: hiddenChannels, alpha: 0.9, beta: 0.85 }
        });
    };

    const handleContinueExperiment = () => {
        if (activeExperiment) sendCommand('experiment', 'continue', { 
            EXPERIMENT_NAME: activeExperiment,
            EPISODES_TO_ADD: episodesToAdd
        });
    };

    const handleLoadExperiment = () => { if (activeExperiment) sendCommand('inference', 'load_experiment', { experiment_name: activeExperiment }); };
    const handleResetSimulation = () => sendCommand('inference', 'reset');
    const togglePlayPause = () => {
        const command = inferenceStatus === 'running' ? 'pause' : 'play';
        sendCommand('inference', command);
    };

    const handleVizChange = (value: string | null) => {
        if (value) {
            setSelectedViz(value);
        }
    };

    const progressPercent = trainingProgress ? (trainingProgress.current_episode / trainingProgress.total_episodes) * 100 : 0;

    return (
        <Box className={classes.sider}>
            <Box className={classes.header}>
                <Text size="lg" fw={700}>Laboratorio Aetheria</Text>
                <Button onClick={connect} size="xs" variant="outline" leftSection={<IconPlug size={14}/>} loading={connectionStatus === 'connecting'}>
                    {connectionStatus === 'connected' ? 'Conectado' : 'Conectar'}
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
                        <Group grow>
                            <Button onClick={togglePlayPause} leftSection={inferenceStatus === 'running' ? <IconPlayerPause size={16} /> : <IconPlayerPlay size={16} />} color={inferenceStatus === 'running' ? 'yellow' : 'green'}>
                                {inferenceStatus === 'running' ? 'Pausar' : 'Iniciar'}
                            </Button>
                            <Button leftSection={<IconRefresh size={14} />} variant="default" onClick={handleResetSimulation}>
                                Reiniciar
                            </Button>
                        </Group>
                        <Select 
                            label="Mapa de Visualización" 
                            data={vizOptions} 
                            value={selectedViz} 
                            onChange={handleVizChange} 
                        />
                    </Stack>

                    <Stack gap="sm">
                        <Text size="xs" fw={700} className={classes.sectionTitle}>EXPERIMENTOS</Text>
                        <Button leftSection={<IconUpload size={14} />} variant="default" onClick={handleLoadExperiment} disabled={!activeExperiment}>
                            Cargar Modelo Seleccionado
                        </Button>
                        <ScrollArea style={{ height: 200, border: '1px solid var(--mantine-color-dark-4)', borderRadius: 'var(--mantine-radius-sm)' }}>
                            {experimentsData?.map((exp) => (
                                <NavLink key={exp.name} href={`#${exp.name}`} label={exp.name}
                                    description={`Modelo: ${exp.config.MODEL_ARCHITECTURE} | Total Eps: ${exp.config.TOTAL_EPISODES || 'N/A'}`}
                                    active={exp.name === activeExperiment}
                                    onClick={() => setActiveExperiment(exp.name)}
                                    className={classes.navLink}
                                />
                            ))}
                        </ScrollArea>
                    </Stack>

                    <Stack gap="sm">
                        <Text size="xs" fw={700} className={classes.sectionTitle}>ENTRENAMIENTO</Text>
                        <NumberInput label="Episodios a Añadir/Entrenar" value={episodesToAdd} onChange={(val) => setEpisodesToAdd(Number(val) || 0)} min={1} step={100} />
                        <Button leftSection={<IconPlayerPlay size={14} />} variant="default" onClick={handleContinueExperiment} disabled={!activeExperiment || trainingStatus === 'running'}>
                            Continuar Entrenamiento
                        </Button>
                        <Divider label="O crear uno nuevo" labelPosition="center" my="sm" />
                        <Select label="Arquitectura del Modelo" data={modelOptions} value={selectedModel} onChange={setSelectedModel} />
                        <Group grow>
                            <NumberInput label="Grid Size" value={gridSize} onChange={(val) => setGridSize(Number(val) || 0)} min={16} step={16} />
                            <NumberInput label="QCA Steps" value={qcaSteps} onChange={(val) => setQcaSteps(Number(val) || 0)} min={1} />
                        </Group>
                        <Group grow>
                            <NumberInput label="d_state" value={dState} onChange={(val) => setDState(Number(val) || 0)} min={2} />
                            <NumberInput label="Hidden Ch." value={hiddenChannels} onChange={(val) => setHiddenChannels(Number(val) || 0)} min={4} />
                        </Group>
                        <NumberInput label="Learning Rate" value={learningRate} onChange={(val) => setLearningRate(Number(val) || 0)} precision={5} step={0.00001} format="decimal" />
                        <Button onClick={handleCreateExperiment} loading={trainingStatus === 'running'} disabled={!selectedModel}>
                            Crear Nuevo Experimento
                        </Button>
                    </Stack>
                </Stack>
            </ScrollArea>
        </Box>
    );
}
