// frontend/src/components/LogOverlay.tsx
import { Paper, ScrollArea, Text, Box, Title, Center, Stack, Tabs, Collapse, Button, Group } from '@mantine/core';
import { useWebSocket } from '../hooks/useWebSocket';
import classes from './LogOverlay.module.css';
import { useEffect, useRef, useState } from 'react';
import { IconInfoCircle, IconChevronUp, IconChevronDown, IconFileText, IconChartBar } from '@tabler/icons-react';
import { useDisclosure } from '@mantine/hooks';

export function LogOverlay() {
    const { allLogs, trainingStatus, inferenceStatus, trainingProgress, experimentsData, activeExperiment } = useWebSocket();
    const viewport = useRef<HTMLDivElement>(null);
    const [opened, { toggle }] = useDisclosure(false);
    const [activeTab, setActiveTab] = useState<string | null>('logs');

    // Scroll automático al final
    useEffect(() => {
        if (viewport.current && opened) {
            viewport.current.scrollTo({ top: viewport.current.scrollHeight, behavior: 'smooth' });
        }
    }, [allLogs, opened]);

    const hasLogs = allLogs && allLogs.length > 0;
    
    // Filtrar logs por tipo
    const trainingLogs = allLogs?.filter(log => log.includes('[Entrenamiento]') || log.includes('[Error]')) || [];
    const simulationLogs = allLogs?.filter(log => log.includes('[Simulación]')) || [];
    const allOtherLogs = allLogs?.filter(log => 
        !log.includes('[Entrenamiento]') && 
        !log.includes('[Error]') && 
        !log.includes('[Simulación]')
    ) || [];

    // Obtener información del experimento activo
    const currentExperiment = experimentsData?.find(exp => exp.name === activeExperiment);
    const experimentInfo = currentExperiment ? {
        name: currentExperiment.name,
        architecture: currentExperiment.config?.MODEL_ARCHITECTURE || 'N/A',
        lr: currentExperiment.config?.LR_RATE_M || 'N/A',
        gridSize: currentExperiment.config?.GRID_SIZE_TRAINING || 'N/A',
        qcaSteps: currentExperiment.config?.QCA_STEPS_TRAINING || 'N/A',
        totalEpisodes: currentExperiment.config?.TOTAL_EPISODES || 'N/A',
        hasCheckpoint: currentExperiment.has_checkpoint || false,
        checkpointPath: currentExperiment.checkpoint_path || null,
        modelParams: currentExperiment.config?.MODEL_PARAMS || {}
    } : null;

    return (
        <Box 
            style={{
                position: 'fixed',
                bottom: 0,
                left: 0,
                right: 0,
                zIndex: 1000,
                backgroundColor: 'var(--mantine-color-dark-7)',
                borderTop: '1px solid var(--mantine-color-dark-4)',
                boxShadow: '0 -2px 10px rgba(0, 0, 0, 0.3)'
            }}
        >
            <Group justify="space-between" p="xs" style={{ borderBottom: opened ? '1px solid var(--mantine-color-dark-4)' : 'none' }}>
                <Group gap="xs">
                    <Button
                        variant="subtle"
                        size="xs"
                        onClick={toggle}
                        leftSection={opened ? <IconChevronUp size={16} /> : <IconChevronDown size={16} />}
                    >
                        {opened ? 'Ocultar' : 'Mostrar'} Logs
                    </Button>
                    {hasLogs && (
                        <Text size="xs" c="dimmed">
                            {allLogs.length} mensajes
                        </Text>
                    )}
                </Group>
            </Group>
            
            <Collapse in={opened}>
                <Box style={{ height: '300px', display: 'flex', flexDirection: 'column' }}>
                    <Tabs value={activeTab} onChange={setActiveTab} style={{ flex: 1, display: 'flex', flexDirection: 'column' }}>
                        <Tabs.List>
                            <Tabs.Tab value="logs" leftSection={<IconFileText size={14} />}>
                                Todos los Logs ({allLogs?.length || 0})
                            </Tabs.Tab>
                            <Tabs.Tab value="training" leftSection={<IconChartBar size={14} />}>
                                Entrenamiento ({trainingLogs.length})
                            </Tabs.Tab>
                            <Tabs.Tab value="simulation" leftSection={<IconInfoCircle size={14} />}>
                                Simulación ({simulationLogs.length})
                            </Tabs.Tab>
                            <Tabs.Tab value="info" leftSection={<IconInfoCircle size={14} />}>
                                Información
                            </Tabs.Tab>
                        </Tabs.List>

                        <Tabs.Panel value="logs" style={{ flex: 1, overflow: 'hidden' }}>
                            <Paper withBorder className={classes.logPaper} style={{ height: '100%' }}>
                                <ScrollArea className={classes.scrollArea} viewportRef={viewport} style={{ height: '100%' }}>
                                    {!hasLogs ? (
                                        <Center h="100%">
                                            <Stack align="center" gap="xs">
                                                <IconInfoCircle size={32} color="gray" />
                                                <Text c="dimmed" size="sm">
                                                    {(trainingStatus === 'running' || inferenceStatus === 'running')
                                                        ? "Esperando logs..." 
                                                        : "Inicie un entrenamiento o carga un modelo para ver los logs."}
                                                </Text>
                                            </Stack>
                                        </Center>
                                    ) : (
                                        allLogs.map((log, index) => (
                                            <Text key={index} component="pre" size="xs" className={classes.logLine}>
                                                {log}
                                            </Text>
                                        ))
                                    )}
                                </ScrollArea>
                            </Paper>
                        </Tabs.Panel>

                        <Tabs.Panel value="training" style={{ flex: 1, overflow: 'hidden' }}>
                            <Paper withBorder className={classes.logPaper} style={{ height: '100%' }}>
                                <ScrollArea className={classes.scrollArea} viewportRef={viewport} style={{ height: '100%' }}>
                                    {trainingLogs.length === 0 ? (
                                        <Center h="100%">
                                            <Text c="dimmed" size="sm">No hay logs de entrenamiento</Text>
                                        </Center>
                                    ) : (
                                        trainingLogs.map((log, index) => (
                                            <Text key={index} component="pre" size="xs" className={classes.logLine}>
                                                {log}
                                            </Text>
                                        ))
                                    )}
                                    {trainingProgress && (
                                        <Box p="sm" style={{ borderTop: '1px solid var(--mantine-color-dark-4)', marginTop: 'auto' }}>
                                            <Text size="sm" fw={500}>Progreso Actual:</Text>
                                            <Text size="xs" c="dimmed">
                                                Episodio {trainingProgress.current_episode}/{trainingProgress.total_episodes} | 
                                                Pérdida: {trainingProgress.avg_loss.toFixed(6)}
                                            </Text>
                                        </Box>
                                    )}
                                </ScrollArea>
                            </Paper>
                        </Tabs.Panel>

                        <Tabs.Panel value="simulation" style={{ flex: 1, overflow: 'hidden' }}>
                            <Paper withBorder className={classes.logPaper} style={{ height: '100%' }}>
                                <ScrollArea className={classes.scrollArea} viewportRef={viewport} style={{ height: '100%' }}>
                                    {simulationLogs.length === 0 ? (
                                        <Center h="100%">
                                            <Text c="dimmed" size="sm">No hay logs de simulación</Text>
                                        </Center>
                                    ) : (
                                        simulationLogs.map((log, index) => (
                                            <Text key={index} component="pre" size="xs" className={classes.logLine}>
                                                {log}
                                            </Text>
                                        ))
                                    )}
                                </ScrollArea>
                            </Paper>
                        </Tabs.Panel>

                        <Tabs.Panel value="info" style={{ flex: 1, overflow: 'hidden' }}>
                            <Paper withBorder className={classes.logPaper} style={{ height: '100%', padding: 'md' }}>
                                <ScrollArea style={{ height: '100%' }}>
                                    {experimentInfo ? (
                                        <Stack gap="sm">
                                            <Title order={5}>Información del Experiment</Title>
                                            <Box>
                                                <Text size="sm" fw={500}>Nombre:</Text>
                                                <Text size="xs" c="dimmed">{experimentInfo.name}</Text>
                                            </Box>
                                            <Box>
                                                <Text size="sm" fw={500}>Arquitectura:</Text>
                                                <Text size="xs" c="dimmed">{experimentInfo.architecture}</Text>
                                            </Box>
                                            <Box>
                                                <Text size="sm" fw={500}>Learning Rate:</Text>
                                                <Text size="xs" c="dimmed">{experimentInfo.lr}</Text>
                                            </Box>
                                            <Box>
                                                <Text size="sm" fw={500}>Grid Size (Training):</Text>
                                                <Text size="xs" c="dimmed">{experimentInfo.gridSize}</Text>
                                            </Box>
                                            <Box>
                                                <Text size="sm" fw={500}>QCA Steps:</Text>
                                                <Text size="xs" c="dimmed">{experimentInfo.qcaSteps}</Text>
                                            </Box>
                                            <Box>
                                                <Text size="sm" fw={500}>Total Episodios:</Text>
                                                <Text size="xs" c="dimmed">{experimentInfo.totalEpisodes}</Text>
                                            </Box>
                                            <Box>
                                                <Text size="sm" fw={500}>Estado:</Text>
                                                <Text size="xs" c={experimentInfo.hasCheckpoint ? "green" : "orange"}>
                                                    {experimentInfo.hasCheckpoint ? "✓ Tiene checkpoints" : "⚠ Sin entrenar"}
                                                </Text>
                                            </Box>
                                            {Object.keys(experimentInfo.modelParams).length > 0 && (
                                                <Box>
                                                    <Text size="sm" fw={500}>Parámetros del Modelo:</Text>
                                                    <Text size="xs" c="dimmed" component="pre" style={{ marginTop: '0.5rem' }}>
                                                        {JSON.stringify(experimentInfo.modelParams, null, 2)}
                                                    </Text>
                                                </Box>
                                            )}
                                        </Stack>
                                    ) : (
                                        <Center h="100%">
                                            <Stack align="center" gap="xs">
                                                <IconInfoCircle size={32} color="gray" />
                                                <Text c="dimmed" size="sm">
                                                    Selecciona un experimento para ver su información
                                                </Text>
                                            </Stack>
                                        </Center>
                                    )}
                                </ScrollArea>
                            </Paper>
                        </Tabs.Panel>
                    </Tabs>
                </Box>
            </Collapse>
        </Box>
    );
}
