// frontend/src/components/LogOverlay.tsx
import { ActionIcon, Box, Button, Center, Collapse, Group, Paper, ScrollArea, Stack, Tabs, Text, TextInput, Title } from '@mantine/core';
import { useDisclosure } from '@mantine/hooks';
import { IconChartBar, IconChevronDown, IconChevronUp, IconFileText, IconGripVertical, IconInfoCircle, IconMinus, IconSend, IconTerminal } from '@tabler/icons-react';
import { useCallback, useEffect, useRef, useState } from 'react';
import { useWebSocket } from '../../hooks/useWebSocket';
import classes from './LogOverlay.module.css';

export function LogOverlay() {
    const { allLogs, trainingStatus, inferenceStatus, trainingProgress, experimentsData, activeExperiment, sendCommand } = useWebSocket();
    const viewport = useRef<HTMLDivElement>(null);
    const [opened, { toggle }] = useDisclosure(false);
    const [minimized, setMinimized] = useState(false);
    const [activeTab, setActiveTab] = useState<string | null>('logs');
    const [height, setHeight] = useState(300);
    const [isResizing, setIsResizing] = useState(false);
    const containerRef = useRef<HTMLDivElement>(null);
    const [commandInput, setCommandInput] = useState('');

    const handleCommandSubmit = (e: React.FormEvent) => {
        e.preventDefault();
        if (!commandInput.trim()) return;

        // Parse command: "scope command arg1=val1 arg2=val2" or just "command arg1=val1" (default scope=system)
        const parts = commandInput.trim().split(/\s+/);
        let scope = 'system';
        let command = parts[0];
        let argsStartIndex = 1;

        // Simple heuristic: if first part is a known scope, use it
        const knownScopes = ['system', 'inference', 'training', 'simulation'];
        if (knownScopes.includes(parts[0]) && parts.length > 1) {
            scope = parts[0];
            command = parts[1];
            argsStartIndex = 2;
        }

        const args: Record<string, any> = {};
        for (let i = argsStartIndex; i < parts.length; i++) {
            const argPart = parts[i];
            if (argPart.includes('=')) {
                const [key, val] = argPart.split('=');
                // Try to parse number or boolean
                if (val === 'true') args[key] = true;
                else if (val === 'false') args[key] = false;
                else if (!isNaN(Number(val))) args[key] = Number(val);
                else args[key] = val;
            } else {
                // Positional args or flags treated as true
                args[argPart] = true;
            }
        }

        sendCommand(scope, command, args);
        setCommandInput('');
    };

    // Scroll automático al final
    useEffect(() => {
        if (viewport.current && opened && !minimized) {
            viewport.current.scrollTo({ top: viewport.current.scrollHeight, behavior: 'smooth' });
        }
    }, [allLogs, opened, minimized]);

    // Manejar redimensionamiento
    const handleMouseDown = useCallback((e: React.MouseEvent) => {
        e.preventDefault();
        setIsResizing(true);
    }, []);

    useEffect(() => {
        const handleMouseMove = (e: MouseEvent) => {
            if (!isResizing) return;
            const newHeight = window.innerHeight - e.clientY;
            // Limitar altura entre 100px y 80% de la ventana
            const minHeight = 100;
            const maxHeight = window.innerHeight * 0.8;
            setHeight(Math.max(minHeight, Math.min(maxHeight, newHeight)));
        };

        const handleMouseUp = () => {
            setIsResizing(false);
        };

        if (isResizing) {
            document.addEventListener('mousemove', handleMouseMove);
            document.addEventListener('mouseup', handleMouseUp);
            return () => {
                document.removeEventListener('mousemove', handleMouseMove);
                document.removeEventListener('mouseup', handleMouseUp);
            };
        }
    }, [isResizing]);

    // Si está minimizado, colapsar también
    useEffect(() => {
        if (minimized && opened) {
            // No hacer nada, solo mantener minimizado
        }
    }, [minimized, opened]);

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
            ref={containerRef}
            style={{
                position: 'fixed',
                bottom: minimized ? 0 : 0,
                left: 0,
                right: 0,
                zIndex: 1000,
                backgroundColor: 'var(--mantine-color-dark-7)',
                borderTop: '1px solid var(--mantine-color-dark-4)',
                boxShadow: '0 -2px 10px rgba(0, 0, 0, 0.3)',
                transition: minimized ? 'none' : 'height 0.2s ease'
            }}
        >
            {/* Handle de redimensionamiento */}
            {opened && !minimized && (
                <Box
                    onMouseDown={handleMouseDown}
                    style={{
                        height: '4px',
                        cursor: 'ns-resize',
                        backgroundColor: isResizing ? 'var(--mantine-color-blue-6)' : 'var(--mantine-color-dark-4)',
                        position: 'relative',
                        zIndex: 1001,
                        transition: 'background-color 0.2s'
                    }}
                >
                    <Center h="100%">
                        <IconGripVertical size={12} style={{ opacity: 0.5 }} />
                    </Center>
                </Box>
            )}
            
            <Group justify="space-between" p="xs" style={{ borderBottom: opened && !minimized ? '1px solid var(--mantine-color-dark-4)' : 'none' }}>
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
                <Group gap="xs">
                    {opened && (
                        <ActionIcon
                            variant="subtle"
                            size="sm"
                            onClick={() => {
                                setMinimized(!minimized);
                                if (minimized) {
                                    // Al expandir, restaurar altura
                                    setHeight(300);
                                }
                            }}
                            title={minimized ? "Expandir" : "Minimizar completamente"}
                        >
                            {minimized ? <IconChevronUp size={16} /> : <IconMinus size={16} />}
                        </ActionIcon>
                    )}
                </Group>
            </Group>
            
            <Collapse in={opened && !minimized}>
                <Box style={{ height: `${height}px`, display: 'flex', flexDirection: 'column', maxHeight: '80vh' }}>
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
                    
                    {/* Command Console Input */}
                    <Box p="xs" style={{ borderTop: '1px solid var(--mantine-color-dark-4)', backgroundColor: 'var(--mantine-color-dark-6)' }}>
                        <form onSubmit={handleCommandSubmit}>
                            <TextInput
                                placeholder="Enter command (e.g., 'system toggle_logs', 'inference play')"
                                value={commandInput}
                                onChange={(e) => setCommandInput(e.currentTarget.value)}
                                size="xs"
                                leftSection={<IconTerminal size={14} />}
                                rightSection={
                                    <ActionIcon size="xs" variant="subtle" type="submit" disabled={!commandInput.trim()}>
                                        <IconSend size={12} />
                                    </ActionIcon>
                                }
                                rightSectionWidth={30}
                            />
                        </form>
                    </Box>
                </Box>
            </Collapse>
        </Box>
    );
}
