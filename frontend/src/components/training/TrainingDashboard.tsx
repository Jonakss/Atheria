// frontend/src/components/TrainingDashboard.tsx
import { Paper, Title, Group, Text, Progress, Badge, Stack, Box, Center } from '@mantine/core';
import { LineChart } from '@mantine/charts';
import { useWebSocket } from '../../hooks/useWebSocket';
import { useEffect, useState, useRef } from 'react';
import { IconTrendingDown, IconTrendingUp, IconClock } from '@tabler/icons-react';
import { motion } from 'framer-motion';
import { hoverLift, scaleIn } from '../../utils/animations';
import classes from '../ui/LogOverlay.module.css';

interface TrainingDataPoint {
    episode: number;
    loss: number;
    reward?: number;
    timestamp: number;
}

interface TrainingProgress {
    current_episode: number;
    total_episodes: number;
    avg_loss: number;
    avg_reward?: number;
}

export function TrainingDashboard() {
    const { trainingStatus, trainingProgress, activeExperiment } = useWebSocket();
    const [trainingHistory, setTrainingHistory] = useState<TrainingDataPoint[]>([]);
    const [chartReady, setChartReady] = useState(false);
    const chartContainerRef = useRef<HTMLDivElement>(null);
    
    // Verificar dimensiones del contenedor usando ResizeObserver
    useEffect(() => {
        if (!chartContainerRef.current) return;
        
        const resizeObserver = new ResizeObserver((entries) => {
            for (const entry of entries) {
                const { width, height } = entry.contentRect;
                if (width > 0 && height > 0) {
                    setChartReady(true);
                } else {
                    setChartReady(false);
                }
            }
        });
        
        resizeObserver.observe(chartContainerRef.current);
        
        // También verificar inmediatamente
        const checkNow = () => {
            if (chartContainerRef.current) {
                const { width, height } = chartContainerRef.current.getBoundingClientRect();
                if (width > 0 && height > 0) {
                    setChartReady(true);
                }
            }
        };
        const timer = setTimeout(checkNow, 50);
        
        return () => {
            resizeObserver.disconnect();
            clearTimeout(timer);
        };
    }, [trainingHistory.length]);

    // Actualizar historial cuando hay nuevo progreso
    useEffect(() => {
        if (trainingProgress) {
            setTrainingHistory(prev => {
                const newPoint: TrainingDataPoint = {
                    episode: trainingProgress.current_episode,
                    loss: trainingProgress.avg_loss,
                    reward: trainingProgress.avg_reward,
                    timestamp: Date.now()
                };
                
                // Evitar duplicados del mismo episodio
                const filtered = prev.filter(p => p.episode !== newPoint.episode);
                return [...filtered, newPoint].slice(-100); // Mantener últimos 100 puntos
            });
        }
    }, [trainingProgress]);

    // Limpiar historial cuando el entrenamiento termina
    useEffect(() => {
        if (trainingStatus === 'idle') {
            // No limpiar inmediatamente, mantener los datos por un tiempo
        }
    }, [trainingStatus]);

    if (trainingStatus !== 'running' && trainingHistory.length === 0) {
        return null; // No mostrar si no hay entrenamiento activo ni historial
    }

    const progressPercent = trainingProgress 
        ? (trainingProgress.current_episode / trainingProgress.total_episodes) * 100 
        : 0;

    const latestLoss = trainingHistory.length > 0 
        ? trainingHistory[trainingHistory.length - 1].loss 
        : trainingProgress?.avg_loss || 0;

    const lossTrend = trainingHistory.length >= 2
        ? trainingHistory[trainingHistory.length - 1].loss - trainingHistory[0].loss
        : 0;

    return (
        <motion.div
            variants={scaleIn}
            initial="hidden"
            animate="visible"
            style={{ position: 'fixed', right: 20, top: 20, zIndex: 10 }}
        >
            <motion.div variants={hoverLift} whileHover="hover" whileTap="rest">
                <Paper 
                    shadow="xl" 
                    p="md" 
                    className={classes.overlay}
                    style={{ 
                        width: 520, 
                        height: 480, 
                        backgroundColor: 'var(--mantine-color-dark-7)',
                        border: '1px solid var(--mantine-color-dark-4)',
                        transition: 'all 0.3s ease-in-out'
                    }}
                >
            <Stack gap="sm">
                <Group justify="space-between" align="center">
                    <Title order={5}>Dashboard de Entrenamiento</Title>
                    <Badge 
                        color={trainingStatus === 'running' ? 'green' : 'gray'} 
                        variant="light"
                    >
                        {trainingStatus === 'running' ? 'Activo' : 'Inactivo'}
                    </Badge>
                </Group>

                {activeExperiment && (
                    <Text size="sm" c="dimmed">
                        {activeExperiment}
                    </Text>
                )}

                {trainingProgress ? (
                    <>
                        <Box>
                            <Group justify="space-between" mb="xs">
                                <Text size="sm" fw={500}>Progreso</Text>
                                <Text size="sm" c="dimmed">
                                    {trainingProgress.current_episode} / {trainingProgress.total_episodes}
                                </Text>
                            </Group>
                            <Progress 
                                value={progressPercent} 
                                size="lg" 
                                radius="xl"
                                color={trainingStatus === 'running' ? 'blue' : 'gray'}
                            />
                        </Box>

                        <Group grow>
                            <Paper p="xs" withBorder>
                                <Stack gap={4} align="center">
                                    <Text size="xs" c="dimmed">Pérdida Actual</Text>
                                    <Group gap={4}>
                                        <Text size="lg" fw={700}>
                                            {latestLoss.toFixed(4)}
                                        </Text>
                                        {lossTrend < 0 ? (
                                            <IconTrendingDown size={16} color="green" />
                                        ) : lossTrend > 0 ? (
                                            <IconTrendingUp size={16} color="red" />
                                        ) : null}
                                    </Group>
                                </Stack>
                            </Paper>

                            {trainingProgress.avg_reward !== undefined && (
                                <Paper p="xs" withBorder>
                                    <Stack gap={4} align="center">
                                        <Text size="xs" c="dimmed">Recompensa</Text>
                                        <Text size="lg" fw={700}>
                                            {trainingProgress.avg_reward.toFixed(4)}
                                        </Text>
                                    </Stack>
                                </Paper>
                            )}

                            <Paper p="xs" withBorder>
                                <Stack gap={4} align="center">
                                    <Text size="xs" c="dimmed">Completado</Text>
                                    <Text size="lg" fw={700}>
                                        {progressPercent.toFixed(1)}%
                                    </Text>
                                </Stack>
                            </Paper>
                        </Group>

                        {trainingHistory.length > 1 && (
                            <Box ref={chartContainerRef} style={{ height: 200, minWidth: 300, minHeight: 200, width: '100%', display: 'block' }}>
                                <Text size="xs" c="dimmed" mb="xs">Evolución del Entrenamiento</Text>
                                {chartReady && trainingHistory.length > 0 && chartContainerRef.current ? (
                                    (() => {
                                        const rect = chartContainerRef.current?.getBoundingClientRect();
                                        if (rect && rect.width > 0 && rect.height > 0) {
                                            return (
                                <LineChart
                                    h={180}
                                                    w="100%"
                                    data={trainingHistory}
                                    dataKey="episode"
                                    series={[
                                        { 
                                            name: 'loss', 
                                            color: 'red.6',
                                            label: 'Pérdida'
                                        },
                                        ...(trainingHistory.some(p => p.reward !== undefined) ? [{
                                            name: 'reward',
                                            color: 'green.6',
                                            label: 'Recompensa'
                                        }] : [])
                                    ]}
                                    curveType="natural"
                                    withDots={false}
                                    withLegend={true}
                                    gridAxis="xy"
                                                    style={{ minWidth: 0, minHeight: 0 }}
                                                />
                                            );
                                        }
                                        return null;
                                    })()
                                ) : (
                                    <Center h={180}>
                                        <Text c="dimmed" size="xs">Cargando gráfico...</Text>
                                    </Center>
                                )}
                            </Box>
                        )}

                        {trainingHistory.length === 0 && (
                            <Box style={{ height: 200 }} p="md">
                                <Stack align="center" justify="center" h="100%">
                                    <IconClock size={32} color="gray" />
                                    <Text size="sm" c="dimmed" ta="center">
                                        Esperando datos de entrenamiento...
                                    </Text>
                                </Stack>
                            </Box>
                        )}
                    </>
                ) : (
                    <Box style={{ height: 200 }} p="md">
                        <Stack align="center" justify="center" h="100%">
                            <Text size="sm" c="dimmed" ta="center">
                                No hay entrenamiento activo
                            </Text>
                        </Stack>
                    </Box>
                )}
            </Stack>
                </Paper>
            </motion.div>
        </motion.div>
    );
}

