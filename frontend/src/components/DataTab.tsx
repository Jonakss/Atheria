// frontend/src/components/DataTab.tsx
import { useState, useEffect, useMemo, useRef } from 'react';
import { 
    Grid, Paper, Title, Text, Group, Badge, Stack, Box, 
    Card, Progress, RingProgress, Center, Table, ScrollArea 
} from '@mantine/core';
import { LineChart, AreaChart, BarChart } from '@mantine/charts';
import { useWebSocket } from '../hooks/useWebSocket';
import { 
    IconTrendingUp, IconTrendingDown, IconActivity, 
    IconChartLine, IconDatabase, IconInfoCircle 
} from '@tabler/icons-react';

interface SimulationStats {
    step: number;
    density: { mean: number; std: number; min: number; max: number };
    phase: { mean: number; std: number; min: number; max: number };
    timestamp: number;
}

export function DataTab() {
    const { simData, trainingProgress, activeExperiment, experimentsData } = useWebSocket();
    const [simulationHistory, setSimulationHistory] = useState<SimulationStats[]>([]);
    const [selectedMetric, setSelectedMetric] = useState<string>('density');
    const [chartsReady, setChartsReady] = useState(false);
    const chartContainerRef = useRef<HTMLDivElement>(null);
    
    // Verificar que el contenedor tenga dimensiones antes de renderizar charts usando ResizeObserver
    useEffect(() => {
        if (!chartContainerRef.current) return;
        
        const resizeObserver = new ResizeObserver((entries) => {
            for (const entry of entries) {
                const { width, height } = entry.contentRect;
                if (width > 0 && height > 0) {
                    setChartsReady(true);
                } else {
                    setChartsReady(false);
                }
            }
        });
        
        resizeObserver.observe(chartContainerRef.current);
        
        // También verificar inmediatamente
        const checkNow = () => {
            if (chartContainerRef.current) {
                const { width, height } = chartContainerRef.current.getBoundingClientRect();
                if (width > 0 && height > 0) {
                    setChartsReady(true);
                }
            }
        };
        const timer = setTimeout(checkNow, 50);
        
        return () => {
            resizeObserver.disconnect();
            clearTimeout(timer);
        };
    }, []); // Removed simulationHistory.length dependency to prevent infinite loops

    // Calcular estadísticas del estado actual
    const currentStats = useMemo(() => {
        if (!simData?.map_data || !Array.isArray(simData.map_data)) return null;

        const mapData = simData.map_data;
        // Aplanar el array 2D
        const flatData: number[] = [];
        for (const row of mapData) {
            if (Array.isArray(row)) {
                flatData.push(...row);
            }
        }
        
        if (flatData.length === 0) return null;
        
        const mean = flatData.reduce((a, b) => a + b, 0) / flatData.length;
        const variance = flatData.reduce((sum, val) => sum + Math.pow(val - mean, 2), 0) / flatData.length;
        const std = Math.sqrt(variance);
        const min = Math.min(...flatData);
        const max = Math.max(...flatData);

        return {
            mean: mean || 0,
            std: std || 0,
            min: min || 0,
            max: max || 0,
            variance: variance || 0
        };
    }, [simData]);

    // Actualizar historial de simulación
    useEffect(() => {
        if (simData && currentStats) {
            setSimulationHistory(prev => {
                const newPoint: SimulationStats = {
                    step: simData.step || 0,
                    density: currentStats,
                    phase: { mean: 0, std: 0, min: 0, max: 0 }, // TODO: calcular de hist_data
                    timestamp: Date.now()
                };
                
                // Evitar duplicados
                const filtered = prev.filter(p => p.step !== newPoint.step);
                return [...filtered, newPoint].slice(-200); // Mantener últimos 200 puntos
            });
        }
    }, [simData, currentStats]);

    // Encontrar experimento activo
    const currentExperiment = activeExperiment 
        ? experimentsData?.find(exp => exp.name === activeExperiment) 
        : null;

    return (
        <ScrollArea h="100%" p="md">
            <Stack gap="md">
                {/* Header con información del experimento */}
                <Paper p="md" withBorder>
                    <Group justify="space-between">
                        <div>
                            <Title order={4}>Datos y Estadísticas</Title>
                            <Text size="sm" c="dimmed" mt={4}>
                                {activeExperiment ? `Experimento: ${activeExperiment}` : 'No hay experimento activo'}
                            </Text>
                        </div>
                        {simData && (
                            <Badge size="lg" color="blue" variant="light">
                                Paso: {simData.step || 0}
                            </Badge>
                        )}
                    </Group>
                </Paper>

                {/* Estadísticas en tiempo real */}
                {currentStats && (
                    <Grid>
                        <Grid.Col span={{ base: 12, sm: 6, md: 3 }}>
                            <Card withBorder p="md">
                                <Stack gap="xs" align="center">
                                    <Text size="xs" c="dimmed">Densidad Media</Text>
                                    <Text size="xl" fw={700}>
                                        {currentStats.mean.toFixed(4)}
                                    </Text>
                                    <Progress 
                                        value={currentStats.mean * 100} 
                                        size="sm" 
                                        color="blue"
                                        style={{ width: '100%' }}
                                    />
                                </Stack>
                            </Card>
                        </Grid.Col>
                        <Grid.Col span={{ base: 12, sm: 6, md: 3 }}>
                            <Card withBorder p="md">
                                <Stack gap="xs" align="center">
                                    <Text size="xs" c="dimmed">Desviación Estándar</Text>
                                    <Text size="xl" fw={700}>
                                        {currentStats.std.toFixed(4)}
                                    </Text>
                                    <Progress 
                                        value={Math.min(currentStats.std * 100, 100)} 
                                        size="sm" 
                                        color="orange"
                                        style={{ width: '100%' }}
                                    />
                                </Stack>
                            </Card>
                        </Grid.Col>
                        <Grid.Col span={{ base: 12, sm: 6, md: 3 }}>
                            <Card withBorder p="md">
                                <Stack gap="xs" align="center">
                                    <Text size="xs" c="dimmed">Mínimo</Text>
                                    <Text size="xl" fw={700}>
                                        {currentStats.min.toFixed(4)}
                                    </Text>
                                </Stack>
                            </Card>
                        </Grid.Col>
                        <Grid.Col span={{ base: 12, sm: 6, md: 3 }}>
                            <Card withBorder p="md">
                                <Stack gap="xs" align="center">
                                    <Text size="xs" c="dimmed">Máximo</Text>
                                    <Text size="xl" fw={700}>
                                        {currentStats.max.toFixed(4)}
                                    </Text>
                                </Stack>
                            </Card>
                        </Grid.Col>
                    </Grid>
                )}

                {/* Gráficos de evolución temporal */}
                {simulationHistory.length > 1 && (
                    <Grid ref={chartContainerRef}>
                        <Grid.Col span={12}>
                            <Paper p="md" withBorder>
                                <Title order={5} mb="md">Evolución Temporal de la Simulación</Title>
                                <Box style={{ minWidth: 300, minHeight: 300, width: '100%', display: 'block' }}>
                                    {chartsReady && simulationHistory.length > 0 && chartContainerRef.current ? (
                                        (() => {
                                            const rect = chartContainerRef.current?.getBoundingClientRect();
                                            if (rect && rect.width > 0 && rect.height > 0) {
                                                return (
                                <LineChart
                                    h={300}
                                                        w="100%"
                                    data={simulationHistory.map(s => ({
                                        step: s.step,
                                        'Densidad Media': s.density.mean,
                                        'Desviación Estándar': s.density.std,
                                        'Mínimo': s.density.min,
                                        'Máximo': s.density.max
                                    }))}
                                    dataKey="step"
                                    series={[
                                        { 
                                            name: 'Densidad Media', 
                                            color: 'blue.6',
                                            label: 'Densidad Media'
                                        },
                                        { 
                                            name: 'Desviación Estándar', 
                                            color: 'orange.6',
                                            label: 'Desviación Estándar'
                                        },
                                        { 
                                            name: 'Mínimo', 
                                            color: 'red.6',
                                            label: 'Mínimo'
                                        },
                                        { 
                                            name: 'Máximo', 
                                            color: 'green.6',
                                            label: 'Máximo'
                                        }
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
                                        <Center h={300}>
                                            <Text c="dimmed" size="sm">Cargando gráfico...</Text>
                                        </Center>
                                    )}
                                </Box>
                            </Paper>
                        </Grid.Col>
                        <Grid.Col span={12}>
                            <Paper p="md" withBorder>
                                <Title order={5} mb="md">Distribución de Valores</Title>
                                <Box style={{ minWidth: 300, minHeight: 250, width: '100%', display: 'block' }}>
                                    {chartsReady && simulationHistory.length > 0 && chartContainerRef.current ? (
                                        (() => {
                                            const rect = chartContainerRef.current?.getBoundingClientRect();
                                            if (rect && rect.width > 0 && rect.height > 0) {
                                                return (
                                <AreaChart
                                    h={250}
                                                        w="100%"
                                    data={simulationHistory.map(s => ({
                                        step: s.step,
                                        'Varianza': s.density.variance
                                    }))}
                                    dataKey="step"
                                    series={[
                                        { 
                                            name: 'Varianza', 
                                            color: 'purple.6',
                                            label: 'Varianza'
                                        }
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
                                        <Center h={250}>
                                            <Text c="dimmed" size="sm">Cargando gráfico...</Text>
                                        </Center>
                                    )}
                                </Box>
                            </Paper>
                        </Grid.Col>
                    </Grid>
                )}

                {/* Información del experimento */}
                {currentExperiment && (
                    <Paper p="md" withBorder>
                        <Title order={5} mb="md">Configuración del Experimento</Title>
                        <Table>
                            <Table.Tbody>
                                <Table.Tr>
                                    <Table.Td><Text fw={500}>Arquitectura</Text></Table.Td>
                                    <Table.Td>{currentExperiment.config?.MODEL_ARCHITECTURE || 'N/A'}</Table.Td>
                                </Table.Tr>
                                <Table.Tr>
                                    <Table.Td><Text fw={500}>d_state</Text></Table.Td>
                                    <Table.Td>{currentExperiment.config?.MODEL_PARAMS?.d_state || 'N/A'}</Table.Td>
                                </Table.Tr>
                                <Table.Tr>
                                    <Table.Td><Text fw={500}>Hidden Channels</Text></Table.Td>
                                    <Table.Td>{currentExperiment.config?.MODEL_PARAMS?.hidden_channels || 'N/A'}</Table.Td>
                                </Table.Tr>
                                <Table.Tr>
                                    <Table.Td><Text fw={500}>Learning Rate</Text></Table.Td>
                                    <Table.Td>{currentExperiment.config?.LR_RATE_M || 'N/A'}</Table.Td>
                                </Table.Tr>
                                <Table.Tr>
                                    <Table.Td><Text fw={500}>Total Episodios</Text></Table.Td>
                                    <Table.Td>{currentExperiment.config?.TOTAL_EPISODES || 'N/A'}</Table.Td>
                                </Table.Tr>
                                <Table.Tr>
                                    <Table.Td><Text fw={500}>Estado</Text></Table.Td>
                                    <Table.Td>
                                        <Badge color={currentExperiment.has_checkpoint ? 'green' : 'orange'}>
                                            {currentExperiment.has_checkpoint ? 'Entrenado' : 'Sin entrenar'}
                                        </Badge>
                                    </Table.Td>
                                </Table.Tr>
                            </Table.Tbody>
                        </Table>
                    </Paper>
                )}

                {/* Progreso de entrenamiento */}
                {trainingProgress && (
                    <Paper p="md" withBorder>
                        <Title order={5} mb="md">Progreso de Entrenamiento</Title>
                        <Stack gap="md">
                            <div>
                                <Group justify="space-between" mb="xs">
                                    <Text size="sm">Episodio {trainingProgress.current_episode} / {trainingProgress.total_episodes}</Text>
                                    <Text size="sm" fw={500}>
                                        {((trainingProgress.current_episode / trainingProgress.total_episodes) * 100).toFixed(1)}%
                                    </Text>
                                </Group>
                                <Progress 
                                    value={(trainingProgress.current_episode / trainingProgress.total_episodes) * 100} 
                                    size="lg"
                                    animated
                                />
                            </div>
                            <Grid>
                                <Grid.Col span={6}>
                                    <Card withBorder p="sm">
                                        <Text size="xs" c="dimmed">Pérdida Actual</Text>
                                        <Text size="lg" fw={700}>{trainingProgress.avg_loss.toFixed(4)}</Text>
                                    </Card>
                                </Grid.Col>
                                {trainingProgress.avg_reward !== undefined && (
                                    <Grid.Col span={6}>
                                        <Card withBorder p="sm">
                                            <Text size="xs" c="dimmed">Recompensa</Text>
                                            <Text size="lg" fw={700}>{trainingProgress.avg_reward.toFixed(4)}</Text>
                                        </Card>
                                    </Grid.Col>
                                )}
                            </Grid>
                        </Stack>
                    </Paper>
                )}

                {/* Estado vacío */}
                {!simData && !trainingProgress && (
                    <Paper p="xl" withBorder>
                        <Center>
                            <Stack align="center" gap="md">
                                <IconDatabase size={48} color="gray" />
                                <Text c="dimmed" ta="center">
                                    No hay datos disponibles. Carga un modelo o inicia un entrenamiento para ver estadísticas.
                                </Text>
                            </Stack>
                        </Center>
                    </Paper>
                )}
            </Stack>
        </ScrollArea>
    );
}

