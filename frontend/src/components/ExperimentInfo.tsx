// frontend/src/components/ExperimentInfo.tsx
import { Paper, Stack, Group, Text, Badge, Box, Divider } from '@mantine/core';
import { IconInfoCircle, IconTransfer, IconBrain, IconSettings } from '@tabler/icons-react';
import { useWebSocket } from '../hooks/useWebSocket';

export function ExperimentInfo() {
    const { activeExperiment, experimentsData } = useWebSocket();
    
    if (!activeExperiment) {
        return (
            <Paper p="sm" withBorder>
                <Group gap="xs">
                    <IconInfoCircle size={16} />
                    <Text size="sm" c="dimmed">No hay experimento seleccionado</Text>
                </Group>
            </Paper>
        );
    }
    
    const experiment = experimentsData?.find(exp => exp.name === activeExperiment);
    if (!experiment) {
        return (
            <Paper p="sm" withBorder>
                <Group gap="xs">
                    <IconInfoCircle size={16} />
                    <Text size="sm" c="dimmed">Experimento no encontrado</Text>
                </Group>
            </Paper>
        );
    }
    
    const config = experiment.config || {};
    const modelParams = config.MODEL_PARAMS || {};
    
    return (
        <Paper p="md" withBorder style={{ backgroundColor: 'var(--mantine-color-dark-7)' }}>
            <Stack gap="sm">
                <Group justify="space-between" align="flex-start">
                    <Group gap="xs">
                        <IconBrain size={18} color="var(--mantine-color-blue-6)" />
                        <Text size="sm" fw={700} c="blue">{activeExperiment}</Text>
                        <Badge 
                            color={experiment.has_checkpoint ? 'green' : 'orange'} 
                            variant="light"
                            size="sm"
                        >
                            {experiment.has_checkpoint ? '✓ Entrenado' : '○ Sin entrenar'}
                        </Badge>
                    </Group>
                </Group>
                
                <Divider />
                
                <Stack gap="xs">
                    <Group gap="xs">
                        <IconSettings size={14} />
                        <Text size="xs" fw={600} c="dimmed">ARQUITECTURA</Text>
                    </Group>
                    <Box pl="md">
                        <Text size="sm" fw={500}>{config.MODEL_ARCHITECTURE || 'N/A'}</Text>
                    </Box>
                </Stack>
                
                <Stack gap="xs">
                    <Text size="xs" fw={600} c="dimmed">HIPERPARÁMETROS</Text>
                    <Box pl="md">
                        <Group gap="md">
                            <Box>
                                <Text size="xs" c="dimmed">d_state</Text>
                                <Text size="sm" fw={500}>{modelParams.d_state || config.D_STATE || 'N/A'}</Text>
                            </Box>
                            <Box>
                                <Text size="xs" c="dimmed">Hidden Channels</Text>
                                <Text size="sm" fw={500}>{modelParams.hidden_channels || config.HIDDEN_CHANNELS || 'N/A'}</Text>
                            </Box>
                            <Box>
                                <Text size="xs" c="dimmed">Learning Rate</Text>
                                <Text size="sm" fw={500}>{config.LR_RATE_M || 'N/A'}</Text>
                            </Box>
                            <Box>
                                <Text size="xs" c="dimmed">Grid Training</Text>
                                <Text size="sm" fw={500}>{config.GRID_SIZE_TRAINING || 'N/A'}</Text>
                            </Box>
                            <Box>
                                <Text size="xs" c="dimmed">Grid Inference</Text>
                                <Text size="sm" fw={500} c="blue">256</Text>
                            </Box>
                            <Box>
                                <Text size="xs" c="dimmed">Gamma Decay</Text>
                                <Text size="sm" fw={500} c={config.GAMMA_DECAY === 0 ? 'gray' : 'orange'}>
                                    {config.GAMMA_DECAY || 0.01}
                                    {config.GAMMA_DECAY === 0 ? ' (Cerrado)' : ' (Abierto)'}
                                </Text>
                            </Box>
                            <Box>
                                <Text size="xs" c="dimmed">Inicialización</Text>
                                <Text size="sm" fw={500} c="cyan">
                                    {config.INITIAL_STATE_MODE_INFERENCE || 'complex_noise'}
                                </Text>
                            </Box>
                        </Group>
                    </Box>
                </Stack>
                
                {config.TOTAL_EPISODES && (
                    <Stack gap="xs">
                        <Text size="xs" fw={600} c="dimmed">ENTRENAMIENTO</Text>
                        <Box pl="md">
                            <Text size="sm">
                                <Text component="span" c="dimmed">Episodios: </Text>
                                <Text component="span" fw={500}>{config.TOTAL_EPISODES}</Text>
                            </Text>
                        </Box>
                    </Stack>
                )}
                
                {config.LOAD_FROM_EXPERIMENT && (
                    <Stack gap="xs">
                        <Group gap="xs">
                            <IconTransfer size={14} />
                            <Text size="xs" fw={600} c="dimmed">TRANSFER LEARNING</Text>
                        </Group>
                        <Box pl="md">
                            <Badge variant="dot" color="blue" size="sm">
                                {config.LOAD_FROM_EXPERIMENT}
                            </Badge>
                        </Box>
                    </Stack>
                )}
            </Stack>
        </Paper>
    );
}

