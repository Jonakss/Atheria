// frontend/src/components/LabSider.tsx
import { useState, useEffect } from 'react';
import { 
    Box, NavLink, Group, Text, ScrollArea, Title,
    NumberInput, TextInput, Select, Switch, Button, Divider
} from '@mantine/core';
import { useForm } from '@mantine/form';
import { IconPlayerPlay, IconBrain } from '@tabler/icons-react';
import { useWebSocket } from '../context/WebSocketContext'; // ¡¡CORRECCIÓN!! Nombre del hook
import classes from './LabSider.module.css';

interface Experiment {
    name: string;
    config: {
        MODEL_ARCHITECTURE?: string;
        LR_RATE_M?: number;
        [key: string]: any;
    };
}

export function LabSider() {
    const { sendCommand, experimentsData, trainingStatus } = useWebSocket();
    const [experiments, setExperiments] = useState<Experiment[]>([]);

    useEffect(() => {
        if (experimentsData) {
            setExperiments(experimentsData);
        }
    }, [experimentsData]);

    const form = useForm({
        initialValues: {
            experiment_name: 'SNN_UNET-D8-H32-G64-LR1e-4',
            model_architecture: 'SNN_UNET',
            d_state: 8,
            hidden_channels: 32,
            grid_size_training: 64,
            lr_rate_m: 0.0001,
            continue_training: false,
        },
    });

    const handleCreateExperiment = (values: typeof form.values) => {
        sendCommand('lab', 'create_experiment', { params: values });
    };

    return (
        <Box className={classes.sider}>
            <div className={classes.header}>
                <Title order={3}>Atheria Lab</Title>
                <Text size="sm" c="dimmed">Control de Entrenamiento</Text>
            </div>

            <ScrollArea className={classes.scrollArea}>
                <Title order={5} className={classes.sectionTitle}>Experimentos</Title>
                {experiments.length > 0 ? (
                    experiments.map((exp) => (
                        <NavLink
                            key={exp.name}
                            href="#"
                            label={
                                <Text size="sm" truncate>{exp.name}</Text>
                            }
                            description={
                                <Text size="xs" c="dimmed">
                                    {exp.config.MODEL_ARCHITECTURE || 'N/A'} - LR: {exp.config.LR_RATE_M || 'N/A'}
                                </Text>
                            }
                            leftSection={<IconBrain size="1rem" stroke={1.5} />}
                            variant="subtle"
                            className={classes.navLink}
                        />
                    ))
                ) : (
                    <Text size="sm" c="dimmed" ta="center" p="md">No se encontraron experimentos.</Text>
                )}
            </ScrollArea>

            <div className={classes.formSection}>
                <Divider my="sm" label="Crear Nuevo Experimento" labelPosition="center" />
                <form onSubmit={form.onSubmit(handleCreateExperiment)}>
                    <TextInput label="Nombre del Experimento" required {...form.getInputProps('experiment_name')} />
                    <Select
                        label="Arquitectura del Modelo"
                        required
                        data={['UNET', 'SNN_UNET', 'DEEP_QCA', 'MLP', 'UNET_UNITARY']}
                        {...form.getInputProps('model_architecture')}
                    />
                    <Group grow>
                        <NumberInput label="d_state" min={2} max={16} step={1} {...form.getInputProps('d_state')} />
                        <NumberInput label="Hidden Channels" min={8} max={64} step={8} {...form.getInputProps('hidden_channels')} />
                    </Group>
                    <Group grow>
                        <NumberInput label="Grid Size" min={16} max={128} step={16} {...form.getInputProps('grid_size_training')} />
                        <NumberInput label="Learning Rate" step={0.00001} decimalScale={5} {...form.getInputProps('lr_rate_m')} />
                    </Group>
                    <Switch mt="md" label="Continuar Entrenamiento" {...form.getInputProps('continue_training', { type: 'checkbox' })} />
                    <Button type="submit" fullWidth mt="md" leftSection={<IconPlayerPlay size={16} />} disabled={trainingStatus === 'running'}>
                        {trainingStatus === 'running' ? 'Entrenando...' : 'Lanzar Entrenamiento'}
                    </Button>
                </form>
            </div>
        </Box>
    );
}
