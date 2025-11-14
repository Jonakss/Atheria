// frontend/src/components/LabSider.tsx
import { useState } from 'react';
import { Box, Button, NavLink, ScrollArea, Select, Stack, Text, TextInput, Group } from '@mantine/core';
import { IconRefresh, IconUpload } from '@tabler/icons-react';
import { useWebSocket } from '../hooks/useWebSocket';
import { modelOptions, vizOptions } from '../utils/vizOptions';
import classes from './LabSider.module.css';

export function LabSider() {
    const { sendCommand, experimentsData, trainingStatus } = useWebSocket();
    const [activeExperiment, setActiveExperiment] = useState<string | null>(null);
    const [selectedModel, setSelectedModel] = useState<string | null>('UNET');
    const [learningRate, setLearningRate] = useState('0.0001');
    const [selectedViz, setSelectedViz] = useState<string>('density_map');

    const handleCreateExperiment = () => {
        const lr = parseFloat(learningRate);
        const expName = `TEST_${selectedModel}-LR${lr.toExponential(0)}`;
        sendCommand('experiment', 'create', { EXPERIMENT_NAME: expName, MODEL_ARCHITECTURE: selectedModel, LR_RATE_M: lr });
    };

    const handleVizChange = (value: string | null) => {
        if (value) {
            setSelectedViz(value);
            sendCommand('simulation', 'set_viz', { viz_type: value });
        }
    };

    const handleLoadExperiment = () => {
        if (activeExperiment) {
            sendCommand('inference', 'load_experiment', { experiment_name: activeExperiment });
        }
    };

    const handleResetSimulation = () => {
        sendCommand('inference', 'reset');
    };

    return (
        <Box className={classes.sider}>
            <Box className={classes.header}>
                <Text size="lg" fw={700}>Laboratorio</Text>
                <Text size="xs" c="dimmed">Gestión de simulación y entrenamiento</Text>
            </Box>

            {/* --- ¡¡NUEVO!! Layout con Stack para mejor espaciado --- */}
            <Stack gap="xl" mt="md" style={{ flex: 1, overflow: 'hidden' }}>

                {/* --- SECCIÓN DE INFERENCIA --- */}
                <Stack gap="sm">
                    <Text size="xs" fw={700} className={classes.sectionTitle}>INFERENCIA</Text>
                    <Group grow>
                        <Button leftSection={<IconUpload size={14} />} variant="default" onClick={handleLoadExperiment} disabled={!activeExperiment}>
                            Cargar Modelo
                        </Button>
                        <Button leftSection={<IconRefresh size={14} />} variant="default" onClick={handleResetSimulation}>
                            Reiniciar Sim
                        </Button>
                    </Group>
                    <Select
                        label="Mapa de Visualización"
                        data={vizOptions}
                        value={selectedViz}
                        onChange={handleVizChange}
                    />
                </Stack>

                {/* --- SECCIÓN DE EXPERIMENTOS --- */}
                <Stack gap="xs" style={{ flex: 1, minHeight: 0 }}>
                    <Text size="xs" fw={700} className={classes.sectionTitle}>EXPERIMENTOS</Text>
                    <ScrollArea style={{ flex: 1 }}>
                        {experimentsData?.map((exp) => (
                            <NavLink
                                key={exp.name}
                                href={`#${exp.name}`}
                                label={exp.name}
                                description={`Modelo: ${exp.config.MODEL_ARCHITECTURE}, LR: ${exp.config.LR_RATE_M}`}
                                active={exp.name === activeExperiment}
                                onClick={() => setActiveExperiment(exp.name)}
                                className={classes.navLink}
                            />
                        ))}
                    </ScrollArea>
                </Stack>

                {/* --- SECCIÓN DE ENTRENAMIENTO --- */}
                <Stack gap="sm">
                    <Text size="xs" fw={700} className={classes.sectionTitle}>ENTRENAMIENTO</Text>
                    <Select
                        label="Arquitectura del Modelo"
                        data={modelOptions}
                        value={selectedModel}
                        onChange={setSelectedModel}
                    />
                    <TextInput
                        label="Tasa de Aprendizaje (LR)"
                        value={learningRate}
                        onChange={(event) => setLearningRate(event.currentTarget.value)}
                    />
                    <Button onClick={handleCreateExperiment} loading={trainingStatus === 'running'} disabled={!selectedModel}>
                        Crear y Entrenar
                    </Button>
                </Stack>

            </Stack>
        </Box>
    );
}
