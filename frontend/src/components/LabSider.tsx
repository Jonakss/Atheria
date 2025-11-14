// frontend/src/components/LabSider.tsx
import { useState } from 'react';
import { Box, Button, NavLink, ScrollArea, Select, Stack, Text, TextInput } from '@mantine/core';
import { useWebSocket } from '../context/WebSocketContext';
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
        
        sendCommand('experiment', 'create', {
            EXPERIMENT_NAME: expName,
            MODEL_ARCHITECTURE: selectedModel,
            LR_RATE_M: lr,
        });
    };

    const handleVizChange = (value: string | null) => {
        if (value) {
            setSelectedViz(value);
            sendCommand('simulation', 'set_viz', { viz_type: value });
        }
    };

    return (
        <Box className={classes.sider}>
            <Box className={classes.header}>
                <Text size="lg" fw={700}>Laboratorio</Text>
                <Text size="xs" c="dimmed">Gestión de experimentos y visualización</Text>
            </Box>

            <ScrollArea className={classes.scrollArea}>
                <Text size="xs" fw={700} className={classes.sectionTitle}>EXPERIMENTOS</Text>
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

            <Stack className={classes.formSection}>
                <Select
                    label="Arquitectura del Modelo"
                    placeholder="Selecciona un modelo"
                    data={modelOptions}
                    value={selectedModel}
                    onChange={setSelectedModel}
                />
                <TextInput
                    label="Tasa de Aprendizaje (LR)"
                    placeholder="e.g., 0.0001"
                    value={learningRate}
                    onChange={(event) => setLearningRate(event.currentTarget.value)}
                />
                <Button
                    onClick={handleCreateExperiment}
                    loading={trainingStatus === 'running'}
                    disabled={!selectedModel}
                >
                    Crear y Entrenar
                </Button>
            </Stack>

            <Stack className={classes.formSection}>
                 <Select
                    label="Mapa de Visualización"
                    data={vizOptions}
                    value={selectedViz}
                    onChange={handleVizChange}
                />
            </Stack>
        </Box>
    );
}
