// frontend/src/components/TransferLearningWizard.tsx
import { useState, useEffect, useMemo } from 'react';
import {
    Modal, Stepper, Button, Group, Stack, Text, Card, Badge,
    Select, NumberInput, TextInput, Alert, Divider, Table,
    Paper, Tooltip, ActionIcon
} from '@mantine/core';
import { useWebSocket } from '../../hooks/useWebSocket';
import {
    IconCheck, IconX, IconInfoCircle, IconArrowRight,
    IconTransfer, IconSettings, IconFile, IconStar
} from '@tabler/icons-react';
import { notifications } from '@mantine/notifications';

interface ExperimentConfig {
    MODEL_ARCHITECTURE: string;
    GRID_SIZE_TRAINING: number;
    LR_RATE_M: number;
    MODEL_PARAMS: {
        d_state: number;
        hidden_channels: number;
        alpha?: number;
        beta?: number;
    };
    GAMMA_DECAY: number;
    QCA_STEPS_TRAINING: number;
    INITIAL_STATE_MODE_INFERENCE: string;
    TOTAL_EPISODES?: number;
}

interface TransferLearningWizardProps {
    opened: boolean;
    onClose: () => void;
}

export function TransferLearningWizard({ opened, onClose }: TransferLearningWizardProps) {
    const { experimentsData, sendCommand } = useWebSocket();
    const [activeStep, setActiveStep] = useState(0);
    const [baseExperiment, setBaseExperiment] = useState<string | null>(null);
    const [baseConfig, setBaseConfig] = useState<ExperimentConfig | null>(null);
    const [newConfig, setNewConfig] = useState<ExperimentConfig | null>(null);
    const [newExperimentName, setNewExperimentName] = useState('');
    const [episodesToAdd, setEpisodesToAdd] = useState(1000);

    // Filtrar experimentos que tienen checkpoints
    const availableExperiments = useMemo(() => {
        return experimentsData?.filter(exp => exp.has_checkpoint) || [];
    }, [experimentsData]);

    // Cargar configuración cuando se selecciona un experimento base
    useEffect(() => {
        if (baseExperiment && activeStep >= 1) {
            const exp = experimentsData?.find(e => e.name === baseExperiment);
            // El backend ahora devuelve config anidado
            const expConfig = exp?.config;
            if (expConfig) {
                const config: ExperimentConfig = {
                    MODEL_ARCHITECTURE: expConfig.MODEL_ARCHITECTURE || 'UNET',
                    GRID_SIZE_TRAINING: expConfig.GRID_SIZE_TRAINING || 64,
                    LR_RATE_M: expConfig.LR_RATE_M || 0.0001,
                    MODEL_PARAMS: expConfig.MODEL_PARAMS || { d_state: 8, hidden_channels: 32 },
                    GAMMA_DECAY: expConfig.GAMMA_DECAY || 0.01,
                    QCA_STEPS_TRAINING: expConfig.QCA_STEPS_TRAINING || 10,
                    INITIAL_STATE_MODE_INFERENCE: expConfig.INITIAL_STATE_MODE_INFERENCE || 'complex_noise',
                    TOTAL_EPISODES: expConfig.TOTAL_EPISODES || 0
                };
                setBaseConfig(config);
                setNewConfig({ ...config }); // Copiar como punto de partida
                
                // Generar nombre sugerido
                const suggestedName = generateExperimentName(config);
                setNewExperimentName(suggestedName);
            }
        }
    }, [baseExperiment, activeStep, experimentsData]);

    const generateExperimentName = (config: ExperimentConfig): string => {
        const arch = config.MODEL_ARCHITECTURE;
        const dState = config.MODEL_PARAMS.d_state;
        const hChannels = config.MODEL_PARAMS.hidden_channels;
        const gridSize = config.GRID_SIZE_TRAINING;
        const lr = config.LR_RATE_M;
        return `${arch}-d${dState}-h${hChannels}-g${gridSize}-lr${lr.toExponential(0)}`;
    };

    const applyProgressionTemplate = (template: 'standard' | 'fine_tune' | 'aggressive') => {
        if (!baseConfig || !newConfig) return;

        const updated = { ...newConfig };

        switch (template) {
            case 'standard':
                // Grid size aumenta 2x, LR se mantiene
                updated.GRID_SIZE_TRAINING = baseConfig.GRID_SIZE_TRAINING * 2;
                updated.LR_RATE_M = baseConfig.LR_RATE_M;
                break;
            case 'fine_tune':
                // Grid size igual, LR se reduce ligeramente
                updated.GRID_SIZE_TRAINING = baseConfig.GRID_SIZE_TRAINING;
                updated.LR_RATE_M = baseConfig.LR_RATE_M * 0.5;
                break;
            case 'aggressive':
                // Grid size aumenta 4x, LR se ajusta
                updated.GRID_SIZE_TRAINING = baseConfig.GRID_SIZE_TRAINING * 4;
                updated.LR_RATE_M = baseConfig.LR_RATE_M * 0.8;
                break;
        }

        setNewConfig(updated);
        setNewExperimentName(generateExperimentName(updated));
    };

    const handleNext = () => {
        if (activeStep === 0 && !baseExperiment) {
            notifications.show({
                title: 'Selecciona un experimento',
                message: 'Debes seleccionar un experimento base para continuar',
                color: 'orange',
            });
            return;
        }
        if (activeStep === 1 && !newConfig) {
            notifications.show({
                title: 'Error',
                message: 'No se pudo cargar la configuración',
                color: 'red',
            });
            return;
        }
        setActiveStep(activeStep + 1);
    };

    const handleBack = () => {
        setActiveStep(activeStep - 1);
    };

    const handleCreate = () => {
        if (!baseExperiment || !newConfig || !newExperimentName) {
            notifications.show({
                title: 'Error',
                message: 'Faltan datos requeridos',
                color: 'red',
            });
            return;
        }

        // Verificar si el nombre ya existe
        const existingExp = experimentsData?.find(e => e.name === newExperimentName);
        if (existingExp) {
            if (!window.confirm(`⚠️ El experimento "${newExperimentName}" ya existe. ¿Deseas continuar de todas formas?`)) {
                return;
            }
        }

        const args: Record<string, any> = {
            EXPERIMENT_NAME: newExperimentName,
            MODEL_ARCHITECTURE: newConfig.MODEL_ARCHITECTURE,
            LR_RATE_M: newConfig.LR_RATE_M,
            GRID_SIZE_TRAINING: newConfig.GRID_SIZE_TRAINING,
            QCA_STEPS_TRAINING: newConfig.QCA_STEPS_TRAINING,
            TOTAL_EPISODES: episodesToAdd,
            MODEL_PARAMS: newConfig.MODEL_PARAMS,
            GAMMA_DECAY: newConfig.GAMMA_DECAY,
            INITIAL_STATE_MODE_INFERENCE: newConfig.INITIAL_STATE_MODE_INFERENCE,
            LOAD_FROM_EXPERIMENT: baseExperiment
        };

        sendCommand('experiment', 'create', args);
        
        notifications.show({
            title: 'Transfer Learning iniciado',
            message: `Creando experimento "${newExperimentName}" desde "${baseExperiment}"`,
            color: 'green',
        });

        // Reset y cerrar
        setActiveStep(0);
        setBaseExperiment(null);
        setBaseConfig(null);
        setNewConfig(null);
        setNewExperimentName('');
        onClose();
    };

    const handleClose = () => {
        setActiveStep(0);
        setBaseExperiment(null);
        setBaseConfig(null);
        setNewConfig(null);
        setNewExperimentName('');
        onClose();
    };

    const baseExperimentData = baseExperiment 
        ? experimentsData?.find(e => e.name === baseExperiment)
        : null;

    const configChanged = useMemo(() => {
        if (!baseConfig || !newConfig) return false;
        return (
            baseConfig.GRID_SIZE_TRAINING !== newConfig.GRID_SIZE_TRAINING ||
            baseConfig.LR_RATE_M !== newConfig.LR_RATE_M ||
            baseConfig.MODEL_PARAMS.d_state !== newConfig.MODEL_PARAMS.d_state ||
            baseConfig.MODEL_PARAMS.hidden_channels !== newConfig.MODEL_PARAMS.hidden_channels ||
            baseConfig.GAMMA_DECAY !== newConfig.GAMMA_DECAY ||
            baseConfig.QCA_STEPS_TRAINING !== newConfig.QCA_STEPS_TRAINING
        );
    }, [baseConfig, newConfig]);

    return (
        <Modal
            opened={opened}
            onClose={handleClose}
            title={
                <Group gap="xs">
                    <IconTransfer size={20} />
                    <Text fw={600}>Transfer Learning Wizard</Text>
                </Group>
            }
            size="xl"
            closeOnClickOutside={false}
        >
            <Stepper active={activeStep} onStepClick={setActiveStep} breakpoint="sm">
                <Stepper.Step label="Seleccionar Base" description="Elige el experimento origen">
                    <Stack gap="md" mt="md">
                        <Alert icon={<IconInfoCircle size={16} />} color="blue" variant="light">
                            <Text size="sm">
                                Selecciona un experimento que tenga checkpoints guardados. 
                                Su configuración se cargará automáticamente y podrás ajustarla.
                            </Text>
                        </Alert>

                        <Select
                            label="Experimento Base"
                            placeholder="Selecciona un experimento..."
                            data={availableExperiments.map(exp => ({
                                value: exp.name,
                                label: exp.name
                            }))}
                            value={baseExperiment}
                            onChange={(value) => setBaseExperiment(value)}
                            searchable
                            nothingFound="No hay experimentos con checkpoints"
                        />

                        {baseExperimentData && (
                            <Card withBorder p="md" style={{ backgroundColor: 'var(--mantine-color-dark-7)' }}>
                                <Stack gap="xs">
                                    <Group justify="space-between">
                                        <Text fw={600} size="sm">Información del Experimento Base</Text>
                                        <Badge color="green" variant="light">
                                            ✓ Tiene checkpoints
                                        </Badge>
                                    </Group>
                                    <Divider />
                                    <Group gap="md">
                                        <Text size="xs" c="dimmed">
                                            <strong>Arquitectura:</strong> {baseExperimentData.config?.MODEL_ARCHITECTURE || 'N/A'}
                                        </Text>
                                        <Text size="xs" c="dimmed">
                                            <strong>Grid Size:</strong> {baseExperimentData.config?.GRID_SIZE_TRAINING || 'N/A'}
                                        </Text>
                                        <Text size="xs" c="dimmed">
                                            <strong>LR:</strong> {(baseExperimentData.config?.LR_RATE_M || 0.0001).toExponential(2)}
                                        </Text>
                                    </Group>
                                </Stack>
                            </Card>
                        )}
                    </Stack>
                </Stepper.Step>

                <Stepper.Step label="Configurar" description="Ajusta los parámetros">
                    {baseConfig && newConfig && (
                        <Stack gap="md" mt="md">
                            <Alert icon={<IconInfoCircle size={16} />} color="blue" variant="light">
                                <Text size="sm">
                                    La configuración del experimento base se ha cargado. 
                                    Ajusta los parámetros según tu estrategia de entrenamiento progresivo.
                                </Text>
                            </Alert>

                            {/* Templates rápidos */}
                            <Card withBorder p="sm">
                                <Stack gap="xs">
                                    <Text size="sm" fw={600}>Templates de Progresión</Text>
                                    <Group>
                                        <Button
                                            size="xs"
                                            variant="light"
                                            onClick={() => applyProgressionTemplate('standard')}
                                        >
                                            Estándar (Grid 2x, LR igual)
                                        </Button>
                                        <Button
                                            size="xs"
                                            variant="light"
                                            onClick={() => applyProgressionTemplate('fine_tune')}
                                        >
                                            Fine-tuning (Grid igual, LR 0.5x)
                                        </Button>
                                        <Button
                                            size="xs"
                                            variant="light"
                                            onClick={() => applyProgressionTemplate('aggressive')}
                                        >
                                            Agresivo (Grid 4x, LR 0.8x)
                                        </Button>
                                    </Group>
                                </Stack>
                            </Card>

                            {/* Comparación lado a lado */}
                            <Table>
                                <Table.Thead>
                                    <Table.Tr>
                                        <Table.Th>Parámetro</Table.Th>
                                        <Table.Th>Base</Table.Th>
                                        <Table.Th>Nuevo</Table.Th>
                                        <Table.Th>Cambio</Table.Th>
                                    </Table.Tr>
                                </Table.Thead>
                                <Table.Tbody>
                                    <Table.Tr>
                                        <Table.Td><Text size="sm" fw={500}>Grid Size</Text></Table.Td>
                                        <Table.Td><Text size="sm">{baseConfig.GRID_SIZE_TRAINING}</Text></Table.Td>
                                        <Table.Td>
                                            <NumberInput
                                                value={newConfig.GRID_SIZE_TRAINING}
                                                onChange={(val) => setNewConfig({
                                                    ...newConfig,
                                                    GRID_SIZE_TRAINING: Number(val) || baseConfig.GRID_SIZE_TRAINING
                                                })}
                                                min={16}
                                                max={512}
                                                size="xs"
                                                style={{ width: 100 }}
                                            />
                                        </Table.Td>
                                        <Table.Td>
                                            {newConfig.GRID_SIZE_TRAINING > baseConfig.GRID_SIZE_TRAINING && (
                                                <Badge color="green" size="sm">↑ {((newConfig.GRID_SIZE_TRAINING / baseConfig.GRID_SIZE_TRAINING - 1) * 100).toFixed(0)}%</Badge>
                                            )}
                                            {newConfig.GRID_SIZE_TRAINING < baseConfig.GRID_SIZE_TRAINING && (
                                                <Badge color="orange" size="sm">↓ {((1 - newConfig.GRID_SIZE_TRAINING / baseConfig.GRID_SIZE_TRAINING) * 100).toFixed(0)}%</Badge>
                                            )}
                                            {newConfig.GRID_SIZE_TRAINING === baseConfig.GRID_SIZE_TRAINING && (
                                                <Badge color="gray" size="sm">=</Badge>
                                            )}
                                        </Table.Td>
                                    </Table.Tr>
                                    <Table.Tr>
                                        <Table.Td><Text size="sm" fw={500}>Learning Rate</Text></Table.Td>
                                        <Table.Td><Text size="sm">{baseConfig.LR_RATE_M.toExponential(3)}</Text></Table.Td>
                                        <Table.Td>
                                            <NumberInput
                                                value={newConfig.LR_RATE_M}
                                                onChange={(val) => setNewConfig({
                                                    ...newConfig,
                                                    LR_RATE_M: Number(val) || baseConfig.LR_RATE_M
                                                })}
                                                min={1e-6}
                                                max={1}
                                                step={1e-5}
                                                precision={6}
                                                size="xs"
                                                style={{ width: 120 }}
                                            />
                                        </Table.Td>
                                        <Table.Td>
                                            {newConfig.LR_RATE_M !== baseConfig.LR_RATE_M && (
                                                <Badge color="blue" size="sm">
                                                    {newConfig.LR_RATE_M > baseConfig.LR_RATE_M ? '↑' : '↓'}
                                                </Badge>
                                            )}
                                        </Table.Td>
                                    </Table.Tr>
                                    <Table.Tr>
                                        <Table.Td><Text size="sm" fw={500}>d_state</Text></Table.Td>
                                        <Table.Td><Text size="sm">{baseConfig.MODEL_PARAMS.d_state}</Text></Table.Td>
                                        <Table.Td>
                                            <NumberInput
                                                value={newConfig.MODEL_PARAMS.d_state}
                                                onChange={(val) => setNewConfig({
                                                    ...newConfig,
                                                    MODEL_PARAMS: {
                                                        ...newConfig.MODEL_PARAMS,
                                                        d_state: Number(val) || baseConfig.MODEL_PARAMS.d_state
                                                    }
                                                })}
                                                min={4}
                                                max={32}
                                                size="xs"
                                                style={{ width: 100 }}
                                            />
                                        </Table.Td>
                                        <Table.Td>
                                            {newConfig.MODEL_PARAMS.d_state !== baseConfig.MODEL_PARAMS.d_state && (
                                                <Badge color="blue" size="sm">
                                                    {newConfig.MODEL_PARAMS.d_state > baseConfig.MODEL_PARAMS.d_state ? '↑' : '↓'}
                                                </Badge>
                                            )}
                                        </Table.Td>
                                    </Table.Tr>
                                    <Table.Tr>
                                        <Table.Td><Text size="sm" fw={500}>hidden_channels</Text></Table.Td>
                                        <Table.Td><Text size="sm">{baseConfig.MODEL_PARAMS.hidden_channels}</Text></Table.Td>
                                        <Table.Td>
                                            <NumberInput
                                                value={newConfig.MODEL_PARAMS.hidden_channels}
                                                onChange={(val) => setNewConfig({
                                                    ...newConfig,
                                                    MODEL_PARAMS: {
                                                        ...newConfig.MODEL_PARAMS,
                                                        hidden_channels: Number(val) || baseConfig.MODEL_PARAMS.hidden_channels
                                                    }
                                                })}
                                                min={8}
                                                max={128}
                                                size="xs"
                                                style={{ width: 100 }}
                                            />
                                        </Table.Td>
                                        <Table.Td>
                                            {newConfig.MODEL_PARAMS.hidden_channels !== baseConfig.MODEL_PARAMS.hidden_channels && (
                                                <Badge color="blue" size="sm">
                                                    {newConfig.MODEL_PARAMS.hidden_channels > baseConfig.MODEL_PARAMS.hidden_channels ? '↑' : '↓'}
                                                </Badge>
                                            )}
                                        </Table.Td>
                                    </Table.Tr>
                                    <Table.Tr>
                                        <Table.Td><Text size="sm" fw={500}>Gamma Decay</Text></Table.Td>
                                        <Table.Td><Text size="sm">{baseConfig.GAMMA_DECAY.toFixed(4)}</Text></Table.Td>
                                        <Table.Td>
                                            <NumberInput
                                                value={newConfig.GAMMA_DECAY}
                                                onChange={(val) => setNewConfig({
                                                    ...newConfig,
                                                    GAMMA_DECAY: Number(val) || baseConfig.GAMMA_DECAY
                                                })}
                                                min={0}
                                                max={1}
                                                step={0.001}
                                                precision={4}
                                                size="xs"
                                                style={{ width: 100 }}
                                            />
                                        </Table.Td>
                                        <Table.Td>
                                            {newConfig.GAMMA_DECAY !== baseConfig.GAMMA_DECAY && (
                                                <Badge color="blue" size="sm">
                                                    {newConfig.GAMMA_DECAY > baseConfig.GAMMA_DECAY ? '↑' : '↓'}
                                                </Badge>
                                            )}
                                        </Table.Td>
                                    </Table.Tr>
                                    <Table.Tr>
                                        <Table.Td><Text size="sm" fw={500}>QCA Steps</Text></Table.Td>
                                        <Table.Td><Text size="sm">{baseConfig.QCA_STEPS_TRAINING}</Text></Table.Td>
                                        <Table.Td>
                                            <NumberInput
                                                value={newConfig.QCA_STEPS_TRAINING}
                                                onChange={(val) => setNewConfig({
                                                    ...newConfig,
                                                    QCA_STEPS_TRAINING: Number(val) || baseConfig.QCA_STEPS_TRAINING
                                                })}
                                                min={1}
                                                max={100}
                                                size="xs"
                                                style={{ width: 100 }}
                                            />
                                        </Table.Td>
                                        <Table.Td>
                                            {newConfig.QCA_STEPS_TRAINING !== baseConfig.QCA_STEPS_TRAINING && (
                                                <Badge color="blue" size="sm">
                                                    {newConfig.QCA_STEPS_TRAINING > baseConfig.QCA_STEPS_TRAINING ? '↑' : '↓'}
                                                </Badge>
                                            )}
                                        </Table.Td>
                                    </Table.Tr>
                                </Table.Tbody>
                            </Table>

                            <NumberInput
                                label="Episodios a Entrenar"
                                value={episodesToAdd}
                                onChange={(val) => setEpisodesToAdd(Number(val) || 1000)}
                                min={1}
                                max={10000}
                                description="Número de episodios para el nuevo entrenamiento"
                            />
                        </Stack>
                    )}
                </Stepper.Step>

                <Stepper.Step label="Confirmar" description="Revisa y crea">
                    {baseConfig && newConfig && (
                        <Stack gap="md" mt="md">
                            <Alert icon={<IconInfoCircle size={16} />} color="green" variant="light">
                                <Text size="sm">
                                    Revisa la configuración antes de crear el nuevo experimento con transfer learning.
                                </Text>
                            </Alert>

                            <Card withBorder p="md">
                                <Stack gap="md">
                                    <Group justify="space-between">
                                        <Text fw={600}>Experimento Base</Text>
                                        <Badge>{baseExperiment}</Badge>
                                    </Group>
                                    <Divider />
                                    <Group justify="space-between">
                                        <Text fw={600}>Nuevo Experimento</Text>
                                        <TextInput
                                            value={newExperimentName}
                                            onChange={(e) => setNewExperimentName(e.target.value)}
                                            style={{ flex: 1, maxWidth: 400 }}
                                            size="sm"
                                        />
                                    </Group>
                                    <Divider />
                                    <Stack gap="xs">
                                        <Text size="sm" fw={500}>Resumen de Cambios:</Text>
                                        {!configChanged && (
                                            <Text size="xs" c="dimmed">No hay cambios en la configuración (solo transfer learning)</Text>
                                        )}
                                        {configChanged && (
                                            <Stack gap={4}>
                                                {newConfig.GRID_SIZE_TRAINING !== baseConfig.GRID_SIZE_TRAINING && (
                                                    <Text size="xs">
                                                        Grid Size: {baseConfig.GRID_SIZE_TRAINING} → {newConfig.GRID_SIZE_TRAINING}
                                                    </Text>
                                                )}
                                                {newConfig.LR_RATE_M !== baseConfig.LR_RATE_M && (
                                                    <Text size="xs">
                                                        LR: {baseConfig.LR_RATE_M.toExponential(3)} → {newConfig.LR_RATE_M.toExponential(3)}
                                                    </Text>
                                                )}
                                                {newConfig.MODEL_PARAMS.d_state !== baseConfig.MODEL_PARAMS.d_state && (
                                                    <Text size="xs">
                                                        d_state: {baseConfig.MODEL_PARAMS.d_state} → {newConfig.MODEL_PARAMS.d_state}
                                                    </Text>
                                                )}
                                                {newConfig.MODEL_PARAMS.hidden_channels !== baseConfig.MODEL_PARAMS.hidden_channels && (
                                                    <Text size="xs">
                                                        hidden_channels: {baseConfig.MODEL_PARAMS.hidden_channels} → {newConfig.MODEL_PARAMS.hidden_channels}
                                                    </Text>
                                                )}
                                            </Stack>
                                        )}
                                        <Text size="xs" c="dimmed" mt="xs">
                                            Episodios: {episodesToAdd}
                                        </Text>
                                    </Stack>
                                </Stack>
                            </Card>
                        </Stack>
                    )}
                </Stepper.Step>

                <Stepper.Completed>
                    <Stack gap="md" mt="md" align="center">
                        <IconCheck size={48} color="var(--mantine-color-green-6)" />
                        <Text fw={600} size="lg">¡Experimento creado!</Text>
                        <Text size="sm" c="dimmed" ta="center">
                            El entrenamiento con transfer learning ha sido iniciado.
                            Puedes monitorear el progreso en el panel lateral.
                        </Text>
                    </Stack>
                </Stepper.Completed>
            </Stepper>

            <Group justify="flex-end" mt="xl">
                {activeStep > 0 && (
                    <Button variant="default" onClick={handleBack}>
                        Atrás
                    </Button>
                )}
                {activeStep < 3 && (
                    <Button onClick={handleNext}>
                        Siguiente
                    </Button>
                )}
                {activeStep === 3 && (
                    <Button onClick={handleCreate} leftSection={<IconCheck size={16} />}>
                        Crear Experimento
                    </Button>
                )}
                <Button variant="subtle" onClick={handleClose}>
                    Cancelar
                </Button>
            </Group>
        </Modal>
    );
}

