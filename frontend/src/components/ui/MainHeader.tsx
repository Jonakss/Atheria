// frontend/src/components/MainHeader.tsx
import { Group, Burger, Text, Badge, Tooltip } from '@mantine/core';
import { IconPlayerPlay, IconPlayerPause } from '@tabler/icons-react';
import { useWebSocket } from '../../hooks/useWebSocket';
import { motion } from 'framer-motion';
import { slideInFromTop } from '../../utils/animations';
import classes from './MainHeader.module.css';

interface HeaderProps {
    mobileOpened: boolean;
    desktopOpened: boolean;
    toggleMobile: () => void;
    toggleDesktop: () => void;
}

export function MainHeader({ mobileOpened, desktopOpened, toggleMobile, toggleDesktop }: HeaderProps) {
    const { simData, inferenceStatus, activeExperiment, trainingStatus } = useWebSocket();
    const currentStep = simData?.step ?? null;

    return (
        <motion.div
            variants={slideInFromTop}
            initial="hidden"
            animate="visible"
            style={{ height: '100%', width: '100%' }}
        >
            <Group h="100%" px="md" justify="space-between">
            <Group gap="md">
                <Burger opened={mobileOpened} onClick={toggleMobile} hiddenFrom="sm" size="sm" />
                <Burger opened={desktopOpened} onClick={toggleDesktop} visibleFrom="sm" size="sm" />
                <Group gap="xs">
                <Text fw={500}>Aetheria Simulation Lab</Text>
                    {activeExperiment && (
                        <Badge variant="dot" color="blue" size="sm">
                            {activeExperiment}
                        </Badge>
                    )}
                </Group>
            </Group>
            <Group gap="xs">
                {trainingStatus === 'running' && (
                    <Tooltip label="Entrenamiento en curso">
                        <Badge color="orange" variant="light" size="lg">
                            Entrenando...
                        </Badge>
                    </Tooltip>
                )}
                {currentStep !== null && (
                    <Tooltip label={`Simulación en paso ${currentStep}`}>
                        <Badge 
                            color={inferenceStatus === 'running' ? 'green' : 'gray'} 
                            variant="light"
                            size="lg"
                            leftSection={
                                inferenceStatus === 'running' ? 
                                    <IconPlayerPlay size={12} /> : 
                                    <IconPlayerPause size={12} />
                            }
                        >
                            Paso: {currentStep.toLocaleString()}
                    </Badge>
                    </Tooltip>
                )}
                <Tooltip label={`Simulación ${inferenceStatus === 'running' ? 'en ejecución' : 'pausada'}`}>
                    <Badge 
                        color={inferenceStatus === 'running' ? 'green' : 'yellow'} 
                        variant="light"
                        size="lg"
                        leftSection={
                            inferenceStatus === 'running' ? 
                                <IconPlayerPlay size={12} /> : 
                                <IconPlayerPause size={12} />
                        }
                    >
                    {inferenceStatus === 'running' ? 'Ejecutando' : 'Pausado'}
                </Badge>
                </Tooltip>
            </Group>
        </Group>
        </motion.div>
    );
}
