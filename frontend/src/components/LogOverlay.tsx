// frontend/src/components/LogOverlay.tsx
import { Paper, ScrollArea, Text, Box, Title, Center, Stack } from '@mantine/core';
import { useWebSocket } from '../hooks/useWebSocket';
import classes from './LogOverlay.module.css';
import { useEffect, useRef } from 'react';
import { IconInfoCircle } from '@tabler/icons-react';

export function LogOverlay() {
    const { trainingLog, trainingStatus } = useWebSocket();
    const viewport = useRef<HTMLDivElement>(null);

    // Scroll automático al final
    useEffect(() => {
        if (viewport.current) {
            viewport.current.scrollTo({ top: viewport.current.scrollHeight, behavior: 'smooth' });
        }
    }, [trainingLog]);

    const hasLogs = trainingLog && trainingLog.length > 0;

    return (
        <Box className={classes.container}>
            <Title order={6} className={classes.header}>
                Log de Entrenamiento
            </Title>
            <Paper withBorder className={classes.logPaper}>
                <ScrollArea className={classes.scrollArea} viewportRef={viewport}>
                    {!hasLogs ? (
                        <Center h="100%">
                            <Stack align="center" gap="xs">
                                <IconInfoCircle size={32} color="gray" />
                                <Text c="dimmed" size="sm">
                                    {trainingStatus === 'running' 
                                        ? "Esperando logs del entrenamiento..." 
                                        : "Inicie un nuevo entrenamiento para ver los logs."}
                                </Text>
                            </Stack>
                        </Center>
                    ) : (
                        trainingLog.map((log, index) => (
                            <Text key={index} component="pre" size="xs" className={classes.logLine}>
                                {log}
                            </Text>
                        ))
                    )}
                </ScrollArea>
            </Paper>
        </Box>
    );
}
