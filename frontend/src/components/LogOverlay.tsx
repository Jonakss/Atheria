// frontend/src/components/LogOverlay.tsx
import { Box, ScrollArea, Text, Title } from '@mantine/core';
import { useEffect, useRef } from 'react';
import { useWebSocket } from '../context/WebSocketContext'; // ¡¡CORRECCIÓN!! Nombre del hook
import classes from './LogOverlay.module.css';

export function LogOverlay() {
    const { trainingLog } = useWebSocket();
    const viewport = useRef<HTMLDivElement>(null);

    useEffect(() => {
        // Scroll al final cuando llega un nuevo log
        if (viewport.current) {
            viewport.current.scrollTo({ top: viewport.current.scrollHeight, behavior: 'smooth' });
        }
    }, [trainingLog]);

    if (trainingLog.length === 0) {
        return null; // No mostrar nada si no hay logs
    }

    return (
        <Box className={classes.overlay}>
            <Title order={5} className={classes.title}>Log de Entrenamiento</Title>
            <ScrollArea className={classes.scrollArea} viewportRef={viewport}>
                {trainingLog.map((line, index) => (
                    <Text component="pre" size="xs" key={index} className={classes.logLine}>
                        {line}
                    </Text>
                ))}
            </ScrollArea>
        </Box>
    );
}