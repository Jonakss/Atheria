// frontend/src/components/TimelineControl.tsx
import { useState, useEffect } from 'react';
import { Paper, Stack, Group, Text, Slider, Button, Badge, Tooltip, ActionIcon, Box } from '@mantine/core';
import { IconPlayerPlay, IconPlayerPause, IconPlayerSkipBack, IconPlayerSkipForward, IconHelpCircle } from '@tabler/icons-react';

interface HistoryFrame {
    step: number;
    timestamp: string;
    map_data: number[][];
}

interface TimelineControlProps {
    frames: HistoryFrame[];
    currentFrameIndex: number;
    onFrameChange: (index: number) => void;
    onPlayPause: () => void;
    isPlaying: boolean;
    onClear?: () => void;
    playbackSpeed?: number;
    onPlaybackSpeedChange?: (speed: number) => void;
}

export function TimelineControl({ 
    frames, 
    currentFrameIndex, 
    onFrameChange, 
    onPlayPause, 
    isPlaying,
    onClear,
    playbackSpeed: externalPlaybackSpeed,
    onPlaybackSpeedChange
}: TimelineControlProps) {
    const [internalPlaybackSpeed, setInternalPlaybackSpeed] = useState(10); // FPS
    const playbackSpeed = externalPlaybackSpeed ?? internalPlaybackSpeed;
    
    const handlePlaybackSpeedChange = (speed: number) => {
        setInternalPlaybackSpeed(speed);
        onPlaybackSpeedChange?.(speed);
    };
    
    const currentFrame = frames[currentFrameIndex];
    const hasFrames = frames.length > 0;
    
    const handlePrevious = () => {
        if (currentFrameIndex > 0) {
            onFrameChange(currentFrameIndex - 1);
        }
    };
    
    const handleNext = () => {
        if (currentFrameIndex < frames.length - 1) {
            onFrameChange(currentFrameIndex + 1);
        }
    };
    
    const handleFirst = () => {
        onFrameChange(0);
    };
    
    const handleLast = () => {
        onFrameChange(frames.length - 1);
    };
    
    return (
        <Paper p="sm" withBorder style={{ backgroundColor: 'var(--mantine-color-dark-7)' }}>
            <Stack gap="xs">
                <Group justify="space-between">
                    <Group gap="xs">
                        <Text size="sm" fw={600}>Línea de Tiempo</Text>
                        <Tooltip 
                            label="Navega por el historial de la simulación. Reproduce, pausa o salta a cualquier frame para analizar la evolución temporal."
                            multiline
                            withArrow
                            style={{ maxWidth: 300 }}
                        >
                            <ActionIcon size="xs" variant="subtle" color="gray">
                                <IconHelpCircle size={14} />
                            </ActionIcon>
                        </Tooltip>
                    </Group>
                    {hasFrames && (
                        <Badge size="sm" variant="light" color="blue">
                            {currentFrameIndex + 1} / {frames.length}
                        </Badge>
                    )}
                </Group>
                
                {hasFrames ? (
                    <>
                        <Group grow>
                            <Button
                                variant="light"
                                size="xs"
                                leftSection={<IconPlayerSkipBack size={14} />}
                                onClick={handleFirst}
                                disabled={currentFrameIndex === 0}
                            >
                                Inicio
                            </Button>
                            <Button
                                variant="light"
                                size="xs"
                                leftSection={<IconPlayerSkipBack size={14} />}
                                onClick={handlePrevious}
                                disabled={currentFrameIndex === 0}
                            >
                                Anterior
                            </Button>
                            <Button
                                variant="light"
                                size="xs"
                                leftSection={isPlaying ? <IconPlayerPause size={14} /> : <IconPlayerPlay size={14} />}
                                onClick={onPlayPause}
                            >
                                {isPlaying ? 'Pausar' : 'Reproducir'}
                            </Button>
                            <Button
                                variant="light"
                                size="xs"
                                leftSection={<IconPlayerSkipForward size={14} />}
                                onClick={handleNext}
                                disabled={currentFrameIndex === frames.length - 1}
                            >
                                Siguiente
                            </Button>
                            <Button
                                variant="light"
                                size="xs"
                                leftSection={<IconPlayerSkipForward size={14} />}
                                onClick={handleLast}
                                disabled={currentFrameIndex === frames.length - 1}
                            >
                                Final
                            </Button>
                        </Group>
                        
                        <Box>
                            <Text size="xs" c="dimmed" mb="xs">
                                Frame: {currentFrame?.step ?? 'N/A'} | 
                                Velocidad: {playbackSpeed} FPS
                            </Text>
                            <Slider
                                value={currentFrameIndex}
                                onChange={onFrameChange}
                                min={0}
                                max={frames.length > 0 ? frames.length - 1 : 0}
                                step={1}
                                label={(value) => `Frame ${value + 1}`}
                                thumbSize={16}
                                color="blue"
                                marks={frames.length > 10 ? [
                                    { value: 0, label: '0' },
                                    { value: Math.floor(frames.length / 2), label: `${Math.floor(frames.length / 2)}` },
                                    { value: frames.length - 1, label: `${frames.length - 1}` }
                                ] : undefined}
                            />
                        </Box>
                        
                        {currentFrame && (
                            <Text size="xs" c="dimmed">
                                Timestamp: {new Date(currentFrame.timestamp).toLocaleTimeString()}
                            </Text>
                        )}
                        
                        <Group>
                            <Text size="xs" c="dimmed">Velocidad de reproducción:</Text>
                            <Slider
                                value={playbackSpeed}
                                onChange={handlePlaybackSpeedChange}
                                min={1}
                                max={60}
                                step={1}
                                style={{ flex: 1 }}
                                marks={[
                                    { value: 1, label: '1' },
                                    { value: 10, label: '10' },
                                    { value: 30, label: '30' },
                                    { value: 60, label: '60' }
                                ]}
                            />
                        </Group>
                        
                        {onClear && (
                            <Button
                                variant="subtle"
                                size="xs"
                                color="red"
                                onClick={onClear}
                            >
                                Limpiar Historial
                            </Button>
                        )}
                    </>
                ) : (
                    <Text size="xs" c="dimmed" ta="center" py="md">
                        No hay frames en el historial. Inicia la simulación para comenzar a acumular frames.
                    </Text>
                )}
            </Stack>
        </Paper>
    );
}

