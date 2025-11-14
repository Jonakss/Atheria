// frontend/src/components/MainHeader.tsx
import { Group, Title, Box, Badge, Button, ActionIcon } from '@mantine/core';
import { IconPlayerPlay, IconPlayerPause } from '@tabler/icons-react';
import { useWebSocket } from '../hooks/useWebSocket';

export default function MainHeader() {
    const { connectionStatus, connect, sendCommand, inferenceStatus } = useWebSocket();

    const statusMap = {
        connected: { color: 'green', text: 'Conectado' },
        connecting: { color: 'yellow', text: 'Conectando' },
        disconnected: { color: 'red', text: 'Desconectado' },
        error: { color: 'red', text: 'Error' },
    };

    const currentStatus = statusMap[connectionStatus];

    const toggleInference = () => {
        const command = inferenceStatus === 'running' ? 'pause' : 'play';
        sendCommand('inference', command);
    };

    return (
        <Box style={{ width: '100%' }}>
            <Group justify="space-between" h="100%">
                <Title order={3} c="white">AETHERIA</Title>
                
                <Group>
                    {/* --- ¡¡NUEVO!! Botón de Play/Pause --- */}
                    <ActionIcon onClick={toggleInference} variant="default" size="lg" disabled={connectionStatus !== 'connected'}>
                        {inferenceStatus === 'running' ? <IconPlayerPause size={18} /> : <IconPlayerPlay size={18} />}
                    </ActionIcon>

                    <Badge color={currentStatus.color} variant="light">
                        {currentStatus.text}
                    </Badge>
                    <Button 
                        variant="default" 
                        size="xs" 
                        onClick={connect}
                        disabled={connectionStatus === 'connected' || connectionStatus === 'connecting'}
                    >
                        Conectar
                    </Button>
                </Group>
            </Group>
        </Box>
    );
}