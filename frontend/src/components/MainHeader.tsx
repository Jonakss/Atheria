// frontend/src/components/MainHeader.tsx
import { Group, Title, Box, Badge, Button } from '@mantine/core';
import { useWebSocket } from '../context/WebSocketContext';

export default function MainHeader() {
    const { connectionStatus, connect } = useWebSocket();

    const statusMap = {
        connected: { color: 'green', text: 'Conectado' },
        connecting: { color: 'yellow', text: 'Conectando' },
        disconnected: { color: 'red', text: 'Desconectado' },
        error: { color: 'red', text: 'Error' },
    };

    const currentStatus = statusMap[connectionStatus];

    // --- ¡¡MEJORA!! El botón de conectar ahora es explícito ---
    const handleConnect = () => {
        // El contexto ahora gestiona la lógica de conexión
        // Esto es solo para que el usuario inicie la acción
        // La lógica real está en el hook, pero aquí podemos llamarla
        // Por ahora, asumimos que el contexto lo reintenta.
        // En el siguiente paso, haremos que el hook exponga `connect`.
    };

    return (
        <Box style={{ width: '100%' }}>
            <Group justify="space-between" h="100%">
                <Title order={3} c="white">AETHERIA</Title>
                
                <Group>
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