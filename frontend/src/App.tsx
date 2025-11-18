// frontend/src/App.tsx
import { AppShell, MantineProvider, Alert, Button } from '@mantine/core';
import { useDisclosure } from '@mantine/hooks';
import { LabSider } from './components/LabSider';
import { MainHeader } from './components/MainHeader';
import { MainTabs } from './components/MainTabs';
import { Box } from '@mantine/core';
import { IconAlertCircle, IconRefresh } from '@tabler/icons-react';
import { WebSocketProvider } from './context/WebSocketContext';
import { useWebSocket } from './hooks/useWebSocket'; // <-- ¡LA IMPORTACIÓN CLAVE!
import { ErrorBoundary } from './components/ErrorBoundary';
import { getWebSocketUrl } from './utils/serverConfig';
import '@mantine/core/styles.css'; // Importa los estilos de Mantine

function AppContent() {
    const { connectionStatus, serverConfig, reconnect, disconnect } = useWebSocket();
    const [mobileOpened, { toggle: toggleMobile }] = useDisclosure();
    const [desktopOpened, { toggle: toggleDesktop }] = useDisclosure(true);

    return (
        <AppShell
            header={{ height: 60 }}
            navbar={{
                width: 400,
                breakpoint: 'sm',
                collapsed: { mobile: !mobileOpened, desktop: !desktopOpened },
            }}
            padding="md"
        >
            <AppShell.Header>
                <MainHeader
                    mobileOpened={mobileOpened}
                    desktopOpened={desktopOpened}
                    toggleMobile={toggleMobile}
                    toggleDesktop={toggleDesktop}
                />
            </AppShell.Header>

            <AppShell.Navbar p="md">
                <LabSider />
            </AppShell.Navbar>

            <AppShell.Main>
                {connectionStatus === 'server_unavailable' && (
                    <Alert
                        icon={<IconAlertCircle size={16} />}
                        title="Servidor no disponible"
                        color="red"
                        mb="md"
                    >
                        No se puede conectar al servidor en <code>{getWebSocketUrl(serverConfig)}</code>.
                        {serverConfig.host === 'localhost' ? (
                            <> Asegúrate de que el servidor esté ejecutándose con <code>python run_server.py</code></>
                        ) : (
                            <> Verifica que el servidor esté ejecutándose y que la configuración sea correcta.</>
                        )}
                        <Button
                            leftSection={<IconRefresh size={16} />}
                            onClick={reconnect}
                            mt="sm"
                            variant="light"
                            color="blue"
                        >
                            Intentar Reconectar
                        </Button>
                    </Alert>
                )}
                {connectionStatus === 'connected' && (
                    <Alert
                        icon={<IconAlertCircle size={16} />}
                        title="Conectado"
                        color="green"
                        mb="md"
                    >
                        Conectado al servidor en <code>{getWebSocketUrl(serverConfig)}</code>.
                        <Button
                            leftSection={<IconRefresh size={16} />}
                            onClick={disconnect}
                            mt="sm"
                            variant="light"
                            color="red"
                        >
                            Desconectar
                        </Button>
                    </Alert>
                )}
                <Box style={{ flex: 1, position: 'relative', height: '100%' }}>
                    <MainTabs />
                </Box>
            </AppShell.Main>
        </AppShell>
    );
}

function App() {
    return (
        // Envolvemos todo en el Provider de Mantine para estilos consistentes
        <MantineProvider defaultColorScheme="dark">
            {/* Error Boundary para capturar errores y evitar pantalla gris */}
            <ErrorBoundary>
                {/* ¡AQUÍ ESTÁ LA MAGIA! Envolvemos la App en nuestro WebSocketProvider */}
                <WebSocketProvider>
                    <AppContent />
                </WebSocketProvider>
            </ErrorBoundary>
        </MantineProvider>
    );
}

export default App;