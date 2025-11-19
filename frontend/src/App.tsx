// frontend/src/App.tsx
import { AppShell, Burger, Group, Alert, Button } from '@mantine/core';
import { useDisclosure } from '@mantine/hooks';
import { motion } from 'framer-motion';
import { LabSider } from './components/ui/LabSider';
import { MainHeader } from './components/ui/MainHeader';
import { MainTabs } from './components/ui/MainTabs';
import { Box } from '@mantine/core';
import { IconAlertCircle, IconRefresh } from '@tabler/icons-react';
import { useWebSocket } from './hooks/useWebSocket';
import { ErrorBoundary } from './components/ui/ErrorBoundary';
import { getWebSocketUrl } from './utils/serverConfig';
import { fadeIn } from './utils/animations';
import '@mantine/core/styles.css';

function AppContent() {
    const { connectionStatus, serverConfig, reconnect } = useWebSocket();
    const [mobileOpened, { toggle: toggleMobile }] = useDisclosure();
    const [desktopOpened, { toggle: toggleDesktop }] = useDisclosure(true);

    return (
        <motion.div
            variants={fadeIn}
            initial="hidden"
            animate="visible"
            style={{ height: '100vh', width: '100%' }}
        >
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
                <Box style={{ flex: 1, position: 'relative', height: '100%' }}>
                    <MainTabs />
                </Box>
            </AppShell.Main>
        </AppShell>
        </motion.div>
    );
}

function App() {
    return (
        // Error Boundary para capturar errores y evitar pantalla gris
        // El MantineProvider ya está en main.tsx con el tema personalizado
        <ErrorBoundary>
            <AppContent />
        </ErrorBoundary>
    );
}

export default App;