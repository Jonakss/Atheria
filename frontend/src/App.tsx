// frontend/src/App.tsx
import { MantineProvider, AppShell, Burger, Group } from '@mantine/core';
import { useDisclosure } from '@mantine/hooks';
import { Notifications } from '@mantine/notifications';
import { WebSocketProvider } from './context/WebSocketContext';
import { LabSider } from './components/LabSider'; // ¡¡CORRECCIÓN!! Importación nombrada
import { PanZoomCanvas } from './components/PanZoomCanvas';
import MainHeader from './components/MainHeader';
import { LogOverlay } from './components/LogOverlay';
import '@mantine/core/styles.css';
import '@mantine/notifications/styles.css';

function App() {
    const [opened, { toggle }] = useDisclosure();

    return (
        <MantineProvider defaultColorScheme="dark">
            <Notifications />
            <WebSocketProvider>
                <AppShell
                    header={{ height: 60 }}
                    navbar={{ width: 350, breakpoint: 'sm', collapsed: { mobile: !opened } }}
                    padding="md"
                >
                    <AppShell.Header>
                        <Group h="100%" px="md">
                            <Burger opened={opened} onClick={toggle} hiddenFrom="sm" size="sm" />
                            <MainHeader />
                        </Group>
                    </AppShell.Header>

                    <AppShell.Navbar p="md">
                        <LabSider />
                    </AppShell.Navbar>

                    <AppShell.Main>
                        <PanZoomCanvas />
                        <LogOverlay />
                    </AppShell.Main>
                </AppShell>
            </WebSocketProvider>
        </MantineProvider>
    );
}

export default App;
