// frontend/src/App.tsx
import { AppShell, Burger, Group, NavLink, Box, Tabs } from '@mantine/core';
import { useDisclosure } from '@mantine/hooks';
import { MainHeader } from './components/MainHeader';
import { LabSider } from './components/LabSider';
import { PanZoomCanvas } from './components/PanZoomCanvas';
import { LogOverlay } from './components/LogOverlay';
import { HistogramPanel } from './components/HistogramPanel';
import { IconChartBar, IconFileText } from '@tabler/icons-react';

function App() {
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
                <Tabs defaultValue="inference" style={{ height: '100%', display: 'flex', flexDirection: 'column' }}>
                    <Tabs.List>
                        <Tabs.Tab value="inference" leftSection={<IconChartBar size={14} />}>
                            Inferencia y Visualización
                        </Tabs.Tab>
                        <Tabs.Tab value="training" leftSection={<IconFileText size={14} />}>
                            Log de Entrenamiento
                        </Tabs.Tab>
                    </Tabs.List>

                    <Tabs.Panel value="inference" style={{ flex: 1, position: 'relative' }}>
                        <PanZoomCanvas />
                        <HistogramPanel />
                    </Tabs.Panel>

                    <Tabs.Panel value="training" style={{ flex: 1, position: 'relative', paddingTop: '1rem' }}>
                        {/* El LogOverlay ahora es relativo a este panel */}
                        <LogOverlay />
                    </Tabs.Panel>
                </Tabs>
            </AppShell.Main>
        </AppShell>
    );
}

export default App;