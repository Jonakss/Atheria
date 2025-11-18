// frontend/src/components/MainTabs.tsx
import { Tabs, Box } from '@mantine/core';
import { IconChartBar, IconDatabase, IconBook, IconChartScatter, IconHistory, IconSettings } from '@tabler/icons-react';
import { VisualizationTab } from './VisualizationTab';
import { DataTab } from './DataTab';
import { LogbookTab } from './LogbookTab';
import { AnalysisTab } from './AnalysisTab';
import { HistoryViewer } from './HistoryViewer';
import { InferenceConfigTab } from './InferenceConfigTab';

export function MainTabs() {
    return (
        <Tabs defaultValue="visualization" style={{ height: '100%', display: 'flex', flexDirection: 'column' }}>
            <Tabs.List>
                <Tabs.Tab value="visualization" leftSection={<IconChartBar size={16} />}>
                    Visualización
                </Tabs.Tab>
                <Tabs.Tab value="analysis" leftSection={<IconChartScatter size={16} />}>
                    Análisis t-SNE
                </Tabs.Tab>
                <Tabs.Tab value="data" leftSection={<IconDatabase size={16} />}>
                    Datos y Estadísticas
                </Tabs.Tab>
                <Tabs.Tab value="logbook" leftSection={<IconBook size={16} />}>
                    Bitácora
                </Tabs.Tab>
                <Tabs.Tab value="history" leftSection={<IconHistory size={16} />}>
                    Historia
                </Tabs.Tab>
                <Tabs.Tab value="inference" leftSection={<IconSettings size={16} />}>
                    Configuración
                </Tabs.Tab>
            </Tabs.List>

            <Tabs.Panel value="visualization" style={{ flex: 1, position: 'relative', overflow: 'hidden' }}>
                <VisualizationTab />
            </Tabs.Panel>

            <Tabs.Panel value="analysis" style={{ flex: 1, position: 'relative', overflow: 'auto' }}>
                <AnalysisTab />
            </Tabs.Panel>

            <Tabs.Panel value="data" style={{ flex: 1, position: 'relative', overflow: 'auto' }}>
                <DataTab />
            </Tabs.Panel>

            <Tabs.Panel value="logbook" style={{ flex: 1, position: 'relative', overflow: 'hidden', display: 'flex', flexDirection: 'column' }}>
                <LogbookTab />
            </Tabs.Panel>

            <Tabs.Panel value="history" style={{ flex: 1, position: 'relative', overflow: 'auto' }}>
                <HistoryViewer />
            </Tabs.Panel>

            <Tabs.Panel value="inference" style={{ flex: 1, position: 'relative', overflow: 'auto' }}>
                <InferenceConfigTab />
            </Tabs.Panel>
        </Tabs>
    );
}

