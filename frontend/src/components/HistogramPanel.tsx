// frontend/src/components/HistogramPanel.tsx
import { Paper, Select, Title, Box, Center, Text, Stack } from '@mantine/core';
import { BarChart } from '@mantine/charts';
import { useWebSocket } from '../hooks/useWebSocket';
import { useState } from 'react';
import classes from './LogOverlay.module.css'; // Reutilizamos el estilo
import { IconInfoCircle } from '@tabler/icons-react';

export function HistogramPanel() {
    const { simData } = useWebSocket();
    const [selectedHist, setSelectedHist] = useState('density');

    const hasHistData = simData?.hist_data && Object.keys(simData.hist_data).length > 0;

    if (!hasHistData) {
        return (
            <Paper shadow="md" p="sm" className={classes.overlay} style={{ width: 450, height: 350, left: 20, right: 'auto' }}>
                <Center h="100%">
                    <Stack align="center" gap="xs">
                        <IconInfoCircle size={32} color="gray" />
                        <Text c="dimmed" size="sm">
                            Esperando datos de simulación...
                        </Text>
                    </Stack>
                </Center>
            </Paper>
        );
    }

    const histOptions = Object.keys(simData.hist_data).map(key => ({
        value: key,
        label: key.charAt(0).toUpperCase() + key.slice(1)
    }));

    const currentData = simData.hist_data[selectedHist] || [];

    return (
        <Paper shadow="md" p="sm" className={classes.overlay} style={{ width: 450, height: 350, left: 20, right: 'auto', minHeight: 350 }}>
            <Box style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: 'var(--mantine-spacing-sm)' }}>
                <Title order={6}>Histogramas</Title>
                <Select
                    size="xs"
                    value={selectedHist}
                    onChange={(value) => setSelectedHist(value || 'density')}
                    data={histOptions}
                />
            </Box>
            <Box style={{ flexGrow: 1, minHeight: 280 }}>
                <BarChart
                    h="100%"
                    data={currentData}
                    dataKey="bin"
                    series={[{ name: 'count', color: 'blue.6' }]}
                    tickLine="y"
                    withXAxis={false}
                />
            </Box>
        </Paper>
    );
}
