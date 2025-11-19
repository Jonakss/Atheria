// frontend/src/components/HistogramPanel.tsx
import { Paper, Select, Title, Box, Center, Text, Stack, Tooltip, ActionIcon, Group } from '@mantine/core';
import { BarChart } from '@mantine/charts';
import { useWebSocket } from '../../hooks/useWebSocket';
import { useState, useEffect, useRef } from 'react';
import { motion } from 'framer-motion';
import { hoverLift } from '../../utils/animations';
import classes from '../ui/LogOverlay.module.css'; // Reutilizamos el estilo
import { IconInfoCircle, IconHelpCircle } from '@tabler/icons-react';

export function HistogramPanel() {
    const { simData } = useWebSocket();
    const [selectedHist, setSelectedHist] = useState('density');
    const [chartReady, setChartReady] = useState(false);
    const chartContainerRef = useRef<HTMLDivElement>(null);

    const hasHistData = simData?.hist_data && Object.keys(simData.hist_data || {}).length > 0;
    const currentData = hasHistData && simData?.hist_data ? (simData.hist_data[selectedHist] || []) : [];
    
    // Verificar dimensiones del contenedor usando ResizeObserver
    useEffect(() => {
        if (!chartContainerRef.current) return;
        
        const resizeObserver = new ResizeObserver((entries) => {
            for (const entry of entries) {
                const { width, height } = entry.contentRect;
                if (width > 0 && height > 0) {
                    setChartReady(true);
                } else {
                    setChartReady(false);
                }
            }
        });
        
        resizeObserver.observe(chartContainerRef.current);
        
        // También verificar inmediatamente
        const checkNow = () => {
            if (chartContainerRef.current) {
                const { width, height } = chartContainerRef.current.getBoundingClientRect();
                if (width > 0 && height > 0) {
                    setChartReady(true);
                }
            }
        };
        const timer = setTimeout(checkNow, 50);
        
        return () => {
            resizeObserver.disconnect();
            clearTimeout(timer);
        };
    }, [selectedHist, currentData.length]);

    if (!hasHistData || !simData?.hist_data) {
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

    return (
        <motion.div variants={hoverLift} whileHover="hover" style={{ position: 'absolute', left: 20, top: 'auto' }}>
            <Paper 
                shadow="md" 
                p="sm" 
                className={classes.overlay} 
                style={{ 
                    width: 450, 
                    height: 350, 
                    minHeight: 350,
                    transition: 'all 0.3s ease-in-out'
                }}
            >
            <Box style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: 'var(--mantine-spacing-sm)' }}>
                <Group gap="xs">
                <Title order={6}>Histogramas</Title>
                    <Tooltip 
                        label="Distribución de valores en el estado cuántico. Útil para entender la estadística del sistema (media, varianza, outliers)."
                        multiline
                        width={300}
                        withArrow
                    >
                        <ActionIcon size="xs" variant="subtle" color="gray">
                            <IconHelpCircle size={14} />
                        </ActionIcon>
                    </Tooltip>
                </Group>
                <Select
                    size="xs"
                    value={selectedHist}
                    onChange={(value) => setSelectedHist(value || 'density')}
                    data={histOptions}
                />
            </Box>
            <Box ref={chartContainerRef} style={{ flexGrow: 1, minHeight: 280, minWidth: 300, width: '100%', position: 'relative' }}>
                {chartReady && currentData && currentData.length > 0 && chartContainerRef.current ? (
                    (() => {
                        const rect = chartContainerRef.current?.getBoundingClientRect();
                        if (rect && rect.width > 0 && rect.height > 0) {
                            return (
                    <BarChart
                        h={280}
                        w="100%"
                        data={currentData}
                        dataKey="bin"
                        series={[{ name: 'count', color: 'blue.6' }]}
                        tickLine="y"
                        withXAxis={false}
                                    style={{ minWidth: 0, minHeight: 0 }}
                    />
                            );
                        }
                        return null;
                    })()
                ) : (
                    <Center h={280}>
                        <Text c="dimmed" size="sm">No hay datos disponibles</Text>
                    </Center>
                )}
                    </Box>
            </Paper>
        </motion.div>
    );
}
