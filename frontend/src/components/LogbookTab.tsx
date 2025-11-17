// frontend/src/components/LogbookTab.tsx
import { useState, useEffect, useRef } from 'react';
import { 
    Paper, Text, ScrollArea, Group, Badge, Stack, Box, 
    Tabs, Button, TextInput, ActionIcon, Select 
} from '@mantine/core';
import { useWebSocket } from '../hooks/useWebSocket';
import { 
    IconSearch, IconX, IconFilter, IconDownload,
    IconClock, IconInfoCircle, IconAlertTriangle, IconCheck,
    IconArrowUp, IconArrowDown
} from '@tabler/icons-react';

type LogType = 'all' | 'training' | 'simulation' | 'error' | 'info';

interface LogEntry {
    id: string;
    message: string;
    type: LogType;
    timestamp: number;
    source: 'training' | 'simulation' | 'system';
}

export function LogbookTab() {
    const { allLogs, trainingLog } = useWebSocket();
    const [filteredLogs, setFilteredLogs] = useState<LogEntry[]>([]);
    const [selectedType, setSelectedType] = useState<LogType>('all');
    const [searchQuery, setSearchQuery] = useState('');
    const [autoScroll, setAutoScroll] = useState(false); // Desactivado por defecto
    const scrollRef = useRef<HTMLDivElement>(null);
    const scrollViewportRef = useRef<HTMLDivElement>(null);
    const logsEndRef = useRef<HTMLDivElement>(null);

    // Procesar logs y convertirlos a LogEntry
    useEffect(() => {
        const processedLogs: LogEntry[] = allLogs.map((log, index) => {
            const id = `log-${index}-${Date.now()}`;
            let type: LogType = 'info';
            let source: 'training' | 'simulation' | 'system' = 'system';

            // Detectar tipo de log
            if (log.includes('[Entrenamiento]') || log.includes('[Error]')) {
                source = 'training';
                if (log.includes('[Error]') || log.toLowerCase().includes('error')) {
                    type = 'error';
                } else {
                    type = 'training';
                }
            } else if (log.includes('[Simulación]')) {
                source = 'simulation';
                type = 'simulation';
            } else if (log.toLowerCase().includes('error') || log.toLowerCase().includes('❌')) {
                type = 'error';
            }

            return {
                id,
                message: log,
                type,
                timestamp: Date.now() - (allLogs.length - index) * 100, // Aproximado
                source
            };
        });

        // Aplicar filtros
        let filtered = processedLogs;

        if (selectedType !== 'all') {
            filtered = filtered.filter(log => log.type === selectedType);
        }

        if (searchQuery) {
            const query = searchQuery.toLowerCase();
            filtered = filtered.filter(log => 
                log.message.toLowerCase().includes(query)
            );
        }

        setFilteredLogs(filtered);
    }, [allLogs, selectedType, searchQuery]);

    // Auto-scroll al final solo si está habilitado y el usuario no ha hecho scroll manual
    useEffect(() => {
        if (autoScroll && scrollViewportRef.current && logsEndRef.current) {
            const viewport = scrollViewportRef.current;
            // Solo hacer scroll si ya está cerca del final (dentro de 100px)
            const isNearBottom = viewport.scrollHeight - viewport.scrollTop - viewport.clientHeight < 100;
            if (isNearBottom) {
                logsEndRef.current.scrollIntoView({ behavior: 'smooth' });
            }
        }
    }, [filteredLogs, autoScroll]);

    // Función para ir arriba
    const scrollToTop = () => {
        if (scrollViewportRef.current) {
            scrollViewportRef.current.scrollTo({ top: 0, behavior: 'smooth' });
        }
    };

    // Función para ir abajo
    const scrollToBottom = () => {
        if (logsEndRef.current) {
            logsEndRef.current.scrollIntoView({ behavior: 'smooth' });
        }
    };

    const getLogIcon = (type: LogType) => {
        switch (type) {
            case 'error':
                return <IconAlertTriangle size={14} color="red" />;
            case 'training':
                return <IconInfoCircle size={14} color="blue" />;
            case 'simulation':
                return <IconClock size={14} color="green" />;
            default:
                return <IconInfoCircle size={14} color="gray" />;
        }
    };

    const getLogColor = (type: LogType) => {
        switch (type) {
            case 'error':
                return 'red';
            case 'training':
                return 'blue';
            case 'simulation':
                return 'green';
            default:
                return 'gray';
        }
    };

    const formatTimestamp = (timestamp: number) => {
        const date = new Date(timestamp);
        return date.toLocaleTimeString();
    };

    const exportLogs = () => {
        const content = filteredLogs.map(log => 
            `[${formatTimestamp(log.timestamp)}] [${log.type.toUpperCase()}] ${log.message}`
        ).join('\n');
        
        const blob = new Blob([content], { type: 'text/plain' });
        const url = URL.createObjectURL(blob);
        const a = document.createElement('a');
        a.href = url;
        a.download = `logbook-${new Date().toISOString()}.txt`;
        a.click();
        URL.revokeObjectURL(url);
    };

    return (
        <Stack gap="md" h="100%" p="md" style={{ overflow: 'hidden', display: 'flex', flexDirection: 'column' }}>
            {/* Header con controles */}
            <Paper p="md" withBorder>
                <Group justify="space-between" mb="md">
                    <Group>
                        <Text fw={700} size="lg">Bitácora del Sistema</Text>
                        <Badge color="blue" variant="light">
                            {filteredLogs.length} entradas
                        </Badge>
                    </Group>
                    <Group>
                        <Button
                            leftSection={<IconArrowUp size={16} />}
                            variant="light"
                            size="sm"
                            onClick={scrollToTop}
                            disabled={filteredLogs.length === 0}
                        >
                            Ir Arriba
                        </Button>
                        <Button
                            leftSection={<IconArrowDown size={16} />}
                            variant="light"
                            size="sm"
                            onClick={scrollToBottom}
                            disabled={filteredLogs.length === 0}
                        >
                            Ir Abajo
                        </Button>
                        <Button
                            leftSection={<IconDownload size={16} />}
                            variant="light"
                            size="sm"
                            onClick={exportLogs}
                            disabled={filteredLogs.length === 0}
                        >
                            Exportar
                        </Button>
                    </Group>
                </Group>

                <Group>
                    <TextInput
                        placeholder="Buscar en logs..."
                        leftSection={<IconSearch size={16} />}
                        rightSection={
                            searchQuery && (
                                <ActionIcon onClick={() => setSearchQuery('')}>
                                    <IconX size={16} />
                                </ActionIcon>
                            )
                        }
                        value={searchQuery}
                        onChange={(e) => setSearchQuery(e.currentTarget.value)}
                        style={{ flex: 1 }}
                    />
                    <Group>
                        <Select
                            placeholder="Filtrar por tipo"
                            leftSection={<IconFilter size={16} />}
                            data={[
                                { value: 'all', label: 'Todos' },
                                { value: 'training', label: 'Entrenamiento' },
                                { value: 'simulation', label: 'Simulación' },
                                { value: 'error', label: 'Errores' },
                                { value: 'info', label: 'Información' }
                            ]}
                            value={selectedType}
                            onChange={(value) => setSelectedType(value as LogType)}
                            style={{ width: 200 }}
                        />
                        <Button
                            variant={autoScroll ? 'filled' : 'light'}
                            size="sm"
                            onClick={() => setAutoScroll(!autoScroll)}
                        >
                            Auto-scroll: {autoScroll ? 'ON' : 'OFF'}
                        </Button>
                    </Group>
                </Group>
            </Paper>

            {/* Lista de logs */}
            <Paper p="md" withBorder style={{ flex: 1, display: 'flex', flexDirection: 'column', minHeight: 0 }}>
                <ScrollArea 
                    h="100%" 
                    viewportRef={scrollViewportRef}
                    style={{ flex: 1 }}
                >
                    <Stack gap="xs" style={{ paddingBottom: '1rem' }}>
                        {filteredLogs.length === 0 ? (
                            <Box p="xl">
                                <Text c="dimmed" ta="center">
                                    No hay logs que coincidan con los filtros seleccionados.
                                </Text>
                            </Box>
                        ) : (
                            filteredLogs.map((log) => (
                                <Paper
                                    key={log.id}
                                    p="xs"
                                    withBorder
                                    style={{
                                        backgroundColor: log.type === 'error' 
                                            ? 'var(--mantine-color-red-0)' 
                                            : 'transparent'
                                    }}
                                >
                                    <Group gap="xs" align="flex-start" wrap="nowrap">
                                        {getLogIcon(log.type)}
                                        <Badge size="xs" color={getLogColor(log.type)} variant="light">
                                            {log.type}
                                        </Badge>
                                        <Text size="xs" c="dimmed" style={{ minWidth: 80 }}>
                                            {formatTimestamp(log.timestamp)}
                                        </Text>
                                        <Text size="sm" style={{ flex: 1, wordBreak: 'break-word' }}>
                                            {log.message}
                                        </Text>
                                    </Group>
                                </Paper>
                            ))
                        )}
                        <div ref={logsEndRef} />
                    </Stack>
                </ScrollArea>
            </Paper>
        </Stack>
    );
}

