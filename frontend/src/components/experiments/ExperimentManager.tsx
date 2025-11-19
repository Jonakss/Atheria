// frontend/src/components/ExperimentManager.tsx
import { useState, useMemo } from 'react';
import { 
    Paper, Stack, Group, Text, Badge, Button, Modal, 
    Table, ScrollArea, Tooltip, ActionIcon, Divider,
    Card, Timeline, Alert, Select, Box
} from '@mantine/core';
import { useDisclosure } from '@mantine/hooks';
import { useWebSocket } from '../../hooks/useWebSocket';
import { 
    IconInfoCircle, IconTransfer, IconTrash, IconDownload,
    IconChartLine, IconDatabase, IconX, IconCheck, IconClock
} from '@tabler/icons-react';

interface ExperimentNode {
    name: string;
    config: any;
    hasCheckpoint: boolean;
    loadFrom?: string;
    children: string[]; // Experimentos que usan este como base
}

export function ExperimentManager() {
    const { experimentsData, activeExperiment, setActiveExperiment, sendCommand } = useWebSocket();
    const [opened, { open, close }] = useDisclosure(false);
    const [selectedExp, setSelectedExp] = useState<string | null>(null);
    const [viewMode, setViewMode] = useState<'list' | 'tree'>('tree');
    const [sortBy, setSortBy] = useState<'created' | 'updated' | 'name' | 'training_time'>('created');

    // Construir árbol de relaciones de transfer learning
    const experimentTree = useMemo(() => {
        if (!experimentsData) return {};

        const tree: Record<string, ExperimentNode> = {};
        
        // Primero, crear nodos para todos los experimentos
        experimentsData.forEach(exp => {
            tree[exp.name] = {
                name: exp.name,
                config: exp.config,
                hasCheckpoint: exp.has_checkpoint || false,
                loadFrom: exp.config?.LOAD_FROM_EXPERIMENT || undefined,
                children: []
            };
        });

        // Luego, construir las relaciones padre-hijo
        Object.values(tree).forEach(node => {
            if (node.loadFrom && tree[node.loadFrom]) {
                tree[node.loadFrom].children.push(node.name);
            }
        });

        return tree;
    }, [experimentsData]);

    // Encontrar la raíz de un experimento (el primero en la cadena)
    const findRoot = (expName: string): string => {
        const node = experimentTree[expName];
        if (!node || !node.loadFrom) return expName;
        return findRoot(node.loadFrom);
    };

    // Obtener toda la cadena de transfer learning
    const getTransferChain = (expName: string): string[] => {
        const chain: string[] = [];
        let current = expName;
        
        while (current && experimentTree[current]) {
            chain.unshift(current);
            current = experimentTree[current].loadFrom || '';
        }
        
        return chain;
    };

    // Ordenar experimentos según el criterio seleccionado
    const sortedExperiments = useMemo(() => {
        if (!experimentsData) return [];
        
        const sorted = [...experimentsData];
        sorted.sort((a, b) => {
            switch (sortBy) {
                case 'created':
                    const aCreated = a.created_at || '';
                    const bCreated = b.created_at || '';
                    return bCreated.localeCompare(aCreated); // Más recientes primero
                case 'updated':
                    const aUpdated = a.updated_at || '';
                    const bUpdated = b.updated_at || '';
                    return bUpdated.localeCompare(aUpdated); // Más recientes primero
                case 'name':
                    return a.name.localeCompare(b.name);
                case 'training_time':
                    const aTime = a.total_training_time || 0;
                    const bTime = b.total_training_time || 0;
                    return bTime - aTime; // Más tiempo primero
                default:
                    return 0;
            }
        });
        return sorted;
    }, [experimentsData, sortBy]);
    
    // Reconstruir árbol con experimentos ordenados
    const experimentTreeSorted = useMemo(() => {
        if (!sortedExperiments) return {};
        
        const tree: Record<string, ExperimentNode> = {};
        
        sortedExperiments.forEach(exp => {
            tree[exp.name] = {
                name: exp.name,
                config: exp.config,
                hasCheckpoint: exp.has_checkpoint || false,
                loadFrom: exp.config?.LOAD_FROM_EXPERIMENT || undefined,
                children: []
            };
        });
        
        Object.values(tree).forEach(node => {
            if (node.loadFrom && tree[node.loadFrom]) {
                tree[node.loadFrom].children.push(node.name);
            }
        });
        
        return tree;
    }, [sortedExperiments]);
    
    // Agrupar experimentos por raíz (usando árbol ordenado)
    const groupedExperiments = useMemo(() => {
        const groups: Record<string, string[]> = {};
        
        Object.keys(experimentTreeSorted).forEach(expName => {
            const root = findRoot(expName);
            if (!groups[root]) {
                groups[root] = [];
            }
            groups[root].push(expName);
        });
        
        // Ordenar grupos por el primer experimento del grupo
        const sortedGroups: Record<string, string[]> = {};
        const groupKeys = Object.keys(groups).sort((a, b) => {
            const aExp = sortedExperiments.find(e => e.name === a);
            const bExp = sortedExperiments.find(e => e.name === b);
            if (!aExp || !bExp) return 0;
            
            switch (sortBy) {
                case 'created':
                    return (bExp.created_at || '').localeCompare(aExp.created_at || '');
                case 'updated':
                    return (bExp.updated_at || '').localeCompare(aExp.updated_at || '');
                case 'name':
                    return a.localeCompare(b);
                case 'training_time':
                    return (bExp.total_training_time || 0) - (aExp.total_training_time || 0);
                default:
                    return 0;
            }
        });
        
        groupKeys.forEach(key => {
            sortedGroups[key] = groups[key];
        });
        
        return sortedGroups;
    }, [experimentTreeSorted, sortedExperiments, sortBy]);

    const formatTime = (seconds: number): string => {
        if (seconds < 60) return `${Math.round(seconds)}s`;
        if (seconds < 3600) return `${Math.round(seconds / 60)}m`;
        const hours = Math.floor(seconds / 3600);
        const minutes = Math.round((seconds % 3600) / 60);
        return minutes > 0 ? `${hours}h ${minutes}m` : `${hours}h`;
    };
    
    const handleViewDetails = (expName: string) => {
        setSelectedExp(expName);
        open();
    };

    const handleDeleteExperiment = (expName: string) => {
        const exp = experimentTree[expName];
        const hasChildren = exp?.children.length > 0;
        
        let message = `¿Estás seguro de que quieres eliminar el experimento "${expName}"?\n\n`;
        message += `Esta acción eliminará:\n`;
        message += `• La configuración del experimento\n`;
        message += `• Todos los checkpoints asociados\n`;
        if (hasChildren) {
            message += `\n⚠️ ADVERTENCIA: Este experimento es base para otros experimentos:\n`;
            exp.children.forEach(child => {
                message += `  - ${child}\n`;
            });
            message += `\nLos experimentos hijos NO se eliminarán, pero perderán la referencia de transfer learning.`;
        }
        message += `\n\nEsta acción NO se puede deshacer.`;
        
        if (confirm(message)) {
            sendCommand('experiment', 'delete', { EXPERIMENT_NAME: expName });
        }
    };

    const selectedExperiment = selectedExp ? experimentTreeSorted[selectedExp] : null;
    const transferChain = selectedExp ? getTransferChain(selectedExp) : [];

    return (
        <>
            <Paper p="sm" withBorder>
                <Group justify="space-between" mb="xs">
                    <Group gap="xs">
                        <IconDatabase size={16} />
                        <Text size="sm" fw={600}>Gestión</Text>
                        <Badge size="xs" variant="light" color="blue">
                            {experimentsData?.length || 0}
                        </Badge>
                    </Group>
                    <Group gap="xs">
                        <Select
                            value={sortBy}
                            onChange={(val) => val && setSortBy(val as 'created' | 'updated' | 'name' | 'training_time')}
                            data={[
                                { value: 'created', label: 'Fecha Creación' },
                                { value: 'updated', label: 'Última Actualización' },
                                { value: 'name', label: 'Nombre' },
                                { value: 'training_time', label: 'Tiempo Entrenamiento' }
                            ]}
                            size="xs"
                            style={{ width: 140 }}
                        />
                        <Select
                            value={viewMode}
                            onChange={(val) => val && setViewMode(val as 'list' | 'tree')}
                            data={[
                                { value: 'list', label: 'Lista' },
                                { value: 'tree', label: 'Árbol' }
                            ]}
                            size="xs"
                            style={{ width: 80 }}
                        />
                        <ActionIcon
                            size="sm"
                            variant="light"
                            onClick={() => sendCommand('system', 'refresh_experiments', {})}
                        >
                            <IconCheck size={14} />
                        </ActionIcon>
                    </Group>
                </Group>

                <ScrollArea h={300}>
                    {viewMode === 'tree' ? (
                        <Stack gap="xs">
                            {Object.entries(groupedExperiments).map(([root, exps]) => {
                                const rootNode = experimentTreeSorted[root];
                                const isActive = root === activeExperiment;
                                
                                return (
                                    <Card key={root} withBorder p="xs" style={{ 
                                        backgroundColor: isActive ? 'var(--mantine-color-blue-0)' : 'transparent'
                                    }}>
                                        <Stack gap="xs">
                                            <Group justify="space-between" gap="xs">
                                                <Group gap="xs" style={{ flex: 1, minWidth: 0 }}>
                                                    <Badge 
                                                        size="xs"
                                                        color={rootNode.hasCheckpoint ? 'green' : 'orange'}
                                                        variant="light"
                                                    >
                                                        {rootNode.hasCheckpoint ? '✓' : '○'}
                                                    </Badge>
                                                    <Box style={{ flex: 1, minWidth: 0 }}>
                                                        <Text 
                                                            size="xs" 
                                                            fw={isActive ? 700 : 500}
                                                            style={{ cursor: 'pointer' }}
                                                            truncate="end"
                                                            onClick={() => setActiveExperiment(root)}
                                                        >
                                                            {root}
                                                        </Text>
                                                        {(() => {
                                                            const exp = sortedExperiments.find(e => e.name === root);
                                                            if (!exp) return null;
                                                            const created = exp.created_at ? new Date(exp.created_at).toLocaleDateString() : '';
                                                            const trainingTime = exp.total_training_time ? formatTime(exp.total_training_time) : '';
                                                            return (
                                                                <Text size="xs" c="dimmed" truncate="end">
                                                                    {trainingTime ? `${created} • ${trainingTime}` : created}
                                                                </Text>
                                                            );
                                                        })()}
                                                    </Box>
                                                    {isActive && (
                                                        <Badge size="xs" color="blue">Activo</Badge>
                                                    )}
                                                </Group>
                                                <Group gap={4}>
                                                    <Tooltip label="Ver detalles">
                                                        <ActionIcon
                                                            size="xs"
                                                            variant="subtle"
                                                            onClick={() => handleViewDetails(root)}
                                                        >
                                                            <IconInfoCircle size={12} />
                                                        </ActionIcon>
                                                    </Tooltip>
                                                    <Tooltip label="Eliminar experimento">
                                                        <ActionIcon
                                                            size="xs"
                                                            variant="subtle"
                                                            color="red"
                                                            onClick={(e) => {
                                                                e.stopPropagation();
                                                                handleDeleteExperiment(root);
                                                            }}
                                                        >
                                                            <IconTrash size={12} />
                                                        </ActionIcon>
                                                    </Tooltip>
                                                </Group>
                                            </Group>
                                            
                                            {exps.length > 1 && (
                                                <Box pl="md" style={{ borderLeft: '2px solid var(--mantine-color-blue-6)' }}>
                                                    <Stack gap={4}>
                                                        {exps.slice(1).map((expName, idx) => {
                                                            const node = experimentTreeSorted[expName];
                                                            const isChildActive = expName === activeExperiment;
                                                            
                                                            return (
                                                                <Group key={expName} gap="xs" wrap="nowrap">
                                                                    <IconTransfer size={12} color="var(--mantine-color-blue-6)" />
                                                                    <Text 
                                                                        size="xs" 
                                                                        fw={isChildActive ? 600 : 400}
                                                                        c={isChildActive ? 'blue' : undefined}
                                                                        style={{ flex: 1, minWidth: 0, cursor: 'pointer' }}
                                                                        truncate="end"
                                                                        onClick={() => setActiveExperiment(expName)}
                                                                    >
                                                                        {expName}
                                                                    </Text>
                                                                    {node.hasCheckpoint && (
                                                                        <Badge size="xs" color="green" variant="dot">✓</Badge>
                                                                    )}
                                                                </Group>
                                                            );
                                                        })}
                                                    </Stack>
                                                </Box>
                                            )}
                                        </Stack>
                                    </Card>
                                );
                            })}
                        </Stack>
                    ) : (
                        <Table fontSize="xs" verticalSpacing="xs">
                            <Table.Thead>
                                <Table.Tr>
                                    <Table.Th style={{ fontSize: '0.7rem' }}>Nombre</Table.Th>
                                    <Table.Th style={{ fontSize: '0.7rem' }}>Estado</Table.Th>
                                    <Table.Th style={{ fontSize: '0.7rem' }}>Creado</Table.Th>
                                    <Table.Th style={{ fontSize: '0.7rem' }}>Tiempo Entrenamiento</Table.Th>
                                    <Table.Th style={{ fontSize: '0.7rem' }}>Transfer</Table.Th>
                                    <Table.Th style={{ fontSize: '0.7rem' }}>Acciones</Table.Th>
                                </Table.Tr>
                            </Table.Thead>
                            <Table.Tbody>
                                {sortedExperiments.map(exp => {
                                    const node = experimentTreeSorted[exp.name];
                                    const isActive = exp.name === activeExperiment;
                                    
                                    return (
                                        <Table.Tr 
                                            key={exp.name}
                                            style={{ 
                                                backgroundColor: isActive ? 'var(--mantine-color-blue-0)' : 'transparent',
                                                cursor: 'pointer'
                                            }}
                                            onClick={() => setActiveExperiment(exp.name)}
                                        >
                                            <Table.Td>
                                                <Group gap="xs" wrap="nowrap">
                                                    <Text size="xs" fw={isActive ? 700 : 500} style={{ flex: 1, minWidth: 0 }} truncate="end">
                                                        {exp.name}
                                                    </Text>
                                                    {isActive && <Badge size="xs" color="blue">Activo</Badge>}
                                                </Group>
                                            </Table.Td>
                                            <Table.Td>
                                                <Badge 
                                                    size="xs"
                                                    color={exp.has_checkpoint ? 'green' : 'orange'}
                                                    variant="light"
                                                >
                                                    {exp.has_checkpoint ? '✓' : '○'}
                                                </Badge>
                                            </Table.Td>
                                            <Table.Td>
                                                <Text size="xs" c="dimmed">
                                                    {exp.created_at ? new Date(exp.created_at).toLocaleDateString() : 'N/A'}
                                                </Text>
                                            </Table.Td>
                                            <Table.Td>
                                                <Group gap={4}>
                                                    <IconClock size={12} />
                                                    <Text size="xs" c="dimmed">
                                                        {exp.total_training_time ? formatTime(exp.total_training_time) : '0s'}
                                                    </Text>
                                                </Group>
                                            </Table.Td>
                                            <Table.Td>
                                                {node.loadFrom ? (
                                                    <Group gap={4} wrap="nowrap">
                                                        <IconTransfer size={10} />
                                                        <Text size="xs" c="dimmed" style={{ maxWidth: 100 }} truncate="end">
                                                            {node.loadFrom}
                                                        </Text>
                                                    </Group>
                                                ) : (
                                                    <Text size="xs" c="dimmed">-</Text>
                                                )}
                                            </Table.Td>
                                            <Table.Td>
                                                <Group gap={4}>
                                                    <Tooltip label="Ver detalles">
                                                        <ActionIcon
                                                            size="xs"
                                                            variant="subtle"
                                                            onClick={(e) => {
                                                                e.stopPropagation();
                                                                handleViewDetails(exp.name);
                                                            }}
                                                        >
                                                            <IconInfoCircle size={12} />
                                                        </ActionIcon>
                                                    </Tooltip>
                                                    <Tooltip label="Eliminar experimento">
                                                        <ActionIcon
                                                            size="xs"
                                                            variant="subtle"
                                                            color="red"
                                                            onClick={(e) => {
                                                                e.stopPropagation();
                                                                handleDeleteExperiment(exp.name);
                                                            }}
                                                        >
                                                            <IconTrash size={12} />
                                                        </ActionIcon>
                                                    </Tooltip>
                                                </Group>
                                            </Table.Td>
                                        </Table.Tr>
                                    );
                                })}
                            </Table.Tbody>
                        </Table>
                    )}
                </ScrollArea>
            </Paper>

            {/* Modal de detalles */}
            <Modal
                opened={opened}
                onClose={close}
                title={`Detalles: ${selectedExp}`}
                size="lg"
            >
                {selectedExperiment && (
                    <Stack gap="md">
                        {/* Cadena de transfer learning */}
                        {transferChain.length > 1 && (
                            <Alert icon={<IconTransfer size={16} />} color="blue" variant="light">
                                <Text size="sm" fw={500} mb="xs">Cadena de Transfer Learning:</Text>
                                <Timeline active={transferChain.length - 1} bulletSize={12} lineWidth={2}>
                                    {transferChain.map((expName, idx) => (
                                        <Timeline.Item
                                            key={expName}
                                            bullet={idx === 0 ? <IconDatabase size={10} /> : <IconTransfer size={10} />}
                                            title={expName}
                                        >
                                            {idx > 0 && (
                                                <Text size="xs" c="dimmed">
                                                    Transfer desde: {transferChain[idx - 1]}
                                                </Text>
                                            )}
                                        </Timeline.Item>
                                    ))}
                                </Timeline>
                            </Alert>
                        )}

                        {/* Información temporal */}
                        {(() => {
                            const exp = sortedExperiments.find(e => e.name === selectedExp);
                            if (!exp) return null;
                            
                            return (
                                <Paper p="md" withBorder>
                                    <Text fw={600} mb="md">Información Temporal</Text>
                                    <Table>
                                        <Table.Tbody>
                                            <Table.Tr>
                                                <Table.Td><Text fw={500}>Creado:</Text></Table.Td>
                                                <Table.Td>
                                                    {exp.created_at ? new Date(exp.created_at).toLocaleString() : 'N/A'}
                                                </Table.Td>
                                            </Table.Tr>
                                            <Table.Tr>
                                                <Table.Td><Text fw={500}>Última Actualización:</Text></Table.Td>
                                                <Table.Td>
                                                    {exp.updated_at ? new Date(exp.updated_at).toLocaleString() : 'N/A'}
                                                </Table.Td>
                                            </Table.Tr>
                                            <Table.Tr>
                                                <Table.Td><Text fw={500}>Tiempo Total de Entrenamiento:</Text></Table.Td>
                                                <Table.Td>
                                                    <Group gap={4}>
                                                        <IconClock size={14} />
                                                        <Text>
                                                            {exp.total_training_time ? formatTime(exp.total_training_time) : '0s'}
                                                        </Text>
                                                    </Group>
                                                </Table.Td>
                                            </Table.Tr>
                                            {exp.last_training_time && (
                                                <Table.Tr>
                                                    <Table.Td><Text fw={500}>Última Sesión de Entrenamiento:</Text></Table.Td>
                                                    <Table.Td>
                                                        {new Date(exp.last_training_time).toLocaleString()}
                                                    </Table.Td>
                                                </Table.Tr>
                                            )}
                                        </Table.Tbody>
                                    </Table>
                                </Paper>
                            );
                        })()}
                        
                        {/* Información del experimento */}
                        <Paper p="md" withBorder>
                            <Text fw={600} mb="md">Configuración</Text>
                            <Table>
                                <Table.Tbody>
                                    <Table.Tr>
                                        <Table.Td><Text fw={500}>Arquitectura</Text></Table.Td>
                                        <Table.Td>{selectedExperiment.config?.MODEL_ARCHITECTURE || 'N/A'}</Table.Td>
                                    </Table.Tr>
                                    <Table.Tr>
                                        <Table.Td><Text fw={500}>d_state</Text></Table.Td>
                                        <Table.Td>{selectedExperiment.config?.MODEL_PARAMS?.d_state || 'N/A'}</Table.Td>
                                    </Table.Tr>
                                    <Table.Tr>
                                        <Table.Td><Text fw={500}>Hidden Channels</Text></Table.Td>
                                        <Table.Td>{selectedExperiment.config?.MODEL_PARAMS?.hidden_channels || 'N/A'}</Table.Td>
                                    </Table.Tr>
                                    <Table.Tr>
                                        <Table.Td><Text fw={500}>Learning Rate</Text></Table.Td>
                                        <Table.Td>{selectedExperiment.config?.LR_RATE_M || 'N/A'}</Table.Td>
                                    </Table.Tr>
                                    <Table.Tr>
                                        <Table.Td><Text fw={500}>Total Episodios</Text></Table.Td>
                                        <Table.Td>{selectedExperiment.config?.TOTAL_EPISODES || 'N/A'}</Table.Td>
                                    </Table.Tr>
                                    <Table.Tr>
                                        <Table.Td><Text fw={500}>Grid Size</Text></Table.Td>
                                        <Table.Td>{selectedExperiment.config?.GRID_SIZE_TRAINING || 'N/A'}</Table.Td>
                                    </Table.Tr>
                                    <Table.Tr>
                                        <Table.Td><Text fw={500}>Estado</Text></Table.Td>
                                        <Table.Td>
                                            <Badge 
                                                color={selectedExperiment.hasCheckpoint ? 'green' : 'orange'}
                                            >
                                                {selectedExperiment.hasCheckpoint ? 'Entrenado' : 'Sin entrenar'}
                                            </Badge>
                                        </Table.Td>
                                    </Table.Tr>
                                    {selectedExperiment.loadFrom && (
                                        <Table.Tr>
                                            <Table.Td><Text fw={500}>Transfer Desde</Text></Table.Td>
                                            <Table.Td>
                                                <Group gap="xs">
                                                    <IconTransfer size={14} />
                                                    <Text>{selectedExperiment.loadFrom}</Text>
                                                </Group>
                                            </Table.Td>
                                        </Table.Tr>
                                    )}
                                    {selectedExperiment.children.length > 0 && (
                                        <Table.Tr>
                                            <Table.Td><Text fw={500}>Usado por</Text></Table.Td>
                                            <Table.Td>
                                                <Stack gap="xs">
                                                    {selectedExperiment.children.map(child => (
                                                        <Badge key={child} variant="dot" color="blue">
                                                            {child}
                                                        </Badge>
                                                    ))}
                                                </Stack>
                                            </Table.Td>
                                        </Table.Tr>
                                    )}
                                </Table.Tbody>
                            </Table>
                        </Paper>

                        <Group justify="space-between">
                            <Button 
                                variant="light" 
                                color="red"
                                leftSection={<IconTrash size={16} />}
                                onClick={() => {
                                    close();
                                    handleDeleteExperiment(selectedExp!);
                                }}
                            >
                                Eliminar
                            </Button>
                            <Group>
                                <Button variant="light" onClick={close}>
                                    Cerrar
                                </Button>
                                <Button 
                                    onClick={() => {
                                        if (selectedExp) {
                                            setActiveExperiment(selectedExp);
                                        }
                                        close();
                                    }}
                                >
                                    Seleccionar
                                </Button>
                            </Group>
                        </Group>
                    </Stack>
                )}
            </Modal>
        </>
    );
}

