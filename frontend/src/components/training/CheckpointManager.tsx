// frontend/src/components/CheckpointManager.tsx
import { useState, useEffect, useMemo } from 'react';
import { 
    Paper, Stack, Group, Text, Badge, Button, Modal, 
    Table, ScrollArea, Tooltip, ActionIcon, Divider,
    Card, Alert, TextInput, Textarea, 
    Tabs, Progress, Center, Menu
} from '@mantine/core';
import { useDisclosure } from '@mantine/hooks';
import { useWebSocket } from '../hooks/useWebSocket';
import { 
    IconDownload, IconTrash, IconInfoCircle, IconFile,
    IconClock, IconX, IconEdit, IconBook,
    IconSearch, IconStar, IconDotsVertical,
    IconRefresh
} from '@tabler/icons-react';
import { notifications } from '@mantine/notifications';

interface CheckpointInfo {
    filename: string;
    episode: number;
    size: number;
    modified: string;
    is_best: boolean;
}

interface ExperimentNote {
    id: string;
    timestamp: number;
    content: string;
    checkpoint?: string; // Checkpoint asociado
}

export function CheckpointManager() {
    const { activeExperiment, sendCommand, experimentsData } = useWebSocket();
    const [opened, { open, close }] = useDisclosure(false);
    const [checkpoints, setCheckpoints] = useState<CheckpointInfo[]>([]);
    const [notes, setNotes] = useState<ExperimentNote[]>([]);
    const [newNote, setNewNote] = useState('');
    const [selectedCheckpoint, setSelectedCheckpoint] = useState<string | null>(null);
    const [loading, setLoading] = useState(false);
    const [searchQuery, setSearchQuery] = useState('');
    const [activeTab, setActiveTab] = useState<string>('checkpoints');

    // Cargar checkpoints y notas cuando se abre el modal
    useEffect(() => {
        if (opened && activeExperiment) {
            loadCheckpoints();
            loadNotes();
        }
    }, [opened, activeExperiment]);

    useEffect(() => {
        // Escuchar eventos de checkpoints actualizados
        const handleCheckpointsUpdate = (event: CustomEvent) => {
            setCheckpoints(event.detail || []);
            setLoading(false);
            if (event.detail && event.detail.length > 0) {
                notifications.show({
                    title: 'Checkpoints actualizados',
                    message: `Se encontraron ${event.detail.length} checkpoints`,
                    color: 'green',
                    autoClose: 2000,
                });
            }
        };
        
        // Escuchar cambio de pestaña
        const handleSwitchToNotes = () => {
            setActiveTab('notes');
        };
        
        window.addEventListener('checkpoints_updated', handleCheckpointsUpdate as EventListener);
        window.addEventListener('switch_to_notes_tab', handleSwitchToNotes);
        
        return () => {
            window.removeEventListener('checkpoints_updated', handleCheckpointsUpdate as EventListener);
            window.removeEventListener('switch_to_notes_tab', handleSwitchToNotes);
        };
    }, []);

    const loadCheckpoints = async () => {
        if (!activeExperiment) return;
        setLoading(true);
        sendCommand('experiment', 'list_checkpoints', { EXPERIMENT_NAME: activeExperiment });
    };

    const loadNotes = () => {
        if (!activeExperiment) return;
        const savedNotes = localStorage.getItem(`experiment_notes_${activeExperiment}`);
        if (savedNotes) {
            try {
                setNotes(JSON.parse(savedNotes));
            } catch (e) {
                console.error('Error loading notes:', e);
            }
        }
    };

    const saveNotes = (updatedNotes: ExperimentNote[]) => {
        if (!activeExperiment) return;
        localStorage.setItem(`experiment_notes_${activeExperiment}`, JSON.stringify(updatedNotes));
        setNotes(updatedNotes);
    };

    const addNote = () => {
        if (!newNote.trim() || !activeExperiment) return;
        
        const note: ExperimentNote = {
            id: `note-${Date.now()}`,
            timestamp: Date.now(),
            content: newNote,
            checkpoint: selectedCheckpoint || undefined
        };
        
        saveNotes([...notes, note]);
        setNewNote('');
        setSelectedCheckpoint(null);
        notifications.show({
            title: 'Nota agregada',
            message: 'La nota se ha guardado correctamente',
            color: 'green',
            autoClose: 2000,
        });
    };

    const deleteNote = (noteId: string) => {
        saveNotes(notes.filter(n => n.id !== noteId));
        notifications.show({
            title: 'Nota eliminada',
            message: 'La nota se ha eliminado correctamente',
            color: 'blue',
            autoClose: 2000,
        });
    };

    const deleteCheckpoint = (checkpointName: string) => {
        if (!activeExperiment) return;
        const checkpoint = checkpoints.find(c => c.filename === checkpointName);
        const isBest = checkpoint?.is_best;
        
        if (window.confirm(
            `¿Eliminar checkpoint "${checkpointName}"?\n\n` +
            `Episodio: ${checkpoint?.episode}\n` +
            `${isBest ? '⚠️ Este es el mejor checkpoint.\n' : ''}` +
            `Esta acción no se puede deshacer.`
        )) {
            sendCommand('experiment', 'delete_checkpoint', { 
                EXPERIMENT_NAME: activeExperiment,
                CHECKPOINT_NAME: checkpointName
            });
            // Recargar después de un breve delay
            setTimeout(() => loadCheckpoints(), 500);
            notifications.show({
                title: 'Checkpoint eliminado',
                message: `Se está eliminando ${checkpointName}`,
                color: 'orange',
                autoClose: 2000,
            });
        }
    };

    const formatFileSize = (bytes: number) => {
        if (bytes < 1024) return `${bytes} B`;
        if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(2)} KB`;
        if (bytes < 1024 * 1024 * 1024) return `${(bytes / (1024 * 1024)).toFixed(2)} MB`;
        return `${(bytes / (1024 * 1024 * 1024)).toFixed(2)} GB`;
    };

    const formatDate = (dateString: string) => {
        const date = new Date(dateString);
        const now = new Date();
        const diffMs = now.getTime() - date.getTime();
        const diffMins = Math.floor(diffMs / 60000);
        const diffHours = Math.floor(diffMs / 3600000);
        const diffDays = Math.floor(diffMs / 86400000);

        if (diffMins < 1) return 'Hace un momento';
        if (diffMins < 60) return `Hace ${diffMins} min`;
        if (diffHours < 24) return `Hace ${diffHours} h`;
        if (diffDays < 7) return `Hace ${diffDays} días`;
        return date.toLocaleDateString() + ' ' + date.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });
    };

    // Filtrar checkpoints según búsqueda
    const filteredCheckpoints = useMemo(() => {
        if (!searchQuery.trim()) return checkpoints;
        const query = searchQuery.toLowerCase();
        return checkpoints.filter(c => 
            c.filename.toLowerCase().includes(query) ||
            c.episode.toString().includes(query)
        );
    }, [checkpoints, searchQuery]);

    const currentExperiment = activeExperiment 
        ? experimentsData?.find(exp => exp.name === activeExperiment) 
        : null;

    const bestCheckpoint = checkpoints.find(c => c.is_best);
    const latestCheckpoint = checkpoints.length > 0 
        ? checkpoints.reduce((latest, current) => 
            current.episode > latest.episode ? current : latest
          )
        : null;
    const totalSize = checkpoints.reduce((sum, c) => sum + c.size, 0);

    const handleCleanup = () => {
        if (!activeExperiment) return;
        if (window.confirm(`¿Estás seguro de eliminar los checkpoints antiguos de "${activeExperiment}"? Se mantendrán los 5 más recientes y el mejor.`)) {
             sendCommand('experiment', 'cleanup_checkpoints', { EXPERIMENT_NAME: activeExperiment });
             // Recargar después de un breve delay
             setTimeout(() => loadCheckpoints(), 1000);
        }
    };

    return (
        <>
            <Button
                variant="light"
                size="sm"
                leftSection={<IconFile size={14} />}
                onClick={open}
                disabled={!activeExperiment}
                fullWidth
            >
                Gestionar Checkpoints
            </Button>

            <Modal
                opened={opened}
                onClose={close}
                title={
                    <Group gap="xs">
                        <IconFile size={20} />
                        <Text fw={600}>Checkpoints y Notas</Text>
                        {activeExperiment && (
                            <Badge variant="light" size="lg">
                                {activeExperiment}
                            </Badge>
                        )}
                    </Group>
                }
                size="xl"
                styles={{ body: { padding: 'var(--mantine-spacing-md)' } }}
            >
                {activeExperiment && checkpoints.length > 0 && (
                     <Alert 
                        variant="light" 
                        color="blue" 
                        title="Estado del Almacenamiento" 
                        icon={<IconInfoCircle size={16} />}
                        style={{ marginBottom: '1rem' }}
                    >
                        <Group justify="space-between" align="center">
                            <Text size="sm">
                                Total ocupado: <b>{formatFileSize(totalSize)}</b> en {checkpoints.length} checkpoints.
                            </Text>
                            <Button 
                                size="xs" 
                                variant="subtle" 
                                color="red" 
                                leftSection={<IconTrash size={14} />}
                                onClick={handleCleanup}
                            >
                                Limpiar antiguos
                            </Button>
                        </Group>
                    </Alert>
                )}

                <Tabs value={activeTab} onChange={(value) => value && setActiveTab(value)}>
                    <Tabs.List>
                        <Tabs.Tab 
                            value="checkpoints" 
                            leftSection={<IconFile size={16} />}
                        >
                            Checkpoints
                            {checkpoints.length > 0 && (
                                <Badge size="xs" variant="filled" ml={8}>
                                    {checkpoints.length}
                                </Badge>
                            )}
                        </Tabs.Tab>
                        <Tabs.Tab 
                            value="notes" 
                            leftSection={<IconBook size={16} />}
                        >
                            Notas
                            {notes.length > 0 && (
                                <Badge size="xs" variant="filled" ml={8}>
                                    {notes.length}
                                </Badge>
                            )}
                        </Tabs.Tab>
                    </Tabs.List>

                    <Tabs.Panel value="checkpoints" pt="md">
                        <Stack gap="md">
                            {/* Información del experimento */}
                            {currentExperiment && (
                                <Card withBorder p="md" style={{ backgroundColor: 'var(--mantine-color-dark-7)' }}>
                                    <Stack gap="xs">
                                        <Group justify="space-between">
                                            <Group gap="xs">
                                                <Text fw={600} size="sm">Experimento</Text>
                                                <Badge 
                                                    color={currentExperiment.has_checkpoint ? 'green' : 'orange'}
                                                    variant="light"
                                                >
                                                    {currentExperiment.has_checkpoint ? '✓ Entrenado' : '○ Sin entrenar'}
                                                </Badge>
                                            </Group>
                                        </Group>
                                        <Group gap="md">
                                            <Text size="xs" c="dimmed">
                                                <strong>Arquitectura:</strong> {currentExperiment.config?.MODEL_ARCHITECTURE || 'N/A'}
                                    </Text>
                                        </Group>
                                    </Stack>
                                </Card>
                            )}

                            {/* Estadísticas rápidas */}
                            {checkpoints.length > 0 && (
                                <Group grow>
                                    <Card withBorder p="sm" style={{ backgroundColor: 'var(--mantine-color-dark-7)' }}>
                                        <Stack gap={4} align="center">
                                            <Text size="xs" c="dimmed">Total Checkpoints</Text>
                                            <Text size="xl" fw={700}>{checkpoints.length}</Text>
                                        </Stack>
                                    </Card>
                                    {bestCheckpoint && (
                                        <Card withBorder p="sm" style={{ backgroundColor: 'var(--mantine-color-green-9)', opacity: 0.2 }}>
                                            <Stack gap={4} align="center">
                                                <Group gap={4}>
                                                    <IconStar size={14} />
                                                    <Text size="xs" c="dimmed">Mejor</Text>
                                                </Group>
                                                <Text size="xl" fw={700}>Ep. {bestCheckpoint.episode}</Text>
                                            </Stack>
                                        </Card>
                                    )}
                                    {latestCheckpoint && (
                                        <Card withBorder p="sm" style={{ backgroundColor: 'var(--mantine-color-dark-7)' }}>
                                            <Stack gap={4} align="center">
                                                <Text size="xs" c="dimmed">Último</Text>
                                                <Text size="xl" fw={700}>Ep. {latestCheckpoint.episode}</Text>
                                            </Stack>
                                        </Card>
                                    )}
                                    <Card withBorder p="sm" style={{ backgroundColor: 'var(--mantine-color-dark-7)' }}>
                                        <Stack gap={4} align="center">
                                            <Text size="xs" c="dimmed">Tamaño Total</Text>
                                            <Text size="lg" fw={600}>{formatFileSize(totalSize)}</Text>
                                        </Stack>
                                    </Card>
                                </Group>
                            )}

                            {/* Barra de búsqueda y acciones */}
                            <Group justify="space-between">
                                <TextInput
                                    placeholder="Buscar por nombre o episodio..."
                                    leftSection={<IconSearch size={16} />}
                                    value={searchQuery}
                                    onChange={(e) => setSearchQuery(e.target.value)}
                                    style={{ flex: 1 }}
                                    size="sm"
                                />
                                <Button
                                    size="sm"
                                    variant="light"
                                    leftSection={<IconRefresh size={16} />}
                                    onClick={loadCheckpoints}
                                    loading={loading}
                                >
                                    Actualizar
                                </Button>
                            </Group>

                            {loading && checkpoints.length === 0 && (
                                <Center p="xl">
                                    <Stack align="center" gap="md">
                                        <Progress value={100} animated />
                                        <Text size="sm" c="dimmed">Cargando checkpoints...</Text>
                                    </Stack>
                                </Center>
                            )}

                            {!loading && checkpoints.length === 0 && (
                                <Paper p="xl" withBorder>
                                    <Stack align="center" gap="md">
                                        <IconFile size={64} color="var(--mantine-color-gray-6)" />
                                        <Text c="dimmed" ta="center" fw={500}>
                                            No hay checkpoints disponibles
                                        </Text>
                                        <Text size="xs" c="dimmed" ta="center">
                                            Los checkpoints se guardan automáticamente durante el entrenamiento.
                                            <br />
                                            Inicia un entrenamiento para generar checkpoints.
                                        </Text>
                                    </Stack>
                                </Paper>
                            )}

                            {!loading && filteredCheckpoints.length === 0 && checkpoints.length > 0 && (
                                <Alert icon={<IconSearch size={16} />} color="yellow" variant="light">
                                    No se encontraron checkpoints que coincidan con "{searchQuery}"
                                </Alert>
                            )}

                            {!loading && filteredCheckpoints.length > 0 && (
                                <ScrollArea h={450}>
                                        <Table highlightOnHover>
                                        <Table.Thead>
                                            <Table.Tr>
                                                <Table.Th style={{ width: 100 }}>Estado</Table.Th>
                                                <Table.Th style={{ width: 100 }}>Episodio</Table.Th>
                                                <Table.Th>Archivo</Table.Th>
                                                <Table.Th style={{ width: 100 }}>Tamaño</Table.Th>
                                                <Table.Th style={{ width: 150 }}>Modificado</Table.Th>
                                                <Table.Th style={{ width: 120 }}>Acciones</Table.Th>
                                            </Table.Tr>
                                        </Table.Thead>
                                        <Table.Tbody>
                                            {filteredCheckpoints
                                                .sort((a, b) => b.episode - a.episode)
                                                    .map((ckpt) => (
                                                    <Table.Tr 
                                                        key={ckpt.filename}
                                                        style={{ 
                                                        backgroundColor: ckpt.is_best 
                                                            ? 'var(--mantine-color-green-9)' 
                                                            : undefined,
                                                        opacity: ckpt.is_best ? 0.15 : 1
                                                        }}
                                                    >
                                                    <Table.Td>
                                                                {ckpt.is_best ? (
                                                            <Badge 
                                                                size="sm" 
                                                                color="green" 
                                                                leftSection={<IconStar size={12} />}
                                                                variant="filled"
                                                            >
                                                                        Mejor
                                                                    </Badge>
                                                                ) : (
                                                                    <Badge size="sm" color="gray" variant="light">
                                                                        Normal
                                                                    </Badge>
                                                            )}
                                                    </Table.Td>
                                                    <Table.Td>
                                                        <Text size="sm" fw={ckpt.is_best ? 700 : 500}>
                                                                {ckpt.episode}
                                                            </Text>
                                                        </Table.Td>
                                                        <Table.Td>
                                                            <Group gap="xs" wrap="nowrap">
                                                                <IconFile size={14} />
                                                            <Tooltip label={ckpt.filename}>
                                                                <Text size="sm" style={{ maxWidth: 250 }} truncate="end">
                                                                    {ckpt.filename}
                                                                </Text>
                                                            </Tooltip>
                                                            </Group>
                                                    </Table.Td>
                                                    <Table.Td>
                                                        <Text size="sm">{formatFileSize(ckpt.size)}</Text>
                                                    </Table.Td>
                                                    <Table.Td>
                                                        <Group gap={4} wrap="nowrap">
                                                                <IconClock size={12} />
                                                        <Text size="xs" c="dimmed">{formatDate(ckpt.modified)}</Text>
                                                            </Group>
                                                    </Table.Td>
                                                    <Table.Td>
                                                        <Group gap={4}>
                                                            <Menu shadow="md" width={200}>
                                                                <Menu.Target>
                                                                    <ActionIcon variant="light" size="sm">
                                                                        <IconDotsVertical size={14} />
                                                                    </ActionIcon>
                                                                </Menu.Target>
                                                                <Menu.Dropdown>
                                                                    <Menu.Item
                                                                        leftSection={<IconDownload size={14} />}
                                                                        onClick={() => {
                                                                            sendCommand('experiment', 'download_checkpoint', {
                                                                                EXPERIMENT_NAME: activeExperiment,
                                                                                CHECKPOINT_NAME: ckpt.filename
                                                                            });
                                                                            notifications.show({
                                                                                title: 'Descarga iniciada',
                                                                                message: `Descargando ${ckpt.filename}`,
                                                                                color: 'blue',
                                                                                autoClose: 2000,
                                                                            });
                                                                        }}
                                                                    >
                                                                        Descargar
                                                                    </Menu.Item>
                                                                    <Menu.Item
                                                                        leftSection={<IconEdit size={14} />}
                                                                    onClick={() => {
                                                                        setSelectedCheckpoint(ckpt.filename);
                                                                            setActiveTab('notes');
                                                                    }}
                                                                >
                                                                        Agregar nota
                                                                    </Menu.Item>
                                                                    <Menu.Divider />
                                                                    <Menu.Item
                                                                        leftSection={<IconTrash size={14} />}
                                                                    color="red"
                                                                    onClick={() => deleteCheckpoint(ckpt.filename)}
                                                                >
                                                                        Eliminar
                                                                    </Menu.Item>
                                                                </Menu.Dropdown>
                                                            </Menu>
                                                        </Group>
                                                    </Table.Td>
                                                </Table.Tr>
                                            ))}
                                        </Table.Tbody>
                                    </Table>
                                </ScrollArea>
                            )}
                        </Stack>
                    </Tabs.Panel>

                    <Tabs.Panel value="notes" pt="md">
                        <Stack gap="md">
                            <Alert icon={<IconInfoCircle size={16} />} color="blue" variant="light">
                                <Text size="sm">
                                    Documenta observaciones sobre checkpoints específicos o el experimento en general.
                                    Las notas se guardan localmente en tu navegador.
                                </Text>
                            </Alert>
                            
                            {selectedCheckpoint && (
                                <Alert 
                                    icon={<IconFile size={16} />} 
                                    color="yellow" 
                                    variant="light"
                                    withCloseButton
                                    onClose={() => setSelectedCheckpoint(null)}
                                >
                            <Group justify="space-between">
                                    <Text size="sm">
                                        <strong>Checkpoint seleccionado:</strong> {selectedCheckpoint}
                                            </Text>
                                    </Group>
                                        </Alert>
                                    )}
                            
                            <Group>
                                    <Textarea
                                    placeholder="Escribe una nota sobre este checkpoint o experimento..."
                                        value={newNote}
                                    onChange={(e) => setNewNote(e.target.value)}
                                    minRows={3}
                                    style={{ flex: 1 }}
                                    />
                            </Group>
                            
                            <Group>
                                <Button
                                    onClick={addNote}
                                    disabled={!newNote.trim()}
                                    leftSection={<IconEdit size={14} />}
                                >
                                            Agregar Nota
                                        </Button>
                                {selectedCheckpoint && (
                                    <Button
                                        variant="light"
                                        onClick={() => setSelectedCheckpoint(null)}
                                        leftSection={<IconX size={14} />}
                                    >
                                        Limpiar Selección
                                    </Button>
                                )}
                                    </Group>

                            <Divider label="Notas guardadas" labelPosition="center" />

                            {notes.length === 0 ? (
                                <Paper p="xl" withBorder>
                                    <Stack align="center" gap="md">
                                        <IconBook size={64} color="var(--mantine-color-gray-6)" />
                                        <Text c="dimmed" ta="center" fw={500}>
                                            No hay notas aún
                                        </Text>
                                        <Text size="xs" c="dimmed" ta="center">
                                            Agrega una nota para documentar tus observaciones sobre el experimento o checkpoints específicos.
                                        </Text>
                                    </Stack>
                                </Paper>
                            ) : (
                                <ScrollArea h={450}>
                                    <Stack gap="md">
                                        {notes
                                            .sort((a, b) => b.timestamp - a.timestamp)
                                            .map((note) => (
                                            <Card key={note.id} withBorder p="md" style={{ backgroundColor: 'var(--mantine-color-dark-7)' }}>
                                                <Stack gap="xs">
                                                    <Group justify="space-between">
                                                        <Group gap="xs">
                                                            <IconClock size={14} />
                                                            <Text size="xs" c="dimmed">
                                                            {new Date(note.timestamp).toLocaleString()}
                                                        </Text>
                                                            {note.checkpoint && (
                                                                <Badge size="xs" variant="light" color="blue" leftSection={<IconFile size={10} />}>
                                                                    {note.checkpoint}
                                                                </Badge>
                                                            )}
                                                        </Group>
                                                        <ActionIcon
                                                            size="sm"
                                                            variant="light"
                                                            color="red"
                                                            onClick={() => deleteNote(note.id)}
                                                        >
                                                            <IconTrash size={14} />
                                                        </ActionIcon>
                                                    </Group>
                                                    <Text size="sm" style={{ whiteSpace: 'pre-wrap' }}>
                                                        {note.content}
                                                    </Text>
                                                </Stack>
                                            </Card>
                                        ))}
                                    </Stack>
                                </ScrollArea>
                            )}
                        </Stack>
                    </Tabs.Panel>
                </Tabs>
            </Modal>
        </>
    );
}
