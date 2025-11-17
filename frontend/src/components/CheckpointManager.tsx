// frontend/src/components/CheckpointManager.tsx
import { useState, useEffect } from 'react';
import { 
    Paper, Stack, Group, Text, Badge, Button, Modal, 
    Table, ScrollArea, Tooltip, ActionIcon, Divider,
    Card, Timeline, Alert, Select, Box, Textarea, 
    Tabs, FileButton, Progress
} from '@mantine/core';
import { useDisclosure } from '@mantine/hooks';
import { useWebSocket } from '../hooks/useWebSocket';
import { 
    IconDownload, IconTrash, IconInfoCircle, IconFile,
    IconClock, IconCheck, IconX, IconEdit, IconBook
} from '@tabler/icons-react';

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
            setCheckpoints(event.detail);
            setLoading(false);
        };
        
        window.addEventListener('checkpoints_updated', handleCheckpointsUpdate as EventListener);
        
        return () => {
            window.removeEventListener('checkpoints_updated', handleCheckpointsUpdate as EventListener);
        };
    }, []);

    const loadCheckpoints = async () => {
        if (!activeExperiment) return;
        setLoading(true);
        // Enviar comando para listar checkpoints
        sendCommand('experiment', 'list_checkpoints', { EXPERIMENT_NAME: activeExperiment });
    };

    const loadNotes = () => {
        if (!activeExperiment) return;
        // Cargar notas desde localStorage (o del backend si lo implementamos)
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
    };

    const deleteNote = (noteId: string) => {
        saveNotes(notes.filter(n => n.id !== noteId));
    };

    const deleteCheckpoint = (checkpointName: string) => {
        if (!activeExperiment) return;
        if (confirm(`¿Eliminar checkpoint "${checkpointName}"? Esta acción no se puede deshacer.`)) {
            sendCommand('experiment', 'delete_checkpoint', { 
                EXPERIMENT_NAME: activeExperiment,
                CHECKPOINT_NAME: checkpointName
            });
            loadCheckpoints();
        }
    };

    const formatFileSize = (bytes: number) => {
        if (bytes < 1024) return `${bytes} B`;
        if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(2)} KB`;
        return `${(bytes / (1024 * 1024)).toFixed(2)} MB`;
    };

    const formatDate = (dateString: string) => {
        return new Date(dateString).toLocaleString();
    };

    const currentExperiment = activeExperiment 
        ? experimentsData?.find(exp => exp.name === activeExperiment) 
        : null;

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
                title={`Checkpoints y Notas: ${activeExperiment}`}
                size="xl"
            >
                <Tabs defaultValue="checkpoints">
                    <Tabs.List>
                        <Tabs.Tab value="checkpoints" leftSection={<IconFile size={16} />}>
                            Checkpoints ({checkpoints.length})
                        </Tabs.Tab>
                        <Tabs.Tab value="notes" leftSection={<IconBook size={16} />}>
                            Notas ({notes.length})
                        </Tabs.Tab>
                    </Tabs.List>

                    <Tabs.Panel value="checkpoints" pt="md">
                        <Stack gap="md">
                            {currentExperiment && (
                                <Alert icon={<IconInfoCircle size={16} />} color="blue" variant="light">
                                    <Text size="sm">
                                        <strong>Experimento:</strong> {activeExperiment}<br />
                                        <strong>Arquitectura:</strong> {currentExperiment.config?.MODEL_ARCHITECTURE}<br />
                                        <strong>Estado:</strong> {currentExperiment.has_checkpoint ? 'Entrenado' : 'Sin entrenar'}
                                    </Text>
                                </Alert>
                            )}

                            <Group justify="space-between">
                                <Text fw={600}>Checkpoints Disponibles</Text>
                                <Button
                                    size="xs"
                                    variant="light"
                                    onClick={loadCheckpoints}
                                    loading={loading}
                                >
                                    Actualizar
                                </Button>
                            </Group>

                            {checkpoints.length === 0 ? (
                                <Paper p="xl" withBorder>
                                    <Stack align="center" gap="md">
                                        <IconFile size={48} color="gray" />
                                        <Text c="dimmed" ta="center">
                                            No hay checkpoints disponibles para este experimento.
                                        </Text>
                                        <Text size="xs" c="dimmed" ta="center">
                                            Los checkpoints se guardan automáticamente durante el entrenamiento.
                                        </Text>
                                    </Stack>
                                </Paper>
                            ) : (
                                <Stack gap="md">
                                    {/* Resumen de checkpoints */}
                                    <Group justify="space-between" p="sm" style={{ backgroundColor: 'var(--mantine-color-dark-7)', borderRadius: '4px' }}>
                                        <Group gap="md">
                                            <Badge color="blue" variant="light">
                                                Total: {checkpoints.length}
                                            </Badge>
                                            {checkpoints.find(c => c.is_best) && (
                                                <Badge color="green" variant="light" leftSection={<IconCheck size={12} />}>
                                                    Mejor: Ep. {checkpoints.find(c => c.is_best)?.episode}
                                                </Badge>
                                            )}
                                            <Badge color="gray" variant="light">
                                                Último: Ep. {Math.max(...checkpoints.map(c => c.episode))}
                                            </Badge>
                                        </Group>
                                        <Text size="xs" c="dimmed">
                                            Tamaño total: {formatFileSize(checkpoints.reduce((sum, c) => sum + c.size, 0))}
                                        </Text>
                                    </Group>
                                    
                                <ScrollArea h={400}>
                                        <Table highlightOnHover>
                                        <Table.Thead>
                                            <Table.Tr>
                                                    <Table.Th>Estado</Table.Th>
                                                    <Table.Th>Episodio</Table.Th>
                                                <Table.Th>Archivo</Table.Th>
                                                <Table.Th>Tamaño</Table.Th>
                                                <Table.Th>Modificado</Table.Th>
                                                <Table.Th>Acciones</Table.Th>
                                            </Table.Tr>
                                        </Table.Thead>
                                        <Table.Tbody>
                                                {checkpoints
                                                    .sort((a, b) => b.episode - a.episode) // Más recientes primero
                                                    .map((ckpt) => (
                                                    <Table.Tr 
                                                        key={ckpt.filename}
                                                        style={{ 
                                                            backgroundColor: ckpt.is_best ? 'var(--mantine-color-green-1)' : undefined 
                                                        }}
                                                    >
                                                    <Table.Td>
                                                        <Group gap="xs">
                                                                {ckpt.is_best ? (
                                                                    <Badge size="sm" color="green" leftSection={<IconCheck size={12} />}>
                                                                        Mejor
                                                                    </Badge>
                                                                ) : (
                                                                    <Badge size="sm" color="gray" variant="light">
                                                                        Normal
                                                                    </Badge>
                                                            )}
                                                        </Group>
                                                    </Table.Td>
                                                    <Table.Td>
                                                            <Text size="sm" fw={ckpt.is_best ? 600 : 400}>
                                                                {ckpt.episode}
                                                            </Text>
                                                        </Table.Td>
                                                        <Table.Td>
                                                            <Group gap="xs" wrap="nowrap">
                                                                <IconFile size={14} />
                                                                <Text size="sm" style={{ maxWidth: 200 }} truncate="end">
                                                                    {ckpt.filename}
                                                                </Text>
                                                            </Group>
                                                    </Table.Td>
                                                    <Table.Td>
                                                        <Text size="sm">{formatFileSize(ckpt.size)}</Text>
                                                    </Table.Td>
                                                    <Table.Td>
                                                            <Group gap={4}>
                                                                <IconClock size={12} />
                                                        <Text size="xs" c="dimmed">{formatDate(ckpt.modified)}</Text>
                                                            </Group>
                                                    </Table.Td>
                                                    <Table.Td>
                                                        <Group gap="xs">
                                                                <Tooltip label="Descargar checkpoint">
                                                                    <ActionIcon
                                                                        size="sm"
                                                                        variant="light"
                                                                        color="blue"
                                                                        onClick={() => {
                                                                            sendCommand('experiment', 'download_checkpoint', {
                                                                                EXPERIMENT_NAME: activeExperiment,
                                                                                CHECKPOINT_NAME: ckpt.filename
                                                                            });
                                                                        }}
                                                                    >
                                                                        <IconDownload size={14} />
                                                                    </ActionIcon>
                                                                </Tooltip>
                                                            <Tooltip label="Usar para nota">
                                                                <ActionIcon
                                                                    size="sm"
                                                                    variant="light"
                                                                    onClick={() => {
                                                                        setSelectedCheckpoint(ckpt.filename);
                                                                            // Cambiar a pestaña de notas usando el estado del modal
                                                                            const event = new CustomEvent('switch_to_notes_tab');
                                                                            window.dispatchEvent(event);
                                                                    }}
                                                                >
                                                                    <IconEdit size={14} />
                                                                </ActionIcon>
                                                            </Tooltip>
                                                                <Tooltip label="Eliminar checkpoint">
                                                                <ActionIcon
                                                                    size="sm"
                                                                    variant="light"
                                                                    color="red"
                                                                    onClick={() => deleteCheckpoint(ckpt.filename)}
                                                                >
                                                                    <IconTrash size={14} />
                                                                </ActionIcon>
                                                            </Tooltip>
                                                        </Group>
                                                    </Table.Td>
                                                </Table.Tr>
                                            ))}
                                        </Table.Tbody>
                                    </Table>
                                </ScrollArea>
                                </Stack>
                            )}
                        </Stack>
                    </Tabs.Panel>

                    <Tabs.Panel value="notes" pt="md">
                        <Stack gap="md">
                            <Alert icon={<IconInfoCircle size={16} />} color="blue" variant="light">
                                <Text size="sm">
                                    Las notas te permiten documentar observaciones sobre checkpoints específicos.
                                    Útiles para recordar qué cambios o mejoras se hicieron en cada punto del entrenamiento.
                                </Text>
                            </Alert>
                            
                            <Group justify="space-between">
                                <Text fw={600}>Notas del Experimento</Text>
                                <Badge variant="light">{notes.length} notas</Badge>
                            </Group>
                            
                                    {selectedCheckpoint && (
                                <Alert color="yellow" variant="light">
                                    <Text size="sm">
                                        <strong>Checkpoint seleccionado:</strong> {selectedCheckpoint}
                                            </Text>
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
                                    >
                                        Limpiar Selección
                                    </Button>
                                )}
                                    </Group>

                            <Divider />

                            {notes.length === 0 ? (
                                <Paper p="xl" withBorder>
                                    <Stack align="center" gap="md">
                                        <IconBook size={48} color="gray" />
                                        <Text c="dimmed" ta="center">
                                            No hay notas aún. Agrega una nota para documentar tus observaciones.
                                        </Text>
                                    </Stack>
                                </Paper>
                            ) : (
                                <ScrollArea h={400}>
                                    <Stack gap="md">
                                        {notes
                                            .sort((a, b) => b.timestamp - a.timestamp) // Más recientes primero
                                            .map((note) => (
                                            <Card key={note.id} withBorder p="md">
                                                <Stack gap="xs">
                                                    <Group justify="space-between">
                                                        <Group gap="xs">
                                                            <IconClock size={14} />
                                                            <Text size="xs" c="dimmed">
                                                            {new Date(note.timestamp).toLocaleString()}
                                                        </Text>
                                                            {note.checkpoint && (
                                                                <Badge size="xs" variant="light" color="blue">
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

