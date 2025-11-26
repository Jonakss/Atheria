// frontend/src/components/training/CheckpointManager.tsx
import {
    BookOpen,
    Clock,
    Download,
    Edit,
    FileText,
    Info,
    MoreVertical, RefreshCw,
    Search, Star,
    Trash2,
    X
} from 'lucide-react';
import { useCallback, useEffect, useMemo, useState } from 'react';
import { useWebSocket } from '../../hooks/useWebSocket';
import { Alert } from '../../modules/Dashboard/components/Alert';
import { Badge } from '../../modules/Dashboard/components/Badge';
import { GlassPanel } from '../../modules/Dashboard/components/GlassPanel';
import { Modal } from '../../modules/Dashboard/components/Modal';
import { Table, TableBody, TableHead, TableRow, TableTd, TableTh } from '../../modules/Dashboard/components/Table';
import { Tab, TabList, TabPanel, Tabs } from '../../modules/Dashboard/components/Tabs';

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
    checkpoint?: string;
}

export function CheckpointManager() {
    const { activeExperiment, sendCommand, experimentsData } = useWebSocket();
    const [opened, setOpened] = useState(false);
    const [checkpoints, setCheckpoints] = useState<CheckpointInfo[]>([]);
    const [notes, setNotes] = useState<ExperimentNote[]>([]);
    const [newNote, setNewNote] = useState('');
    const [selectedCheckpoint, setSelectedCheckpoint] = useState<string | null>(null);
    const [loading, setLoading] = useState(false);
    const [searchQuery, setSearchQuery] = useState('');
    const [activeTab, setActiveTab] = useState<string>('checkpoints');
    const [menuOpen, setMenuOpen] = useState<string | null>(null);

    const loadCheckpoints = useCallback(async () => {
        if (!activeExperiment) return;
        setLoading(true);
        sendCommand('experiment', 'list_checkpoints', { EXPERIMENT_NAME: activeExperiment });
    }, [activeExperiment, sendCommand]);

    const loadNotes = useCallback(() => {
        if (!activeExperiment) return;
        const savedNotes = localStorage.getItem(`experiment_notes_${activeExperiment}`);
        if (savedNotes) {
            try {
                setNotes(JSON.parse(savedNotes));
            } catch (e) {
                console.error('Error loading notes:', e);
            }
        }
    }, [activeExperiment]);

    // Cargar checkpoints y notas cuando se abre el modal
    useEffect(() => {
        if (opened && activeExperiment) {
            loadCheckpoints();
            loadNotes();
        }
    }, [opened, activeExperiment, loadCheckpoints, loadNotes]);

    useEffect(() => {
        const handleCheckpointsUpdate = (event: CustomEvent) => {
            setCheckpoints(event.detail || []);
            setLoading(false);
            if (event.detail && event.detail.length > 0) {
                console.log(`✅ Checkpoints actualizados: ${event.detail.length} encontrados`);
            }
        };
        
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
        console.log('✅ Nota agregada correctamente');
    };

    const deleteNote = (noteId: string) => {
        saveNotes(notes.filter(n => n.id !== noteId));
        console.log('✅ Nota eliminada correctamente');
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
            setTimeout(() => loadCheckpoints(), 500);
            console.log(`✅ Checkpoint "${checkpointName}" eliminado`);
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
             setTimeout(() => loadCheckpoints(), 1000);
        }
    };

    return (
        <>
            <button
                onClick={() => setOpened(true)}
                disabled={!activeExperiment}
                className="w-full flex items-center justify-center gap-2 px-3 py-2 bg-white/5 hover:bg-white/10 border border-white/10 text-xs font-bold text-gray-300 rounded transition-all disabled:opacity-50 disabled:cursor-not-allowed"
            >
                <FileText size={14} />
                Gestionar Checkpoints
            </button>

            <Modal
                opened={opened}
                onClose={() => setOpened(false)}
                title={
                    <div className="flex items-center gap-2">
                        <FileText size={16} />
                        <span>Checkpoints y Notas</span>
                        {activeExperiment && (
                            <Badge variant="light" size="sm">
                                {activeExperiment}
                            </Badge>
                        )}
                    </div>
                }
                size="xl"
                closeOnClickOutside={true}
            >
                {activeExperiment && checkpoints.length > 0 && (
                    <Alert 
                        variant="light" 
                        color="blue" 
                        title="Estado del Almacenamiento" 
                        icon={<Info size={16} />}
                        className="mb-4"
                    >
                        <div className="flex items-center justify-between">
                            <span className="text-xs">
                                Total ocupado: <strong>{formatFileSize(totalSize)}</strong> en {checkpoints.length} checkpoints.
                            </span>
                            <button 
                                onClick={handleCleanup}
                                className="flex items-center gap-1.5 px-2 py-1 bg-red-500/10 hover:bg-red-500/20 border border-red-500/30 text-red-400 text-xs font-bold rounded transition-all"
                            >
                                <Trash2 size={12} />
                                Limpiar antiguos
                            </button>
                        </div>
                    </Alert>
                )}

                <Tabs value={activeTab} onTabChange={(value) => setActiveTab(value || 'checkpoints')}>
                    <TabList>
                        <Tab 
                            value="checkpoints"
                            label="Checkpoints"
                            icon={<FileText size={14} />}
                            rightSection={checkpoints.length > 0 ? (
                                <Badge size="xs" variant="filled" color="blue">
                                    {checkpoints.length}
                                </Badge>
                            ) : undefined}
                        />
                        <Tab 
                            value="notes"
                            label="Notas"
                            icon={<BookOpen size={14} />}
                            rightSection={notes.length > 0 ? (
                                <Badge size="xs" variant="filled" color="blue">
                                    {notes.length}
                                </Badge>
                            ) : undefined}
                        />
                    </TabList>

                    <TabPanel value="checkpoints">
                        <div className="space-y-4">
                            {/* Información del experimento */}
                            {currentExperiment && (
                                <GlassPanel className="p-3 bg-[#0a0a0a]">
                                    <div className="space-y-2">
                                        <div className="flex items-center justify-between">
                                            <div className="flex items-center gap-2">
                                                <span className="text-xs font-bold text-gray-300">Experimento</span>
                                                <Badge 
                                                    color={currentExperiment.has_checkpoint ? 'green' : 'orange'}
                                                    variant="light"
                                                    size="sm"
                                                >
                                                    {currentExperiment.has_checkpoint ? '✓ Entrenado' : '○ Sin entrenar'}
                                                </Badge>
                                            </div>
                                        </div>
                                        <div className="flex items-center gap-4">
                                            <span className="text-[10px] text-gray-500">
                                                <strong>Arquitectura:</strong> {currentExperiment.config?.MODEL_ARCHITECTURE || 'N/A'}
                                            </span>
                                        </div>
                                    </div>
                                </GlassPanel>
                            )}

                            {/* Estadísticas rápidas */}
                            {checkpoints.length > 0 && (
                                <div className="grid grid-cols-4 gap-2">
                                    <GlassPanel className="p-2 bg-[#0a0a0a]">
                                        <div className="text-center space-y-1">
                                            <div className="text-[10px] text-gray-500">Total Checkpoints</div>
                                            <div className="text-lg font-bold text-gray-200">{checkpoints.length}</div>
                                        </div>
                                    </GlassPanel>
                                    {bestCheckpoint && (
                                        <GlassPanel className="p-2 bg-emerald-500/20">
                                            <div className="text-center space-y-1">
                                                <div className="flex items-center justify-center gap-1 text-[10px] text-gray-500">
                                                    <Star size={12} />
                                                    <span>Mejor</span>
                                                </div>
                                                <div className="text-lg font-bold text-emerald-400">Ep. {bestCheckpoint.episode}</div>
                                            </div>
                                        </GlassPanel>
                                    )}
                                    {latestCheckpoint && (
                                        <GlassPanel className="p-2 bg-[#0a0a0a]">
                                            <div className="text-center space-y-1">
                                                <div className="text-[10px] text-gray-500">Último</div>
                                                <div className="text-lg font-bold text-gray-200">Ep. {latestCheckpoint.episode}</div>
                                            </div>
                                        </GlassPanel>
                                    )}
                                    <GlassPanel className="p-2 bg-[#0a0a0a]">
                                        <div className="text-center space-y-1">
                                            <div className="text-[10px] text-gray-500">Tamaño Total</div>
                                            <div className="text-sm font-bold text-gray-200">{formatFileSize(totalSize)}</div>
                                        </div>
                                    </GlassPanel>
                                </div>
                            )}

                            {/* Barra de búsqueda y acciones */}
                            <div className="flex items-center gap-2">
                                <div className="flex-1 relative">
                                    <Search size={14} className="absolute left-2 top-1/2 -translate-y-1/2 text-gray-500" />
                                    <input
                                        type="text"
                                        placeholder="Buscar por nombre o episodio..."
                                        value={searchQuery}
                                        onChange={(e) => setSearchQuery(e.target.value)}
                                        className="w-full pl-8 pr-3 py-1.5 bg-white/5 border border-white/10 rounded text-xs text-gray-300 placeholder-gray-600 focus:outline-none focus:border-blue-500/50"
                                    />
                                </div>
                                <button
                                    onClick={loadCheckpoints}
                                    disabled={loading}
                                    className="flex items-center gap-1.5 px-3 py-1.5 bg-white/5 hover:bg-white/10 border border-white/10 text-xs font-bold text-gray-300 rounded transition-all disabled:opacity-50"
                                >
                                    <RefreshCw size={12} className={loading ? 'animate-spin' : ''} />
                                    Actualizar
                                </button>
                            </div>

                            {loading && checkpoints.length === 0 && (
                                <div className="flex flex-col items-center justify-center py-8 space-y-4">
                                    <div className="w-full h-1 bg-gray-800 rounded-full overflow-hidden">
                                        <div className="h-full bg-blue-500 animate-pulse" style={{ width: '100%' }} />
                                    </div>
                                    <span className="text-xs text-gray-500">Cargando checkpoints...</span>
                                </div>
                            )}

                            {!loading && checkpoints.length === 0 && (
                                <GlassPanel className="p-8 border border-white/10">
                                    <div className="flex flex-col items-center space-y-4">
                                        <FileText size={48} className="text-gray-600" />
                                        <div className="text-center">
                                            <div className="text-xs font-bold text-gray-500 mb-1">
                                                No hay checkpoints disponibles
                                            </div>
                                            <div className="text-[10px] text-gray-600 text-center">
                                                Los checkpoints se guardan automáticamente durante el entrenamiento.
                                                <br />
                                                Inicia un entrenamiento para generar checkpoints.
                                            </div>
                                        </div>
                                    </div>
                                </GlassPanel>
                            )}

                            {!loading && filteredCheckpoints.length === 0 && checkpoints.length > 0 && (
                                <Alert icon={<Search size={16} />} color="yellow" variant="light">
                                    No se encontraron checkpoints que coincidan con &quot;{searchQuery}&quot;
                                </Alert>
                            )}

                            {!loading && filteredCheckpoints.length > 0 && (
                                <div className="h-[450px] overflow-y-auto custom-scrollbar">
                                    <Table highlightOnHover>
                                        <TableHead>
                                            <TableRow>
                                                <TableTh style={{ width: 100 }}>Estado</TableTh>
                                                <TableTh style={{ width: 100 }}>Episodio</TableTh>
                                                <TableTh>Archivo</TableTh>
                                                <TableTh style={{ width: 100 }}>Tamaño</TableTh>
                                                <TableTh style={{ width: 150 }}>Modificado</TableTh>
                                                <TableTh style={{ width: 120 }}>Acciones</TableTh>
                                            </TableRow>
                                        </TableHead>
                                        <TableBody>
                                            {filteredCheckpoints
                                                .sort((a, b) => b.episode - a.episode)
                                                .map((ckpt) => (
                                                    <TableRow 
                                                        key={ckpt.filename}
                                                        style={{ 
                                                            backgroundColor: ckpt.is_best ? 'rgba(34, 197, 94, 0.15)' : undefined,
                                                            opacity: ckpt.is_best ? 1 : 1
                                                        }}
                                                    >
                                                        <TableTd>
                                                            {ckpt.is_best ? (
                                                                <Badge 
                                                                    size="sm" 
                                                                    color="green" 
                                                                    leftSection={<Star size={10} />}
                                                                    variant="filled"
                                                                >
                                                                    Mejor
                                                                </Badge>
                                                            ) : (
                                                                <Badge size="sm" color="gray" variant="light">
                                                                    Normal
                                                                </Badge>
                                                            )}
                                                        </TableTd>
                                                        <TableTd>
                                                            <span className={`text-xs ${ckpt.is_best ? 'font-bold' : 'font-medium'}`}>
                                                                {ckpt.episode}
                                                            </span>
                                                        </TableTd>
                                                        <TableTd>
                                                            <div className="flex items-center gap-2">
                                                                <FileText size={12} className="text-gray-500" />
                                                                <span className="text-xs max-w-[250px] truncate" title={ckpt.filename}>
                                                                    {ckpt.filename}
                                                                </span>
                                                            </div>
                                                        </TableTd>
                                                        <TableTd>
                                                            <span className="text-xs">{formatFileSize(ckpt.size)}</span>
                                                        </TableTd>
                                                        <TableTd>
                                                            <div className="flex items-center gap-1">
                                                                <Clock size={10} className="text-gray-500" />
                                                                <span className="text-[10px] text-gray-500">{formatDate(ckpt.modified)}</span>
                                                            </div>
                                                        </TableTd>
                                                        <TableTd>
                                                            <div className="relative flex items-center gap-1">
                                                                <button
                                                                    onClick={() => setMenuOpen(menuOpen === ckpt.filename ? null : ckpt.filename)}
                                                                    className="p-1 bg-white/5 hover:bg-white/10 border border-white/10 rounded transition-colors"
                                                                >
                                                                    <MoreVertical size={12} />
                                                                </button>
                                                                {menuOpen === ckpt.filename && (
                                                                    <div className="absolute right-0 top-8 z-10 w-48 bg-[#0a0a0a] border border-white/10 rounded shadow-lg">
                                                                        <div className="p-1 space-y-1">
                                                                            <button
                                                                                onClick={() => {
                                                                                    sendCommand('experiment', 'download_checkpoint', {
                                                                                        EXPERIMENT_NAME: activeExperiment,
                                                                                        CHECKPOINT_NAME: ckpt.filename
                                                                                    });
                                                                                    setMenuOpen(null);
                                                                                    console.log(`✅ Descargando ${ckpt.filename}`);
                                                                                }}
                                                                                className="w-full flex items-center gap-2 px-2 py-1.5 text-xs text-gray-300 hover:bg-white/5 rounded transition-colors"
                                                                            >
                                                                                <Download size={12} />
                                                                                Descargar
                                                                            </button>
                                                                            <button
                                                                                onClick={() => {
                                                                                    setSelectedCheckpoint(ckpt.filename);
                                                                                    setActiveTab('notes');
                                                                                    setMenuOpen(null);
                                                                                }}
                                                                                className="w-full flex items-center gap-2 px-2 py-1.5 text-xs text-gray-300 hover:bg-white/5 rounded transition-colors"
                                                                            >
                                                                                <Edit size={12} />
                                                                                Agregar nota
                                                                            </button>
                                                                            <div className="h-px bg-white/10 my-1" />
                                                                            <button
                                                                                onClick={() => {
                                                                                    deleteCheckpoint(ckpt.filename);
                                                                                    setMenuOpen(null);
                                                                                }}
                                                                                className="w-full flex items-center gap-2 px-2 py-1.5 text-xs text-red-400 hover:bg-red-500/10 rounded transition-colors"
                                                                            >
                                                                                <Trash2 size={12} />
                                                                                Eliminar
                                                                            </button>
                                                                        </div>
                                                                    </div>
                                                                )}
                                                            </div>
                                                        </TableTd>
                                                    </TableRow>
                                                ))}
                                        </TableBody>
                                    </Table>
                                </div>
                            )}
                        </div>
                    </TabPanel>

                    <TabPanel value="notes">
                        <div className="space-y-4">
                            <Alert icon={<Info size={16} />} color="blue" variant="light">
                                <span className="text-xs">
                                    Documenta observaciones sobre checkpoints específicos o el experimento en general.
                                    Las notas se guardan localmente en tu navegador.
                                </span>
                            </Alert>
                            
                            {selectedCheckpoint && (
                                <Alert 
                                    icon={<FileText size={16} />} 
                                    color="yellow" 
                                    variant="light"
                                    withCloseButton
                                    onClose={() => setSelectedCheckpoint(null)}
                                >
                                    <div className="flex items-center justify-between">
                                        <span className="text-xs">
                                            <strong>Checkpoint seleccionado:</strong> {selectedCheckpoint}
                                        </span>
                                    </div>
                                </Alert>
                            )}
                            
                            <div className="flex flex-col gap-2">
                                <textarea
                                    placeholder="Escribe una nota sobre este checkpoint o experimento..."
                                    value={newNote}
                                    onChange={(e) => setNewNote(e.target.value)}
                                    rows={3}
                                    className="w-full px-3 py-2 bg-white/5 border border-white/10 rounded text-xs text-gray-300 placeholder-gray-600 focus:outline-none focus:border-blue-500/50 resize-none"
                                />
                            </div>
                            
                            <div className="flex items-center gap-2">
                                <button
                                    onClick={addNote}
                                    disabled={!newNote.trim()}
                                    className="flex items-center gap-1.5 px-3 py-1.5 bg-blue-500/10 hover:bg-blue-500/20 border border-blue-500/30 text-blue-400 text-xs font-bold rounded transition-all disabled:opacity-50 disabled:cursor-not-allowed"
                                >
                                    <Edit size={12} />
                                    Agregar Nota
                                </button>
                                {selectedCheckpoint && (
                                    <button
                                        onClick={() => setSelectedCheckpoint(null)}
                                        className="flex items-center gap-1.5 px-3 py-1.5 bg-white/5 hover:bg-white/10 border border-white/10 text-xs font-bold text-gray-300 rounded transition-all"
                                    >
                                        <X size={12} />
                                        Limpiar Selección
                                    </button>
                                )}
                            </div>

                            <div className="h-px bg-white/10 my-2" />

                            {notes.length === 0 ? (
                                <GlassPanel className="p-8 border border-white/10">
                                    <div className="flex flex-col items-center space-y-4">
                                        <BookOpen size={48} className="text-gray-600" />
                                        <div className="text-center">
                                            <div className="text-xs font-bold text-gray-500 mb-1">
                                                No hay notas aún
                                            </div>
                                            <div className="text-[10px] text-gray-600 text-center">
                                                Agrega una nota para documentar tus observaciones sobre el experimento o checkpoints específicos.
                                            </div>
                                        </div>
                                    </div>
                                </GlassPanel>
                            ) : (
                                <div className="h-[450px] overflow-y-auto custom-scrollbar space-y-2">
                                    {notes
                                        .sort((a, b) => b.timestamp - a.timestamp)
                                        .map((note) => (
                                            <GlassPanel key={note.id} className="p-3 bg-[#0a0a0a]">
                                                <div className="space-y-2">
                                                    <div className="flex items-center justify-between">
                                                        <div className="flex items-center gap-2">
                                                            <Clock size={12} className="text-gray-500" />
                                                            <span className="text-[10px] text-gray-500">
                                                                {new Date(note.timestamp).toLocaleString()}
                                                            </span>
                                                            {note.checkpoint && (
                                                                <Badge size="xs" variant="light" color="blue" leftSection={<FileText size={8} />}>
                                                                    {note.checkpoint}
                                                                </Badge>
                                                            )}
                                                        </div>
                                                        <button
                                                            onClick={() => deleteNote(note.id)}
                                                            className="p-1 bg-red-500/10 hover:bg-red-500/20 border border-red-500/30 text-red-400 rounded transition-colors"
                                                        >
                                                            <Trash2 size={12} />
                                                        </button>
                                                    </div>
                                                    <div className="text-xs whitespace-pre-wrap">
                                                        {note.content}
                                                    </div>
                                                </div>
                                            </GlassPanel>
                                        ))}
                                </div>
                            )}
                        </div>
                    </TabPanel>
                </Tabs>
            </Modal>
        </>
    );
}
