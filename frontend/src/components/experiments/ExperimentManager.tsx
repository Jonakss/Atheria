// frontend/src/components/ExperimentManager.tsx
import { useState, useMemo, useEffect } from 'react';
import { Database, Info, Trash2, Check, Clock, ArrowRightLeft, X, Save, UploadCloud } from 'lucide-react';
import { useWebSocket } from '../../hooks/useWebSocket';
import { GlassPanel } from '../../modules/Dashboard/components/GlassPanel';

interface ExperimentNode {
    name: string;
    config: any;
    hasCheckpoint: boolean;
    loadFrom?: string;
    children: string[]; // Experimentos que usan este como base
}

export function ExperimentManager() {
    const { experimentsData, activeExperiment, setActiveExperiment, sendCommand, inferenceSnapshots } = useWebSocket();
    const [opened, setOpened] = useState(false);
    const [selectedExp, setSelectedExp] = useState<string | null>(null);
    const [viewMode, setViewMode] = useState<'list' | 'tree'>('tree');
    const [sortBy, setSortBy] = useState<'created' | 'updated' | 'name' | 'training_time'>('created');

    const open = () => setOpened(true);
    const close = () => setOpened(false);

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
    // Protección contra ciclos para evitar stack overflow
    // Esta función debe ser memoizada y usar el árbol correcto
    const findRoot = useMemo(() => {
        return (expName: string, tree: Record<string, ExperimentNode>, visited: Set<string> = new Set()): string => {
            // Si ya visitamos este nodo, hay un ciclo - retornar el nodo actual
            if (visited.has(expName)) {
                console.warn(`⚠️ Ciclo detectado en transfer learning para "${expName}". Retornando nodo actual.`);
                return expName;
            }
            
            const node = tree[expName];
            if (!node || !node.loadFrom) return expName;
            
            // Agregar a visitados y continuar recursión
            visited.add(expName);
            return findRoot(node.loadFrom, tree, visited);
        };
    }, []);

    // Obtener toda la cadena de transfer learning
    // Protección contra ciclos para evitar stack overflow
    const getTransferChain = (expName: string, tree: Record<string, ExperimentNode>): string[] => {
        const chain: string[] = [];
        const visited = new Set<string>();
        let current = expName;
        
        while (current && tree[current]) {
            // Si ya visitamos este nodo, hay un ciclo - romper el bucle
            if (visited.has(current)) {
                console.warn(`⚠️ Ciclo detectado en transfer learning para "${expName}". Cadena truncada.`);
                break;
            }
            
            visited.add(current);
            chain.unshift(current);
            current = tree[current].loadFrom || '';
            
            // Protección adicional: límite máximo de longitud de cadena
            if (chain.length > 100) {
                console.warn(`⚠️ Cadena de transfer learning muy larga para "${expName}". Cadena truncada.`);
                break;
            }
        }
        
        return chain;
    };

    // Ordenar experimentos según el criterio seleccionado
    const sortedExperiments = useMemo(() => {
        if (!experimentsData) return [];
        
        const sorted = [...experimentsData];
        sorted.sort((a, b) => {
            switch (sortBy) {
                case 'created': {
                    const aCreated = a.created_at || '';
                    const bCreated = b.created_at || '';
                    return bCreated.localeCompare(aCreated); // Más recientes primero
                }
                case 'updated': {
                    const aUpdated = a.updated_at || '';
                    const bUpdated = b.updated_at || '';
                    return bUpdated.localeCompare(aUpdated); // Más recientes primero
                }
                case 'name':
                    return a.name.localeCompare(b.name);
                case 'training_time': {
                    const aTime = a.total_training_time || 0;
                    const bTime = b.total_training_time || 0;
                    return bTime - aTime; // Más tiempo primero
                }
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
        
        // Función helper para encontrar raíz usando el árbol ordenado
        // Cache para evitar recalcular raíces y detectar ciclos solo una vez
        const rootCache = new Map<string, string>();
        const cycleDetected = new Set<string>();
        
        const findRootInTree = (expName: string, visited: Set<string> = new Set()): string => {
            // Si ya calculamos la raíz de este nodo, retornarla del cache
            if (rootCache.has(expName)) {
                return rootCache.get(expName)!;
            }
            
            // Si ya visitamos este nodo, hay un ciclo - retornar el nodo actual
            if (visited.has(expName)) {
                // Solo loguear el ciclo una vez por experimento
                if (!cycleDetected.has(expName)) {
                    if (process.env.NODE_ENV === 'development') {
                        console.warn(`⚠️ Ciclo detectado en transfer learning para "${expName}". Retornando nodo actual.`);
                    }
                    cycleDetected.add(expName);
                }
                rootCache.set(expName, expName);
                return expName;
            }
            
            const node = experimentTreeSorted[expName];
            if (!node || !node.loadFrom) {
                rootCache.set(expName, expName);
                return expName;
            }
            
            // Agregar a visitados y continuar recursión
            visited.add(expName);
            const root = findRootInTree(node.loadFrom, visited);
            rootCache.set(expName, root);
            return root;
        };
        
        Object.keys(experimentTreeSorted).forEach(expName => {
            const root = findRootInTree(expName);
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

    useEffect(() => {
        if (opened && selectedExp) {
            sendCommand('snapshot', 'list_snapshots', { experiment_name: selectedExp });
        }
    }, [opened, selectedExp, sendCommand]);

    const handleLoadSnapshot = (filepath_pt: string) => {
        if (confirm('¿Cargar este snapshot? La simulación actual se detendrá y su estado se reemplazará.')) {
            sendCommand('snapshot', 'load_snapshot', { filepath_pt });
            close();
        }
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
    const transferChain = selectedExp ? getTransferChain(selectedExp, experimentTree) : [];

    return (
        <>
            <GlassPanel className="p-3">
                {/* Header */}
                <div className="flex items-center justify-between mb-3">
                    <div className="flex items-center gap-2">
                        <Database size={14} className="text-gray-400" />
                        <span className="text-xs font-bold text-gray-300">Gestión</span>
                        <span className="px-1.5 py-0.5 rounded text-[10px] font-bold border bg-blue-500/10 text-blue-400 border-blue-500/30">
                            {experimentsData?.length || 0}
                        </span>
                    </div>
                    <div className="flex items-center gap-2">
                        <select
                            value={sortBy}
                            onChange={(e) => setSortBy(e.target.value as 'created' | 'updated' | 'name' | 'training_time')}
                            className="px-2 py-1 bg-white/5 border border-white/10 rounded text-[10px] text-gray-300 focus:outline-none focus:border-blue-500/50"
                            style={{ width: 140 }}
                        >
                            <option value="created">Fecha Creación</option>
                            <option value="updated">Última Actualización</option>
                            <option value="name">Nombre</option>
                            <option value="training_time">Tiempo Entrenamiento</option>
                        </select>
                        <select
                            value={viewMode}
                            onChange={(e) => setViewMode(e.target.value as 'list' | 'tree')}
                            className="px-2 py-1 bg-white/5 border border-white/10 rounded text-[10px] text-gray-300 focus:outline-none focus:border-blue-500/50"
                            style={{ width: 80 }}
                        >
                            <option value="list">Lista</option>
                            <option value="tree">Árbol</option>
                        </select>
                        <button
                            onClick={() => sendCommand('system', 'refresh_experiments', {})}
                            className="p-1.5 text-gray-400 hover:text-gray-200 hover:bg-white/5 rounded transition-all"
                            title="Refrescar"
                        >
                            <Check size={12} />
                        </button>
                    </div>
                </div>

                {/* Scrollable Content */}
                <div className="max-h-[300px] overflow-y-auto custom-scrollbar">
                    {viewMode === 'tree' ? (
                        <div className="space-y-2">
                            {Object.entries(groupedExperiments).map(([root, exps]) => {
                                const rootNode = experimentTreeSorted[root];
                                const isActive = root === activeExperiment;
                                
                                return (
                                    <GlassPanel 
                                        key={root} 
                                        className={`p-2 ${isActive ? 'bg-blue-500/5 border-blue-500/30' : ''}`}
                                    >
                                        <div className="space-y-2">
                                            <div className="flex items-center justify-between gap-2">
                                                <div className="flex items-center gap-2 flex-1 min-w-0">
                                                    <span className={`px-1.5 py-0.5 rounded text-[10px] font-bold border shrink-0 ${
                                                        rootNode.hasCheckpoint 
                                                            ? 'bg-teal-500/10 text-teal-400 border-teal-500/30' 
                                                            : 'bg-pink-500/10 text-pink-400 border-pink-500/30'
                                                    }`}>
                                                        {rootNode.hasCheckpoint ? '✓' : '○'}
                                                    </span>
                                                    <div className="flex-1 min-w-0">
                                                        <div className="flex items-center gap-2">
                                                            <div 
                                                                className={`text-xs cursor-pointer truncate flex-1 ${isActive ? 'font-bold text-blue-400' : 'font-medium text-gray-300'}`}
                                                                onClick={() => setActiveExperiment(root)}
                                                            >
                                                                {root}
                                                            </div>
                                                            {/* Mostrar indicador de transfer si el root tiene loadFrom */}
                                                            {(() => {
                                                                const rootExp = sortedExperiments.find(e => e.name === root);
                                                                const rootHasTransfer = rootExp?.config?.LOAD_FROM_EXPERIMENT;
                                                                if (rootHasTransfer) {
                                                                    return (
                                                                        <div className="flex items-center gap-1 px-1.5 py-0.5 rounded text-[10px] font-bold border bg-purple-500/10 text-purple-400 border-purple-500/30 shrink-0" title={`Transfer desde: ${rootHasTransfer}`}>
                                                                            <ArrowRightLeft size={8} />
                                                                            <span className="truncate max-w-[60px]" title={rootHasTransfer}>{rootHasTransfer}</span>
                                                                        </div>
                                                                    );
                                                                }
                                                                return null;
                                                            })()}
                                                        </div>
                                                        {(() => {
                                                            const exp = sortedExperiments.find(e => e.name === root);
                                                            if (!exp) return null;
                                                            const created = exp.created_at ? new Date(exp.created_at).toLocaleDateString() : '';
                                                            const trainingTime = exp.total_training_time ? formatTime(exp.total_training_time) : '';
                                                            return (
                                                                <div className="text-[10px] text-gray-500 truncate">
                                                                    {trainingTime ? `${created} • ${trainingTime}` : created}
                                                                </div>
                                                            );
                                                        })()}
                                                    </div>
                                                    {isActive && (
                                                        <span className="px-1.5 py-0.5 rounded text-[10px] font-bold border bg-blue-500/10 text-blue-400 border-blue-500/30 shrink-0">
                                                            Activo
                                                        </span>
                                                    )}
                                                </div>
                                                <div className="flex items-center gap-1 shrink-0">
                                                    <button
                                                        onClick={() => handleViewDetails(root)}
                                                        className="p-1 text-gray-400 hover:text-gray-200 hover:bg-white/5 rounded transition-all"
                                                        title="Ver detalles"
                                                    >
                                                        <Info size={12} />
                                                    </button>
                                                    <button
                                                        onClick={(e) => {
                                                            e.stopPropagation();
                                                            handleDeleteExperiment(root);
                                                        }}
                                                        className="p-1 text-red-400 hover:text-red-300 hover:bg-red-500/10 rounded transition-all"
                                                        title="Eliminar experimento"
                                                    >
                                                        <Trash2 size={12} />
                                                    </button>
                                                </div>
                                            </div>
                                            
                                            {exps.length > 1 && (
                                                <div className="pl-4 border-l-2 border-blue-500/30">
                                                    <div className="space-y-1">
                                                        {exps.slice(1).map((expName) => {
                                                            const node = experimentTreeSorted[expName];
                                                            const isChildActive = expName === activeExperiment;
                                                            
                                                            return (
                                                                <div key={expName} className="flex items-center gap-2">
                                                                    <ArrowRightLeft size={10} className="text-blue-400 shrink-0" />
                                                                    <div 
                                                                        className={`text-xs flex-1 min-w-0 truncate cursor-pointer ${isChildActive ? 'font-bold text-blue-400' : 'font-normal text-gray-400'}`}
                                                                        onClick={() => setActiveExperiment(expName)}
                                                                    >
                                                                        {expName}
                                                                    </div>
                                                                    {node.hasCheckpoint && (
                                                                        <span className="px-1 py-0.5 rounded text-[10px] font-bold border bg-teal-500/10 text-teal-400 border-teal-500/30 shrink-0">
                                                                            ✓
                                                                        </span>
                                                                    )}
                                                                </div>
                                                            );
                                                        })}
                                                    </div>
                                                </div>
                                            )}
                                        </div>
                                    </GlassPanel>
                                );
                            })}
                        </div>
                    ) : (
                        <div className="overflow-x-auto">
                            <table className="w-full text-left">
                                <thead>
                                    <tr className="border-b border-white/10">
                                        <th className="text-[10px] font-bold text-gray-400 uppercase tracking-wider py-2 pr-4">Nombre</th>
                                        <th className="text-[10px] font-bold text-gray-400 uppercase tracking-wider py-2 pr-4">Estado</th>
                                        <th className="text-[10px] font-bold text-gray-400 uppercase tracking-wider py-2 pr-4">Creado</th>
                                        <th className="text-[10px] font-bold text-gray-400 uppercase tracking-wider py-2 pr-4">Tiempo Entrenamiento</th>
                                        <th className="text-[10px] font-bold text-gray-400 uppercase tracking-wider py-2 pr-4">Transfer</th>
                                        <th className="text-[10px] font-bold text-gray-400 uppercase tracking-wider py-2">Acciones</th>
                                    </tr>
                                </thead>
                                <tbody>
                                    {sortedExperiments.map(exp => {
                                        const node = experimentTreeSorted[exp.name];
                                        const isActive = exp.name === activeExperiment;
                                        
                                        return (
                                            <tr 
                                                key={exp.name}
                                                className={`border-b border-white/5 cursor-pointer hover:bg-white/5 ${isActive ? 'bg-blue-500/5' : ''}`}
                                                onClick={() => setActiveExperiment(exp.name)}
                                            >
                                                <td className="py-2 pr-4">
                                                    <div className="flex items-center gap-2">
                                                        <span className={`text-xs flex-1 min-w-0 truncate ${isActive ? 'font-bold text-blue-400' : 'font-medium text-gray-300'}`}>
                                                            {exp.name}
                                                        </span>
                                                        {node.loadFrom && (
                                                            <div className="flex items-center gap-1 px-1.5 py-0.5 rounded text-[10px] font-bold border bg-purple-500/10 text-purple-400 border-purple-500/30 shrink-0" title={`Transfer desde: ${node.loadFrom}`}>
                                                                <ArrowRightLeft size={8} />
                                                                <span className="truncate max-w-[60px]" title={node.loadFrom}>{node.loadFrom}</span>
                                                            </div>
                                                        )}
                                                        {isActive && (
                                                            <span className="px-1.5 py-0.5 rounded text-[10px] font-bold border bg-blue-500/10 text-blue-400 border-blue-500/30 shrink-0">
                                                                Activo
                                                            </span>
                                                        )}
                                                    </div>
                                                </td>
                                                <td className="py-2 pr-4">
                                                    <span className={`px-1.5 py-0.5 rounded text-[10px] font-bold border ${
                                                        exp.has_checkpoint 
                                                            ? 'bg-teal-500/10 text-teal-400 border-teal-500/30' 
                                                            : 'bg-pink-500/10 text-pink-400 border-pink-500/30'
                                                    }`}>
                                                        {exp.has_checkpoint ? '✓' : '○'}
                                                    </span>
                                                </td>
                                                <td className="py-2 pr-4">
                                                    <span className="text-xs text-gray-500">
                                                        {exp.created_at ? new Date(exp.created_at).toLocaleDateString() : 'N/A'}
                                                    </span>
                                                </td>
                                                <td className="py-2 pr-4">
                                                    <div className="flex items-center gap-1">
                                                        <Clock size={10} className="text-gray-500" />
                                                        <span className="text-xs text-gray-500">
                                                            {exp.total_training_time ? formatTime(exp.total_training_time) : '0s'}
                                                        </span>
                                                    </div>
                                                </td>
                                                <td className="py-2 pr-4">
                                                    {node.loadFrom ? (
                                                        <div className="flex items-center gap-1 px-1.5 py-0.5 rounded text-[10px] font-bold border bg-purple-500/10 text-purple-400 border-purple-500/30" title={`Transfer desde: ${node.loadFrom}`}>
                                                            <ArrowRightLeft size={10} />
                                                            <span className="text-xs truncate max-w-[100px]" title={node.loadFrom}>
                                                                {node.loadFrom}
                                                            </span>
                                                        </div>
                                                    ) : (
                                                        <span className="text-xs text-gray-500">-</span>
                                                    )}
                                                </td>
                                                <td className="py-2">
                                                    <div className="flex items-center gap-1">
                                                        <button
                                                            onClick={(e) => {
                                                                e.stopPropagation();
                                                                handleViewDetails(exp.name);
                                                            }}
                                                            className="p-1 text-gray-400 hover:text-gray-200 hover:bg-white/5 rounded transition-all"
                                                            title="Ver detalles"
                                                        >
                                                            <Info size={12} />
                                                        </button>
                                                        <button
                                                            onClick={(e) => {
                                                                e.stopPropagation();
                                                                handleDeleteExperiment(exp.name);
                                                            }}
                                                            className="p-1 text-red-400 hover:text-red-300 hover:bg-red-500/10 rounded transition-all"
                                                            title="Eliminar experimento"
                                                        >
                                                            <Trash2 size={12} />
                                                        </button>
                                                    </div>
                                                </td>
                                            </tr>
                                        );
                                    })}
                                </tbody>
                            </table>
                        </div>
                    )}
                </div>
            </GlassPanel>

            {/* Modal de detalles */}
            {opened && (
                <div className="fixed inset-0 z-[100] flex items-center justify-center pointer-events-none">
                    {/* Backdrop */}
                    <div 
                        className="absolute inset-0 bg-black/80 backdrop-blur-sm pointer-events-auto"
                        onClick={close}
                    />
                    
                    {/* Modal */}
                    <GlassPanel className="relative w-[600px] max-h-[80vh] pointer-events-auto flex flex-col overflow-hidden">
                        {/* Header */}
                        <div className="flex items-center justify-between p-4 border-b border-white/10 shrink-0">
                            <h2 className="text-sm font-bold text-gray-200">Detalles: {selectedExp}</h2>
                            <button
                                onClick={close}
                                className="p-1.5 text-gray-400 hover:text-gray-200 transition-colors rounded hover:bg-white/5"
                            >
                                <X size={18} />
                            </button>
                        </div>
                        
                        {/* Content */}
                        <div className="flex-1 overflow-y-auto custom-scrollbar p-4 space-y-4">
                            {selectedExperiment && (
                                <>
                                    {/* Cadena de transfer learning */}
                                    {transferChain.length > 1 && (
                                        <div className="p-3 bg-blue-500/10 border border-blue-500/30 rounded">
                                            <div className="flex items-center gap-2 mb-2">
                                                <ArrowRightLeft size={14} className="text-blue-400" />
                                                <span className="text-xs font-bold text-blue-400">Cadena de Transfer Learning:</span>
                                            </div>
                                            <div className="space-y-2 pl-6 border-l-2 border-blue-500/30">
                                                {transferChain.map((expName, idx) => (
                                                    <div key={expName} className="relative">
                                                        <div className="flex items-center gap-2">
                                                            <div className={`w-2 h-2 rounded-full ${idx === transferChain.length - 1 ? 'bg-blue-400' : 'bg-blue-500/50'}`} />
                                                            <span className="text-xs font-medium text-gray-300">{expName}</span>
                                                        </div>
                                                        {idx > 0 && (
                                                            <div className="text-[10px] text-gray-500 pl-4 mt-0.5">
                                                                Transfer desde: {transferChain[idx - 1]}
                                                            </div>
                                                        )}
                                                    </div>
                                                ))}
                                            </div>
                                        </div>
                                    )}

                                    {/* Información temporal */}
                                    {(() => {
                                        const exp = sortedExperiments.find(e => e.name === selectedExp);
                                        if (!exp) return null;
                                        
                                        return (
                                            <GlassPanel className="p-4">
                                                <h3 className="text-xs font-bold text-gray-300 mb-3 uppercase tracking-wider">Información Temporal</h3>
                                                <table className="w-full">
                                                    <tbody className="space-y-2">
                                                        <tr>
                                                            <td className="text-xs font-medium text-gray-400 py-1 pr-4">Creado:</td>
                                                            <td className="text-xs text-gray-300 py-1">
                                                                {exp.created_at ? new Date(exp.created_at).toLocaleString() : 'N/A'}
                                                            </td>
                                                        </tr>
                                                        <tr>
                                                            <td className="text-xs font-medium text-gray-400 py-1 pr-4">Última Actualización:</td>
                                                            <td className="text-xs text-gray-300 py-1">
                                                                {exp.updated_at ? new Date(exp.updated_at).toLocaleString() : 'N/A'}
                                                            </td>
                                                        </tr>
                                                        <tr>
                                                            <td className="text-xs font-medium text-gray-400 py-1 pr-4">Tiempo Total de Entrenamiento:</td>
                                                            <td className="text-xs text-gray-300 py-1">
                                                                <div className="flex items-center gap-1">
                                                                    <Clock size={12} className="text-gray-500" />
                                                                    <span>{exp.total_training_time ? formatTime(exp.total_training_time) : '0s'}</span>
                                                                </div>
                                                            </td>
                                                        </tr>
                                                        {exp.last_training_time && (
                                                            <tr>
                                                                <td className="text-xs font-medium text-gray-400 py-1 pr-4">Última Sesión de Entrenamiento:</td>
                                                                <td className="text-xs text-gray-300 py-1">
                                                                    {new Date(exp.last_training_time).toLocaleString()}
                                                                </td>
                                                            </tr>
                                                        )}
                                                    </tbody>
                                                </table>
                                            </GlassPanel>
                                        );
                                    })()}

                                    {/* Snapshots de Inferencia */}
                                    <GlassPanel className="p-4">
                                        <h3 className="text-xs font-bold text-gray-300 mb-3 uppercase tracking-wider flex items-center gap-2">
                                            <Save size={12} />
                                            Snapshots de Inferencia
                                        </h3>
                                        {inferenceSnapshots && inferenceSnapshots.length > 0 ? (
                                            <div className="space-y-2 max-h-[150px] overflow-y-auto custom-scrollbar pr-2">
                                                {inferenceSnapshots.map((snapshot, index) => (
                                                    <div key={index} className="flex items-center justify-between p-2 bg-white/5 rounded">
                                                        <div>
                                                            <div className="text-xs font-mono text-gray-300">Paso: {snapshot.step.toLocaleString()}</div>
                                                            <div className="text-[10px] text-gray-500">{new Date(snapshot.timestamp).toLocaleString()}</div>
                                                        </div>
                                                        <button
                                                            onClick={() => handleLoadSnapshot(snapshot.filepath_pt)}
                                                            className="flex items-center gap-1.5 px-2 py-1 bg-blue-500/10 hover:bg-blue-500/20 border border-blue-500/30 text-blue-400 text-[10px] font-bold rounded transition-all"
                                                        >
                                                            <UploadCloud size={12} />
                                                            Cargar
                                                        </button>
                                                    </div>
                                                ))}
                                            </div>
                                        ) : (
                                            <p className="text-xs text-gray-500 italic">No hay snapshots para este experimento.</p>
                                        )}
                                    </GlassPanel>
                                    
                                    {/* Información del experimento */}
                                    <GlassPanel className="p-4">
                                        <h3 className="text-xs font-bold text-gray-300 mb-3 uppercase tracking-wider">Configuración</h3>
                                        <table className="w-full">
                                            <tbody>
                                                <tr>
                                                    <td className="text-xs font-medium text-gray-400 py-1 pr-4">Arquitectura</td>
                                                    <td className="text-xs text-gray-300 py-1">{selectedExperiment.config?.MODEL_ARCHITECTURE || 'N/A'}</td>
                                                </tr>
                                                <tr>
                                                    <td className="text-xs font-medium text-gray-400 py-1 pr-4">d_state</td>
                                                    <td className="text-xs text-gray-300 py-1">{selectedExperiment.config?.MODEL_PARAMS?.d_state || 'N/A'}</td>
                                                </tr>
                                                <tr>
                                                    <td className="text-xs font-medium text-gray-400 py-1 pr-4">Hidden Channels</td>
                                                    <td className="text-xs text-gray-300 py-1">{selectedExperiment.config?.MODEL_PARAMS?.hidden_channels || 'N/A'}</td>
                                                </tr>
                                                <tr>
                                                    <td className="text-xs font-medium text-gray-400 py-1 pr-4">Learning Rate</td>
                                                    <td className="text-xs text-gray-300 py-1">{selectedExperiment.config?.LR_RATE_M || 'N/A'}</td>
                                                </tr>
                                                <tr>
                                                    <td className="text-xs font-medium text-gray-400 py-1 pr-4">Total Episodios</td>
                                                    <td className="text-xs text-gray-300 py-1">{selectedExperiment.config?.TOTAL_EPISODES || 'N/A'}</td>
                                                </tr>
                                                <tr>
                                                    <td className="text-xs font-medium text-gray-400 py-1 pr-4">Grid Size</td>
                                                    <td className="text-xs text-gray-300 py-1">{selectedExperiment.config?.GRID_SIZE_TRAINING || 'N/A'}</td>
                                                </tr>
                                                <tr>
                                                    <td className="text-xs font-medium text-gray-400 py-1 pr-4">Estado</td>
                                                    <td className="text-xs py-1">
                                                        <span className={`px-2 py-0.5 rounded text-[10px] font-bold border ${
                                                            selectedExperiment.hasCheckpoint 
                                                                ? 'bg-teal-500/10 text-teal-400 border-teal-500/30' 
                                                                : 'bg-pink-500/10 text-pink-400 border-pink-500/30'
                                                        }`}>
                                                            {selectedExperiment.hasCheckpoint ? 'Entrenado' : 'Sin entrenar'}
                                                        </span>
                                                    </td>
                                                </tr>
                                                {selectedExperiment.loadFrom && (
                                                    <tr>
                                                        <td className="text-xs font-medium text-gray-400 py-1 pr-4">Transfer Desde</td>
                                                        <td className="text-xs text-gray-300 py-1">
                                                            <div className="flex items-center gap-1">
                                                                <ArrowRightLeft size={12} className="text-gray-500" />
                                                                <span>{selectedExperiment.loadFrom}</span>
                                                            </div>
                                                        </td>
                                                    </tr>
                                                )}
                                                {selectedExperiment.children.length > 0 && (
                                                    <tr>
                                                        <td className="text-xs font-medium text-gray-400 py-1 pr-4">Usado por</td>
                                                        <td className="text-xs py-1">
                                                            <div className="flex flex-wrap gap-1">
                                                                {selectedExperiment.children.map(child => (
                                                                    <span key={child} className="px-2 py-0.5 rounded text-[10px] font-medium border bg-blue-500/10 text-blue-400 border-blue-500/30">
                                                                        {child}
                                                                    </span>
                                                                ))}
                                                            </div>
                                                        </td>
                                                    </tr>
                                                )}
                                                {/* Información del motor y dispositivo */}
                                                <tr>
                                                    <td className="text-xs font-medium text-gray-400 py-1 pr-4">Motor</td>
                                                    <td className="text-xs py-1">
                                                        {(() => {
                                                            const exp = sortedExperiments.find(e => e.name === selectedExp);
                                                            const useNative = exp?.config?.USE_NATIVE_ENGINE;
                                                            if (useNative) {
                                                                return (
                                                                    <span className="px-2 py-0.5 rounded text-[10px] font-bold border bg-emerald-500/10 text-emerald-400 border-emerald-500/30">
                                                                        ⚡ Nativo (C++)
                                                                    </span>
                                                                );
                                                            } else {
                                                                return (
                                                                    <span className="px-2 py-0.5 rounded text-[10px] font-bold border bg-amber-500/10 text-amber-400 border-amber-500/30">
                                                                        🐍 Python
                                                                    </span>
                                                                );
                                                            }
                                                        })()}
                                                    </td>
                                                </tr>
                                                <tr>
                                                    <td className="text-xs font-medium text-gray-400 py-1 pr-4">Dispositivo</td>
                                                    <td className="text-xs py-1">
                                                        {(() => {
                                                            const exp = sortedExperiments.find(e => e.name === selectedExp);
                                                            const device = exp?.config?.TRAINING_DEVICE || 'cpu';
                                                            const isCuda = device.toLowerCase() === 'cuda';
                                                            return (
                                                                <span className={`px-2 py-0.5 rounded text-[10px] font-bold border ${
                                                                    isCuda 
                                                                        ? 'bg-purple-500/10 text-purple-400 border-purple-500/30' 
                                                                        : 'bg-gray-500/10 text-gray-400 border-gray-500/30'
                                                                }`}>
                                                                    {isCuda ? '🎮 CUDA (Gráfica)' : '💻 CPU'}
                                                                </span>
                                                            );
                                                        })()}
                                                    </td>
                                                </tr>
                                            </tbody>
                                        </table>
                                    </GlassPanel>
                                </>
                            )}
                        </div>
                        
                        {/* Footer */}
                        <div className="flex items-center justify-between p-4 border-t border-white/10 shrink-0 gap-3">
                            <button
                                onClick={() => {
                                    close();
                                    if (selectedExp) {
                                        handleDeleteExperiment(selectedExp);
                                    }
                                }}
                                className="flex items-center gap-2 px-4 py-2 bg-red-500/10 hover:bg-red-500/20 border border-red-500/30 text-red-400 text-xs font-bold rounded transition-all"
                            >
                                <Trash2 size={14} />
                                Eliminar
                            </button>
                            <div className="flex gap-2">
                                <button
                                    onClick={close}
                                    className="px-4 py-2 bg-white/5 hover:bg-white/10 border border-white/10 text-gray-300 text-xs font-bold rounded transition-all"
                                >
                                    Cerrar
                                </button>
                                <button
                                    onClick={() => {
                                        if (selectedExp) {
                                            setActiveExperiment(selectedExp);
                                        }
                                        close();
                                    }}
                                    className="px-4 py-2 bg-blue-500/10 hover:bg-blue-500/20 border border-blue-500/30 text-blue-300 text-xs font-bold rounded transition-all"
                                >
                                    Seleccionar
                                </button>
                            </div>
                        </div>
                    </GlassPanel>
                </div>
            )}
        </>
    );
}
