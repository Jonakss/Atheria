// frontend/src/context/WebSocketContext.tsx
import { createContext, useState, useEffect, useRef, useCallback, ReactNode } from 'react';
import { decompressIfNeeded, decodeBinaryFrame, processDecodedPayload } from '../utils/dataDecompression';

/**
 * Descomprime arrays comprimidos dentro de un payload (formato antiguo o nuevo).
 */
function decompressPayloadArrays(payload: any): any {
    if (!payload || typeof payload !== 'object') {
        return payload;
    }
    
    const result: any = { ...payload };
    
    // Descomprimir map_data si existe
    if (result.map_data) {
        result.map_data = decompressIfNeeded(result.map_data);
    }
    
    // Descomprimir complex_3d_data si existe
    if (result.complex_3d_data && typeof result.complex_3d_data === 'object') {
        if (result.complex_3d_data.real) {
            result.complex_3d_data.real = decompressIfNeeded(result.complex_3d_data.real);
        }
        if (result.complex_3d_data.imag) {
            result.complex_3d_data.imag = decompressIfNeeded(result.complex_3d_data.imag);
        }
    }
    
    // Descomprimir flow_data si existe
    if (result.flow_data && typeof result.flow_data === 'object') {
        if (result.flow_data.dx) {
            result.flow_data.dx = decompressIfNeeded(result.flow_data.dx);
        }
        if (result.flow_data.dy) {
            result.flow_data.dy = decompressIfNeeded(result.flow_data.dy);
        }
        if (result.flow_data.magnitude) {
            result.flow_data.magnitude = decompressIfNeeded(result.flow_data.magnitude);
        }
    }
    
    // Descomprimir phase_hsv_data si existe
    if (result.phase_hsv_data && typeof result.phase_hsv_data === 'object') {
        if (result.phase_hsv_data.hue) {
            result.phase_hsv_data.hue = decompressIfNeeded(result.phase_hsv_data.hue);
        }
        if (result.phase_hsv_data.saturation) {
            result.phase_hsv_data.saturation = decompressIfNeeded(result.phase_hsv_data.saturation);
        }
        if (result.phase_hsv_data.value) {
            result.phase_hsv_data.value = decompressIfNeeded(result.phase_hsv_data.value);
        }
    }
    
    return result;
}
import { getServerConfig, getWebSocketUrl, saveServerConfig, type ServerConfig } from '../utils/serverConfig';

interface SimData {
    complex_3d_data?: {
        real: number[][];
        imag: number[][];
    };
    map_data?: number[][];
    hist_data?: Record<string, Array<{ bin: string; count: number }>>;
    poincare_coords?: number[][];
    step?: number | null;
    timestamp?: number;
    simulation_info?: {
        step?: number;
        is_paused?: boolean;
        live_feed_enabled?: boolean;
        gamma_decay?: number;
        fps?: number;
        epoch?: number;
        epoch_metrics?: {
            energy?: number;
            clustering?: number;
            symmetry?: number;
        };
    };
    phase_attractor?: any;
    flow_data?: {
        dx: number[][];
        dy: number[][];
        magnitude?: number[][];
    };
    phase_hsv_data?: {
        hue: number[][];
        saturation: number[][];
        value: number[][];
    };
    roi_info?: any;
}

interface TrainingProgress {
    current_episode: number;
    total_episodes: number;
    avg_loss: number;
    avg_reward?: number;
}

interface TrainingSnapshot {
    episode: number;
    step?: number;
    map_data: number[][];
    timestamp?: number;
    loss?: number;
    metrics?: {
        survival?: number;
        symmetry?: number;
        complexity?: number;
    };
}

interface CompileStatus {
    is_compiled: boolean;
    is_native: boolean;  // ← INDICADOR DE MOTOR NATIVO
    model_name: string;
    compiles_enabled: boolean;
    device_str?: string;  // CPU/CUDA - CORREGIDO: usar device_str para consistencia
}

interface WebSocketContextType {
    sendCommand: (scope: string, command: string, args?: Record<string, any>) => void;
    connectionStatus: 'connecting' | 'connected' | 'disconnected' | 'server_unavailable';
    experimentsData: any[] | null;
    trainingStatus: 'idle' | 'running';
    inferenceStatus: 'paused' | 'running';
    analysisStatus: 'idle' | 'running' | 'completed' | 'cancelled' | 'error'; // Estado del análisis
    analysisType: 'universe_atlas' | 'cell_chemistry' | null; // Tipo de análisis actual
    selectedViz: string;
    setSelectedViz: (viz: string) => void;
    connect: () => void;
    reconnect: () => void; // Función para reconexión manual
    disconnect: () => void; // Función para desconectar manualmente
    simData: SimData | null;
    trainingLog: string[];
    allLogs: string[]; // Logs unificados
    trainingProgress: TrainingProgress | null;
    activeExperiment: string | null;
    setActiveExperiment: (name: string | null) => void;
    ws: WebSocket | null; // Exponer WebSocket para escuchar mensajes personalizados
    snapshotCount: number; // Contador de snapshots capturados
    trainingSnapshots: TrainingSnapshot[]; // Snapshots de entrenamiento
    serverConfig: ServerConfig; // Configuración del servidor
    updateServerConfig: (config: Partial<ServerConfig>) => void; // Actualizar configuración
    compileStatus: CompileStatus | null; // Estado de compilación/motor
    liveFeedEnabled: boolean; // Estado del live feed (sincronizado con backend)
    setLiveFeedEnabled: (enabled: boolean) => void; // Función para cambiar live feed
}

export const WebSocketContext = createContext<WebSocketContextType | undefined>(undefined);

export const WebSocketProvider = ({ children }: { children: ReactNode }) => {
    const [connectionStatus, setConnectionStatus] = useState<'connecting' | 'connected' | 'disconnected' | 'server_unavailable'>('disconnected');
    const ws = useRef<WebSocket | null>(null);
    const effectRan = useRef(false);
    const [liveFeedEnabled, setLiveFeedEnabledState] = useState<boolean>(true); // Por defecto habilitado
    const reconnectAttempts = useRef(0);
    const lastErrorLog = useRef(0);
    const isManualClose = useRef(false); // Flag para indicar si el cierre fue manual
    const [serverConfig, setServerConfigState] = useState<ServerConfig>(getServerConfig());
    // Usar una referencia para acceder al valor actual de serverConfig en callbacks
    const serverConfigRef = useRef<ServerConfig>(serverConfig);

    // --- Hooks de Estado para manejar la lógica de la aplicación ---
    const [experimentsData, setExperimentsData] = useState<any[] | null>(null);
    const [trainingStatus, setTrainingStatus] = useState<'idle' | 'running'>('idle');
    const [inferenceStatus, setInferenceStatus] = useState<'paused' | 'running'>('paused');
    const [selectedViz, setSelectedViz] = useState<string>('density'); // Valor inicial
    const [simData, setSimData] = useState<SimData | null>(null);
    const [trainingLog, setTrainingLog] = useState<string[]>([]);
    const [allLogs, setAllLogs] = useState<string[]>([]); // Logs unificados
    const [trainingProgress, setTrainingProgress] = useState<TrainingProgress | null>(null);
    const [activeExperiment, setActiveExperiment] = useState<string | null>(null);
    const [snapshotCount, setSnapshotCount] = useState<number>(0);
    const [trainingSnapshots, setTrainingSnapshots] = useState<TrainingSnapshot[]>([]);
    const [analysisStatus, setAnalysisStatus] = useState<'idle' | 'running' | 'completed' | 'cancelled' | 'error'>('idle');
    const [analysisType, setAnalysisType] = useState<'universe_atlas' | 'cell_chemistry' | null>(null);
    const [compileStatus, setCompileStatus] = useState<CompileStatus | null>(null);
    // Usar una referencia para acceder al valor actual de activeExperiment en callbacks
    const activeExperimentRef = useRef<string | null>(null);
    
    // Mantener las referencias actualizadas
    useEffect(() => {
        activeExperimentRef.current = activeExperiment;
    }, [activeExperiment]);
    
    useEffect(() => {
        serverConfigRef.current = serverConfig;
    }, [serverConfig]);

    // Función para actualizar la configuración del servidor
    const updateServerConfig = useCallback((config: Partial<ServerConfig>) => {
        const currentConfig = serverConfigRef.current;
        const newConfig = { ...currentConfig, ...config };
        setServerConfigState(newConfig);
        saveServerConfig(newConfig);
        // Cerrar conexión actual para reconectar con la nueva configuración
        if (ws.current) {
            isManualClose.current = true; // Marcar como cierre manual
            ws.current.close(1000); // Código 1000 = cierre normal
        }
    }, []);

    const connect = useCallback((isManual = false) => {
        if (ws.current?.readyState === WebSocket.OPEN) return;
        
        // Si es una reconexión manual, resetear los intentos
        if (isManual) {
            reconnectAttempts.current = 0;
        }
        
        // Limpiar conexión anterior si existe
        if (ws.current) {
            ws.current.onerror = null;
            ws.current.onclose = null;
            ws.current.close();
        }
        
        setConnectionStatus('connecting');
        // Usar la referencia para obtener siempre la configuración más reciente
        const currentConfig = serverConfigRef.current;
        const wsUrl = getWebSocketUrl(currentConfig);
        const socket = new WebSocket(wsUrl);

        socket.onopen = () => {
            reconnectAttempts.current = 0;
            setConnectionStatus('connected');
            // Solo loguear en desarrollo
            if (process.env.NODE_ENV === 'development') {
                console.log("✓ WebSocket conectado");
            }
        };
        
        socket.onclose = (event) => {
            // Si fue un cierre manual (cambio de configuración), no reconectar automáticamente
            if (isManualClose.current) {
                isManualClose.current = false; // Resetear el flag
                setConnectionStatus('disconnected');
                reconnectAttempts.current = 0; // Resetear intentos
                return;
            }
            
            // No loguear si fue un cierre limpio (código 1000)
            if (event.code !== 1000) {
                reconnectAttempts.current += 1;
                
                // Solo loguear errores ocasionalmente (cada 10 segundos)
                const now = Date.now();
                if (now - lastErrorLog.current > 10000) {
                    if (reconnectAttempts.current === 1) {
                        console.warn("⚠ Servidor no disponible. Intentando reconectar...");
                    }
                    lastErrorLog.current = now;
                }
                
                // Después de 3 intentos, marcar como servidor no disponible y detener reconexión automática
                if (reconnectAttempts.current >= 3) {
                    setConnectionStatus('server_unavailable');
                    console.warn("⚠ Se agotaron los intentos de reconexión automática. Usa el botón de reconexión para intentar nuevamente.");
                } else {
                    setConnectionStatus('disconnected');
                    // Reintentar conexión automáticamente solo si no hemos alcanzado el límite
                    setTimeout(connect, 3000);
                }
            } else {
                setConnectionStatus('disconnected');
            }
        };
        
        socket.onerror = (error) => {
            // Solo loguear errores persistentes, no durante la conexión inicial
            const now = Date.now();
            if (now - lastErrorLog.current > 10000 && connectionStatus !== 'connecting') {
                if (process.env.NODE_ENV === 'development') {
                    console.debug("Error de WebSocket (ignorado durante conexión inicial)");
                }
                lastErrorLog.current = now;
            }
        };

        socket.onmessage = async (event) => {
            try {
                let message: any;
                
                // Detectar si es un frame binario (ArrayBuffer/Blob) o texto (JSON)
                if (event.data instanceof ArrayBuffer || event.data instanceof Blob) {
                    // Frame binario (formato optimizado)
                    const arrayBuffer = event.data instanceof Blob 
                        ? await event.data.arrayBuffer() 
                        : event.data;
                    
                    try {
                        // Decodificar frame binario (CBOR o JSON con datos binarios)
                        const frameData = await decodeBinaryFrame(new Uint8Array(arrayBuffer));
                        
                        // Si tiene estructura de frame optimizado (metadata + arrays)
                        if (frameData.metadata && frameData.arrays) {
                            // Procesar arrays binarios y reconstruir payload
                            // Capturar simData actual para differential compression
                            const currentSimData = simData;
                            message = {
                                type: frameData.metadata.type || 'simulation_frame',
                                payload: processDecodedPayload(frameData, currentSimData)
                            };
                        } else if (frameData.type) {
                            // Estructura simple con type y payload
                            // Descomprimir arrays dentro del payload si existen
                            message = {
                                type: frameData.type,
                                payload: decompressPayloadArrays(frameData.payload || frameData)
                            };
                        } else {
                            // Frame sin estructura conocida, intentar procesar directamente
                            message = {
                                type: 'simulation_frame',
                                payload: decompressPayloadArrays(frameData)
                            };
                        }
                    } catch (error) {
                        console.error('Error decodificando frame binario:', error);
                        // Fallback: intentar como JSON string
                        const text = new TextDecoder('utf-8', { fatal: false }).decode(new Uint8Array(arrayBuffer));
                        if (text) {
                            message = JSON.parse(text);
                        } else {
                            throw error;
                        }
                    }
                } else if (typeof event.data === 'string') {
                    // Mensaje JSON (formato antiguo o fallback)
                    message = JSON.parse(event.data);
                    
                    // Descomprimir arrays en el payload si existen
                    if (message.payload) {
                        message.payload = decompressPayloadArrays(message.payload);
                    }
                } else {
                    // Datos desconocidos
                    console.warn('Tipo de dato WebSocket desconocido:', typeof event.data);
                    return;
                }
                
                // Solo loguear en desarrollo y no para cada frame o log
                if (process.env.NODE_ENV === 'development' && 
                    message.type !== 'simulation_frame' && 
                    message.type !== 'simulation_log') {
                    console.debug("Mensaje recibido:", message.type);
                }
                
                const { type, payload } = message;
                switch (type) {
                    case 'initial_state':
                        setExperimentsData(payload.experiments);
                        setTrainingStatus(payload.training_status);
                        setInferenceStatus(payload.inference_status);
                        break;
                    case 'experiments_updated':
                        // Actualizar lista de experimentos cuando se crea/elimina un experimento
                        setExperimentsData(payload.experiments);
                        // Si el experimento activo fue eliminado, limpiar selección
                        const currentActive = activeExperimentRef.current;
                        if (currentActive && !payload.experiments?.find((e: any) => e.name === currentActive)) {
                            setActiveExperiment(null);
                        }
                        break;
                    case 'training_status_update':
                        setTrainingStatus(payload.status);
                        // Limpiar snapshots cuando el entrenamiento termina (opcional)
                        // if (payload.status === 'idle') {
                        //     setTrainingSnapshots([]);
                        // }
                        break;
                    case 'training_snapshot':
                        // Agregar nuevo snapshot de entrenamiento
                        if (payload.snapshot) {
                            setTrainingSnapshots(prev => {
                                const newSnapshot: TrainingSnapshot = payload.snapshot;
                                // Mantener máximo 50 snapshots para evitar acumulación excesiva
                                const updated = [...prev, newSnapshot].slice(-50);
                                return updated;
                            });
                        }
                        break;
                    case 'inference_status_update':
                        setInferenceStatus(payload.status);
                        // Almacenar compile_status si está disponible
                        if (payload.compile_status) {
                            const newCompileStatus = payload.compile_status as CompileStatus;
                            console.log('📥 WebSocketContext - Recibido compile_status:', JSON.stringify(newCompileStatus, null, 2));
                            setCompileStatus(newCompileStatus);
                        } else {
                            console.warn('⚠️ WebSocketContext - inference_status_update sin compile_status. Payload completo:', payload);
                        }
                        break;
                    case 'simulation_frame':
                    case 'simulation_frame_binary':
                        // Payload ya está descomprimido por decompressPayloadArrays o processDecodedPayload
                        // IMPORTANTE: Preservar step, timestamp y simulation_info
                        // Usar función de actualización para evitar condiciones de carrera
                        try {
                            const finalPayload = {
                                ...payload,
                                step: payload.step ?? payload.simulation_info?.step ?? null,
                                timestamp: payload.timestamp ?? Date.now(),
                                simulation_info: payload.simulation_info
                            };
                            
                            // Usar función de actualización para evitar sobrescribir actualizaciones más recientes
                            setSimData(prev => {
                                // Si hay un timestamp y el payload nuevo es más antiguo, ignorarlo
                                if (prev?.timestamp && finalPayload.timestamp && 
                                    finalPayload.timestamp < prev.timestamp) {
                                    return prev;
                                }
                                return finalPayload;
                            });
                        } catch (error) {
                            console.error("Error procesando simulation_frame:", error);
                        }
                        break;
                    case 'simulation_state_update':
                        // Actualización de estado sin datos de visualización (cuando live feed está desactivado)
                        // Actualizar solo step y simulation_info, preservando otros datos existentes
                        // Usar función de actualización para evitar condiciones de carrera
                        try {
                            setSimData(prev => {
                                const newStep = payload.step ?? payload.simulation_info?.step ?? prev?.step ?? null;
                                const newTimestamp = payload.timestamp ?? prev?.timestamp ?? Date.now();
                                
                                // Si hay un timestamp y el payload nuevo es más antiguo, ignorarlo
                                if (prev?.timestamp && newTimestamp < prev.timestamp) {
                                    return prev;
                                }
                                
                                // Sincronizar estado del live feed desde simulation_info
                                const newLiveFeedEnabled = payload.simulation_info?.live_feed_enabled !== undefined 
                                    ? payload.simulation_info.live_feed_enabled 
                                    : (prev?.simulation_info?.live_feed_enabled ?? true);
                                if (newLiveFeedEnabled !== (prev?.simulation_info?.live_feed_enabled ?? true)) {
                                    setLiveFeedEnabledState(newLiveFeedEnabled);
                                }
                                
                                return {
                                    ...prev,
                                    step: newStep,
                                    timestamp: newTimestamp,
                                    simulation_info: {
                                        ...(payload.simulation_info ?? prev?.simulation_info ?? {}),
                                        live_feed_enabled: newLiveFeedEnabled
                                    }
                                    // No actualizar map_data, hist_data, etc. - estos solo vienen con simulation_frame
                                };
                            });
                        } catch (error) {
                            console.error("Error procesando simulation_state_update:", error);
                        }
                        break;
                    case 'training_log':
                        // OPTIMIZACIÓN DE MEMORIA: Limitar tamaño de logs para evitar memory leaks
                        const MAX_LOGS_TRAINING = 500; // Reducido de 1000 a 500 para ahorrar memoria
                        setTrainingLog(prev => {
                            const newLogs = [...prev, payload];
                            return newLogs.slice(-MAX_LOGS_TRAINING); // Mantener solo últimos N logs
                        });
                        setAllLogs(prev => {
                            const newLogs = [...prev, payload];
                            return newLogs.slice(-MAX_LOGS_TRAINING); // Mantener solo últimos N logs
                        });
                        break;
                    case 'simulation_log':
                        // OPTIMIZACIÓN DE MEMORIA: Limitar tamaño de logs para evitar memory leaks
                        const MAX_LOGS = 500; // Reducido de 1000 a 500 para ahorrar memoria
                        setAllLogs(prev => {
                            const newLogs = [...prev, payload];
                            return newLogs.slice(-MAX_LOGS); // Mantener solo últimos N logs
                        });
                        break;
                    case 'training_progress':
                        setTrainingProgress(payload);
                        break;
                    case 'notification':
                        // Mostrar notificación del servidor
                        const { status, message } = payload;
                        const title = status === 'error' ? 'Error' : status === 'warning' ? 'Advertencia' : status === 'success' ? 'Éxito' : 'Información';
                        console.log(`[${title}] ${message}`);
                        // TODO: Implementar sistema de notificaciones con Tailwind
                        break;
                    case 'snapshot_count':
                        // Actualizar contador de snapshots
                        setSnapshotCount(payload.count || 0);
                        break;
                    case 'analysis_status_update':
                        // Actualizar estado de análisis
                        setAnalysisStatus(payload.status || 'idle');
                        setAnalysisType(payload.type || null);
                        break;
                    case 'live_feed_status_update':
                        // Sincronizar estado del live feed
                        // NO limpiar datos aquí - el backend enviará un frame inicial si es necesario
                        setLiveFeedEnabledState(payload.enabled ?? true);
                        break;
                    case 'history_files_list':
                        // Lista de archivos de historia recibida
                        // Se manejará en HistoryViewer
                        window.dispatchEvent(new CustomEvent('history_files_list', { detail: payload.files }));
                        break;
                    case 'history_file_loaded':
                        // Archivo de historia cargado
                        window.dispatchEvent(new CustomEvent('history_file_loaded', { detail: payload }));
                        break;
                    case 'history_saved':
                        // Historia guardada exitosamente
                        window.dispatchEvent(new CustomEvent('history_saved', { detail: payload }));
                        break;
                    case 'checkpoints_list':
                        // Lista de checkpoints recibida
                        window.dispatchEvent(new CustomEvent('checkpoints_updated', { detail: payload.checkpoints }));
                        break;
                }
            } catch (error) {
                console.error("Error procesando mensaje:", error);
            }
        };
        ws.current = socket;
    }, []); // No depender de serverConfig, usar la referencia en su lugar

    useEffect(() => {
        if (effectRan.current === false) {
            connect(false); // Conexión inicial automática
            return () => {
                effectRan.current = true;
                ws.current?.close();
            };
        }
    }, [connect]);
    
    // Función para reconexión manual
    const reconnect = useCallback(() => {
        connect(true); // Reconexión manual, resetea los intentos
    }, [connect]);
    
    // Función para desconectar manualmente
    const disconnect = useCallback(() => {
        if (ws.current) {
            isManualClose.current = true; // Marcar como cierre manual
            ws.current.close(1000); // Código 1000 = cierre normal
            setConnectionStatus('disconnected');
        }
    }, []);
    
    const sendCommand = useCallback((scope: string, command: string, args: Record<string, any> = {}) => {
        if (ws.current?.readyState === WebSocket.OPEN) {
            ws.current.send(JSON.stringify({ scope, command, args }));
        } else {
            // Solo loguear en desarrollo y si no es un estado esperado
            if (process.env.NODE_ENV === 'development' && 
                connectionStatus === 'disconnected' && 
                ws.current?.readyState !== WebSocket.CONNECTING) {
                // Solo loguear si realmente hay un problema, no durante la conexión inicial
                console.debug("No se puede enviar comando: WebSocket no conectado", { scope, command });
            }
        }
    }, [connectionStatus]);
    
    // Función para cambiar el estado del live feed
    const setLiveFeedEnabled = useCallback((enabled: boolean) => {
        setLiveFeedEnabledState(enabled);
        sendCommand('simulation', 'set_live_feed', { enabled });
    }, [sendCommand]);

    const value = {
        sendCommand,
        connectionStatus,
        experimentsData,
        trainingStatus,
        inferenceStatus,
        analysisStatus,
        analysisType,
        selectedViz,
        setSelectedViz,
        connect,
        reconnect, // Función para reconexión manual
        disconnect, // Función para desconectar manualmente
        simData,
        trainingLog,
        allLogs,
        trainingProgress,
        activeExperiment,
        setActiveExperiment,
        ws: ws.current, // Exponer WebSocket para mensajes personalizados
        snapshotCount,
        trainingSnapshots,
        serverConfig,
        updateServerConfig,
        compileStatus, // Estado de compilación/motor (indica si es nativo)
        liveFeedEnabled, // Estado del live feed (sincronizado con backend)
        setLiveFeedEnabled, // Función para cambiar live feed
    };

    return (
        <WebSocketContext.Provider value={value}>
            {children}
        </WebSocketContext.Provider>
    );
};
