// frontend/src/context/WebSocketContext.tsx
import { createContext, useState, useEffect, useRef, useCallback, ReactNode } from 'react';
import { notifications } from '@mantine/notifications';
import { decompressIfNeeded } from '../utils/dataDecompression';
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
    simData: SimData | null;
    trainingLog: string[];
    allLogs: string[]; // Logs unificados
    trainingProgress: TrainingProgress | null;
    activeExperiment: string | null;
    setActiveExperiment: (name: string | null) => void;
    ws: WebSocket | null; // Exponer WebSocket para escuchar mensajes personalizados
    snapshotCount: number; // Contador de snapshots capturados
    serverConfig: ServerConfig; // Configuración del servidor
    updateServerConfig: (config: Partial<ServerConfig>) => void; // Actualizar configuración
}

export const WebSocketContext = createContext<WebSocketContextType | undefined>(undefined);

export const WebSocketProvider = ({ children }: { children: ReactNode }) => {
    const [connectionStatus, setConnectionStatus] = useState<'connecting' | 'connected' | 'disconnected' | 'server_unavailable'>('disconnected');
    const ws = useRef<WebSocket | null>(null);
    const effectRan = useRef(false);
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
    const [analysisStatus, setAnalysisStatus] = useState<'idle' | 'running' | 'completed' | 'cancelled' | 'error'>('idle');
    const [analysisType, setAnalysisType] = useState<'universe_atlas' | 'cell_chemistry' | null>(null);
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

        socket.onmessage = (event) => {
            try {
                const message = JSON.parse(event.data);
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
                        break;
                    case 'inference_status_update':
                        setInferenceStatus(payload.status);
                        break;
                    case 'simulation_frame':
                        // Descomprimir datos si están comprimidos
                        // IMPORTANTE: Preservar step, timestamp y simulation_info
                        // Usar función de actualización para evitar condiciones de carrera
                        try {
                        const decompressedPayload = {
                            ...payload,
                                step: payload.step ?? payload.simulation_info?.step ?? null, // Asegurar que step esté presente
                                timestamp: payload.timestamp ?? Date.now(),
                                simulation_info: payload.simulation_info,
                            map_data: payload.map_data ? decompressIfNeeded(payload.map_data) : undefined,
                            complex_3d_data: payload.complex_3d_data ? {
                                real: decompressIfNeeded(payload.complex_3d_data.real),
                                imag: decompressIfNeeded(payload.complex_3d_data.imag)
                            } : undefined,
                            flow_data: payload.flow_data ? {
                                dx: decompressIfNeeded(payload.flow_data.dx),
                                dy: decompressIfNeeded(payload.flow_data.dy),
                                magnitude: decompressIfNeeded(payload.flow_data.magnitude)
                            } : undefined,
                            phase_hsv_data: payload.phase_hsv_data ? {
                                hue: decompressIfNeeded(payload.phase_hsv_data.hue),
                                saturation: decompressIfNeeded(payload.phase_hsv_data.saturation),
                                value: decompressIfNeeded(payload.phase_hsv_data.value)
                            } : undefined
                        };
                            // Usar función de actualización para evitar sobrescribir actualizaciones más recientes
                            setSimData(prev => {
                                // Si hay un timestamp y el payload nuevo es más antiguo, ignorarlo
                                if (prev?.timestamp && decompressedPayload.timestamp && 
                                    decompressedPayload.timestamp < prev.timestamp) {
                                    return prev;
                                }
                                return decompressedPayload;
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
                                
                                return {
                                    ...prev,
                                    step: newStep,
                                    timestamp: newTimestamp,
                                    simulation_info: payload.simulation_info ?? prev?.simulation_info
                                    // No actualizar map_data, hist_data, etc. - estos solo vienen con simulation_frame
                                };
                            });
                        } catch (error) {
                            console.error("Error procesando simulation_state_update:", error);
                        }
                        break;
                    case 'training_log':
                        setTrainingLog(prev => [...prev, payload]);
                        setAllLogs(prev => [...prev, payload]);
                        break;
                    case 'simulation_log':
                        setAllLogs(prev => [...prev, payload]);
                        break;
                    case 'training_progress':
                        setTrainingProgress(payload);
                        break;
                    case 'notification':
                        // Mostrar notificación del servidor
                        const { status, message } = payload;
                        notifications.show({
                            title: status === 'error' ? 'Error' : status === 'warning' ? 'Advertencia' : status === 'success' ? 'Éxito' : 'Información',
                            message: message,
                            color: status === 'error' ? 'red' : status === 'warning' ? 'orange' : status === 'success' ? 'green' : 'blue',
                            autoClose: status === 'error' ? 5000 : 3000,
                        });
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
                        // Cuando se desactiva el live feed, limpiar datos de visualización
                        // para evitar mostrar datos antiguos
                        if (!payload.enabled) {
                            setSimData(prev => {
                                if (!prev) return prev;
                                // Mantener solo step, timestamp y simulation_info
                                // Limpiar todos los datos de visualización
                                return {
                                    step: prev.step,
                                    timestamp: prev.timestamp,
                                    simulation_info: prev.simulation_info,
                                    // Limpiar datos de visualización
                                    map_data: undefined,
                                    hist_data: undefined,
                                    poincare_coords: undefined,
                                    phase_attractor: undefined,
                                    flow_data: undefined,
                                    phase_hsv_data: undefined,
                                    complex_3d_data: undefined
                                };
                            });
                        }
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
        simData,
        trainingLog,
        allLogs,
        trainingProgress,
        activeExperiment,
        setActiveExperiment,
        ws: ws.current, // Exponer WebSocket para mensajes personalizados
        snapshotCount,
        serverConfig,
        updateServerConfig,
    };

    return (
        <WebSocketContext.Provider value={value}>
            {children}
        </WebSocketContext.Provider>
    );
};
