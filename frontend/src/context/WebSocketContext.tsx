// frontend/src/context/WebSocketContext.tsx
import { createContext, useState, useEffect, useRef, useCallback, ReactNode } from 'react';
import { notifications } from '@mantine/notifications';

interface SimData {
    complex_3d_data?: {
        real: number[][];
        imag: number[][];
    };
    map_data?: number[][];
    hist_data?: Record<string, Array<{ bin: string; count: number }>>;
    poincare_coords?: number[][];
    step?: number;
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
    selectedViz: string;
    setSelectedViz: (viz: string) => void;
    connect: () => void;
    simData: SimData | null;
    trainingLog: string[];
    allLogs: string[]; // Logs unificados
    trainingProgress: TrainingProgress | null;
    activeExperiment: string | null;
    setActiveExperiment: (name: string | null) => void;
    ws: WebSocket | null; // Exponer WebSocket para escuchar mensajes personalizados
    snapshotCount: number; // Contador de snapshots capturados
}

export const WebSocketContext = createContext<WebSocketContextType | undefined>(undefined);

export const WebSocketProvider = ({ children }: { children: ReactNode }) => {
    const [connectionStatus, setConnectionStatus] = useState<'connecting' | 'connected' | 'disconnected' | 'server_unavailable'>('disconnected');
    const ws = useRef<WebSocket | null>(null);
    const effectRan = useRef(false);
    const reconnectAttempts = useRef(0);
    const lastErrorLog = useRef(0);

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
    // Usar una referencia para acceder al valor actual de activeExperiment en callbacks
    const activeExperimentRef = useRef<string | null>(null);
    
    // Mantener la referencia actualizada
    useEffect(() => {
        activeExperimentRef.current = activeExperiment;
    }, [activeExperiment]);

    const connect = useCallback(() => {
        if (ws.current?.readyState === WebSocket.OPEN) return;
        
        // Limpiar conexión anterior si existe
        if (ws.current) {
            ws.current.onerror = null;
            ws.current.onclose = null;
            ws.current.close();
        }
        
        setConnectionStatus('connecting');
        const socket = new WebSocket(`ws://localhost:8000/ws`);

        socket.onopen = () => {
            reconnectAttempts.current = 0;
            setConnectionStatus('connected');
            // Solo loguear en desarrollo
            if (process.env.NODE_ENV === 'development') {
                console.log("✓ WebSocket conectado");
            }
        };
        
        socket.onclose = (event) => {
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
                
                // Después de varios intentos, marcar como servidor no disponible
                if (reconnectAttempts.current > 3) {
                    setConnectionStatus('server_unavailable');
                } else {
                    setConnectionStatus('disconnected');
                }
                
                // Reintentar conexión
                setTimeout(connect, 3000);
            } else {
                setConnectionStatus('disconnected');
            }
        };
        
        socket.onerror = (error) => {
            // Solo loguear errores ocasionalmente para no saturar la consola
            const now = Date.now();
            if (now - lastErrorLog.current > 10000) {
                // No loguear el objeto Event completo, solo un mensaje simple
                lastErrorLog.current = now;
            }
        };

        socket.onmessage = (event) => {
            try {
                const message = JSON.parse(event.data);
                // Solo loguear en desarrollo y no para cada frame
                if (process.env.NODE_ENV === 'development' && message.type !== 'simulation_frame') {
                    console.log("Mensaje recibido:", message.type);
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
                        setSimData(payload);
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
    }, []);

    useEffect(() => {
        if (effectRan.current === false) {
            connect();
            return () => {
                effectRan.current = true;
                ws.current?.close();
            };
        }
    }, [connect]);
    
    const sendCommand = useCallback((scope: string, command: string, args: Record<string, any> = {}) => {
        if (ws.current?.readyState === WebSocket.OPEN) {
            ws.current.send(JSON.stringify({ scope, command, args }));
        } else {
            // Solo loguear si no es un estado esperado (servidor no disponible)
            if (connectionStatus !== 'server_unavailable' && connectionStatus !== 'connecting') {
                console.warn("No se puede enviar comando: WebSocket no conectado");
            }
        }
    }, [connectionStatus]);

    const value = {
        sendCommand,
        connectionStatus,
        experimentsData,
        trainingStatus,
        inferenceStatus,
        selectedViz,
        setSelectedViz,
        connect,
        simData,
        trainingLog,
        allLogs,
        trainingProgress,
        activeExperiment,
        setActiveExperiment,
        ws: ws.current, // Exponer WebSocket para mensajes personalizados
        snapshotCount,
    };

    return (
        <WebSocketContext.Provider value={value}>
            {children}
        </WebSocketContext.Provider>
    );
};
