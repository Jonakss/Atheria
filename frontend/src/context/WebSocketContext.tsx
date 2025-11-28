// frontend/src/context/WebSocketContext.tsx
import { ReactNode, useCallback, useEffect, useRef, useState } from 'react';
import { decodeBinaryFrame, decompressIfNeeded, processDecodedPayload } from '../utils/dataDecompression';
import { getServerConfig, getWebSocketUrl, saveServerConfig, type ServerConfig } from '../utils/serverConfig';
import { saveFrameToTimeline } from '../utils/timelineStorage';
import {
    CompileStatus,
    InferenceSnapshot,
    SimData,
    TrainingProgress,
    TrainingSnapshot,
    WebSocketContext
} from './WebSocketContextDefinition';

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
        if (process.env.NODE_ENV === 'development') {
            const isComp = result.map_data.compressed === true;
            console.log(`🔍 WebSocketContext: map_data recibido. Compressed: ${isComp}, Type: ${typeof result.map_data}, Keys: ${Object.keys(result.map_data)}`);
        }
        result.map_data = decompressIfNeeded(result.map_data);
        
        if (process.env.NODE_ENV === 'development') {
            const isArray = Array.isArray(result.map_data);
            const length = isArray ? result.map_data.length : 'N/A';
            console.log(`✅ WebSocketContext: map_data procesado. IsArray: ${isArray}, Length: ${length}`);
        }
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

export const WebSocketProvider = ({ children }: { children: ReactNode }) => {
    const [connectionStatus, setConnectionStatus] = useState<'connecting' | 'connected' | 'disconnected' | 'server_unavailable'>('disconnected');
    const ws = useRef<WebSocket | null>(null);
    const effectRan = useRef(false);
    const [liveFeedEnabled, setLiveFeedEnabledState] = useState<boolean>(true); // Por defecto habilitado
    const reconnectAttempts = useRef(0);
    const lastErrorLog = useRef(0);
    const isManualClose = useRef(false); // Flag para indicar si el cierre fue manual
    const pendingBinaryFormat = useRef<string | undefined>(undefined); // Formato esperado para datos binarios (msgpack/cbor/json)
    const [serverConfig, setServerConfigState] = useState<ServerConfig>(getServerConfig());
    // Usar una referencia para acceder al valor actual de serverConfig en callbacks
    const serverConfigRef = useRef<ServerConfig>(serverConfig);
    const simDataRef = useRef<SimData | null>(null);
    const connectionStatusRef = useRef(connectionStatus);

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
    const [trainingCheckpoints, setTrainingCheckpoints] = useState<any[]>([]);
    const [inferenceSnapshots, setInferenceSnapshots] = useState<InferenceSnapshot[]>([]);
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

    useEffect(() => {
        simDataRef.current = simData;
    }, [simData]);

    useEffect(() => {
        connectionStatusRef.current = connectionStatus;
    }, [connectionStatus]);

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
        
        socket.onerror = () => {
            // Solo loguear errores persistentes, no durante la conexión inicial
            const now = Date.now();
            if (now - lastErrorLog.current > 10000 && connectionStatusRef.current !== 'connecting') {
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
                        // Leer metadata JSON primero (si existe como mensaje separado anterior)
                        // En el nuevo formato, primero recibimos metadata JSON, luego datos binarios
                        // Para este caso (datos binarios directos), decodificar directamente
                        const format = pendingBinaryFormat.current; // Formato esperado desde metadata
                        const frameData = await decodeBinaryFrame(new Uint8Array(arrayBuffer), format);
                        pendingBinaryFormat.current = undefined; // Limpiar formato pendiente
                        
                        // Si tiene estructura de frame optimizado (metadata + arrays)
                        if (frameData.metadata && frameData.arrays) {
                            // Procesar arrays binarios y reconstruir payload
                            // Capturar simData actual para differential compression
                            const currentSimData = simDataRef.current;
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
                    // Mensaje JSON (metadata del servidor o metadata de frame binario)
                    const jsonMessage = JSON.parse(event.data);
                    
                    // Verificar si es metadata para un frame binario (formato híbrido: JSON + binario)
                    if (jsonMessage.type && jsonMessage.type.endsWith('_binary') && jsonMessage.format && jsonMessage.size !== undefined) {
                        // Este es metadata JSON que precede a un frame binario
                        // Almacenar formato esperado y esperar el siguiente mensaje binario
                        pendingBinaryFormat.current = jsonMessage.format; // "msgpack", "cbor", o "json"
                        // No procesar este mensaje como un mensaje completo, esperar el binario
                        return; // Salir y esperar el siguiente mensaje binario
                    }
                    
                    // Mensaje JSON normal (comandos, notificaciones, metadatos del servidor)
                    message = jsonMessage;
                    
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
                        // Si hay compile_status en el estado inicial, establecerlo
                        if (payload.compile_status) {
                            setCompileStatus(payload.compile_status as CompileStatus);
                        }
                        // Restaurar o limpiar experimento activo
                        setActiveExperiment(payload.active_experiment || null);
                        break;
                    case 'experiments_updated': {
                        // Actualizar lista de experimentos cuando se crea/elimina un experimento
                        setExperimentsData(payload.experiments);
                        // Si el experimento activo fue eliminado, limpiar selección
                        const currentActive = activeExperimentRef.current;
                        if (currentActive && !payload.experiments?.find((e: any) => e.name === currentActive)) {
                            setActiveExperiment(null);
                        }
                        break;
                    }
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
                    case 'training_checkpoint_event':
                        // Agregar nuevo checkpoint
                        setTrainingCheckpoints(prev => {
                            // Evitar duplicados por episodio
                            if (prev.some(cp => cp.episode === payload.episode)) return prev;
                            const newCheckpoint = payload;
                            return [newCheckpoint, ...prev].slice(0, 50); // Mantener últimos 50, más recientes primero
                        });
                        break;
                    case 'inference_status_update':
                        setInferenceStatus(payload.status);
                        
                        // CRÍTICO: Actualizar simulation_info si está disponible (FPS, step, etc.)
                        if (payload.simulation_info) {
                            setSimData(prev => ({
                                ...(prev || {}),
                                step: payload.step ?? payload.simulation_info?.step ?? prev?.step ?? null,
                                timestamp: payload.timestamp ?? Date.now(),
                                simulation_info: {
                                    ...(prev?.simulation_info || {}),
                                    ...payload.simulation_info,
                                    is_paused: payload.status === 'paused'
                                }
                            }));
                        }
                        
                        // Almacenar compile_status si está disponible
                        if (payload.compile_status) {
                            const newCompileStatus = payload.compile_status as CompileStatus;
                            if (process.env.NODE_ENV === 'development') {
                                console.log('📥 WebSocketContext - Recibido compile_status:', JSON.stringify(newCompileStatus, null, 2));
                            }
                            setCompileStatus(newCompileStatus);
                        } else {
                            if (process.env.NODE_ENV === 'development') {
                                console.warn('⚠️ WebSocketContext - inference_status_update sin compile_status. Payload completo:', payload);
                            }
                        }
                        
                        // Actualizar experimento activo si viene en el payload
                        if (payload.active_experiment) {
                            setActiveExperiment(payload.active_experiment);
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
                            
                            // DEBUG: Log información del frame recibido
                            if (process.env.NODE_ENV === 'development') {
                                const mapDataShape = finalPayload.map_data 
                                    ? [finalPayload.map_data.length, finalPayload.map_data[0]?.length || 0]
                                    : [0, 0];
                                const mapDataSample = finalPayload.map_data && finalPayload.map_data.length > 0 && finalPayload.map_data[0]?.length > 0
                                    ? finalPayload.map_data[0][0]
                                    : null;
                                
                                console.log(`📥 WebSocketContext: simulation_frame recibido - step: ${finalPayload.step}, map_data shape: [${mapDataShape[0]}, ${mapDataShape[1]}], sample value: ${mapDataSample}`);
                            }
                            
                            // Guardar frame en timeline del navegador (localStorage)
                            // Solo guardar si hay map_data y step válido
                            if (finalPayload.map_data && finalPayload.step !== null && finalPayload.step !== undefined) {
                                try {
                                    // Obtener límite de frames desde localStorage o usar valor por defecto
                                    const maxFrames = parseInt(
                                        localStorage.getItem('atheria_timeline_max_frames') || '100',
                                        10
                                    );
                                    
                                    saveFrameToTimeline(
                                        {
                                            step: finalPayload.step,
                                            timestamp: finalPayload.timestamp || Date.now(),
                                            map_data: finalPayload.map_data,
                                            simulation_info: finalPayload.simulation_info,
                                        },
                                        activeExperimentRef.current,
                                        maxFrames
                                    );
                                } catch (timelineError) {
                                    // Silenciar errores de timeline (puede ser por cuota excedida)
                                    if (process.env.NODE_ENV === 'development') {
                                        console.warn('Error guardando frame en timeline:', timelineError);
                                    }
                                }
                            }
                            
                            // Usar función de actualización para evitar sobrescribir actualizaciones más recientes
                            setSimData(prev => {
                                // Si hay un timestamp y el payload nuevo es más antiguo
                                if (prev?.timestamp && finalPayload.timestamp && 
                                    finalPayload.timestamp < prev.timestamp) {
                                    
                                    // EXCEPCIÓN CRÍTICA: Si el nuevo payload tiene map_data y el previo NO, aceptar los datos visuales
                                    // Esto ocurre porque los status_update son más rápidos que los frames pesados
                                    if (finalPayload.map_data && !prev.map_data) {
                                        if (process.env.NODE_ENV === 'development') {
                                            console.log(`🔄 WebSocketContext: Fusionando map_data tardío - frame step: ${finalPayload.step}, prev step: ${prev.step}`);
                                        }
                                        return {
                                            ...prev, // Mantener estado más reciente (timestamp, info)
                                            map_data: finalPayload.map_data, // Inyectar data visual
                                            // También inyectar otros datos visuales si existen
                                            complex_3d_data: finalPayload.complex_3d_data || prev.complex_3d_data,
                                            flow_data: finalPayload.flow_data || prev.flow_data,
                                            phase_hsv_data: finalPayload.phase_hsv_data || prev.phase_hsv_data
                                        };
                                    }
                                    
                                    return prev;
                                }
                                
                                // EXCEPCIÓN CRÍTICA: Si el nuevo payload tiene map_data y el previo NO, aceptar los datos visuales
                                // Esto ocurre porque los status_update son más rápidos que los frames pesados
                                if (finalPayload.map_data && !prev?.map_data) { // prev?.map_data para manejar el caso inicial donde prev es undefined
                                    if (process.env.NODE_ENV === 'development') {
                                        console.log(`🔄 WebSocketContext: Fusionando map_data tardío - frame step: ${finalPayload.step}, prev step: ${prev?.step}`);
                                    }
                                    return {
                                        ...prev, // Mantener estado más reciente (timestamp, info)
                                        map_data: finalPayload.map_data, // Inyectar data visual
                                        // También inyectar otros datos visuales si existen
                                        complex_3d_data: finalPayload.complex_3d_data || prev?.complex_3d_data,
                                        flow_data: finalPayload.flow_data || prev?.flow_data,
                                        phase_hsv_data: finalPayload.phase_hsv_data || prev?.phase_hsv_data
                                    };
                                }
                                
                                // DEBUG: Log cuando se actualiza simData
                                if (process.env.NODE_ENV === 'development') {
                                    console.log(`🔄 WebSocketContext: Actualizando simData - step: ${finalPayload.step}, map_data presente: ${!!finalPayload.map_data}`);
                                }
                                
                                return finalPayload;
                            });
                        } catch (error) {
                            console.error("Error procesando simulation_frame:", error);
                        }
                        break;
                    case 'simulation_state_update':
                        // Actualización de estado sin datos de visualización (cuando live feed está desactivado)
                        // Actualizar solo step y simulation_info, preservando otros datos existentes (incluyendo map_data)
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
                                
                                // CRÍTICO: Preservar TODOS los datos existentes (map_data, hist_data, etc.)
                                // Solo actualizar step, timestamp y simulation_info
                                return {
                                    ...(prev || {}), // Asegurar que prev existe, si no usar objeto vacío
                                    step: newStep,
                                    timestamp: newTimestamp,
                                    simulation_info: {
                                        ...(prev?.simulation_info || {}), // Preservar simulation_info anterior
                                        ...(payload.simulation_info || {}), // Actualizar con nuevo
                                        live_feed_enabled: newLiveFeedEnabled
                                    }
                                    // map_data, hist_data, etc. se preservan desde prev
                                };
                            });
                        } catch (error) {
                            console.error("Error procesando simulation_state_update:", error);
                        }
                        break;
                    case 'training_log': {
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
                    }
                    case 'simulation_log': {
                        // OPTIMIZACIÓN DE MEMORIA: Limitar tamaño de logs para evitar memory leaks
                        const MAX_LOGS = 500; // Reducido de 1000 a 500 para ahorrar memoria
                        setAllLogs(prev => {
                            // Extraer mensaje formateado si es un objeto, o usar payload si es string
                            const logMessage = typeof payload === 'object' && payload.formatted 
                                ? payload.formatted 
                                : (typeof payload === 'string' ? payload : JSON.stringify(payload));
                                
                            const newLogs = [...prev, logMessage];
                            return newLogs.slice(-MAX_LOGS); // Mantener solo últimos N logs
                        });
                        break;
                    }
                    case 'training_progress':
                        setTrainingProgress(payload);
                        break;
                    case 'notification': {
                        // Mostrar notificación del servidor
                        const { status, message } = payload;
                        const title = status === 'error' ? 'Error' : status === 'warning' ? 'Advertencia' : status === 'success' ? 'Éxito' : 'Información';
                        console.log(`[${title}] ${message}`);
                        // TODO: Implementar sistema de notificaciones con Tailwind
                        break;
                    }
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
                    case 'snapshot_list':
                        setInferenceSnapshots(payload.snapshots || []);
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

    // Función para cambiar el intervalo de pasos
    const setStepsInterval = useCallback((interval: number) => {
        sendCommand('simulation', 'set_steps_interval', { interval });
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
        trainingCheckpoints,
        inferenceSnapshots,
        serverConfig,
        updateServerConfig,
        compileStatus, // Estado de compilación/motor (indica si es nativo)
        liveFeedEnabled, // Estado del live feed (sincronizado con backend)
        setLiveFeedEnabled, // Función para cambiar live feed
        setStepsInterval, // Función para cambiar el intervalo de pasos
    };

    return (
        <WebSocketContext.Provider value={value}>
            {children}
        </WebSocketContext.Provider>
    );
};
