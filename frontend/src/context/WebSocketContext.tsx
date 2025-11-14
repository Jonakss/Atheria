// frontend/src/context/WebSocketContext.tsx
import React, { createContext, useContext, useState, useCallback, useRef } from 'react';
import { notifications } from '@mantine/notifications';

// --- Tipos ---
interface Experiment { name: string; config: any; }
interface SimData { step: number; viz_type: string; map_data: any; }
type ConnectionStatus = 'connecting' | 'connected' | 'disconnected' | 'error';
type TrainingStatus = 'idle' | 'running' | 'finished' | 'error';
type InferenceStatus = 'running' | 'paused'; // ¡¡NUEVO!!

interface WebSocketContextType {
    connectionStatus: ConnectionStatus;
    trainingStatus: TrainingStatus;
    inferenceStatus: InferenceStatus; // ¡¡NUEVO!!
    experimentsData: Experiment[] | null;
    simData: SimData | null;
    trainingLog: string[];
    sendCommand: (scope: string, command: string, args?: any) => void;
    connect: () => void;
}

const WebSocketContext = createContext<WebSocketContextType | undefined>(undefined);

export const WebSocketProvider: React.FC<{ children: React.ReactNode }> = ({ children }) => {
    const [connectionStatus, setConnectionStatus] = useState<ConnectionStatus>('disconnected');
    const [trainingStatus, setTrainingStatus] = useState<TrainingStatus>('idle');
    const [inferenceStatus, setInferenceStatus] = useState<InferenceStatus>('paused'); // ¡¡NUEVO!!
    const [experimentsData, setExperimentsData] = useState<Experiment[] | null>(null);
    const [simData, setSimData] = useState<SimData | null>(null);
    const [trainingLog, setTrainingLog] = useState<string[]>([]);
    const socketRef = useRef<WebSocket | null>(null);

    const connect = useCallback(() => {
        if (socketRef.current && socketRef.current.readyState === WebSocket.OPEN) return;
        setConnectionStatus('connecting');
        const ws = new WebSocket(`ws://${window.location.hostname}:8000/ws`);
        ws.onopen = () => setConnectionStatus('connected');
        ws.onmessage = (event) => {
            const data = JSON.parse(event.data);
            switch (data.type) {
                case 'initial_state':
                    setExperimentsData(data.payload.experiments || []);
                    setTrainingStatus(data.payload.training_status || 'idle');
                    setInferenceStatus(data.payload.inference_status || 'paused');
                    break;
                case 'notification':
                    notifications.show({ title: 'Notificación', message: data.payload.message, color: data.payload.status === 'error' ? 'red' : 'blue' });
                    break;
                case 'training_log':
                    setTrainingLog(prev => [...prev, data.payload]);
                    break;
                case 'simulation_frame':
                    setSimData(data.payload);
                    break;
                // --- ¡¡NUEVO!! Actualizar estado de inferencia ---
                case 'inference_status_update':
                    setInferenceStatus(data.payload.status);
                    break;
            }
        };
        ws.onclose = () => setConnectionStatus('disconnected');
        ws.onerror = () => setConnectionStatus('error');
        socketRef.current = ws;
    }, []);

    const sendCommand = useCallback((scope: string, command: string, args: any = {}) => {
        if (socketRef.current?.readyState === WebSocket.OPEN) {
            socketRef.current.send(JSON.stringify({ scope, command, args }));
        } else {
            notifications.show({ title: 'Error', message: 'No hay conexión.', color: 'red' });
        }
    }, []);

    return (
        <WebSocketContext.Provider value={{
            connectionStatus, trainingStatus, inferenceStatus, experimentsData, simData, trainingLog,
            sendCommand, connect
        }}>
            {children}
        </WebSocketContext.Provider>
    );
};

export const useWebSocket = () => {
    const context = useContext(WebSocketContext);
    if (!context) throw new Error('useWebSocket debe ser usado dentro de un WebSocketProvider');
    return context;
};
