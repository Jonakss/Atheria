// frontend/src/context/WebSocketContext.tsx
import React, { createContext, useState, useCallback, useRef, ReactNode } from 'react';

// --- TYPE DEFINITIONS ---
interface ExperimentConfig {
    MODEL_ARCHITECTURE: string;
    TOTAL_EPISODES?: number;
}
interface ExperimentData {
    name: string;
    config: ExperimentConfig;
}
interface TrainingProgress {
    current_episode: number;
    total_episodes: number;
    avg_loss: number;
}
interface SimData {
    map_data: number[][];
    hist_data: { [key: string]: { bin: string; count: number }[] };
    poincare_coords: number[][];
    viz_type: string;
}

// --- CONTEXT TYPE ---
interface WebSocketContextType {
    connect: () => void;
    sendCommand: (scope: string, cmd: string, payload?: any) => void;
    connectionStatus: string;
    experimentsData: ExperimentData[] | null;
    trainingStatus: string;
    trainingLog: string[];
    trainingProgress: TrainingProgress | null;
    simData: SimData | null;
    inferenceStatus: string;
    selectedViz: string;
    setSelectedViz: (viz: string) => void;
}

export const WebSocketContext = createContext<WebSocketContextType | undefined>(undefined);

// --- PROVIDER COMPONENT ---
export const WebSocketProvider = ({ children }: { children: ReactNode }) => {
    const [connectionStatus, setConnectionStatus] = useState('disconnected');
    const [experimentsData, setExperimentsData] = useState<ExperimentData[] | null>(null);
    const [trainingStatus, setTrainingStatus] = useState('idle');
    const [trainingLog, setTrainingLog] = useState<string[]>([]);
    const [trainingProgress, setTrainingProgress] = useState<TrainingProgress | null>(null);
    const [simData, setSimData] = useState<SimData | null>(null);
    const [inferenceStatus, setInferenceStatus] = useState('paused');
    const [selectedViz, setSelectedViz] = useState('density');
    
    const ws = useRef<WebSocket | null>(null);

    const connect = useCallback(() => {
        if (ws.current && ws.current.readyState === WebSocket.OPEN) return;

        setConnectionStatus('connecting');
        const wsUrl = `ws://${window.location.hostname}:8000/ws`;
        ws.current = new WebSocket(wsUrl);

        ws.current.onopen = () => {
            setConnectionStatus('connected');
            sendCommand('initial_data', 'get_experiments');
        };

        ws.current.onmessage = (event) => {
            const data = JSON.parse(event.data);
            switch (data.type) {
                case 'initial_experiments':
                    setExperimentsData(data.payload);
                    break;
                case 'training_log':
                    setTrainingLog(prev => [...prev, data.payload]);
                    break;
                case 'training_progress':
                    setTrainingProgress(data.payload);
                    break;
                case 'training_status_update':
                    setTrainingStatus(data.payload.status);
                    if (data.payload.status === 'idle') {
                        setTrainingProgress(null);
                        setTrainingLog([]);
                    }
                    break;
                case 'simulation_update':
                    setSimData(data.payload);
                    break;
                case 'inference_status_update':
                    setInferenceStatus(data.payload.status);
                    break;
                case 'notification':
                    // Aquí se podría manejar la notificación, por ejemplo con una librería de toasts
                    console.log(`Notification: ${data.payload.message}`);
                    break;
            }
        };

        ws.current.onclose = () => {
            setConnectionStatus('disconnected');
        };

        ws.current.onerror = (error) => {
            console.error('WebSocket Error:', error);
            setConnectionStatus('disconnected');
        };
    }, []);

    const sendCommand = useCallback((scope: string, cmd: string, payload: any = {}) => {
        if (ws.current && ws.current.readyState === WebSocket.OPEN) {
            ws.current.send(JSON.stringify({ scope, cmd, payload }));
        }
    }, []);

    const handleSetSelectedViz = (viz: string) => {
        setSelectedViz(viz);
        sendCommand('simulation', 'set_viz', { viz_type: viz });
    };

    return (
        <WebSocketContext.Provider value={{
            connect, sendCommand, connectionStatus, experimentsData,
            trainingStatus, trainingLog, trainingProgress, simData, inferenceStatus,
            selectedViz, setSelectedViz: handleSetSelectedViz
        }}>
            {children}
        </WebSocketContext.Provider>
    );
};