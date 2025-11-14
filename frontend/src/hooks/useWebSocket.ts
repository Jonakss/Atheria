// frontend/src/hooks/useWebSocket.ts
import { useContext } from 'react';
import { WebSocketContext } from '../context/WebSocketContext';

export const useWebSocket = () => {
    const context = useContext(WebSocketContext);
    if (context === undefined) {
        throw new Error('useWebSocket must be used within a WebSocketProvider');
    }
    return context;
};