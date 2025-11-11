import { useState, useEffect, useRef, useCallback } from 'react';
import { notifications } from '@mantine/notifications';

// The useWebSocket hook content will go here
const useWebSocket = () => {
  const [status, setStatus] = useState('Conectando...');
  const [stepCount, setStepCount] = useState(0);
  const [simConfig, setSimConfig] = useState<any>({});
  const [metrics, setMetrics] = useState<any>({});
  const [trainingLog, setTrainingLog] = useState('');
  const [availableCheckpoints, setAvailableCheckpoints] = useState<Record<string, string[]>>({});
  const [imageData, setImageData] = useState<string | null>(null);
  const [trainingMetrics, setTrainingMetrics] = useState<any>({});
  const ws = useRef<WebSocket | null>(null);

  const sendCommand = useCallback((cmd: any) => {
    if (ws.current && ws.current.readyState === WebSocket.OPEN) {
      ws.current.send(JSON.stringify(cmd));
    } else {
      console.warn("No se pudo enviar el comando, WebSocket no está abierto.");
      notifications.show({
        title: 'Error de Conexión',
        message: 'WebSocket no está abierto. No se pudo enviar el comando.',
        color: 'red',
      });
    }
  }, []);

  useEffect(() => {
    const proto = window.location.protocol === "https:" ? "wss:" : "ws:";
    const WEBSOCKET_URL = `${proto}//${window.location.hostname}:8000/ws`;

    const connect = () => {
      console.log("Intentando conectar a:", WEBSOCKET_URL);
      setStatus(`Conectando a ${WEBSOCKET_URL}...`);
      ws.current = new WebSocket(WEBSOCKET_URL);

      ws.current.onopen = () => {
        console.log("Conectado al servidor AETHERIA.");
        setStatus("Conectado");
        notifications.show({
          title: 'Conectado',
          message: 'Conexión establecida con el servidor AETHERIA.',
          color: 'green',
        });
        sendCommand({ scope: 'lab', command: 'refresh_checkpoints' });
        sendCommand({ scope: 'sim', command: 'set_viewport', args: { viewport: { x: 0, y: 0, width: 1, height: 1 } } });
      };

      ws.current.onclose = () => {
        console.log("Desconectado del servidor. Reintentando en 3s...");
        setStatus("Desconectado. Reintentando...");
        setTimeout(connect, 3000);
      };

      ws.current.onerror = (err) => {
        console.error("Error de WebSocket:", err);
        setStatus("Error de conexión.");
      };

      ws.current.onmessage = (event) => {
        try {
          const data = JSON.parse(event.data);
          const { type, payload } = data;

          switch (type) {
            case 'status_update':
              setStatus(payload.status);
              setStepCount(payload.step_count);
              setSimConfig(payload.config || {});
              break;
            case 'metrics_update':
              setMetrics(payload);
              break;
            case 'image_update':
              setImageData(`data:image/jpeg;base64,${payload}`);
              break;
            case 'training_log':
              setTrainingLog(prev => prev + payload);
              break;
            case 'training_metrics':
              setTrainingMetrics(payload);
              break;
            case 'checkpoints_update':
              setAvailableCheckpoints(payload.checkpoints || {});
              break;
            case 'notification':
              notifications.show({
                title: payload.title || 'Notificación del Servidor',
                message: payload.message,
                color: payload.color || 'blue',
              });
              break;
            default:
              console.warn('Unknown message type:', type);
          }
        } catch (e) {
          console.error("Error procesando mensaje:", e, event.data);
        }
      };
    };

    connect();

    return () => {
      ws.current?.close();
    };
  }, [sendCommand]);

  return { status, stepCount, simConfig, metrics, trainingLog, setTrainingLog, availableCheckpoints, imageData, trainingMetrics, sendCommand };
};

export default useWebSocket;
