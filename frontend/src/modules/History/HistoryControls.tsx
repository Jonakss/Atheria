import { BackwardIcon, ForwardIcon, PauseIcon, PlayIcon } from '@heroicons/react/24/solid';
import React, { useCallback, useEffect, useState } from 'react';
import { useWebSocket } from '../../contexts/WebSocketContext';

interface HistoryRange {
  available: boolean;
  min_step: number | null;
  max_step: number | null;
  total_frames: number;
  current_step: number;
}

export const HistoryControls: React.FC = () => {
  const { sendMessage, lastMessage } = useWebSocket();
  const [historyRange, setHistoryRange] = useState<HistoryRange>({
    available: false,
    min_step: null,
    max_step: null,
    total_frames: 0,
    current_step: 0,
  });
  const [selectedStep, setSelectedStep] = useState<number>(0);
  const [isPlaying, setIsPlaying] = useState(false);

  // Solicitar rango de historia al iniciar
  useEffect(() => {
    requestHistoryRange();
    const interval = setInterval(requestHistoryRange, 5000); // Actualizar cada 5s
    return () => clearInterval(interval);
  }, []);

  // Escuchar mensajes del WebSocket
  useEffect(() => {
    if (!lastMessage) return;

    try {
      const message = typeof lastMessage === 'string' ? JSON.parse(lastMessage) : lastMessage;

      if (message.type === 'history_range') {
        const range = message.payload;
        setHistoryRange(range);
        if (range.current_step !== undefined) {
          setSelectedStep(range.current_step);
        }
      } else if (message.type === 'inference_status_update') {
        const payload = message.payload;
        setIsPlaying(payload.status === 'running');
        if (payload.step !== undefined) {
          setSelectedStep(payload.step);
        }
      }
    } catch (error) {
      console.error('Error parsing WebSocket message:', error);
    }
  }, [lastMessage]);

  const requestHistoryRange = useCallback(() => {
    sendMessage({
      scope: 'history',
      command: 'get_history_range',
      args: {},
    });
  }, [sendMessage]);

  const handleRestoreStep = useCallback((step: number) => {
    sendMessage({
      scope: 'history',
      command: 'restore_history_step',
      args: { step },
    });
    setSelectedStep(step);
  }, [sendMessage]);

  const handleSliderChange = (event: React.ChangeEvent<HTMLInputElement>) => {
    const step = parseInt(event.target.value, 10);
    setSelectedStep(step);
  };

  const handleSliderRelease = () => {
    handleRestoreStep(selectedStep);
  };

  const handleStepBackward = () => {
    if (!historyRange.available || historyRange.min_step === null) return;
    const newStep = Math.max(historyRange.min_step, selectedStep - 10);
    handleRestoreStep(newStep);
  };

  const handleStepForward = () => {
    if (!historyRange.available || historyRange.max_step === null) return;
    const newStep = Math.min(historyRange.max_step, selectedStep + 10);
    handleRestoreStep(newStep);
  };

  const handlePlayPause = () => {
    if (isPlaying) {
      sendMessage({ scope: 'inference', command: 'pause', args: {} });
    } else {
      sendMessage({ scope: 'inference', command: 'play', args: {} });
    }
  };

  if (!historyRange.available) {
    return (
      <div className="flex items-center justify-center p-4 bg-gray-800/50 rounded-lg border border-gray-700">
        <p className="text-gray-400 text-sm">
          No hay historial disponible. Ejecuta la simulación para generar historial.
        </p>
      </div>
    );
  }

  const { min_step, max_step, total_frames, current_step } = historyRange;

  return (
    <div className="flex flex-col gap-3 p-4 bg-gray-800/50 rounded-lg border border-gray-700">
      {/* Header */}
      <div className="flex items-center justify-between">
        <h3 className="text-sm font-semibold text-gray-200">Timeline de Simulación</h3>
        <div className="text-xs text-gray-400">
          {total_frames} frames ({min_step} - {max_step})
        </div>
      </div>

      {/* Controls */}
      <div className="flex items-center gap-3">
        {/* Step Backward */}
        <button
          onClick={handleStepBackward}
          disabled={selectedStep <= (min_step ?? 0)}
          className="p-2 bg-gray-700 hover:bg-gray-600 disabled:bg-gray-800 disabled:text-gray-600 rounded-lg transition-colors"
          title="Retroceder 10 steps"
        >
          <BackwardIcon className="w-4 h-4" />
        </button>

        {/* Play/Pause */}
        <button
          onClick={handlePlayPause}
          className="p-2 bg-blue-600 hover:bg-blue-500 rounded-lg transition-colors"
          title={isPlaying ? 'Pausar' : 'Reproducir'}
        >
          {isPlaying ? (
            <PauseIcon className="w-4 h-4" />
          ) : (
            <PlayIcon className="w-4 h-4" />
          )}
        </button>

        {/* Step Forward */}
        <button
          onClick={handleStepForward}
          disabled={selectedStep >= (max_step ?? 0)}
          className="p-2 bg-gray-700 hover:bg-gray-600 disabled:bg-gray-800 disabled:text-gray-600 rounded-lg transition-colors"
          title="Avanzar 10 steps"
        >
          <ForwardIcon className="w-4 h-4" />
        </button>

        {/* Slider */}
        <div className="flex-1 flex items-center gap-2">
          <span className="text-xs text-gray-400 min-w-[60px]">
            Step: {selectedStep}
          </span>
          <input
            type="range"
            min={min_step ?? 0}
            max={max_step ?? 0}
            value={selectedStep}
            onChange={handleSliderChange}
            onMouseUp={handleSliderRelease}
            onTouchEnd={handleSliderRelease}
            className="flex-1 h-2 bg-gray-700 rounded-lg appearance-none cursor-pointer slider-thumb"
            style={{
              backgroundImage: `linear-gradient(to right, #3b82f6 0%, #3b82f6 ${
                ((selectedStep - (min_step ?? 0)) / ((max_step ?? 1) - (min_step ?? 0))) * 100
              }%, #4b5563 ${
                ((selectedStep - (min_step ?? 0)) / ((max_step ?? 1) - (min_step ?? 0))) * 100
              }%, #4b5563 100%)`,
            }}
          />
        </div>
      </div>

      {/* Current Step Indicator */}
      {current_step !== selectedStep && (
        <div className="text-xs text-amber-400">
          ⚠️ Visualizando step {selectedStep} (actual: {current_step})
        </div>
      )}
    </div>
  );
};
