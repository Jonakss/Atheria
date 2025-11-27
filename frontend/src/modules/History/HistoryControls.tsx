import { BackwardIcon, ForwardIcon, PauseIcon, PlayIcon } from '@heroicons/react/24/solid';
import { Eye, EyeOff, RefreshCw, Save, Clock } from 'lucide-react';
import React, { useCallback, useEffect, useMemo, useState } from 'react';
import { useWebSocket } from '../../hooks/useWebSocket';
import { calculateParticleCount } from '../../utils/simulationUtils';

interface HistoryRange {
  available: boolean;
  min_step: number | null;
  max_step: number | null;
  total_frames: number;
  current_step: number;
}

// --- Sub-components ---

const SimulationControls: React.FC<{
  isPlaying: boolean;
  controlsEnabled: boolean;
  onPlayPause: () => void;
  onReset: () => void;
  onSaveSnapshot: () => void;
}> = ({ isPlaying, controlsEnabled, onPlayPause, onReset, onSaveSnapshot }) => (
  <div className="flex items-center gap-2">
    <button
      onClick={onPlayPause}
      disabled={!controlsEnabled}
      className={`px-3 py-1.5 rounded text-xs font-bold flex items-center gap-2 transition-all border ${
        !controlsEnabled
          ? 'bg-white/5 text-gray-600 border-white/5 cursor-not-allowed opacity-50'
          : isPlaying
            ? 'bg-pink-500/10 text-pink-500 border-pink-500/30 hover:bg-pink-500/20'
            : 'bg-teal-500/10 text-teal-400 border-teal-500/30 hover:bg-teal-500/20'
      }`}
    >
      {isPlaying ? <PauseIcon className="w-3 h-3" /> : <PlayIcon className="w-3 h-3" />}
      {isPlaying ? 'PAUSE' : 'RUN'}
    </button>

    <button
      onClick={onReset}
      disabled={!controlsEnabled}
      className="p-1.5 rounded text-gray-400 hover:text-white hover:bg-white/10 transition-colors"
      title="Reset Simulation"
    >
      <RefreshCw size={14} />
    </button>

    <button
      onClick={onSaveSnapshot}
      disabled={!controlsEnabled}
      className="p-1.5 rounded text-blue-400 hover:text-blue-300 hover:bg-blue-500/10 transition-colors"
      title="Save Snapshot"
    >
      <Save size={14} />
    </button>
  </div>
);

const LiveFeedControl: React.FC<{
  liveFeedEnabled: boolean;
  controlsEnabled: boolean;
  onSetInterval: (interval: number) => void;
  onToggleLiveFeed: () => void;
}> = ({ liveFeedEnabled, controlsEnabled, onSetInterval, onToggleLiveFeed }) => {
  const [showOptions, setShowOptions] = useState(false);
  const [customInterval, setCustomInterval] = useState(10);

  return (
    <div className="relative">
      <div className="flex rounded-md shadow-sm">
        <button
          onClick={onToggleLiveFeed}
          disabled={!controlsEnabled}
          className={`flex items-center gap-1.5 px-2 py-0.5 rounded-l text-[10px] font-bold border transition-all ${
            liveFeedEnabled
              ? 'bg-blue-500/10 text-blue-400 border-blue-500/30'
              : 'bg-gray-800 text-gray-500 border-gray-700'
          }`}
        >
          {liveFeedEnabled ? <Eye size={10} /> : <EyeOff size={10} />}
          <span>{liveFeedEnabled ? 'LIVE' : 'OFF'}</span>
        </button>
        <button
            onClick={() => setShowOptions(!showOptions)}
            disabled={!controlsEnabled}
            className={`px-1 py-0.5 rounded-r border-l-0 text-[10px] font-bold border transition-all ${
                liveFeedEnabled
                ? 'bg-blue-500/10 text-blue-400 border-blue-500/30 hover:bg-blue-500/20'
                : 'bg-gray-800 text-gray-500 border-gray-700 hover:bg-gray-700'
            }`}
        >
            <Clock size={10} />
        </button>
      </div>

      {showOptions && (
        <div className="absolute bottom-full right-0 mb-2 w-48 bg-dark-950 border border-white/10 rounded-lg shadow-xl p-2 z-50">
            <div className="text-[10px] font-bold text-gray-400 mb-2 uppercase">Intervalo de Actualización</div>
            <div className="space-y-1">
                <button onClick={() => { onSetInterval(1); setShowOptions(false); }} className="w-full text-left px-2 py-1 text-xs text-gray-300 hover:bg-white/10 rounded">
                    Cada paso (1)
                </button>
                <button onClick={() => { onSetInterval(10); setShowOptions(false); }} className="w-full text-left px-2 py-1 text-xs text-gray-300 hover:bg-white/10 rounded">
                    Cada 10 pasos
                </button>
                <button onClick={() => { onSetInterval(100); setShowOptions(false); }} className="w-full text-left px-2 py-1 text-xs text-gray-300 hover:bg-white/10 rounded">
                    Cada 100 pasos
                </button>
                <button onClick={() => { onSetInterval(-1); setShowOptions(false); }} className="w-full text-left px-2 py-1 text-xs text-red-400 hover:bg-red-500/10 rounded">
                    Desactivado (-1)
                </button>

                <div className="border-t border-white/10 my-1 pt-1">
                    <div className="flex items-center gap-2">
                        <input
                            type="number"
                            value={customInterval}
                            onChange={(e) => setCustomInterval(parseInt(e.target.value) || 1)}
                            className="w-16 bg-dark-900 border border-white/20 rounded px-1 py-0.5 text-xs text-white"
                            min="1"
                        />
                        <button
                            onClick={() => { onSetInterval(customInterval); setShowOptions(false); }}
                            className="text-[10px] bg-blue-500/20 text-blue-400 px-2 py-0.5 rounded hover:bg-blue-500/30"
                        >
                            Set
                        </button>
                    </div>
                </div>
            </div>
        </div>
      )}
      {showOptions && (
          <div className="fixed inset-0 z-40" onClick={() => setShowOptions(false)}></div>
      )}
    </div>
  );
};

const StatusIndicators: React.FC<{
  fps: number;
  particleCount: string | number;
  liveFeedEnabled: boolean;
  controlsEnabled: boolean;
  onToggleLiveFeed: () => void;
  onSetInterval: (interval: number) => void;
  currentStep: number;
}> = ({ fps, particleCount, liveFeedEnabled, controlsEnabled, onToggleLiveFeed, onSetInterval, currentStep }) => (
  <div className="flex items-center gap-4 text-xs font-mono">
    <div className="flex items-center gap-2">
      <span className="text-[10px] font-bold text-gray-500">FPS</span>
      <span className={fps > 0 ? 'text-teal-400' : 'text-gray-600'}>
        {fps > 0 ? fps.toFixed(1) : '0.0'}
      </span>
    </div>
    <div className="flex items-center gap-2">
      <span className="text-[10px] font-bold text-gray-500">PART.</span>
      <span className={particleCount ? 'text-blue-400' : 'text-gray-600'}>
        {particleCount}
      </span>
    </div>
    <div className="flex items-center gap-2">
        <span className="text-[10px] font-bold text-gray-500">STEP</span>
        <span className="text-white">
            {currentStep.toLocaleString()}
        </span>
    </div>
    <LiveFeedControl
        liveFeedEnabled={liveFeedEnabled}
        controlsEnabled={controlsEnabled}
        onSetInterval={onSetInterval}
        onToggleLiveFeed={onToggleLiveFeed}
    />
  </div>
);

const HistoryTimeline: React.FC<{
  available: boolean;
  minStep: number | null;
  maxStep: number | null;
  selectedStep: number;
  onStepBackward: () => void;
  onStepForward: () => void;
  onSliderChange: (event: React.ChangeEvent<HTMLInputElement>) => void;
  onSliderRelease: () => void;
}> = ({ available, minStep, maxStep, selectedStep, onStepBackward, onStepForward, onSliderChange, onSliderRelease }) => {
  if (!available) {
    return (
      <div className="text-[10px] text-gray-500 pt-2 text-center border-t border-white/5">
        No history available. Run simulation to generate data.
      </div>
    );
  }

  return (
    <div className="flex items-center gap-3 pt-2 border-t border-white/5">
      <button
        onClick={onStepBackward}
        disabled={selectedStep <= (minStep ?? 0)}
        className="p-1 text-gray-400 hover:text-white disabled:opacity-30"
      >
        <BackwardIcon className="w-3 h-3" />
      </button>

      <div className="flex-1 flex items-center gap-2">
        <input
          type="range"
          min={minStep ?? 0}
          max={maxStep ?? 0}
          value={selectedStep}
          onChange={onSliderChange}
          onMouseUp={onSliderRelease}
          onTouchEnd={onSliderRelease}
          className="flex-1 h-1 bg-gray-700 rounded-lg appearance-none cursor-pointer slider-thumb-sm"
          style={{
            backgroundImage: `linear-gradient(to right, #3b82f6 0%, #3b82f6 ${
              ((selectedStep - (minStep ?? 0)) / ((maxStep ?? 1) - (minStep ?? 0))) * 100
            }%, #4b5563 ${
              ((selectedStep - (minStep ?? 0)) / ((maxStep ?? 1) - (minStep ?? 0))) * 100
            }%, #4b5563 100%)`,
          }}
        />
      </div>

      <button
        onClick={onStepForward}
        disabled={selectedStep >= (maxStep ?? 0)}
        className="p-1 text-gray-400 hover:text-white disabled:opacity-30"
      >
        <ForwardIcon className="w-3 h-3" />
      </button>

      <div className="text-[10px] font-mono text-gray-400 min-w-[80px] text-right">
        Step: <span className="text-white">{selectedStep.toLocaleString()}</span>
      </div>
    </div>
  );
};

// --- Main Component ---

interface HistoryControlsProps {
  mode?: 'full' | 'compact';
}

export const HistoryControls: React.FC<HistoryControlsProps> = ({ mode = 'full' }) => {
  const { sendCommand, ws, simData, inferenceStatus, connectionStatus, liveFeedEnabled, setLiveFeedEnabled, setStepsInterval } = useWebSocket();
  const [historyRange, setHistoryRange] = useState<HistoryRange>({
    available: false,
    min_step: null,
    max_step: null,
    total_frames: 0,
    current_step: 0,
  });
  const [selectedStep, setSelectedStep] = useState<number>(0);
  const [isPlaying, setIsPlaying] = useState(false);
  const isConnected = connectionStatus === 'connected';

  const requestHistoryRange = useCallback(() => {
    sendCommand('history', 'get_history_range', {});
  }, [sendCommand]);

  const handleRestoreStep = useCallback((step: number) => {
    sendCommand('history', 'restore_history_step', { step });
    setSelectedStep(step);
  }, [sendCommand]);

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
    const command = isPlaying ? 'pause' : 'play';
    sendCommand('inference', command);
  };

  const handleReset = () => {
    sendCommand('inference', 'reset');
  };

  const handleSaveSnapshot = () => {
    sendCommand('snapshot', 'save_snapshot');
  };

  const handleToggleLiveFeed = () => {
    setLiveFeedEnabled(!liveFeedEnabled);
  };

  const handleSetInterval = (interval: number) => {
      // Si el intervalo es -1, desactivamos el live feed (comportamiento legacy/backend)
      // Pero el backend set_steps_interval con -1 ya hace que no envíe frames.
      // Además podemos desactivar el live feed explícitamente para sincronizar el estado UI.

      setStepsInterval(interval);

      // Si el intervalo es -1, también actualizamos el estado visual de liveFeedEnabled a false
      if (interval === -1) {
          setLiveFeedEnabled(false);
      } else {
          // Si establecemos un intervalo positivo, aseguramos que el live feed esté habilitado (o el backend lo use)
          // Nota: El backend usa live_feed_enabled como switch maestro. steps_interval es para cuando NO es live feed?
          // No, steps_interval es "Configura el intervalo de pasos para el envío de frames cuando live_feed está DESACTIVADO."
          // Wait, let's re-read backend code.

          /*
            async def handle_set_steps_interval(args):
                "Configura el intervalo de pasos para el envío de frames cuando live_feed está DESACTIVADO."
          */

          // Ah, steps_interval is ONLY effective when live_feed is DISABLED?
          // Let's check `simulation_handlers.py`.
          // `handle_set_live_feed` says: "Live feed activado... enviando datos en tiempo real."
          // `handle_set_steps_interval` says: "Configura el intervalo... cuando live_feed está DESACTIVADO."

          // So if user wants "-1 to disable", they mean they want to disable live feed AND set interval to -1 (fullspeed).
          // If they want "10 steps to show", they likely mean they want live feed DISABLED but getting updates every 10 steps?
          // OR they want live feed ENABLED but throttled?

          // In `handle_set_live_feed`: "Live feed activado... simulación continuará... enviando datos en tiempo real." (Implies every step or controlled by something else?)

          // Actually, if `live_feed_enabled` is TRUE, it sends frames. Does it respect `steps_interval`?
          // Let's check where `simulation_frame` is sent.
          // It's usually in the inference loop.

          // Regardless, based on user request: "cambiar de live feed a desactivado con nros de pasoss a mostrar, -1 era desactivado."
          // It implies they used this control to set the interval AND disable live feed (or enable it in a specific mode).

          // If I set interval to 10, I probably want to see updates every 10 steps.
          // If I set to -1, I want NO updates (Disabled).

          // So:
          if (interval === -1) {
              setLiveFeedEnabled(false);
          } else {
              // If setting a specific interval, we probably want to ENABLE live feed?
              // Or does the backend send frames even if live_feed is false IF steps_interval > 0?

              // Backend comment: "Configura el intervalo de pasos para el envío de frames cuando live_feed está DESACTIVADO."
              // So if live_feed is FALSE, it uses steps_interval.
              // So to "show every 10 steps", we should set live_feed = FALSE and steps_interval = 10.
              // To "disable" (-1), set live_feed = FALSE and steps_interval = -1.

              setLiveFeedEnabled(false);
          }
      }
  };

  useEffect(() => {
    requestHistoryRange();
    const interval = setInterval(requestHistoryRange, 5000);
    return () => clearInterval(interval);
  }, [requestHistoryRange]);

  useEffect(() => {
    if (!ws) return;
    const handleMessage = (event: MessageEvent) => {
      try {
        const message = typeof event.data === 'string' ? JSON.parse(event.data) : event.data;
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
    };
    ws.addEventListener('message', handleMessage);
    return () => ws.removeEventListener('message', handleMessage);
  }, [ws]);

  useEffect(() => {
    setIsPlaying(inferenceStatus === 'running');
  }, [inferenceStatus]);

  const fps = simData?.simulation_info?.fps ?? 0;
  const currentStep = simData?.step ?? simData?.simulation_info?.step ?? 0;

  // Use the utility function for particle calculation
  const particleCount = useMemo(() => {
    return calculateParticleCount(simData);
  }, [simData]);

  const controlsEnabled = isConnected;
  const { min_step, max_step } = historyRange;

  return (
    <div className={`flex flex-col gap-2 relative z-30 ${mode === 'compact' ? 'w-fit' : 'w-full'}`}>
      <div className={`flex items-center ${mode === 'compact' ? 'gap-4' : 'justify-between'}`}>
        <SimulationControls
          isPlaying={isPlaying}
          controlsEnabled={controlsEnabled}
          onPlayPause={handlePlayPause}
          onReset={handleReset}
          onSaveSnapshot={handleSaveSnapshot}
        />
        <StatusIndicators
          fps={fps}
          particleCount={particleCount}
          liveFeedEnabled={liveFeedEnabled}
          controlsEnabled={controlsEnabled}
          onToggleLiveFeed={handleToggleLiveFeed}
          onSetInterval={handleSetInterval}
          currentStep={currentStep}
        />
      </div>
      {mode === 'full' && (
        <HistoryTimeline
          available={historyRange.available}
          minStep={min_step}
          maxStep={max_step}
          selectedStep={selectedStep}
          onStepBackward={handleStepBackward}
          onStepForward={handleStepForward}
          onSliderChange={handleSliderChange}
          onSliderRelease={handleSliderRelease}
        />
      )}
    </div>
  );
};
