import { BackwardIcon, ForwardIcon, PauseIcon, PlayIcon } from '@heroicons/react/24/solid';
import { RefreshCw, Save, Eye, EyeOff } from 'lucide-react';
import React, { useCallback, useEffect, useState, useMemo } from 'react';
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

const StatusIndicators: React.FC<{
  fps: number;
  particleCount: string | number;
  liveFeedEnabled: boolean;
  controlsEnabled: boolean;
  onToggleLiveFeed: () => void;
}> = ({ fps, particleCount, liveFeedEnabled, controlsEnabled, onToggleLiveFeed }) => (
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
    <button
      onClick={onToggleLiveFeed}
      disabled={!controlsEnabled}
      className={`flex items-center gap-1.5 px-2 py-0.5 rounded text-[10px] font-bold border transition-all ${
        liveFeedEnabled
          ? 'bg-blue-500/10 text-blue-400 border-blue-500/30'
          : 'bg-gray-800 text-gray-500 border-gray-700'
      }`}
    >
      {liveFeedEnabled ? <Eye size={10} /> : <EyeOff size={10} />}
      <span>{liveFeedEnabled ? 'LIVE' : 'OFF'}</span>
    </button>
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

export const HistoryControls: React.FC = () => {
  const { sendCommand, ws, simData, inferenceStatus, connectionStatus, liveFeedEnabled, setLiveFeedEnabled } = useWebSocket();
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

  // Use the utility function for particle calculation
  const particleCount = useMemo(() => {
    return calculateParticleCount(simData);
  }, [simData]);

  const controlsEnabled = isConnected;
  const { min_step, max_step } = historyRange;

  return (
    <div className="flex flex-col gap-2 p-3 bg-brand-dark/95 backdrop-blur-md rounded-lg border border-white/10 relative z-30">
      <div className="flex items-center justify-between">
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
        />
      </div>
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
    </div>
  );
};
