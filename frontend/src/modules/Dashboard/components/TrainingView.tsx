import { Activity, Brain, Eye, Shield, Zap } from 'lucide-react';
import React, { useEffect, useRef, useState } from 'react';
import {
    CartesianGrid as RechartsCartesianGrid,
    Legend as RechartsLegend,
    Line as RechartsLine,
    LineChart as RechartsLineChart,
    ResponsiveContainer as RechartsResponsiveContainer,
    Tooltip as RechartsTooltip,
    XAxis as RechartsXAxis,
    YAxis as RechartsYAxis
} from 'recharts';
import { useWebSocket } from '../../../hooks/useWebSocket';
import { GlassPanel } from './GlassPanel';
import { MetricItem } from './MetricItem';

// Fix for TS2786: 'Component' cannot be used as a JSX component.
const LineChart = RechartsLineChart as any;
const Line = RechartsLine as any;
const XAxis = RechartsXAxis as any;
const YAxis = RechartsYAxis as any;
const CartesianGrid = RechartsCartesianGrid as any;
const Tooltip = RechartsTooltip as any;
const Legend = RechartsLegend as any;
const ResponsiveContainer = RechartsResponsiveContainer as any;

interface TrainingDataPoint {
  episode: number;
  loss: number;
  survival: number;
  symmetry: number;
  complexity: number;
  combined: number;
}

export const TrainingView: React.FC = () => {
  const { trainingProgress, trainingStatus, trainingCheckpoints, activeExperiment, sendCommand } = useWebSocket();
  const [history, setHistory] = useState<TrainingDataPoint[]>([]);

  // Ref to track last processed episode to avoid duplicates
  const lastEpisodeRef = useRef<number>(-1);

  useEffect(() => {
    if (trainingProgress && trainingProgress.current_episode !== lastEpisodeRef.current) {
        lastEpisodeRef.current = trainingProgress.current_episode;

        const newDataPoint: TrainingDataPoint = {
            episode: trainingProgress.current_episode,
            loss: trainingProgress.avg_loss,
            survival: trainingProgress.survival || 0,
            symmetry: trainingProgress.symmetry || 0,
            complexity: trainingProgress.complexity || 0,
            combined: trainingProgress.combined || 0
        };

        setHistory(prev => {
            // Check if episode is already in history (double safety)
            if (prev.some(p => p.episode === newDataPoint.episode)) return prev;

            const newHistory = [...prev, newDataPoint];
            // Limit history size to last 500 points to prevent memory issues
            if (newHistory.length > 500) return newHistory.slice(newHistory.length - 500);
            return newHistory;
        });
    }
  }, [trainingProgress]);

  // Clear history if we detect a reset (e.g. episode 0)
  useEffect(() => {
     if (trainingProgress && trainingProgress.current_episode === 0 && history.length > 10) {
         // Only clear if we had a substantial history, implying a new run
         if (history[history.length - 1].episode > 0) {
             setHistory([]);
             lastEpisodeRef.current = -1;
         }
     }
  }, [trainingProgress, history]);

  return (
    <div className="absolute inset-0 bg-[#050505] text-gray-200 p-6 flex gap-6 overflow-hidden z-10">
        {/* Left Panel: Metrics */}
        <div className="w-1/4 flex flex-col gap-4 min-w-[250px]">
            <GlassPanel className="p-4 flex flex-col gap-4">
                <h2 className="text-sm font-bold text-gray-400 uppercase tracking-wider flex items-center gap-2">
                    <Activity size={16} /> Training Metrics
                </h2>

                <div className="flex flex-col gap-4">
                    <MetricItem
                        label="Status"
                        value={trainingStatus === 'running' ? 'RUNNING' : 'IDLE'}
                        status={trainingStatus === 'running' ? 'good' : 'neutral'}
                    />
                    <MetricItem
                        label="Episode"
                        value={trainingProgress ? `${trainingProgress.current_episode} / ${trainingProgress.total_episodes}` : 'N/A'}
                    />
                     <MetricItem
                        label="Loss"
                        value={trainingProgress?.avg_loss.toFixed(6) || '0.000000'}
                        status={trainingProgress && trainingProgress.avg_loss < 0.01 ? 'good' : 'neutral'}
                    />
                    <div className="h-px bg-white/5 my-2" />
                    <MetricItem
                        label="Survival"
                        value={trainingProgress?.survival?.toFixed(6) || '0.000000'}
                        unit="(Energy)"
                    />
                     <MetricItem
                        label="Symmetry"
                        value={trainingProgress?.symmetry?.toFixed(6) || '0.000000'}
                        unit="(Geo)"
                    />
                     <MetricItem
                        label="Complexity"
                        value={trainingProgress?.complexity?.toFixed(6) || '0.000000'}
                        unit="(Entropy)"
                    />
                     <MetricItem
                        label="Combined"
                        value={trainingProgress?.combined?.toFixed(6) || '0.000000'}
                        status="warning"
                    />
                </div>
            </GlassPanel>

            <GlassPanel className="p-4 flex-1 overflow-hidden flex flex-col">
                 <h2 className="text-sm font-bold text-gray-400 uppercase tracking-wider mb-4 flex items-center gap-2">
                    <Shield size={16} /> Best Checkpoints
                </h2>
                <div className="text-xs text-gray-500 italic flex-1 overflow-y-auto pr-1">
                    {trainingCheckpoints.length === 0 ? (
                        <div className="flex items-center justify-center h-full opacity-50">
                            Waiting for checkpoint data...
                        </div>
                    ) : (
                        <div className="flex flex-col gap-2">
                            {trainingCheckpoints.map((cp) => (
                                <div key={cp.episode} className="flex items-center justify-between p-2 rounded bg-white/5 border border-white/5 group hover:border-emerald-500/30 transition-colors">
                                    <div className="flex flex-col">
                                        <span className="font-mono text-emerald-400 font-bold">Ep {cp.episode}</span>
                                        <span className="text-[10px] text-gray-500">{new Date(cp.timestamp * 1000).toLocaleTimeString()}</span>
                                    </div>
                                    <div className="flex flex-col items-end gap-1">
                                        <div className="flex items-center gap-1">
                                            {cp.is_best && <span className="text-[10px] bg-amber-500/20 text-amber-300 px-1 rounded border border-amber-500/30">BEST</span>}
                                            {cp.has_snapshot && (
                                                <button
                                                    onClick={() => activeExperiment && sendCommand('experiment', 'load_checkpoint_snapshot', { EXPERIMENT_NAME: activeExperiment, EPISODE: cp.episode })}
                                                    className="p-1 rounded bg-blue-500/20 text-blue-300 hover:bg-blue-500/40 transition-colors flex items-center justify-center opacity-0 group-hover:opacity-100"
                                                    title="Load Snapshot"
                                                >
                                                    <Eye size={12} />
                                                </button>
                                            )}
                                        </div>
                                        <span className="font-mono text-[10px]">L: {cp.metrics?.loss?.toFixed(4) ?? 'N/A'}</span>
                                        {cp.metrics?.combined && <span className="font-mono text-[10px] text-gray-400">C: {cp.metrics.combined.toFixed(4)}</span>}
                                    </div>
                                </div>
                            ))}
                        </div>
                    )}
                </div>
            </GlassPanel>
        </div>

        {/* Right Panel: Charts */}
        <div className="flex-1 flex flex-col gap-4 min-w-0">
            {/* Loss Chart */}
            <GlassPanel className="flex-1 p-4 flex flex-col min-h-0">
                <h2 className="text-sm font-bold text-gray-400 uppercase tracking-wider mb-2 flex items-center gap-2">
                    <Zap size={16} /> Loss History
                </h2>
                <div className="flex-1 w-full min-h-0">
                    <ResponsiveContainer width="100%" height="100%">
                        <LineChart data={history}>
                            <CartesianGrid strokeDasharray="3 3" stroke="#333" />
                            <XAxis
                                dataKey="episode"
                                stroke="#666"
                                fontSize={10}
                                tickFormatter={(value: number) => value.toString()}
                            />
                            <YAxis stroke="#666" fontSize={10} domain={['auto', 'auto']} />
                            <Tooltip
                                contentStyle={{ backgroundColor: '#111', borderColor: '#333', color: '#eee' }}
                                itemStyle={{ fontSize: '12px' }}
                                labelStyle={{ color: '#aaa' }}
                            />
                            <Legend />
                            <Line
                                type="monotone"
                                dataKey="loss"
                                stroke="#ef4444"
                                dot={false}
                                strokeWidth={2}
                                activeDot={{ r: 4 }}
                                name="Total Loss"
                            />
                            <Line
                                type="monotone"
                                dataKey="combined"
                                stroke="#f59e0b"
                                dot={false}
                                strokeWidth={1}
                                strokeDasharray="5 5"
                                name="Combined Metric"
                            />
                        </LineChart>
                    </ResponsiveContainer>
                </div>
            </GlassPanel>

            {/* Metrics Chart */}
            <GlassPanel className="flex-1 p-4 flex flex-col min-h-0">
                 <h2 className="text-sm font-bold text-gray-400 uppercase tracking-wider mb-2 flex items-center gap-2">
                    <Brain size={16} /> Evolution Metrics
                </h2>
                <div className="flex-1 w-full min-h-0">
                    <ResponsiveContainer width="100%" height="100%">
                        <LineChart data={history}>
                            <CartesianGrid strokeDasharray="3 3" stroke="#333" />
                            <XAxis dataKey="episode" stroke="#666" fontSize={10} />
                            <YAxis stroke="#666" fontSize={10} />
                            <Tooltip
                                contentStyle={{ backgroundColor: '#111', borderColor: '#333', color: '#eee' }}
                                itemStyle={{ fontSize: '12px' }}
                                labelStyle={{ color: '#aaa' }}
                            />
                            <Legend />
                            <Line type="monotone" dataKey="survival" stroke="#10b981" dot={false} strokeWidth={2} name="Survival" />
                            <Line type="monotone" dataKey="symmetry" stroke="#3b82f6" dot={false} strokeWidth={2} name="Symmetry" />
                            <Line type="monotone" dataKey="complexity" stroke="#a855f7" dot={false} strokeWidth={2} name="Complexity" />
                        </LineChart>
                    </ResponsiveContainer>
                </div>
            </GlassPanel>
        </div>
    </div>
  );
};
