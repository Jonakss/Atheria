import React, { useState } from 'react';
import { useWebSocketContext } from '../context/WebSocketContext';

interface QuantumToolboxProps {
    className?: string;
}

export const QuantumToolbox: React.FC<QuantumToolboxProps> = ({ className }) => {
    const { sendMessage } = useWebSocketContext();
    const [collapseIntensity, setCollapseIntensity] = useState(0.5);
    const [vortexRadius, setVortexRadius] = useState(5);
    const [waveK, setWaveK] = useState(1.0);

    const handleCollapse = () => {
        sendMessage('tool_action', {
            action: 'collapse',
            params: { intensity: collapseIntensity }
        });
    };

    const handleVortex = () => {
        sendMessage('tool_action', {
            action: 'vortex',
            params: { radius: vortexRadius, strength: 1.0 }
        });
    };

    const handleWave = () => {
        sendMessage('tool_action', {
            action: 'wave',
            params: { k_x: waveK, k_y: waveK }
        });
    };

    return (
        <div className={`p-4 bg-gray-900/80 backdrop-blur-md rounded-xl border border-white/10 ${className}`}>
            <h3 className="text-lg font-bold text-cyan-400 mb-4 flex items-center gap-2">
                <span className="text-xl">🛠️</span> Quantum Toolbox
            </h3>

            <div className="space-y-6">
                {/* Collapse Tool */}
                <div className="space-y-2">
                    <div className="flex justify-between items-center">
                        <label className="text-sm font-medium text-gray-300">IonQ Collapse</label>
                        <span className="text-xs text-cyan-300">{collapseIntensity.toFixed(2)}</span>
                    </div>
                    <input 
                        type="range" 
                        min="0" max="1" step="0.05"
                        value={collapseIntensity}
                        onChange={(e) => setCollapseIntensity(parseFloat(e.target.value))}
                        className="w-full accent-cyan-500 h-1 bg-gray-700 rounded-lg appearance-none cursor-pointer"
                    />
                    <button 
                        onClick={handleCollapse}
                        className="w-full py-1.5 px-3 bg-cyan-900/50 hover:bg-cyan-800/70 border border-cyan-700/50 rounded-lg text-cyan-100 text-sm transition-all active:scale-95 flex items-center justify-center gap-2"
                    >
                        ⚡ Trigger Collapse
                    </button>
                </div>

                {/* Vortex Tool */}
                <div className="space-y-2 border-t border-white/5 pt-4">
                    <div className="flex justify-between items-center">
                        <label className="text-sm font-medium text-gray-300">Quantum Vortex</label>
                        <span className="text-xs text-purple-300">R: {vortexRadius}</span>
                    </div>
                    <input 
                        type="range" 
                        min="1" max="20" step="1"
                        value={vortexRadius}
                        onChange={(e) => setVortexRadius(parseInt(e.target.value))}
                        className="w-full accent-purple-500 h-1 bg-gray-700 rounded-lg appearance-none cursor-pointer"
                    />
                    <button 
                        onClick={handleVortex}
                        className="w-full py-1.5 px-3 bg-purple-900/50 hover:bg-purple-800/70 border border-purple-700/50 rounded-lg text-purple-100 text-sm transition-all active:scale-95 flex items-center justify-center gap-2"
                    >
                        🌀 Inject Vortex
                    </button>
                </div>

                {/* Plane Wave Tool */}
                <div className="space-y-2 border-t border-white/5 pt-4">
                    <div className="flex justify-between items-center">
                        <label className="text-sm font-medium text-gray-300">Plane Wave</label>
                        <span className="text-xs text-green-300">k: {waveK.toFixed(1)}</span>
                    </div>
                    <input 
                        type="range" 
                        min="0.1" max="5.0" step="0.1"
                        value={waveK}
                        onChange={(e) => setWaveK(parseFloat(e.target.value))}
                        className="w-full accent-green-500 h-1 bg-gray-700 rounded-lg appearance-none cursor-pointer"
                    />
                    <button 
                        onClick={handleWave}
                        className="w-full py-1.5 px-3 bg-green-900/50 hover:bg-green-800/70 border border-green-700/50 rounded-lg text-green-100 text-sm transition-all active:scale-95 flex items-center justify-center gap-2"
                    >
                        🌊 Inject Wave
                    </button>
                </div>
            </div>
        </div>
    );
};
