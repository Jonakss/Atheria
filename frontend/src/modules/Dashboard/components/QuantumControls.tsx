import React, { useState } from 'react';
import { useWebSocket } from '../../../hooks/useWebSocket';
import { Modal } from './Modal';
import { QuantumCircuitViewer } from './QuantumCircuitViewer';
import { Activity, Cpu, Play, Clock, Database, AlertTriangle } from 'lucide-react';
import { Select } from './Select';
import { Badge } from './Badge';

interface QuantumControlsProps {
  isOpen: boolean;
  onClose: () => void;
}

// Static backend list to avoid unnecessary complexity if no API endpoint exists
const availableBackends = [
    { value: 'ionq_simulator', label: 'IonQ Simulator (29 Qubits)' },
    { value: 'ionq_aria', label: 'IonQ Aria (25 Qubits)' },
    { value: 'local_mock', label: 'Local Mock (Debug)' }
];

export const QuantumControls: React.FC<QuantumControlsProps> = ({ isOpen, onClose }) => {
  const { sendCommand, quantumStatus } = useWebSocket();
  const [selectedBackend, setSelectedBackend] = useState<string>('ionq_simulator');

  const handleQuantumJump = () => {
    sendCommand('simulation', 'quantum_fast_forward', {
      steps: 1000000,
      backend: selectedBackend
    });
  };

  const isRunning = quantumStatus?.status === 'running' || quantumStatus?.status === 'queued' || quantumStatus?.status === 'submitted';
  const isCompleted = quantumStatus?.status === 'completed';

  return (
    <Modal
      opened={isOpen}
      onClose={onClose}
      title="Quantum Fast Forward"
      size="lg"
    >
      <div className="space-y-6">

        {/* Header / Intro */}
        <div className="flex items-start gap-4 p-4 bg-indigo-950/20 border border-indigo-500/20 rounded-lg">
           <div className="p-2 bg-indigo-500/10 rounded-full shrink-0">
                <Activity size={24} className="text-indigo-400" />
           </div>
           <div>
               <h3 className="text-sm font-bold text-indigo-200 uppercase tracking-wide">Time Warp Protocol</h3>
               <p className="text-xs text-indigo-300/70 mt-1">
                   Execute 1,000,000 simulation steps instantly using Quantum Hamiltonian Simulation.
                   The state is offloaded to a QPU, evolved unitarily, and re-injected into the lattice.
               </p>
           </div>
        </div>

        {/* Configuration */}
        <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
            <div className="space-y-2">
                <label className="text-xs text-slate-400 font-semibold uppercase flex items-center gap-2">
                    <Database size={12} /> Target Backend
                </label>
                <Select
                    value={selectedBackend}
                    onChange={(val) => val && setSelectedBackend(val)}
                    data={availableBackends}
                    disabled={isRunning}
                    className="w-full"
                />
            </div>

             <div className="space-y-2">
                <label className="text-xs text-slate-400 font-semibold uppercase flex items-center gap-2">
                    <Clock size={12} /> Estimated Wall Time
                </label>
                <div className="px-3 py-2 bg-slate-900 border border-slate-800 rounded text-sm text-slate-300 font-mono">
                    ~120ms <span className="text-slate-600">(Quantum)</span> / 0.8s <span className="text-slate-600">(Latency)</span>
                </div>
            </div>
        </div>

        {/* Circuit Viewer */}
        <div className="space-y-2">
             <label className="text-xs text-slate-400 font-semibold uppercase flex items-center gap-2">
                <Cpu size={12} /> Quantum Circuit (Ansatz)
            </label>
            <QuantumCircuitViewer />
        </div>

        {/* Status & Metrics */}
        {(isRunning || isCompleted) && (
            <div className={`p-4 rounded-lg border ${isCompleted ? 'bg-emerald-950/10 border-emerald-500/20' : 'bg-blue-950/10 border-blue-500/20'}`}>
                <div className="flex justify-between items-center mb-3">
                    <span className="text-xs font-bold uppercase tracking-wider text-slate-400">Execution Status</span>
                    <Badge color={isCompleted ? 'green' : 'blue'} variant="filled">
                        {quantumStatus?.status?.toUpperCase()}
                    </Badge>
                </div>

                {quantumStatus?.job_id && (
                     <div className="text-[10px] font-mono text-slate-500 mb-2">Job ID: {quantumStatus.job_id}</div>
                )}

                {/* Progress Bar (Fake or Real) */}
                {isRunning && (
                    <div className="w-full bg-slate-800 rounded-full h-1.5 mb-4 overflow-hidden">
                        <div className="bg-blue-500 h-1.5 rounded-full animate-progress-indeterminate"></div>
                    </div>
                )}

                {/* Metrics */}
                {isCompleted && quantumStatus?.metadata && (
                    <div className="grid grid-cols-2 gap-4 mt-2">
                        <div className="bg-slate-900/50 p-2 rounded border border-slate-800">
                             <span className="block text-[10px] text-slate-500 uppercase">Fidelity</span>
                             <span className="text-emerald-400 font-mono text-sm">{quantumStatus.metadata.fidelity?.toFixed(4) || 'N/A'}</span>
                        </div>
                         <div className="bg-slate-900/50 p-2 rounded border border-slate-800">
                             <span className="block text-[10px] text-slate-500 uppercase">QPU Time</span>
                             <span className="text-blue-400 font-mono text-sm">{quantumStatus.metadata.quantum_execution_time || 'N/A'}</span>
                        </div>
                    </div>
                )}
            </div>
        )}

        {/* Error State */}
        {quantumStatus?.status === 'error' && (
             <div className="p-3 bg-red-950/20 border border-red-500/30 rounded flex items-center gap-3">
                <AlertTriangle size={16} className="text-red-500" />
                <span className="text-sm text-red-200">Execution Failed: {quantumStatus.message || 'Unknown error'}</span>
             </div>
        )}

        {/* Action Button */}
        <div className="pt-2 flex justify-end">
            <button
                onClick={handleQuantumJump}
                disabled={isRunning}
                className={`
                    flex items-center gap-2 px-6 py-2.5 rounded font-bold text-sm tracking-wide transition-all
                    ${isRunning
                        ? 'bg-slate-800 text-slate-500 cursor-not-allowed'
                        : 'bg-indigo-600 hover:bg-indigo-500 text-white shadow-lg shadow-indigo-900/50 hover:shadow-indigo-500/20'
                    }
                `}
            >
                {isRunning ? (
                    <>
                        <div className="w-3 h-3 border-2 border-slate-500 border-t-transparent rounded-full animate-spin"></div>
                        <span>Processing...</span>
                    </>
                ) : (
                    <>
                        <Play size={16} fill="currentColor" />
                        <span>QUANTUM JUMP (1M STEPS)</span>
                    </>
                )}
            </button>
        </div>
      </div>
    </Modal>
  );
};
