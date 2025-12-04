import React from 'react';

// Mock QASM for Quantum Fast Forward (1M Steps)
// Represents a complex Variational Quantum Eigensolver (VQE) circuit
// optimized for time-evolution (Hamiltonian Simulation)
const MOCK_QASM = `OPENQASM 2.0;
include "qelib1.inc";

qreg q[4];
creg c[4];

// Initialization: Time-Evolution Operator U(t)
h q[0];
h q[1];
h q[2];
h q[3];

// Entanglement Layer (Interaction Term)
barrier q;
cx q[0], q[1];
rz(0.45) q[1];
cx q[0], q[1];

cx q[1], q[2];
rz(0.78) q[2];
cx q[1], q[2];

cx q[2], q[3];
rz(0.33) q[3];
cx q[2], q[3];

// Hamiltonian Evolution Steps (k=1 to 1000)
// ... Compressed representation ...
barrier q;
rx(1.57) q[0];
ry(0.22) q[1];
rz(3.14) q[2];
rx(0.99) q[3];

// Final Measurement
barrier q;
measure q[0] -> c[0];
measure q[1] -> c[1];
measure q[2] -> c[2];
measure q[3] -> c[3];
`;

export const QuantumCircuitViewer: React.FC = () => {
  return (
    <div className="bg-slate-950 p-4 rounded-md border border-slate-800 font-mono text-xs overflow-auto max-h-[200px] shadow-inner">
        <div className="flex justify-between items-center mb-2 pb-2 border-b border-slate-800">
            <span className="text-slate-400 font-bold">SOURCE CODE</span>
            <span className="text-emerald-500 text-[10px] bg-emerald-950/30 px-2 py-0.5 rounded border border-emerald-900/50">VALID</span>
        </div>
      <pre className="text-blue-300 leading-relaxed">
        {MOCK_QASM.split('\n').map((line, i) => (
          <div key={i} className="flex">
            <span className="w-8 text-slate-600 select-none text-right pr-3">{i + 1}</span>
            <span className={line.startsWith('//') ? 'text-slate-500 italic' : ''}>
              {line}
            </span>
          </div>
        ))}
      </pre>
    </div>
  );
};
