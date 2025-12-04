import React from 'react';
import { MOCK_QASM } from '../constants/quantum';

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
