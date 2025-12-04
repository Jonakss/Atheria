// Mock QASM for Quantum Fast Forward (1M Steps)
// Represents a complex Variational Quantum Eigensolver (VQE) circuit
// optimized for time-evolution (Hamiltonian Simulation)
export const MOCK_QASM = `OPENQASM 2.0;
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
