import { create } from 'zustand';

interface ExperimentState {
  // Hybrid Simulation State
  hybridMode: boolean;
  injectionInterval: number;
  quantumNoiseRate: number;

  // Actions
  setHybridMode: (enabled: boolean) => void;
  setInjectionInterval: (interval: number) => void;
  setQuantumNoiseRate: (rate: number) => void;

  // Transient UI State (Visual Effects)
  isQuantumInjecting: boolean;
  triggerQuantumInjection: () => void;
}

export const useExperimentStore = create<ExperimentState>((set) => ({
  hybridMode: false,
  injectionInterval: 50,
  quantumNoiseRate: 0.05,

  setHybridMode: (enabled) => set({ hybridMode: enabled }),
  setInjectionInterval: (interval) => set({ injectionInterval: interval }),
  setQuantumNoiseRate: (rate) => set({ quantumNoiseRate: rate }),

  isQuantumInjecting: false,
  triggerQuantumInjection: () => {
    set({ isQuantumInjecting: true });
    // Reset flag after animation duration (e.g. 1000ms)
    setTimeout(() => set({ isQuantumInjecting: false }), 1000);
  }
}));
