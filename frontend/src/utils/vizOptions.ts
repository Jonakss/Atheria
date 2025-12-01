// frontend/src/utils/vizOptions.ts

export const modelOptions = [
  { value: "UNET", label: "UNet" },
  { value: "SNN_UNET", label: "SNN UNet" },
  { value: "DEEP_QCA", label: "Deep QCA" },
  { value: "MLP", label: "MLP" },
  { value: "UNET_UNITARY", label: "UNet Unitary" },
  {
    value: "UNET_UNITARY_RMSNORM",
    label: "UNet Unitary + RMSNorm (más rápido)",
  },
  { value: "UNET_CONVLSTM", label: "UNet ConvLSTM (con memoria)" },
];

export const vizOptions = [
  { value: "density", label: "Densidad |ρ|²" },
  { value: "phase", label: "Fase (Arg)" },
  { value: "phase_hsv", label: "Fase HSV" },
  { value: "energy", label: "Energía" },
  { value: "real", label: "Parte Real" },
  { value: "imag", label: "Parte Imaginaria" },
  { value: "gradient", label: "Gradiente" },
  { value: "spectral", label: "Espectro (FFT)" },
  { value: "physics", label: "Física (Matriz A)" },
  { value: "entropy", label: "Entropía (Complejidad)" },
  { value: "coherence", label: "Coherencia (Estructuras)" },
  { value: "channel_activity", label: "Actividad por Canal" },
  { value: "holographic", label: "Holographic 3D" },
  { value: "history_3d", label: "Evolución Temporal 3D" },
  { value: "complex_3d", label: "Espacio Complejo 3D (Real vs Imag)" },
  { value: "poincare", label: "Gráfico de Poincaré" },
  { value: "poincare_3d", label: "Poincaré 3D" },
  { value: "flow", label: "Flujo (Quiver)" },
  { value: "phase_attractor", label: "Atractor de Fase" },
  { value: "phase_space", label: "Espacio de Fases (PCA)" },
];
