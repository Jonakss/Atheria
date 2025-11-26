// frontend/src/modules/Dashboard/components/epochConfig.ts

export const EPOCH_LABELS = ["VACÍO", "CUÁNTICA", "PARTÍCULAS", "QUÍMICA", "GRAVEDAD", "BIOLOGÍA"];

export const EPOCH_CONFIGS = [
  {
    label: "VACÍO",
    description: "Estado inicial - Ruido cuántico puro",
    gammaDecay: 0.0,
    vizType: "phase",
    color: "purple"
  },
  {
    label: "CUÁNTICA",
    description: "Coherencia cuántica emergente",
    gammaDecay: 0.001,
    vizType: "phase",
    color: "blue"
  },
  {
    label: "PARTÍCULAS",
    description: "Formación de estructuras discretas",
    gammaDecay: 0.01,
    vizType: "density",
    color: "blue"
  },
  {
    label: "QUÍMICA",
    description: "Interacciones moleculares y enlaces",
    gammaDecay: 0.03,
    vizType: "flow",
    color: "teal"
  },
  {
    label: "GRAVEDAD",
    description: "Estructuras gravitatorias y acumulación",
    gammaDecay: 0.05,
    vizType: "spectral",
    color: "pink"
  },
  {
    label: "BIOLOGÍA",
    description: "Sistemas complejos y autoorganización",
    gammaDecay: 0.1,
    vizType: "density",
    color: "green"
  }
];
