// frontend/src/modules/Dashboard/components/epochConfig.ts
/**
 * Configuración de las 6 épocas de Atheria (Evolución del Universo)
 *
 * Cada época representa una fase en la simulación con:
 * - Configuración física (gamma_decay)
 * - Tipo de visualización preferida
 * - Descripción conceptual
 */

export interface EpochConfig {
  label: string;
  description: string;
  gammaDecay: number;
  vizType: string;
  color: string;
}

export const EPOCH_LABELS = [
  "VACÍO",
  "CUÁNTICA",
  "PARTÍCULAS",
  "QUÍMICA",
  "GRAVEDAD",
  "BIOLOGÍA",
];

export const EPOCH_CONFIGS: Record<number, EpochConfig> = {
  0: {
    label: "Vacío Primordial",
    description:
      "Estado inicial de mínima energía. Fluctuaciones cuánticas del vacío.",
    gammaDecay: 0.001,
    vizType: "poincare",
    color: "purple",
  },
  1: {
    label: "Era Cuántica",
    description: "Transiciones de fase cuánticas. Coherencia emergente.",
    gammaDecay: 0.005,
    vizType: "poincare",
    color: "blue",
  },
  2: {
    label: "Era de Partículas",
    description: "Formación de estructuras estables. Simetría rota.",
    gammaDecay: 0.01,
    vizType: "density",
    color: "teal",
  },
  3: {
    label: "Era Química",
    description:
      "Interacciones complejas entre partículas. Enlaces emergentes.",
    gammaDecay: 0.015,
    vizType: "density",
    color: "pink",
  },
  4: {
    label: "Era Gravitacional",
    description: "Colapso gravitacional. Formación de estructuras masivas.",
    gammaDecay: 0.02,
    vizType: "holographic",
    color: "green",
  },
  5: {
    label: "Era Biológica",
    description: "Complejidad autorreplicante. Vida emergente.",
    gammaDecay: 0.025,
    vizType: "holographic",
    color: "green",
  },
};
