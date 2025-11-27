
import { SimData } from "../context/WebSocketContextDefinition";

/**
 * Calculates the number of "active" particles in the simulation map.
 * Particles are considered active if their value is greater than 0.01.
 *
 * @param simData The simulation data containing the map_data grid.
 * @returns A formatted string (e.g., "1.2K" or "500") or "N/A" if data is missing.
 */
export const calculateParticleCount = (simData: SimData | null): string => {
  if (!simData?.map_data) return 'N/A';

  let count = 0;
  for (const row of simData.map_data) {
    if (Array.isArray(row)) {
      for (const val of row) {
        if (typeof val === 'number' && !isNaN(val) && val > 0.01) {
          count++;
        }
      }
    }
  }

  return count > 1000 ? `${(count / 1000).toFixed(1)}K` : count.toString();
};
