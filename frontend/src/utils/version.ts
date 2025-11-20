/**
 * Utilidad para obtener la versión de la aplicación.
 * La versión se inyecta desde package.json durante el build.
 * Sigue Semantic Versioning (SemVer): MAJOR.MINOR.PATCH
 */

export const APP_VERSION: string = import.meta.env.APP_VERSION || '4.0.2';

// Formatear versión para mostrar en UI (ej: 4.0.2 -> Ver. 4.0.2)
export const getFormattedVersion = (): string => {
  const version = APP_VERSION;
  // SemVer format: MAJOR.MINOR.PATCH
  return `v${version}`;
};

// Función para comparar versiones SemVer (retorna -1 si v1 < v2, 0 si igual, 1 si v1 > v2)
export const compareVersions = (v1: string, v2: string): number => {
  const parts1 = v1.split('.').map(Number);
  const parts2 = v2.split('.').map(Number);
  
  for (let i = 0; i < Math.max(parts1.length, parts2.length); i++) {
    const part1 = parts1[i] || 0;
    const part2 = parts2[i] || 0;
    if (part1 < part2) return -1;
    if (part1 > part2) return 1;
  }
  return 0;
};

// Verificar si una versión es compatible (misma major version)
export const isCompatibleVersion = (v1: string, v2: string): boolean => {
  const major1 = parseInt(v1.split('.')[0], 10);
  const major2 = parseInt(v2.split('.')[0], 10);
  return major1 === major2;
};

