/**
 * Utilidad para obtener la versión de la aplicación.
 * La versión se inyecta desde package.json durante el build.
 */

// @ts-ignore - Vite inyecta esta variable
export const APP_VERSION: string = import.meta.env.APP_VERSION || '4.0.2';

// Formatear versión para mostrar en UI (ej: 4.0.2 -> Ver. 4.0.2-RC)
export const getFormattedVersion = (): string => {
  const version = APP_VERSION;
  // Detectar si es una versión RC (Release Candidate)
  const isRC = version.includes('rc') || version.includes('RC');
  return isRC ? `Ver. ${version}` : `Ver. ${version}`;
};

