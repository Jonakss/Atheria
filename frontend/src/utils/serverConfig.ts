// frontend/src/utils/serverConfig.ts
// Utilidades para gestionar la configuración del servidor con persistencia

const SERVER_CONFIG_KEY = 'aetheria_server_config';

export interface ServerConfig {
    host: string;
    port: number;
    protocol: 'ws' | 'wss';
    path?: string; // Path opcional para WebSocket (por defecto '/ws')
}

const DEFAULT_CONFIG: ServerConfig = {
    host: 'localhost',
    port: 8000,
    protocol: 'ws',
    path: '/ws' // Por defecto usa /ws para compatibilidad con servidor local
};

/**
 * API Endpoints configuration
 */
export const API_ENDPOINTS = {
    UPLOAD_MODEL: '/api/upload_model',
    EXPORT_EXPERIMENT: '/api/export_experiment'
};

/**
 * Obtiene la configuración del servidor desde localStorage
 */
export function getServerConfig(): ServerConfig {
    try {
        const stored = localStorage.getItem(SERVER_CONFIG_KEY);
        if (stored) {
            const parsed = JSON.parse(stored);
            // Validar que tenga los campos necesarios
            if (parsed.host && typeof parsed.port === 'number') {
                return {
                    host: parsed.host,
                    port: parsed.port,
                    protocol: parsed.protocol || 'ws',
                    path: parsed.path !== undefined ? parsed.path : '/ws' // Mantener compatibilidad
                };
            }
        }
    } catch (e) {
        console.warn('Error leyendo configuración del servidor:', e);
    }
    return { ...DEFAULT_CONFIG };
}

/**
 * Guarda la configuración del servidor en localStorage
 */
export function saveServerConfig(config: Partial<ServerConfig>): void {
    try {
        const current = getServerConfig();
        const updated = { ...current, ...config };
        localStorage.setItem(SERVER_CONFIG_KEY, JSON.stringify(updated));
    } catch (e) {
        console.error('Error guardando configuración del servidor:', e);
    }
}

/**
 * Construye la URL completa del WebSocket desde la configuración
 */
export function getWebSocketUrl(config?: Partial<ServerConfig>): string {
    const finalConfig = config ? { ...getServerConfig(), ...config } : getServerConfig();
    const protocol = finalConfig.protocol === 'wss' ? 'wss' : 'ws';
    // Usar el path configurado, o '/ws' por defecto, o cadena vacía si es undefined/null
    const path = finalConfig.path !== undefined && finalConfig.path !== null ? finalConfig.path : '/ws';
    return `${protocol}://${finalConfig.host}:${finalConfig.port}${path}`;
}

/**
 * Resetea la configuración a los valores por defecto
 */
export function resetServerConfig(): void {
    try {
        localStorage.removeItem(SERVER_CONFIG_KEY);
    } catch (e) {
        console.error('Error reseteando configuración del servidor:', e);
    }
}
