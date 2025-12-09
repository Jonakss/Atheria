// frontend/src/utils/timelineStorage.ts
/**
 * Utilidades para guardar y cargar frames de simulación en localStorage.
 * Permite mantener una línea de tiempo local de la simulación.
 */

const STORAGE_KEY_PREFIX = "atheria_timeline_";
const DEFAULT_MAX_FRAMES = 25; // Reducido de 100 a 25 para evitar QuotaExceeded
const STORAGE_SIZE_LIMIT = 2 * 1024 * 1024; // Reducido a 2MB para ser conservador

export interface TimelineFrame {
  step: number;
  timestamp: number;
  map_data: number[][];
  simulation_info?: {
    fps?: number;
    gamma_decay?: number;
    epoch?: number;
    [key: string]: any;
  };
}

interface TimelineMetadata {
  max_frames: number;
  total_frames: number;
  min_step: number;
  max_step: number;
  last_updated: number;
}

/**
 * Obtiene la clave de almacenamiento para un experimento
 */
function getStorageKey(experimentName: string | null): string {
  const expKey = experimentName || "default";
  return `${STORAGE_KEY_PREFIX}${expKey}`;
}

/**
 * Limpia frames antiguos si exceden el límite
 */
function cleanupFrames(
  frames: TimelineFrame[],
  maxFrames: number
): TimelineFrame[] {
  if (frames.length <= maxFrames) {
    return frames;
  }

  // Mantener solo los últimos maxFrames frames
  return frames.slice(-maxFrames);
}

/**
 * Guarda un frame en el timeline del navegador
 */
export function saveFrameToTimeline(
  frame: TimelineFrame,
  experimentName: string | null = null,
  maxFrames: number = DEFAULT_MAX_FRAMES
): boolean {
  try {
    const storageKey = getStorageKey(experimentName);
    const existing = loadTimeline(experimentName, maxFrames);

    // Optimizar frame: solo guardar datos esenciales
    const optimizedFrame: TimelineFrame = {
      step: frame.step,
      timestamp: frame.timestamp || Date.now(),
      map_data: frame.map_data, // Ya viene optimizado del backend
      simulation_info: frame.simulation_info
        ? {
            fps: frame.simulation_info.fps,
            gamma_decay: frame.simulation_info.gamma_decay,
            epoch: frame.simulation_info.epoch,
          }
        : undefined,
    };

    // Agregar nuevo frame
    const updatedFrames = [...existing.frames, optimizedFrame];

    // Limpiar frames antiguos si exceden el límite
    const cleanedFrames = cleanupFrames(updatedFrames, maxFrames);

    // Actualizar metadata
    const steps = cleanedFrames.map((f) => f.step);
    const metadata: TimelineMetadata = {
      max_frames: maxFrames,
      total_frames: cleanedFrames.length,
      min_step: steps.length > 0 ? Math.min(...steps) : 0,
      max_step: steps.length > 0 ? Math.max(...steps) : 0,
      last_updated: Date.now(),
    };

    // Intentar guardar
    const dataToStore = {
      metadata,
      frames: cleanedFrames,
    };

    const dataString = JSON.stringify(dataToStore);
    const dataSize = dataString.length * 2; // Aproximación

    // Si excede el límite, reducir más agresivamente
    if (dataSize > STORAGE_SIZE_LIMIT) {
      // Reducir a la mitad
      const reducedFrames = cleanupFrames(
        cleanedFrames,
        Math.floor(maxFrames / 2)
      );
      const reducedMetadata: TimelineMetadata = {
        ...metadata,
        total_frames: reducedFrames.length,
        max_frames: Math.floor(maxFrames / 2),
      };

      const reducedData = {
        metadata: reducedMetadata,
        frames: reducedFrames,
      };

      localStorage.setItem(storageKey, JSON.stringify(reducedData));
      console.warn(
        `Timeline excedió límite de almacenamiento. Reducido a ${reducedFrames.length} frames.`
      );
      return true;
    }

    localStorage.setItem(storageKey, JSON.stringify(dataToStore));
    return true;
  } catch (error) {
    // Si hay error (ej: cuota excedida), intentar limpiar y reducir
    if (error instanceof Error && error.name === "QuotaExceededError") {
      console.warn("Cuota de localStorage excedida. Limpiando timeline...");
      try {
        const existing = loadTimeline(
          experimentName,
          Math.floor(maxFrames / 2)
        );
        // Intentar guardar con menos frames
        if (existing.frames.length > 0) {
          const reduced = cleanupFrames(
            existing.frames,
            Math.floor(maxFrames / 4)
          );
          const reducedData = {
            metadata: {
              max_frames: Math.floor(maxFrames / 4),
              total_frames: reduced.length,
              min_step:
                reduced.length > 0
                  ? Math.min(...reduced.map((f) => f.step))
                  : 0,
              max_step:
                reduced.length > 0
                  ? Math.max(...reduced.map((f) => f.step))
                  : 0,
              last_updated: Date.now(),
            },
            frames: reduced,
          };
          localStorage.setItem(
            getStorageKey(experimentName),
            JSON.stringify(reducedData)
          );
          return true;
        }
      } catch (e) {
        console.error("Error al limpiar timeline:", e);
      }
    }
    console.error("Error guardando frame en timeline:", error);
    // Si falla repetidamente, podríamos deshabilitar el guardado temporalmente aquí
    return false;
  }
}

/**
 * Carga el timeline desde localStorage
 */
export function loadTimeline(
  experimentName: string | null = null,
  maxFrames: number = DEFAULT_MAX_FRAMES
): { frames: TimelineFrame[]; metadata: TimelineMetadata } {
  try {
    const storageKey = getStorageKey(experimentName);
    const stored = localStorage.getItem(storageKey);

    if (!stored) {
      return {
        frames: [],
        metadata: {
          max_frames: maxFrames,
          total_frames: 0,
          min_step: 0,
          max_step: 0,
          last_updated: 0,
        },
      };
    }

    const data = JSON.parse(stored);

    // Validar estructura
    if (!data.frames || !Array.isArray(data.frames)) {
      return {
        frames: [],
        metadata: data.metadata || {
          max_frames: maxFrames,
          total_frames: 0,
          min_step: 0,
          max_step: 0,
          last_updated: 0,
        },
      };
    }

    // Aplicar límite si es necesario
    let frames = data.frames;
    if (frames.length > maxFrames) {
      frames = cleanupFrames(frames, maxFrames);
    }

    return {
      frames,
      metadata: data.metadata || {
        max_frames: maxFrames,
        total_frames: frames.length,
        min_step:
          frames.length > 0
            ? Math.min(...frames.map((f: TimelineFrame) => f.step))
            : 0,
        max_step:
          frames.length > 0
            ? Math.max(...frames.map((f: TimelineFrame) => f.step))
            : 0,
        last_updated: 0,
      },
    };
  } catch (error) {
    console.error("Error cargando timeline:", error);
    return {
      frames: [],
      metadata: {
        max_frames: maxFrames,
        total_frames: 0,
        min_step: 0,
        max_step: 0,
        last_updated: 0,
      },
    };
  }
}

/**
 * Limpia el timeline de un experimento
 */
export function clearTimeline(experimentName: string | null = null): void {
  try {
    const storageKey = getStorageKey(experimentName);
    localStorage.removeItem(storageKey);
  } catch (error) {
    console.error("Error limpiando timeline:", error);
  }
}

/**
 * Obtiene estadísticas del timeline
 */
export function getTimelineStats(
  experimentName: string | null = null
): TimelineMetadata | null {
  try {
    const storageKey = getStorageKey(experimentName);
    const stored = localStorage.getItem(storageKey);

    if (!stored) {
      return null;
    }

    const data = JSON.parse(stored);
    return data.metadata || null;
  } catch (error) {
    console.error("Error obteniendo estadísticas del timeline:", error);
    return null;
  }
}

/**
 * Obtiene un frame por step
 */
export function getFrameByStep(
  step: number,
  experimentName: string | null = null
): TimelineFrame | null {
  const timeline = loadTimeline(experimentName);
  return timeline.frames.find((f) => f.step === step) || null;
}

/**
 * Obtiene frames en un rango de steps
 */
export function getFramesRange(
  startStep: number,
  endStep: number,
  experimentName: string | null = null
): TimelineFrame[] {
  const timeline = loadTimeline(experimentName);
  return timeline.frames.filter(
    (f) => f.step >= startStep && f.step <= endStep
  );
}
