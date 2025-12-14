// frontend/src/utils/timelineStorage.ts
/**
 * Utilidades para guardar y cargar frames de simulación en localStorage.
 * Permite mantener una línea de tiempo local de la simulación.
 */

const STORAGE_KEY_PREFIX = "atheria_timeline_";
const DEFAULT_MAX_FRAMES = 5; // Drastically reduced for 256x256 grids
const STORAGE_SIZE_LIMIT = 4 * 1024 * 1024; // Use 4MB (mostly full usage)

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
    // Check approximate size of the frame
    // Heuristic: estimate total number of numerical elements
    let effectiveMaxFrames = maxFrames;
    if (frame.map_data && Array.isArray(frame.map_data)) {
      let totalElements = frame.map_data.length; // dim 0

      if (totalElements > 0 && Array.isArray(frame.map_data[0])) {
        const dim1 = frame.map_data[0].length;
        totalElements *= dim1;

        if (dim1 > 0 && Array.isArray(frame.map_data[0][0])) {
          const dim2 = frame.map_data[0][0].length;
          totalElements *= dim2;
        }
      }

      // 100x100 = 10k elements. 256x256 = 65k. 256x256x8 (bulk) = 524k.
      // JSON overhead is roughly 10 chars per number + punctuation.
      // 500k elements ~ 5MB.
      if (totalElements > 50000) {
        // 50k elements (approx 500KB)
        // Aggressively reduce max frames for large payloads
        if (totalElements > 200000) {
          // > 200k (e.g. 2MB+) -> 1 or 2 frames max
          effectiveMaxFrames = Math.min(effectiveMaxFrames, 2);
        } else {
          effectiveMaxFrames = Math.min(effectiveMaxFrames, 5);
        }
      }
    }

    const storageKey = getStorageKey(experimentName);
    const existing = loadTimeline(experimentName, effectiveMaxFrames);

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
    const cleanedFrames = cleanupFrames(updatedFrames, effectiveMaxFrames);

    // Actualizar metadata
    const steps = cleanedFrames.map((f) => f.step);
    const metadata: TimelineMetadata = {
      max_frames: effectiveMaxFrames,
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
      // Reducir a 1 frame si es necesario
      const reduceTo = Math.max(1, Math.floor(effectiveMaxFrames / 2));
      const reducedFrames = cleanupFrames(cleanedFrames, reduceTo);

      const reducedMetadata: TimelineMetadata = {
        ...metadata,
        total_frames: reducedFrames.length,
        max_frames: reduceTo,
      };

      const reducedData = {
        metadata: reducedMetadata,
        frames: reducedFrames,
      };

      // Si aun así es demasiado grande (un solo frame > 5MB), abortar
      if (JSON.stringify(reducedData).length * 2 > STORAGE_SIZE_LIMIT) {
        console.warn(
          "Frame too large for localStorage, skipping timeline save."
        );
        return false;
      }

      localStorage.setItem(storageKey, JSON.stringify(reducedData));
      console.warn(
        `Timeline excedió límite de almacenamiento. Reducido a ${reducedFrames.length} frames.`
      );
      return true;
    }

    localStorage.setItem(storageKey, JSON.stringify(dataToStore));
    return true;
  } catch (error) {
    if (error instanceof Error && error.name === "QuotaExceededError") {
      // Just clear it and try to save ONE frame, or give up
      console.warn("QuotaExceededError caught. Clever clearing...");
      try {
        // Hard clear
        clearTimeline(experimentName);
        // Try saving just this current frame
        const dataToStore = {
          metadata: {
            max_frames: 1,
            total_frames: 1,
            min_step: frame.step,
            max_step: frame.step,
            last_updated: Date.now(),
          },
          frames: [frame],
        };
        localStorage.setItem(
          getStorageKey(experimentName),
          JSON.stringify(dataToStore)
        );
      } catch (e) {
        console.error(
          "Critical timeline failure, disabling saving for this frame."
        );
      }
      return false;
    }
    console.error("Error guardando frame en timeline:", error);
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
