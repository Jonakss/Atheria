import { createContext } from "react";
import { ServerConfig } from "../utils/serverConfig";

export interface SimData {
  complex_3d_data?: {
    real: number[][];
    imag: number[][];
  };
  map_data?: number[][];
  hist_data?: Record<string, Array<{ bin: string; count: number }>>;
  poincare_coords?: number[][];
  step?: number | null;
  timestamp?: number;
  simulation_info?: {
    step?: number;
    start_step?: number;
    total_steps?: number;
    is_paused?: boolean;
    live_feed_enabled?: boolean;
    gamma_decay?: number;
    fps?: number;
    epoch?: number;
    epoch_metrics?: {
      energy?: number;
      clustering?: number;
      symmetry?: number;
    };
    initial_step?: number;
    checkpoint_step?: number;
    checkpoint_episode?: number;
    training_grid_size?: number;
    inference_grid_size?: number;
    grid_scaled?: boolean;
  };
  phase_attractor?: any;
  flow_data?: {
    dx: number[][];
    dy: number[][];
    magnitude?: number[][];
  };
  phase_hsv_data?: {
    hue: number[][];
    saturation: number[][];
    value: number[][];
  };
  roi_info?: any;
  analysis_data?: Array<{ x: number; y: number; step: number }>;
}

export interface QuantumStatus {
  status: "idle" | "submitted" | "queued" | "running" | "completed" | "error";
  job_id?: string;
  data?: any;
  metadata?: {
    quantum_execution_time?: string;
    fidelity?: number;
  };
  message?: string;
}

export interface TrainingProgress {
  current_episode: number;
  total_episodes: number;
  avg_loss: number;
  avg_reward?: number;
  survival?: number;
  symmetry?: number;
  complexity?: number;
  combined?: number;
}

export interface TrainingSnapshot {
  episode: number;
  step?: number;
  map_data: number[][];
  timestamp?: number;
  loss?: number;
  metrics?: {
    survival?: number;
    symmetry?: number;
    complexity?: number;
  };
}

export interface TrainingCheckpoint {
  episode: number;
  is_best: boolean;
  metrics: {
    loss?: number;
    survival?: number;
    symmetry?: number;
    combined?: number;
    complexity?: number;
  };
  timestamp: number;
}

export interface InferenceSnapshot {
  step: number;
  timestamp: string;
  filepath_pt: string;
}

export interface CompileStatus {
  is_compiled: boolean;
  is_native: boolean; // ← INDICADOR DE MOTOR NATIVO
  model_name: string;
  compiles_enabled: boolean;
  device_str?: string; // CPU/CUDA - CORREGIDO: usar device_str para consistencia
  native_version?: string; // Versión del motor C++ (SemVer)
  wrapper_version?: string; // Versión del wrapper Python (SemVer)
  python_version?: string; // Versión del motor Python (SemVer)
  engine_type?: string; // Tipo de motor (CARTESIAN, POLAR, HARMONIC, HOLOGRAPHIC, etc.)
}

export interface WebSocketContextType {
  sendCommand: (
    scope: string,
    command: string,
    args?: Record<string, any>
  ) => void;
  connectionStatus:
    | "connecting"
    | "connected"
    | "disconnected"
    | "server_unavailable";
  experimentsData: any[] | null;
  trainingStatus: "idle" | "running";
  inferenceStatus: "paused" | "running";
  analysisStatus: "idle" | "running" | "completed" | "cancelled" | "error"; // Estado del análisis
  analysisType: "universe_atlas" | "cell_chemistry" | null; // Tipo de análisis actual
  selectedViz: string;
  setSelectedViz: (viz: string) => void;
  connect: () => void;
  reconnect: () => void; // Función para reconexión manual
  disconnect: () => void; // Función para desconectar manualmente
  simData: SimData | null;
  trainingLog: string[];
  allLogs: string[]; // Logs unificados
  trainingProgress: TrainingProgress | null;
  activeExperiment: string | null;
  setActiveExperiment: (name: string | null) => void;
  ws: WebSocket | null; // Exponer WebSocket para escuchar mensajes personalizados
  lastMessage: any | null; // Último mensaje recibido por WebSocket
  snapshotCount: number; // Contador de snapshots capturados
  trainingSnapshots: TrainingSnapshot[]; // Snapshots de entrenamiento
  trainingCheckpoints: TrainingCheckpoint[]; // Checkpoints de entrenamiento
  inferenceSnapshots: InferenceSnapshot[]; // Snapshots de inferencia
  serverConfig: ServerConfig; // Configuración del servidor
  updateServerConfig: (config: Partial<ServerConfig>) => void; // Actualizar configuración
  compileStatus: CompileStatus | null; // Estado de compilación/motor
  liveFeedEnabled: boolean; // Estado del live feed (sincronizado con backend)
  setLiveFeedEnabled: (enabled: boolean) => void; // Función para cambiar live feed
  setStepsInterval: (interval: number) => void; // Función para cambiar el intervalo de pasos
  roiInfo: any; // Información de ROI actual
  quantumStatus: QuantumStatus | null; // Estado de ejecución cuántica
}

export const WebSocketContext = createContext<WebSocketContextType | undefined>(
  undefined
);
