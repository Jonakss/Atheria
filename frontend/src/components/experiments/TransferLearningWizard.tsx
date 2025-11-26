// frontend/src/components/experiments/TransferLearningWizard.tsx
import {
    Check, Info,
    ArrowRightLeft as TransferIcon
} from 'lucide-react';
import { useEffect, useMemo, useState } from 'react';
import { useWebSocket } from '../../hooks/useWebSocket';
import { Alert } from '../../modules/Dashboard/components/Alert';
import { Badge } from '../../modules/Dashboard/components/Badge';
import { GlassPanel } from '../../modules/Dashboard/components/GlassPanel';
import { Modal } from '../../modules/Dashboard/components/Modal';
import { NumberInput } from '../../modules/Dashboard/components/NumberInput';
import { Step, Stepper, StepperCompleted } from '../../modules/Dashboard/components/Stepper';
import { Table, TableBody, TableHead, TableRow, TableTd, TableTh } from '../../modules/Dashboard/components/Table';

interface ExperimentConfig {
    MODEL_ARCHITECTURE: string;
    GRID_SIZE_TRAINING: number;
    LR_RATE_M: number;
    MODEL_PARAMS: {
        d_state: number;
        hidden_channels: number;
        alpha?: number;
        beta?: number;
    };
    GAMMA_DECAY: number;
    QCA_STEPS_TRAINING: number;
    INITIAL_STATE_MODE_INFERENCE: string;
    TOTAL_EPISODES?: number;
}

interface TransferLearningWizardProps {
    opened: boolean;
    onClose: () => void;
}

export function TransferLearningWizard({ opened, onClose }: TransferLearningWizardProps) {
    const { experimentsData, sendCommand } = useWebSocket();
    const [activeStep, setActiveStep] = useState(0);
    const [baseExperiment, setBaseExperiment] = useState<string | null>(null);
    const [baseConfig, setBaseConfig] = useState<ExperimentConfig | null>(null);
    const [newConfig, setNewConfig] = useState<ExperimentConfig | null>(null);
    const [newExperimentName, setNewExperimentName] = useState('');
    const [isExperimentNameManuallyEdited, setIsExperimentNameManuallyEdited] = useState(false);
    const [episodesToAdd, setEpisodesToAdd] = useState(1000);
    const [searchQuery, setSearchQuery] = useState('');

    // Filtrar experimentos que tienen checkpoints
    const availableExperiments = useMemo(() => {
        return experimentsData?.filter(exp => exp.has_checkpoint) || [];
    }, [experimentsData]);

    // Filtrar experimentos por búsqueda
    const filteredExperiments = useMemo(() => {
        if (!searchQuery.trim()) return availableExperiments;
        const query = searchQuery.toLowerCase();
        return availableExperiments.filter(exp => 
            exp.name.toLowerCase().includes(query)
        );
    }, [availableExperiments, searchQuery]);

    // Cargar configuración cuando se selecciona un experimento base
    useEffect(() => {
        if (baseExperiment && activeStep >= 1) {
            const exp = experimentsData?.find(e => e.name === baseExperiment);
            const expConfig = exp?.config;
            if (expConfig) {
                const config: ExperimentConfig = {
                    MODEL_ARCHITECTURE: expConfig.MODEL_ARCHITECTURE || 'UNET',
                    GRID_SIZE_TRAINING: expConfig.GRID_SIZE_TRAINING || 64,
                    LR_RATE_M: expConfig.LR_RATE_M || 0.0001,
                    MODEL_PARAMS: expConfig.MODEL_PARAMS || { d_state: 8, hidden_channels: 32 },
                    GAMMA_DECAY: expConfig.GAMMA_DECAY || 0.01,
                    QCA_STEPS_TRAINING: expConfig.QCA_STEPS_TRAINING || 10,
                    INITIAL_STATE_MODE_INFERENCE: expConfig.INITIAL_STATE_MODE_INFERENCE || 'complex_noise',
                    TOTAL_EPISODES: expConfig.TOTAL_EPISODES || 0
                };
                setBaseConfig(config);
                setNewConfig({ ...config });
                
                // Solo generar nombre automáticamente si el usuario no lo ha editado manualmente
                if (!isExperimentNameManuallyEdited) {
                    const suggestedName = generateExperimentName(config);
                    setNewExperimentName(suggestedName);
                }
            }
        }
    }, [baseExperiment, activeStep, experimentsData]);

    const generateExperimentName = (config: ExperimentConfig): string => {
        const arch = config.MODEL_ARCHITECTURE;
        const dState = config.MODEL_PARAMS.d_state;
        const hChannels = config.MODEL_PARAMS.hidden_channels;
        const gridSize = config.GRID_SIZE_TRAINING;
        const lr = config.LR_RATE_M;
        return `${arch}-d${dState}-h${hChannels}-g${gridSize}-lr${lr.toExponential(0)}`;
    };

    const applyProgressionTemplate = (template: 'standard' | 'fine_tune' | 'aggressive') => {
        if (!baseConfig || !newConfig) return;

        const updated = { ...newConfig };

        switch (template) {
            case 'standard':
                updated.GRID_SIZE_TRAINING = baseConfig.GRID_SIZE_TRAINING * 2;
                updated.LR_RATE_M = baseConfig.LR_RATE_M;
                break;
            case 'fine_tune':
                updated.GRID_SIZE_TRAINING = baseConfig.GRID_SIZE_TRAINING;
                updated.LR_RATE_M = baseConfig.LR_RATE_M * 0.5;
                break;
            case 'aggressive':
                updated.GRID_SIZE_TRAINING = baseConfig.GRID_SIZE_TRAINING * 4;
                updated.LR_RATE_M = baseConfig.LR_RATE_M * 0.8;
                break;
        }

        setNewConfig(updated);
        // Solo regenerar nombre si el usuario no lo ha editado manualmente
        if (!isExperimentNameManuallyEdited) {
            setNewExperimentName(generateExperimentName(updated));
        }
    };

    const handleNext = () => {
        if (activeStep === 0 && !baseExperiment) {
            alert('⚠️ Selecciona un experimento base para continuar');
            return;
        }
        if (activeStep === 1 && !newConfig) {
            alert('❌ Error: No se pudo cargar la configuración');
            return;
        }
        setActiveStep(activeStep + 1);
    };

    const handleBack = () => {
        setActiveStep(activeStep - 1);
    };

    const handleCreate = () => {
        if (!baseExperiment || !newConfig || !newExperimentName) {
            alert('❌ Error: Faltan datos requeridos');
            return;
        }

        const existingExp = experimentsData?.find(e => e.name === newExperimentName);
        if (existingExp) {
            if (!window.confirm(`⚠️ El experimento "${newExperimentName}" ya existe. ¿Deseas continuar de todas formas?`)) {
                return;
            }
        }

        const args: Record<string, any> = {
            EXPERIMENT_NAME: newExperimentName,
            MODEL_ARCHITECTURE: newConfig.MODEL_ARCHITECTURE,
            LR_RATE_M: newConfig.LR_RATE_M,
            GRID_SIZE_TRAINING: newConfig.GRID_SIZE_TRAINING,
            QCA_STEPS_TRAINING: newConfig.QCA_STEPS_TRAINING,
            TOTAL_EPISODES: episodesToAdd,
            MODEL_PARAMS: newConfig.MODEL_PARAMS,
            GAMMA_DECAY: newConfig.GAMMA_DECAY,
            INITIAL_STATE_MODE_INFERENCE: newConfig.INITIAL_STATE_MODE_INFERENCE,
            LOAD_FROM_EXPERIMENT: baseExperiment
        };

        sendCommand('experiment', 'create', args);
        
        console.log(`✅ Transfer Learning iniciado: Creando experimento "${newExperimentName}" desde "${baseExperiment}"`);

        setActiveStep(0);
        setBaseExperiment(null);
        setBaseConfig(null);
        setNewConfig(null);
        setNewExperimentName('');
        setIsExperimentNameManuallyEdited(false);
        onClose();
    };

    const handleClose = () => {
        setActiveStep(0);
        setBaseExperiment(null);
        setBaseConfig(null);
        setNewConfig(null);
        setNewExperimentName('');
        setIsExperimentNameManuallyEdited(false);
        onClose();
    };

    const baseExperimentData = baseExperiment 
        ? experimentsData?.find(e => e.name === baseExperiment)
        : null;

    const configChanged = useMemo(() => {
        if (!baseConfig || !newConfig) return false;
        return (
            baseConfig.GRID_SIZE_TRAINING !== newConfig.GRID_SIZE_TRAINING ||
            baseConfig.LR_RATE_M !== newConfig.LR_RATE_M ||
            baseConfig.MODEL_PARAMS.d_state !== newConfig.MODEL_PARAMS.d_state ||
            baseConfig.MODEL_PARAMS.hidden_channels !== newConfig.MODEL_PARAMS.hidden_channels ||
            baseConfig.GAMMA_DECAY !== newConfig.GAMMA_DECAY ||
            baseConfig.QCA_STEPS_TRAINING !== newConfig.QCA_STEPS_TRAINING
        );
    }, [baseConfig, newConfig]);

    return (
        <Modal
            opened={opened}
            onClose={handleClose}
            title={
                <div className="flex items-center gap-2">
                    <TransferIcon size={16} />
                    <span>Transfer Learning Wizard</span>
                </div>
            }
            size="xl"
            closeOnClickOutside={false}
        >
            <Stepper active={activeStep} onStepClick={setActiveStep}>
                <Step label="Seleccionar Base" description="Elige el experimento origen">
                    <div className="space-y-4 mt-4">
                        <Alert icon={<Info size={16} />} color="blue" variant="light">
                            <span className="text-xs">
                                Selecciona un experimento que tenga checkpoints guardados. 
                                Su configuración se cargará automáticamente y podrás ajustarla.
                            </span>
                        </Alert>

                        <div className="space-y-2">
                            <label className="text-[10px] font-bold text-gray-400 uppercase tracking-wider">
                                Experimento Base
                            </label>
                            <div className="relative">
                                <input
                                    type="text"
                                    placeholder="Buscar experimento..."
                                    value={searchQuery}
                                    onChange={(e) => setSearchQuery(e.target.value)}
                                    className="w-full px-3 py-1.5 bg-white/5 border border-white/10 rounded text-xs text-gray-300 placeholder-gray-600 focus:outline-none focus:border-blue-500/50 mb-2"
                                />
                            </div>
                            <select
                                value={baseExperiment || ''}
                                onChange={(e) => setBaseExperiment(e.target.value || null)}
                                className="w-full px-3 py-1.5 bg-white/5 border border-white/10 rounded text-xs text-gray-300 focus:outline-none focus:border-blue-500/50"
                            >
                                <option value="">Selecciona un experimento...</option>
                                {filteredExperiments.map(exp => (
                                    <option key={exp.name} value={exp.name}>
                                        {exp.name}
                                    </option>
                                ))}
                            </select>
                        </div>

                        {baseExperimentData && (
                            <GlassPanel className="p-3 bg-[#0a0a0a]">
                                <div className="space-y-2">
                                    <div className="flex items-center justify-between">
                                        <span className="text-xs font-bold text-gray-300">Información del Experimento Base</span>
                                        <Badge color="green" variant="light" size="xs">
                                            ✓ Tiene checkpoints
                                        </Badge>
                                    </div>
                                    <div className="h-px bg-white/10 my-2" />
                                    <div className="flex items-center gap-4 flex-wrap">
                                        <span className="text-[10px] text-gray-500">
                                            <strong>Arquitectura:</strong> {baseExperimentData.config?.MODEL_ARCHITECTURE || 'N/A'}
                                        </span>
                                        <span className="text-[10px] text-gray-500">
                                            <strong>Grid Size:</strong> {baseExperimentData.config?.GRID_SIZE_TRAINING || 'N/A'}
                                        </span>
                                        <span className="text-[10px] text-gray-500">
                                            <strong>LR:</strong> {(baseExperimentData.config?.LR_RATE_M || 0.0001).toExponential(2)}
                                        </span>
                                    </div>
                                </div>
                            </GlassPanel>
                        )}
                    </div>
                </Step>

                <Step label="Configurar" description="Ajusta los parámetros">
                    {baseConfig && newConfig && (
                        <div className="space-y-4 mt-4">
                            <Alert icon={<Info size={16} />} color="blue" variant="light">
                                <span className="text-xs">
                                    La configuración del experimento base se ha cargado. 
                                    Ajusta los parámetros según tu estrategia de entrenamiento progresivo.
                                </span>
                            </Alert>

                            {/* Templates rápidos */}
                            <GlassPanel className="p-3 border border-white/10">
                                <div className="space-y-2">
                                    <span className="text-xs font-bold text-gray-300">Templates de Progresión</span>
                                    <div className="flex items-center gap-2 flex-wrap">
                                        <button
                                            onClick={() => applyProgressionTemplate('standard')}
                                            className="px-2 py-1 bg-white/5 hover:bg-white/10 border border-white/10 text-[10px] font-bold text-gray-300 rounded transition-all"
                                        >
                                            Estándar (Grid 2x, LR igual)
                                        </button>
                                        <button
                                            onClick={() => applyProgressionTemplate('fine_tune')}
                                            className="px-2 py-1 bg-white/5 hover:bg-white/10 border border-white/10 text-[10px] font-bold text-gray-300 rounded transition-all"
                                        >
                                            Fine-tuning (Grid igual, LR 0.5x)
                                        </button>
                                        <button
                                            onClick={() => applyProgressionTemplate('aggressive')}
                                            className="px-2 py-1 bg-white/5 hover:bg-white/10 border border-white/10 text-[10px] font-bold text-gray-300 rounded transition-all"
                                        >
                                            Agresivo (Grid 4x, LR 0.8x)
                                        </button>
                                    </div>
                                </div>
                            </GlassPanel>

                            {/* Comparación lado a lado */}
                            <div className="overflow-x-auto">
                                <Table highlightOnHover>
                                    <TableHead>
                                        <TableRow>
                                            <TableTh>Parámetro</TableTh>
                                            <TableTh>Base</TableTh>
                                            <TableTh>Nuevo</TableTh>
                                            <TableTh>Cambio</TableTh>
                                        </TableRow>
                                    </TableHead>
                                    <TableBody>
                                        <TableRow>
                                            <TableTd><span className="text-xs font-medium">Grid Size</span></TableTd>
                                            <TableTd><span className="text-xs">{baseConfig.GRID_SIZE_TRAINING}</span></TableTd>
                                            <TableTd>
                                                <NumberInput
                                                    value={newConfig.GRID_SIZE_TRAINING}
                                                    onChange={(val) => setNewConfig({
                                                        ...newConfig,
                                                        GRID_SIZE_TRAINING: Number(val) || baseConfig.GRID_SIZE_TRAINING
                                                    })}
                                                    min={16}
                                                    max={512}
                                                    size="xs"
                                                    style={{ width: 100 }}
                                                />
                                            </TableTd>
                                            <TableTd>
                                                {newConfig.GRID_SIZE_TRAINING > baseConfig.GRID_SIZE_TRAINING && (
                                                    <Badge color="green" size="sm">↑ {((newConfig.GRID_SIZE_TRAINING / baseConfig.GRID_SIZE_TRAINING - 1) * 100).toFixed(0)}%</Badge>
                                                )}
                                                {newConfig.GRID_SIZE_TRAINING < baseConfig.GRID_SIZE_TRAINING && (
                                                    <Badge color="orange" size="sm">↓ {((1 - newConfig.GRID_SIZE_TRAINING / baseConfig.GRID_SIZE_TRAINING) * 100).toFixed(0)}%</Badge>
                                                )}
                                                {newConfig.GRID_SIZE_TRAINING === baseConfig.GRID_SIZE_TRAINING && (
                                                    <Badge color="gray" size="sm">=</Badge>
                                                )}
                                            </TableTd>
                                        </TableRow>
                                        <TableRow>
                                            <TableTd><span className="text-xs font-medium">Learning Rate</span></TableTd>
                                            <TableTd><span className="text-xs">{baseConfig.LR_RATE_M.toExponential(3)}</span></TableTd>
                                            <TableTd>
                                                <NumberInput
                                                    value={newConfig.LR_RATE_M}
                                                    onChange={(val) => setNewConfig({
                                                        ...newConfig,
                                                        LR_RATE_M: Number(val) || baseConfig.LR_RATE_M
                                                    })}
                                                    min={1e-6}
                                                    max={1}
                                                    step={1e-5}
                                                    size="xs"
                                                    style={{ width: 120 }}
                                                />
                                            </TableTd>
                                            <TableTd>
                                                {newConfig.LR_RATE_M !== baseConfig.LR_RATE_M && (
                                                    <Badge color="blue" size="sm">
                                                        {newConfig.LR_RATE_M > baseConfig.LR_RATE_M ? '↑' : '↓'}
                                                    </Badge>
                                                )}
                                            </TableTd>
                                        </TableRow>
                                        <TableRow>
                                            <TableTd><span className="text-xs font-medium">d_state</span></TableTd>
                                            <TableTd><span className="text-xs">{baseConfig.MODEL_PARAMS.d_state}</span></TableTd>
                                            <TableTd>
                                                <NumberInput
                                                    value={newConfig.MODEL_PARAMS.d_state}
                                                    onChange={(val) => setNewConfig({
                                                        ...newConfig,
                                                        MODEL_PARAMS: {
                                                            ...newConfig.MODEL_PARAMS,
                                                            d_state: Number(val) || baseConfig.MODEL_PARAMS.d_state
                                                        }
                                                    })}
                                                    min={4}
                                                    max={32}
                                                    size="xs"
                                                    style={{ width: 100 }}
                                                />
                                            </TableTd>
                                            <TableTd>
                                                {newConfig.MODEL_PARAMS.d_state !== baseConfig.MODEL_PARAMS.d_state && (
                                                    <Badge color="blue" size="sm">
                                                        {newConfig.MODEL_PARAMS.d_state > baseConfig.MODEL_PARAMS.d_state ? '↑' : '↓'}
                                                    </Badge>
                                                )}
                                            </TableTd>
                                        </TableRow>
                                        <TableRow>
                                            <TableTd><span className="text-xs font-medium">hidden_channels</span></TableTd>
                                            <TableTd><span className="text-xs">{baseConfig.MODEL_PARAMS.hidden_channels}</span></TableTd>
                                            <TableTd>
                                                <NumberInput
                                                    value={newConfig.MODEL_PARAMS.hidden_channels}
                                                    onChange={(val) => setNewConfig({
                                                        ...newConfig,
                                                        MODEL_PARAMS: {
                                                            ...newConfig.MODEL_PARAMS,
                                                            hidden_channels: Number(val) || baseConfig.MODEL_PARAMS.hidden_channels
                                                        }
                                                    })}
                                                    min={8}
                                                    max={128}
                                                    size="xs"
                                                    style={{ width: 100 }}
                                                />
                                            </TableTd>
                                            <TableTd>
                                                {newConfig.MODEL_PARAMS.hidden_channels !== baseConfig.MODEL_PARAMS.hidden_channels && (
                                                    <Badge color="blue" size="sm">
                                                        {newConfig.MODEL_PARAMS.hidden_channels > baseConfig.MODEL_PARAMS.hidden_channels ? '↑' : '↓'}
                                                    </Badge>
                                                )}
                                            </TableTd>
                                        </TableRow>
                                        <TableRow>
                                            <TableTd><span className="text-xs font-medium">Gamma Decay</span></TableTd>
                                            <TableTd><span className="text-xs">{baseConfig.GAMMA_DECAY.toFixed(4)}</span></TableTd>
                                            <TableTd>
                                                <NumberInput
                                                    value={newConfig.GAMMA_DECAY}
                                                    onChange={(val) => setNewConfig({
                                                        ...newConfig,
                                                        GAMMA_DECAY: Number(val) || baseConfig.GAMMA_DECAY
                                                    })}
                                                    min={0}
                                                    max={1}
                                                    step={0.001}
                                                    size="xs"
                                                    style={{ width: 100 }}
                                                />
                                            </TableTd>
                                            <TableTd>
                                                {newConfig.GAMMA_DECAY !== baseConfig.GAMMA_DECAY && (
                                                    <Badge color="blue" size="sm">
                                                        {newConfig.GAMMA_DECAY > baseConfig.GAMMA_DECAY ? '↑' : '↓'}
                                                    </Badge>
                                                )}
                                            </TableTd>
                                        </TableRow>
                                        <TableRow>
                                            <TableTd><span className="text-xs font-medium">QCA Steps</span></TableTd>
                                            <TableTd><span className="text-xs">{baseConfig.QCA_STEPS_TRAINING}</span></TableTd>
                                            <TableTd>
                                                <NumberInput
                                                    value={newConfig.QCA_STEPS_TRAINING}
                                                    onChange={(val) => setNewConfig({
                                                        ...newConfig,
                                                        QCA_STEPS_TRAINING: Number(val) || baseConfig.QCA_STEPS_TRAINING
                                                    })}
                                                    min={1}
                                                    max={100}
                                                    size="xs"
                                                    style={{ width: 100 }}
                                                />
                                            </TableTd>
                                            <TableTd>
                                                {newConfig.QCA_STEPS_TRAINING !== baseConfig.QCA_STEPS_TRAINING && (
                                                    <Badge color="blue" size="sm">
                                                        {newConfig.QCA_STEPS_TRAINING > baseConfig.QCA_STEPS_TRAINING ? '↑' : '↓'}
                                                    </Badge>
                                                )}
                                            </TableTd>
                                        </TableRow>
                                    </TableBody>
                                </Table>
                            </div>

                            <div className="space-y-2">
                                <label className="text-[10px] font-bold text-gray-400 uppercase tracking-wider">
                                    Episodios a Entrenar
                                </label>
                                <NumberInput
                                    value={episodesToAdd}
                                    onChange={(val) => setEpisodesToAdd(Number(val) || 1000)}
                                    min={1}
                                    max={10000}
                                    size="sm"
                                    placeholder="Número de episodios"
                                />
                                <div className="text-[10px] text-gray-600">
                                    Número de episodios para el nuevo entrenamiento
                                </div>
                            </div>
                        </div>
                    )}
                </Step>

                <Step label="Confirmar" description="Revisa y crea">
                    {baseConfig && newConfig && (
                        <div className="space-y-4 mt-4">
                            <Alert icon={<Info size={16} />} color="green" variant="light">
                                <span className="text-xs">
                                    Revisa la configuración antes de crear el nuevo experimento con transfer learning.
                                </span>
                            </Alert>

                            <GlassPanel className="p-4 border border-white/10">
                                <div className="space-y-4">
                                    <div className="flex items-center justify-between">
                                        <span className="text-xs font-bold text-gray-300">Experimento Base</span>
                                        <Badge>{baseExperiment}</Badge>
                                    </div>
                                    <div className="h-px bg-white/10" />
                                    <div className="flex items-center justify-between">
                                        <span className="text-xs font-bold text-gray-300">Nuevo Experimento</span>
                                        <input
                                            type="text"
                                            value={newExperimentName}
                                            onChange={(e) => {
                                                setNewExperimentName(e.target.value);
                                                setIsExperimentNameManuallyEdited(true);
                                            }}
                                            className="flex-1 max-w-[400px] px-3 py-1.5 bg-white/5 border border-white/10 rounded text-xs text-gray-300 placeholder-gray-600 focus:outline-none focus:border-blue-500/50 ml-4"
                                        />
                                    </div>
                                    <div className="h-px bg-white/10" />
                                    <div className="space-y-2">
                                        <span className="text-xs font-medium text-gray-300">Resumen de Cambios:</span>
                                        {!configChanged && (
                                            <div className="text-[10px] text-gray-500">
                                                No hay cambios en la configuración (solo transfer learning)
                                            </div>
                                        )}
                                        {configChanged && (
                                            <div className="space-y-1">
                                                {newConfig.GRID_SIZE_TRAINING !== baseConfig.GRID_SIZE_TRAINING && (
                                                    <div className="text-[10px] text-gray-400">
                                                        Grid Size: {baseConfig.GRID_SIZE_TRAINING} → {newConfig.GRID_SIZE_TRAINING}
                                                    </div>
                                                )}
                                                {newConfig.LR_RATE_M !== baseConfig.LR_RATE_M && (
                                                    <div className="text-[10px] text-gray-400">
                                                        LR: {baseConfig.LR_RATE_M.toExponential(3)} → {newConfig.LR_RATE_M.toExponential(3)}
                                                    </div>
                                                )}
                                                {newConfig.MODEL_PARAMS.d_state !== baseConfig.MODEL_PARAMS.d_state && (
                                                    <div className="text-[10px] text-gray-400">
                                                        d_state: {baseConfig.MODEL_PARAMS.d_state} → {newConfig.MODEL_PARAMS.d_state}
                                                    </div>
                                                )}
                                                {newConfig.MODEL_PARAMS.hidden_channels !== baseConfig.MODEL_PARAMS.hidden_channels && (
                                                    <div className="text-[10px] text-gray-400">
                                                        hidden_channels: {baseConfig.MODEL_PARAMS.hidden_channels} → {newConfig.MODEL_PARAMS.hidden_channels}
                                                    </div>
                                                )}
                                            </div>
                                        )}
                                        <div className="text-[10px] text-gray-500 mt-2">
                                            Episodios: {episodesToAdd}
                                        </div>
                                    </div>
                                </div>
                            </GlassPanel>
                        </div>
                    )}
                </Step>

                <StepperCompleted>
                    <div className="flex flex-col items-center justify-center space-y-4 mt-4">
                        <Check size={48} className="text-emerald-400" />
                        <span className="text-base font-bold text-gray-200">¡Experimento creado!</span>
                        <div className="text-xs text-gray-500 text-center">
                            El entrenamiento con transfer learning ha sido iniciado.
                            <br />
                            Puedes monitorear el progreso en el panel lateral.
                        </div>
                    </div>
                </StepperCompleted>
            </Stepper>

            <div className="flex items-center justify-end gap-2 mt-6 pt-4 border-t border-white/10">
                {activeStep > 0 && (
                    <button
                        onClick={handleBack}
                        className="px-3 py-1.5 bg-white/5 hover:bg-white/10 border border-white/10 text-xs font-bold text-gray-300 rounded transition-all"
                    >
                        Atrás
                    </button>
                )}
                {activeStep < 3 && (
                    <button
                        onClick={handleNext}
                        className="px-3 py-1.5 bg-blue-500/10 hover:bg-blue-500/20 border border-blue-500/30 text-blue-400 text-xs font-bold rounded transition-all"
                    >
                        Siguiente
                    </button>
                )}
                {activeStep === 3 && (
                    <button
                        onClick={handleCreate}
                        className="flex items-center gap-1.5 px-3 py-1.5 bg-emerald-500/10 hover:bg-emerald-500/20 border border-emerald-500/30 text-emerald-400 text-xs font-bold rounded transition-all"
                    >
                        <Check size={12} />
                        Crear Experimento
                    </button>
                )}
                <button
                    onClick={handleClose}
                    className="px-3 py-1.5 bg-white/5 hover:bg-white/10 border border-white/10 text-xs font-bold text-gray-300 rounded transition-all"
                >
                    Cancelar
                </button>
            </div>
        </Modal>
    );
}
