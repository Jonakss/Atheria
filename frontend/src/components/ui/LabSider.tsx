// frontend/src/components/ui/LabSider.tsx
import React, { useState } from 'react';
import { Play, Pause, RefreshCw, Upload, ArrowRightLeft, ChevronRight, FlaskConical, Brain, BarChart3 } from 'lucide-react';
import { useWebSocket } from '../../hooks/useWebSocket';
import { modelOptions, vizOptions } from '../../utils/vizOptions';
import { ExperimentManager } from '../experiments/ExperimentManager';
// import { CheckpointManager } from '../training/CheckpointManager'; // TODO: Migrar a Tailwind
import { ExperimentInfo } from '../experiments/ExperimentInfo';
// import { TransferLearningWizard } from '../experiments/TransferLearningWizard'; // TODO: Migrar a Tailwind
import { GlassPanel } from '../../modules/Dashboard/components/GlassPanel';

type LabSection = 'inference' | 'training' | 'analysis';

export function LabSider() {
    const { 
        sendCommand, experimentsData, trainingStatus, trainingProgress, 
        inferenceStatus, connectionStatus, connect, disconnect, selectedViz, setSelectedViz,
        activeExperiment, setActiveExperiment
    } = useWebSocket();
    
    const isConnected = connectionStatus === 'connected';
    const [activeSection, setActiveSection] = useState<LabSection>('inference');
    
    // Estados para los inputs de entrenamiento
    const [selectedModel, setSelectedModel] = useState<string>('UNET');
    const [learningRate, setLearningRate] = useState(0.0001);
    const [gridSize, setGridSize] = useState(64);
    const [qcaSteps, setQcaSteps] = useState(16);
    const [dState, setDState] = useState(8);
    const [hiddenChannels, setHiddenChannels] = useState(32);
    const [episodesToAdd, setEpisodesToAdd] = useState(100);
    const [transferFromExperiment, setTransferFromExperiment] = useState<string | null>(null);
    const [gammaDecay, setGammaDecay] = useState(0.01);
    const [initialStateMode, setInitialStateMode] = useState('complex_noise');
    const [transferWizardOpened, setTransferWizardOpened] = useState(false);

    // Encontrar el experimento activo
    const currentExperiment = activeExperiment 
        ? experimentsData?.find(exp => exp.name === activeExperiment) 
        : null;

    const handleCreateExperiment = () => {
        if (!isConnected) {
            alert('⚠️ No hay conexión con el servidor. Conecta primero.');
            return;
        }
        
        if (!selectedModel) {
            alert('⚠️ Por favor selecciona una arquitectura de modelo.');
            return;
        }
        
        if (gridSize < 16 || gridSize > 512) {
            alert('⚠️ El tamaño de grid debe estar entre 16 y 512.');
            return;
        }
        
        if (learningRate <= 0 || learningRate > 1) {
            alert('⚠️ El learning rate debe estar entre 0 y 1.');
            return;
        }
        
        if (episodesToAdd < 1) {
            alert('⚠️ Debes especificar al menos 1 episodio.');
            return;
        }
        
        const expName = `${selectedModel}-d${dState}-h${hiddenChannels}-g${gridSize}-lr${learningRate.toExponential(0)}`;
        const existingExp = experimentsData?.find(e => e.name === expName);
        if (existingExp) {
            if (!confirm(`⚠️ El experimento "${expName}" ya existe. ¿Deseas continuar de todas formas?`)) {
                return;
            }
        }
        
        const args: Record<string, any> = { 
            EXPERIMENT_NAME: expName, 
            MODEL_ARCHITECTURE: selectedModel, 
            LR_RATE_M: learningRate,
            GRID_SIZE_TRAINING: gridSize, 
            QCA_STEPS_TRAINING: qcaSteps,
            TOTAL_EPISODES: episodesToAdd,
            MODEL_PARAMS: { d_state: dState, hidden_channels: hiddenChannels, alpha: 0.9, beta: 0.85 },
            GAMMA_DECAY: gammaDecay,
            INITIAL_STATE_MODE_INFERENCE: initialStateMode
        };
        
        if (transferFromExperiment) {
            args.LOAD_FROM_EXPERIMENT = transferFromExperiment;
        }
        
        sendCommand('experiment', 'create', args);
        setActiveSection('training'); // Cambiar a sección de entrenamiento
    };

    const handleContinueExperiment = () => {
        if (!isConnected) {
            alert('⚠️ No hay conexión con el servidor.');
            return;
        }
        
        if (!activeExperiment) {
            alert('⚠️ Por favor selecciona un experimento primero.');
            return;
        }
        
        if (episodesToAdd < 1) {
            alert('⚠️ Debes especificar al menos 1 episodio para añadir.');
            return;
        }
        
        if (trainingStatus === 'running') {
            alert('⚠️ Ya hay un entrenamiento en curso. Espera a que termine.');
            return;
        }
        
        const exp = experimentsData?.find(e => e.name === activeExperiment);
        if (exp && !exp.has_checkpoint) {
            alert('⚠️ Este experimento no tiene checkpoints. Debes entrenarlo primero antes de continuar.');
            return;
        }
        
        if (!confirm(`¿Continuar entrenamiento de "${activeExperiment}" añadiendo ${episodesToAdd} episodios más?`)) {
            return;
        }
        
        sendCommand('experiment', 'continue', { 
            EXPERIMENT_NAME: activeExperiment,
            EPISODES_TO_ADD: episodesToAdd
        });
    };

    const handleLoadExperiment = () => { 
        if (!isConnected) return;
        if (activeExperiment) {
            const exp = experimentsData?.find(e => e.name === activeExperiment);
            if (exp && !exp.has_checkpoint) return;
            sendCommand('inference', 'load_experiment', { experiment_name: activeExperiment }); 
        }
    };
    
    const handleResetSimulation = () => {
        if (!isConnected) return;
        sendCommand('inference', 'reset');
    };
    
    const togglePlayPause = () => {
        if (!isConnected) return;
        const command = inferenceStatus === 'running' ? 'pause' : 'play';
        sendCommand('inference', command);
    };

    const handleConnectDisconnect = () => {
        if (connectionStatus === 'connected') {
            disconnect();
        } else {
            connect();
        }
    };

    const handleVizChange = (e: React.ChangeEvent<HTMLSelectElement>) => {
        const value = e.target.value;
        if (value && isConnected) {
            setSelectedViz(value);
            sendCommand('simulation', 'set_viz', { viz_type: value });
        }
    };

    const progressPercent = trainingProgress ? (trainingProgress.current_episode / trainingProgress.total_episodes) * 100 : 0;

    const sectionButtons = [
        { id: 'inference' as LabSection, icon: FlaskConical, label: 'Inferencia', color: 'blue' },
        { id: 'training' as LabSection, icon: Brain, label: 'Entrenamiento', color: 'emerald' },
        { id: 'analysis' as LabSection, icon: BarChart3, label: 'Análisis', color: 'amber' },
    ];

    return (
        <div className="h-full w-full flex flex-col text-gray-300">
            {/* Layout: Sidebar Vertical + Contenido - Sin header duplicado, integrado con dashboard */}
            <div className="flex-1 flex overflow-hidden">
                {/* Sidebar Vertical de Secciones - Similar a NavigationSidebar */}
                <aside className="w-12 border-r border-white/5 bg-[#050505] flex flex-col items-center py-3 gap-2 shrink-0">
                    {sectionButtons.map((section) => {
                        const Icon = section.icon;
                        const isActive = activeSection === section.id;
                        
                        return (
                            <button
                                key={section.id}
                                onClick={() => setActiveSection(section.id)}
                                className={`w-8 h-8 rounded flex items-center justify-center transition-all relative group ${
                                    isActive 
                                        ? section.id === 'inference' ? 'bg-blue-500/10 text-blue-400' :
                                          section.id === 'training' ? 'bg-emerald-500/10 text-emerald-400' :
                                          'bg-amber-500/10 text-amber-400'
                                        : 'text-gray-600 hover:text-gray-300 hover:bg-white/5'
                                }`}
                                title={section.label}
                            >
                                <Icon size={16} strokeWidth={2} />
                                {/* Indicador Activo */}
                                {isActive && (
                                    <div className={`absolute left-0 top-1/2 -translate-y-1/2 w-0.5 h-4 rounded-r ${
                                        section.id === 'inference' ? 'bg-blue-500' :
                                        section.id === 'training' ? 'bg-emerald-500' :
                                        'bg-amber-500'
                                    }`} />
                                )}
                            </button>
                        );
                    })}
                </aside>

                {/* Contenido Scrollable */}
                <div className="flex-1 overflow-y-auto custom-scrollbar">
                    {/* Progreso de Entrenamiento - Siempre visible si está corriendo */}
                    {trainingStatus === 'running' && (
                        <div className="m-4 mb-0 bg-white/5 border border-white/10 rounded-lg p-3 space-y-2">
                            <div className="flex items-center justify-between text-[10px] font-bold text-gray-400 uppercase tracking-wider">
                                <span>PROGRESO</span>
                                <span className="text-emerald-400">ACTIVO</span>
                            </div>
                            <div className="h-2 w-full bg-gray-800 rounded-full overflow-hidden">
                                <div 
                                    className="h-full bg-emerald-500 transition-all duration-300"
                                    style={{ width: `${progressPercent}%` }}
                                />
                            </div>
                            {trainingProgress && (
                                <div className="flex items-center justify-between text-xs text-gray-400">
                                    <span>Episodio {trainingProgress.current_episode}/{trainingProgress.total_episodes}</span>
                                    <span className="font-mono">Loss: {trainingProgress.avg_loss.toFixed(6)}</span>
                                </div>
                            )}
                            <button
                                onClick={() => isConnected && sendCommand('experiment', 'stop', {})}
                                disabled={!isConnected}
                                className="w-full py-1.5 bg-red-500/10 hover:bg-red-500/20 border border-red-500/30 text-red-400 text-xs font-bold rounded transition-all disabled:opacity-50 disabled:cursor-not-allowed"
                            >
                                Detener Entrenamiento
                            </button>
                        </div>
                    )}

                    <div className="p-4 space-y-6">
                    {/* SECCIÓN: INFERENCIA */}
                    {activeSection === 'inference' && (
                        <div className="space-y-4">
                            {/* Experimento Activo */}
                            <div className="space-y-3">
                                <div className="text-[10px] font-bold text-gray-400 uppercase tracking-wider">EXPERIMENTO ACTIVO</div>
                                <ExperimentInfo />
                                
                                <button
                                    onClick={handleLoadExperiment}
                                    disabled={!isConnected || !activeExperiment || !currentExperiment?.has_checkpoint || inferenceStatus === 'running'}
                                    className={`w-full flex items-center justify-center gap-2 px-3 py-2 rounded text-xs font-bold border transition-all ${
                                        currentExperiment?.has_checkpoint && inferenceStatus !== 'running' && isConnected
                                            ? 'bg-blue-500/10 text-blue-400 border-blue-500/30 hover:bg-blue-500/20'
                                            : 'bg-white/5 text-gray-500 border-white/10'
                                    } disabled:opacity-50 disabled:cursor-not-allowed`}
                                >
                                    <Upload size={14} />
                                    Cargar Modelo
                                </button>
                            </div>

                            {/* Controles de Inferencia */}
                            <div className="space-y-3 pt-3 border-t border-white/5">
                                <div className="text-[10px] font-bold text-gray-400 uppercase tracking-wider">CONTROLES</div>
                                <div className="grid grid-cols-2 gap-2">
                                    <button
                                        onClick={togglePlayPause}
                                        disabled={!isConnected || (!activeExperiment || !currentExperiment?.has_checkpoint) && inferenceStatus !== 'running'}
                                        className={`flex items-center justify-center gap-2 px-3 py-2 rounded text-xs font-bold border transition-all ${
                                            inferenceStatus === 'running'
                                                ? 'bg-amber-500/10 text-amber-500 border-amber-500/30 hover:bg-amber-500/20'
                                                : 'bg-green-500/10 text-green-400 border-green-500/30 hover:bg-green-500/20'
                                        } disabled:opacity-50 disabled:cursor-not-allowed`}
                                    >
                                        {inferenceStatus === 'running' ? <Pause size={14} /> : <Play size={14} />}
                                        {inferenceStatus === 'running' ? 'Pausar' : 'Iniciar'}
                                    </button>
                                    <button
                                        onClick={handleResetSimulation}
                                        disabled={!isConnected || !activeExperiment || !currentExperiment?.has_checkpoint}
                                        className="flex items-center justify-center gap-2 px-3 py-2 bg-white/5 hover:bg-white/10 border border-white/10 text-gray-300 text-xs font-bold rounded transition-all disabled:opacity-50 disabled:cursor-not-allowed"
                                    >
                                        <RefreshCw size={14} />
                                        Reiniciar
                                    </button>
                                </div>
                                
                                <div>
                                    <label className="block text-[10px] text-gray-400 mb-1 uppercase">Visualización</label>
                                    <select
                                        value={selectedViz || 'density'}
                                        onChange={handleVizChange}
                                        disabled={!isConnected}
                                        className="w-full px-3 py-1.5 bg-white/5 border border-white/10 rounded text-xs text-gray-300 focus:outline-none focus:border-blue-500/50 disabled:opacity-50"
                                    >
                                        {vizOptions.map(opt => (
                                            <option key={opt.value} value={opt.value}>{opt.label}</option>
                                        ))}
                                    </select>
                                </div>
                            </div>

                            {/* Gestión de Experimentos (Compacta) */}
                            <div className="space-y-3 pt-3 border-t border-white/5">
                                <div className="text-[10px] font-bold text-gray-400 uppercase tracking-wider">GESTIÓN</div>
                                <ExperimentManager />
                                {/* <CheckpointManager /> TODO: Migrar a Tailwind */}
                            </div>
                        </div>
                    )}

                    {/* SECCIÓN: ENTRENAMIENTO */}
                    {activeSection === 'training' && (
                        <div className="space-y-4">
                            {/* Continuar Entrenamiento */}
                            {activeExperiment && (
                                <div className="space-y-3 p-3 bg-white/5 border border-white/10 rounded">
                                    <div className="text-[10px] font-bold text-gray-400 uppercase tracking-wider">Continuar: {activeExperiment}</div>
                                    <div>
                                        <label className="block text-[10px] text-gray-400 mb-1 uppercase">Episodios a Añadir</label>
                                        <input
                                            type="number"
                                            value={episodesToAdd}
                                            onChange={(e) => setEpisodesToAdd(Number(e.target.value) || 0)}
                                            min={1}
                                            step={100}
                                            disabled={!isConnected}
                                            className="w-full px-3 py-1.5 bg-white/5 border border-white/10 rounded text-xs text-gray-300 focus:outline-none focus:border-blue-500/50 disabled:opacity-50 font-mono"
                                        />
                                    </div>
                                    <button
                                        onClick={handleContinueExperiment}
                                        disabled={!isConnected || !activeExperiment || trainingStatus === 'running'}
                                        className="w-full flex items-center justify-center gap-2 px-3 py-2 bg-emerald-500/10 hover:bg-emerald-500/20 border border-emerald-500/30 text-emerald-400 text-xs font-bold rounded transition-all disabled:opacity-50 disabled:cursor-not-allowed"
                                    >
                                        <Play size={14} />
                                        Continuar Entrenamiento
                                    </button>
                                </div>
                            )}

                            <div className="h-px bg-white/5 my-3" />

                            {/* Crear Nuevo Experimento */}
                            <div className="space-y-3">
                                <div className="text-[10px] font-bold text-gray-400 uppercase tracking-wider">Nuevo Experimento</div>
                                
                                <button
                                    onClick={() => setTransferWizardOpened(true)}
                                    className="w-full flex items-center justify-center gap-2 px-3 py-2 bg-blue-500/10 hover:bg-blue-500/20 border border-blue-500/20 text-blue-300 text-xs font-bold rounded transition-all"
                                >
                                    <ArrowRightLeft size={14} />
                                    Transfer Learning (Wizard)
                                </button>

                                <div className="space-y-3 pt-3 border-t border-white/5">
                                    <div>
                                        <label className="block text-[10px] text-gray-400 mb-1 uppercase">Transfer Learning (Opcional)</label>
                                        <select
                                            value={transferFromExperiment || ''}
                                            onChange={(e) => setTransferFromExperiment(e.target.value || null)}
                                            disabled={!isConnected}
                                            className="w-full px-3 py-1.5 bg-white/5 border border-white/10 rounded text-xs text-gray-300 focus:outline-none focus:border-blue-500/50 disabled:opacity-50"
                                        >
                                            <option value="">Ninguno (desde cero)</option>
                                            {experimentsData?.filter(exp => exp.has_checkpoint).map(exp => (
                                                <option key={exp.name} value={exp.name}>
                                                    {exp.name}
                                                </option>
                                            ))}
                                        </select>
                                    </div>

                                    <div className="grid grid-cols-2 gap-2">
                                        <div>
                                            <label className="block text-[10px] text-gray-400 mb-1 uppercase">Arquitectura</label>
                                            <select
                                                value={selectedModel}
                                                onChange={(e) => setSelectedModel(e.target.value)}
                                                disabled={!isConnected}
                                                className="w-full px-3 py-1.5 bg-white/5 border border-white/10 rounded text-xs text-gray-300 focus:outline-none focus:border-blue-500/50 disabled:opacity-50"
                                            >
                                                {modelOptions.map(opt => (
                                                    <option key={opt.value} value={opt.value}>{opt.label}</option>
                                                ))}
                                            </select>
                                        </div>
                                        <div>
                                            <label className="block text-[10px] text-gray-400 mb-1 uppercase">Episodios</label>
                                            <input
                                                type="number"
                                                value={episodesToAdd}
                                                onChange={(e) => setEpisodesToAdd(Number(e.target.value) || 0)}
                                                min={1}
                                                step={100}
                                                disabled={!isConnected}
                                                className="w-full px-3 py-1.5 bg-white/5 border border-white/10 rounded text-xs text-gray-300 focus:outline-none focus:border-blue-500/50 disabled:opacity-50 font-mono"
                                            />
                                        </div>
                                    </div>

                                    <div className="grid grid-cols-2 gap-2">
                                        <div>
                                            <label className="block text-[10px] text-gray-400 mb-1 uppercase">Grid Size</label>
                                            <input
                                                type="number"
                                                value={gridSize}
                                                onChange={(e) => setGridSize(Number(e.target.value) || 0)}
                                                min={16}
                                                step={16}
                                                disabled={!isConnected}
                                                className="w-full px-3 py-1.5 bg-white/5 border border-white/10 rounded text-xs text-gray-300 focus:outline-none focus:border-blue-500/50 disabled:opacity-50 font-mono"
                                            />
                                        </div>
                                        <div>
                                            <label className="block text-[10px] text-gray-400 mb-1 uppercase">QCA Steps</label>
                                            <input
                                                type="number"
                                                value={qcaSteps}
                                                onChange={(e) => setQcaSteps(Number(e.target.value) || 0)}
                                                min={1}
                                                disabled={!isConnected}
                                                className="w-full px-3 py-1.5 bg-white/5 border border-white/10 rounded text-xs text-gray-300 focus:outline-none focus:border-blue-500/50 disabled:opacity-50 font-mono"
                                            />
                                        </div>
                                    </div>

                                    <div className="grid grid-cols-2 gap-2">
                                        <div>
                                            <label className="block text-[10px] text-gray-400 mb-1 uppercase">d_state</label>
                                            <input
                                                type="number"
                                                value={dState}
                                                onChange={(e) => setDState(Number(e.target.value) || 0)}
                                                min={2}
                                                disabled={!isConnected}
                                                className="w-full px-3 py-1.5 bg-white/5 border border-white/10 rounded text-xs text-gray-300 focus:outline-none focus:border-blue-500/50 disabled:opacity-50 font-mono"
                                            />
                                        </div>
                                        <div>
                                            <label className="block text-[10px] text-gray-400 mb-1 uppercase">Hidden Ch.</label>
                                            <input
                                                type="number"
                                                value={hiddenChannels}
                                                onChange={(e) => setHiddenChannels(Number(e.target.value) || 0)}
                                                min={4}
                                                disabled={!isConnected}
                                                className="w-full px-3 py-1.5 bg-white/5 border border-white/10 rounded text-xs text-gray-300 focus:outline-none focus:border-blue-500/50 disabled:opacity-50 font-mono"
                                            />
                                        </div>
                                    </div>

                                    <div>
                                        <label className="block text-[10px] text-gray-400 mb-1 uppercase">Learning Rate</label>
                                        <input
                                            type="number"
                                            value={learningRate}
                                            onChange={(e) => setLearningRate(Number(e.target.value) || 0)}
                                            step={0.00001}
                                            min={0}
                                            max={1}
                                            disabled={!isConnected}
                                            className="w-full px-3 py-1.5 bg-white/5 border border-white/10 rounded text-xs text-gray-300 focus:outline-none focus:border-blue-500/50 disabled:opacity-50 font-mono"
                                        />
                                    </div>

                                    <div>
                                        <label className="block text-[10px] text-gray-400 mb-1 uppercase">Gamma Decay</label>
                                        <input
                                            type="number"
                                            value={gammaDecay}
                                            onChange={(e) => setGammaDecay(Number(e.target.value) || 0)}
                                            step={0.001}
                                            min={0}
                                            max={1}
                                            disabled={!isConnected}
                                            className="w-full px-3 py-1.5 bg-white/5 border border-white/10 rounded text-xs text-gray-300 focus:outline-none focus:border-blue-500/50 disabled:opacity-50 font-mono"
                                        />
                                        <div className="text-[10px] text-gray-600 mt-1">Sistema abierto (&gt;0) o cerrado (=0)</div>
                                    </div>

                                    <div>
                                        <label className="block text-[10px] text-gray-400 mb-1 uppercase">Inicialización</label>
                                        <select
                                            value={initialStateMode}
                                            onChange={(e) => setInitialStateMode(e.target.value)}
                                            disabled={!isConnected}
                                            className="w-full px-3 py-1.5 bg-white/5 border border-white/10 rounded text-xs text-gray-300 focus:outline-none focus:border-blue-500/50 disabled:opacity-50"
                                        >
                                            <option value="complex_noise">Ruido Complejo (estable)</option>
                                            <option value="random">Aleatorio Normalizado</option>
                                            <option value="zeros">Ceros</option>
                                        </select>
                                    </div>

                                    <button
                                        onClick={handleCreateExperiment}
                                        disabled={!isConnected || !selectedModel || trainingStatus === 'running'}
                                        className="w-full px-3 py-2 bg-emerald-500/10 hover:bg-emerald-500/20 border border-emerald-500/30 text-emerald-400 text-xs font-bold rounded transition-all disabled:opacity-50 disabled:cursor-not-allowed"
                                    >
                                        {transferFromExperiment ? 'Crear con Transfer Learning' : 'Crear y Entrenar'}
                                    </button>

                                    {/* Vista Previa */}
                                    <div className="p-3 bg-white/5 border border-white/10 rounded">
                                        <div className="text-[10px] font-semibold text-gray-400 mb-2">Vista Previa</div>
                                        <div className="space-y-1 text-[10px] text-gray-600 font-mono">
                                            <div><span className="text-gray-400">Nombre:</span> {selectedModel ? `${selectedModel}-d${dState}-h${hiddenChannels}-g${gridSize}-lr${learningRate.toExponential(0)}` : 'N/A'}</div>
                                            <div><span className="text-gray-400">Grid:</span> {gridSize}x{gridSize} | <span className="text-gray-400">Steps:</span> {qcaSteps}</div>
                                            <div><span className="text-gray-400">Episodios:</span> {episodesToAdd} | <span className="text-gray-400">LR:</span> {learningRate.toExponential(2)}</div>
                                            {transferFromExperiment && (
                                                <div className="text-blue-400"><span className="text-gray-400">Transfer:</span> {transferFromExperiment}</div>
                                            )}
                                        </div>
                                    </div>
                                </div>
                            </div>
                        </div>
                    )}

                    {/* SECCIÓN: ANÁLISIS */}
                    {activeSection === 'analysis' && (
                        <div className="space-y-4">
                            <div className="text-[10px] font-bold text-gray-400 uppercase tracking-wider">Análisis de Experimentos</div>
                            
                            {/* Resumen de Experimento Activo */}
                            {activeExperiment && currentExperiment && (
                                <GlassPanel className="p-4">
                                    <div className="text-xs font-bold text-gray-200 mb-3">{activeExperiment}</div>
                                    <div className="grid grid-cols-2 gap-4 text-xs">
                                        <div>
                                            <div className="text-[10px] text-gray-500 uppercase mb-1">Estado</div>
                                            <div className={`font-mono font-medium ${
                                                currentExperiment.has_checkpoint ? 'text-emerald-400' : 'text-gray-500'
                                            }`}>
                                                {currentExperiment.has_checkpoint ? '✓ Entrenado' : '○ Sin entrenar'}
                                            </div>
                                        </div>
                                        <div>
                                            <div className="text-[10px] text-gray-500 uppercase mb-1">Episodios</div>
                                            <div className="font-mono text-gray-300">{currentExperiment.total_episodes || 0}</div>
                                        </div>
                                        <div>
                                            <div className="text-[10px] text-gray-500 uppercase mb-1">Grid</div>
                                            <div className="font-mono text-gray-300">{currentExperiment.grid_size_training || 'N/A'}</div>
                                        </div>
                                        <div>
                                            <div className="text-[10px] text-gray-500 uppercase mb-1">Arquitectura</div>
                                            <div className="font-mono text-gray-300">{currentExperiment.model_architecture || 'N/A'}</div>
                                        </div>
                                    </div>
                                </GlassPanel>
                            )}

                            {/* Métricas de Entrenamiento Activo */}
                            {trainingStatus === 'running' && trainingProgress && (
                                <GlassPanel className="p-4">
                                    <div className="text-xs font-bold text-gray-200 mb-3">Entrenamiento en Curso</div>
                                    <div className="space-y-2 text-xs">
                                        <div className="flex justify-between">
                                            <span className="text-gray-400">Progreso</span>
                                            <span className="font-mono text-emerald-400">{progressPercent.toFixed(1)}%</span>
                                        </div>
                                        <div className="flex justify-between">
                                            <span className="text-gray-400">Episodio</span>
                                            <span className="font-mono text-gray-300">{trainingProgress.current_episode}/{trainingProgress.total_episodes}</span>
                                        </div>
                                        <div className="flex justify-between">
                                            <span className="text-gray-400">Loss Promedio</span>
                                            <span className="font-mono text-amber-400">{trainingProgress.avg_loss.toFixed(6)}</span>
                                        </div>
                                    </div>
                                </GlassPanel>
                            )}

                            {/* Análisis Comparativo - Placeholder */}
                            <GlassPanel className="p-4">
                                <div className="text-xs font-bold text-gray-200 mb-3">Comparación de Experimentos</div>
                                <div className="text-xs text-gray-600 text-center py-4">
                                    Análisis comparativo - Próximamente
                                </div>
                            </GlassPanel>

                            {/* Estadísticas Globales */}
                            <GlassPanel className="p-4">
                                <div className="text-xs font-bold text-gray-200 mb-3">Estadísticas Globales</div>
                                <div className="grid grid-cols-2 gap-4 text-xs">
                                    <div>
                                        <div className="text-[10px] text-gray-500 uppercase mb-1">Total Experimentos</div>
                                        <div className="font-mono text-gray-300 text-lg">{experimentsData?.length || 0}</div>
                                    </div>
                                    <div>
                                        <div className="text-[10px] text-gray-500 uppercase mb-1">Entrenados</div>
                                        <div className="font-mono text-emerald-400 text-lg">
                                            {experimentsData?.filter(e => e.has_checkpoint).length || 0}
                                        </div>
                                    </div>
                                </div>
                            </GlassPanel>
                        </div>
                    )}
                </div>
                </div>
            </div>
            
            {/* Transfer Learning Wizard - TODO: Migrar a Tailwind */}
            {/* <TransferLearningWizard 
                opened={transferWizardOpened}
                onClose={() => setTransferWizardOpened(false)}
            /> */}
        </div>
    );
}
