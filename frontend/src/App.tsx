import { useState, useEffect } from 'react';
import { AppShell, MantineProvider, useMantineColorScheme } from '@mantine/core';
import { useDisclosure } from '@mantine/hooks';
import { notifications } from '@mantine/notifications';
import useWebSocket from './hooks/useWebSocket';
import MainHeader from './components/MainHeader';
import LabSider from './components/LabSider';
import MetricsDashboard from './components/MetricsDashboard';

function App() {
  const [navOpened, { toggle: toggleNav }] = useDisclosure();
  const { colorScheme, toggleColorScheme } = useMantineColorScheme();
  const [navbarCollapsed, { toggle: toggleNavbar }] = useDisclosure(false);
  const [panelsVisible, { toggle: togglePanels }] = useDisclosure(true);
  
  // --- ¡¡NUEVO ESTADO PARA SELECCIÓN DE MODELOS!! ---
  const [checkpointData, setCheckpointData] = useState<Record<string, string[]>>({});
  const [selectedExperiment, setSelectedExperiment] = useState<string | null>(null);
  const [selectedCheckpoint, setSelectedCheckpoint] = useState<string | null>(null);
  // ----------------------------------------------------

  const { status, stepCount, simConfig, metrics, trainingLog, setTrainingLog, availableCheckpoints, imageData, trainingMetrics, sendCommand } = useWebSocket();
  
  // Actualiza el estado cuando llegan nuevos datos de checkpoints
  useEffect(() => {
    if (availableCheckpoints) {
      setCheckpointData(availableCheckpoints);
      
      // Si no hay un experimento seleccionado, selecciona el primero de la lista
      if (!selectedExperiment && Object.keys(availableCheckpoints).length > 0) {
        const firstExperiment = Object.keys(availableCheckpoints)[0];
        setSelectedExperiment(firstExperiment);
      }
    }
  }, [availableCheckpoints, selectedExperiment]);

  // Cuando el experimento cambia, resetea el checkpoint seleccionado
  useEffect(() => {
    setSelectedCheckpoint(null);
  }, [selectedExperiment]);

  const [trainingHistory, setTrainingHistory] = useState<any[]>([]);

  useEffect(() => {
    if (trainingMetrics && trainingMetrics.episode) {
      setTrainingHistory(prevHistory => {
        if (prevHistory.find(p => p.episode === trainingMetrics.episode)) {
          return prevHistory;
        }
        return [...prevHistory, trainingMetrics];
      });
    }
  }, [trainingMetrics]);

  const [expName, setExpName] = useState("NuevoExperimento");
  const [trainModelType, setTrainModelType] = useState("unet");
  const [hiddenChannels, setHiddenChannels] = useState(32);
  const [trainingGridSize, setTrainingGridSize] = useState(64);
  const [learningRate, setLearningRate] = useState(0.000001);
  const [episodes, setEpisodes] = useState(2000);
  const [isTrainingLoading, setIsTrainingLoading] = useState(false);
  const [isRefreshingCheckpoints, setIsRefreshingCheckpoints] = useState(false);

  const startTraining = () => {
    setTrainingHistory([]);
    setIsTrainingLoading(true);
    notifications.show({
      id: 'start-training',
      loading: true,
      title: 'Iniciando Entrenamiento',
      message: 'Enviando comando...',
      autoClose: false,
      withCloseButton: false,
    });

    sendCommand({
      scope: 'lab',
      command: 'start_training',
      args: {
        name: expName,
        model: trainModelType,
        hidden_channels: hiddenChannels,
        training_grid_size: trainingGridSize,
        lr: learningRate,
        episodes: episodes
      }
    });
    setTrainingLog('');
    
    notifications.update({
      id: 'start-training',
      color: 'teal',
      title: 'Entrenamiento Iniciado',
      message: `Entrenamiento '${expName}' iniciado.`,
      autoClose: 5000,
    });
  };

  const stopTraining = () => {
    setIsTrainingLoading(false);
    sendCommand({ scope: 'lab', command: 'stop_training' });
    notifications.show({
      title: 'Entrenamiento Detenido',
      message: 'Comando para detener entrenamiento enviado.',
      color: 'yellow',
    });
  };

  const refreshCheckpoints = () => {
    setIsRefreshingCheckpoints(true);
    sendCommand({ scope: 'lab', command: 'refresh_checkpoints' });
    setTimeout(() => {
      setIsRefreshingCheckpoints(false);
      notifications.show({
        title: 'Modelos Refrescados',
        message: 'Lista de modelos actualizada.',
        color: 'blue',
      });
    }, 1000);
  };

  const loadSimulation = (checkpointPath: string | null) => {
    if (!checkpointPath) return;
    
    sendCommand({ scope: 'sim', command: 'start', args: { model_path: checkpointPath } });
    
    notifications.show({
      id: 'load-experiment',
      loading: true,
      title: 'Cargando Simulación',
      message: `Cargando checkpoint: ${checkpointPath}...`,
      autoClose: false,
      withCloseButton: false,
    });

    // Simula un tiempo de carga para feedback visual
    setTimeout(() => {
      notifications.update({
        id: 'load-experiment',
        color: 'teal',
        title: 'Simulación Cargada',
        message: `Modelo ${checkpointPath} cargado exitosamente.`,
        autoClose: 5000,
      });
    }, 1500);
  };

  return (
    <MantineProvider>
      <AppShell
        header={{ height: 60 }}
        navbar={{ width: navbarCollapsed ? 80 : 300, breakpoint: 'sm', collapsed: { mobile: !navOpened } }}
        padding="xs"
      >
        <AppShell.Header>
          <MainHeader
            navOpened={navOpened}
            toggleNav={toggleNav}
            colorScheme={colorScheme}
            toggleColorScheme={toggleColorScheme}
            currentProject={selectedExperiment || 'Sin Proyecto'}
            panelsVisible={panelsVisible}
            togglePanels={togglePanels}
          />
        </AppShell.Header>

        <AppShell.Navbar p={0}>
          <LabSider
            collapsed={navbarCollapsed}
            toggle={toggleNavbar}
            // --- ¡¡NUEVAS PROPS PARA EL SELECTOR!! ---
            checkpointData={checkpointData}
            selectedExperiment={selectedExperiment}
            setSelectedExperiment={setSelectedExperiment}
            selectedCheckpoint={selectedCheckpoint}
            setSelectedCheckpoint={setSelectedCheckpoint}
            loadSimulation={loadSimulation}
            // -----------------------------------------
            status={status}
            stepCount={stepCount}
            expName={expName}
            setExpName={setExpName}
            trainModelType={trainModelType}
            setTrainModelType={setTrainModelType}
            hiddenChannels={hiddenChannels}
            setHiddenChannels={setHiddenChannels}
            trainingGridSize={trainingGridSize}
            setTrainingGridSize={setTrainingGridSize}
            learningRate={learningRate}
            setLearningRate={setLearningRate}
            episodes={episodes}
            setEpisodes={setEpisodes}
            startTraining={startTraining}
            isTrainingLoading={isTrainingLoading}
            stopTraining={stopTraining}
            refreshCheckpoints={refreshCheckpoints}
            isRefreshingCheckpoints={isRefreshingCheckpoints}
          />
        </AppShell.Navbar>

        <AppShell.Main>
          <MetricsDashboard
            panelsVisible={panelsVisible}
            imageData={imageData}
            sendCommand={sendCommand}
            metrics={metrics}
            simConfig={simConfig}
            trainingHistory={trainingHistory}
            trainingLog={trainingLog}
          />
        </AppShell.Main>
      </AppShell>
    </MantineProvider>
  );
}

export default App;
