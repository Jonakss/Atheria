import React from 'react';
import { Group, Text, Badge, Divider, Fieldset, TextInput, Select, NumberInput, Button, Tooltip, ActionIcon, UnstyledButton, Center, Stack } from '@mantine/core';
import { IconRefresh, IconBulb, IconCheckbox, IconArrowRight, IconArrowLeft, IconPlayerPlay } from '@tabler/icons-react';
import classes from './LabSider.module.css';

interface LabSiderProps {
  collapsed: boolean;
  toggle: () => void;
  
  // --- Nuevas props para selección ---
  checkpointData: Record<string, string[]>;
  selectedExperiment: string | null;
  setSelectedExperiment: (exp: string | null) => void;
  selectedCheckpoint: string | null;
  setSelectedCheckpoint: (ckpt: string | null) => void;
  loadSimulation: (path: string | null) => void;
  // ---------------------------------

  status: string;
  stepCount: number;
  expName: string;
  setExpName: (name: string) => void;
  trainModelType: string;
  setTrainModelType: (type: string) => void;
  hiddenChannels: number;
  setHiddenChannels: (channels: number) => void;
  trainingGridSize: number;
  setTrainingGridSize: (size: number) => void;
  learningRate: number;
  setLearningRate: (rate: number) => void;
  episodes: number;
  setEpisodes: (count: number) => void;
  startTraining: () => void;
  isTrainingLoading: boolean;
  stopTraining: () => void;
  refreshCheckpoints: () => void;
  isRefreshingCheckpoints: boolean;
}

const LabSider: React.FC<LabSiderProps> = (props) => {
  const experimentOptions = Object.keys(props.checkpointData);
  const checkpointOptions = props.selectedExperiment ? props.checkpointData[props.selectedExperiment] || [] : [];

  const mainLinks = [
    { icon: IconBulb, label: 'Estado Simulación', value: props.status, color: props.status === 'running' ? 'green' : 'gray' },
    { icon: IconCheckbox, label: 'Paso Actual', value: props.stepCount.toString() },
  ];

  const mainLinkItems = mainLinks.map((link) => (
    <UnstyledButton key={link.label} className={classes.mainLink}>
      <div className={classes.mainLinkInner}>
        <Tooltip label={link.label} position="right" withArrow>
          <link.icon size={20} className={classes.mainLinkIcon} stroke={1.5} />
        </Tooltip>
        {!props.collapsed && <span>{link.label}</span>}
      </div>
      {!props.collapsed && link.value && (
        <Badge size="sm" variant="filled" className={classes.mainLinkBadge} color={link.color}>
          {link.value}
        </Badge>
      )}
    </UnstyledButton>
  ));

  return (
    <div className={classes.navbarWrapper}>
      <nav className={classes.navbar}>
        <div className={classes.section}>
          <Group className={classes.collectionsHeader} justify="space-between" p="xs">
            {!props.collapsed && <Text size="xs" fw={500} c="dimmed">Simulador</Text>}
            <Tooltip label="Refrescar Experimentos" withArrow position="right">
              <ActionIcon variant="default" size={18} onClick={props.refreshCheckpoints} loading={props.isRefreshingCheckpoints}>
                <IconRefresh size={12} stroke={1.5} />
              </ActionIcon>
            </Tooltip>
          </Group>
          
          {!props.collapsed && (
            <Stack p="xs" gap="sm">
              <Select
                label="Experimento"
                placeholder="Selecciona un experimento"
                value={props.selectedExperiment}
                onChange={props.setSelectedExperiment}
                data={experimentOptions}
                searchable
              />
              <Select
                label="Checkpoint"
                placeholder="Selecciona un checkpoint"
                value={props.selectedCheckpoint}
                onChange={props.setSelectedCheckpoint}
                data={checkpointOptions}
                disabled={!props.selectedExperiment}
                searchable
              />
              <Button
                fullWidth
                leftSection={<IconPlayerPlay size={14} />}
                disabled={!props.selectedCheckpoint}
                onClick={() => props.loadSimulation(props.selectedCheckpoint)}
              >
                Cargar en Simulador
              </Button>
            </Stack>
          )}
        </div>

        <Divider />

        <div className={classes.section}>
          <div className={classes.mainLinks}>{mainLinkItems}</div>
        </div>

        {!props.collapsed && (
          <div className={classes.section}>
            <Fieldset legend="Laboratorio de Entrenamiento">
              <Stack p="xs" gap="sm">
                <TextInput size="xs" label="Nuevo Experimento" value={props.expName} onChange={(e) => props.setExpName(e.currentTarget.value)} />
                <Select
                  size="xs"
                  label="Modelo"
                  value={props.trainModelType}
                  onChange={(value) => props.setTrainModelType(value || 'unet')}
                  data={['unet', 'unet_unitary', 'mlp', 'deep_qca', 'SNN_UNET']}
                />
                <NumberInput size="xs" label="Canales Ocultos" value={props.hiddenChannels} onChange={(val) => props.setHiddenChannels(Number(val))} />
                <NumberInput size="xs" label="Tamaño de Grilla" value={props.trainingGridSize} onChange={(val) => props.setTrainingGridSize(Number(val))} />
                <NumberInput size="xs" label="Tasa de Aprendizaje" value={props.learningRate} onChange={(val) => props.setLearningRate(Number(val))} step={0.0000001} decimalScale={8} />
                <NumberInput size="xs" label="Episodios" value={props.episodes} onChange={(val) => props.setEpisodes(Number(val))} />
                <Button size="xs" onClick={props.startTraining} loading={props.isTrainingLoading} mt="md">Iniciar Entrenamiento</Button>
                <Button size="xs" onClick={props.stopTraining} color="red">Detener</Button>
              </Stack>
            </Fieldset>
          </div>
        )}
      </nav>
      <div className={classes.footer}>
        <Center>
          <ActionIcon onClick={props.toggle} variant="default" size="lg">
            {props.collapsed ? <IconArrowRight size={18} /> : <IconArrowLeft size={18} />}
          </ActionIcon>
        </Center>
      </div>
    </div>
  );
};

export default LabSider;


