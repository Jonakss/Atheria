import { Grid, Paper, Title, Text, ScrollArea, Box } from '@mantine/core';
import { LineChart } from '@mantine/charts';
import PanZoomCanvas from './PanZoomCanvas';

interface MetricsDashboardProps {
  panelsVisible: boolean;
  imageData: string | null;
  sendCommand: (command: any) => void;
  metrics: any;
  simConfig: any;
  trainingHistory: any[];
  trainingLog: string;
}

const MetricsDashboard: React.FC<MetricsDashboardProps> = (props) => {
  const chartData = props.trainingHistory.map(item => ({
    episode: item.episode,
    Loss: item.Loss,
    R_Quietud: item.R_Quietud,
    R_Complejidad_Localizada: item.R_Complejidad_Localizada,
  }));

  return (
    <Grid gutter="xs">
      <Grid.Col span={props.panelsVisible ? 8 : 12}>
        <PanZoomCanvas imageData={props.imageData} sendCommand={props.sendCommand} />
      </Grid.Col>
      
      {props.panelsVisible && (
        <Grid.Col span={4}>
          <Paper withBorder shadow="sm" p="xs" style={{ height: 'calc(100vh - 80px)', display: 'flex', flexDirection: 'column' }}>
            <Title order={5}>Métricas de Simulación</Title>
            <Text size="xs" c="dimmed">Valores actuales de la simulación en vivo.</Text>
            <Grid mt="md">
              <Grid.Col span={6}><Text size="sm">Estabilidad (L2):</Text></Grid.Col>
              <Grid.Col span={6}><Text size="sm" c="blue">{props.metrics?.stability_l2?.toFixed(6)}</Text></Grid.Col>
              <Grid.Col span={6}><Text size="sm">Entropía:</Text></Grid.Col>
              <Grid.Col span={6}><Text size="sm" c="blue">{props.metrics?.entropy?.toFixed(6)}</Text></Grid.Col>
              <Grid.Col span={6}><Text size="sm">Complejidad (LZ):</Text></Grid.Col>
              <Grid.Col span={6}><Text size="sm" c="blue">{props.metrics?.complexity_lz?.toFixed(6)}</Text></Grid.Col>
            </Grid>
            
            <Title order={5} mt="lg">Evolución del Entrenamiento</Title>
            <Box style={{ flex: 1, minHeight: 200 }}>
              <LineChart
                h="100%"
                data={chartData}
                dataKey="episode"
                series={[
                  { name: 'Loss', color: 'red.6' },
                  { name: 'R_Quietud', color: 'blue.6' },
                  { name: 'R_Complejidad_Localizada', color: 'green.6' },
                ]}
                curveType="monotone"
              />
            </Box>

            <Title order={5} mt="lg">Log de Entrenamiento</Title>
            <ScrollArea style={{ flex: 1, marginTop: '10px', border: '1px solid #ccc', borderRadius: '4px', padding: '10px', minHeight: 100 }}>
              <pre style={{ margin: 0, whiteSpace: 'pre-wrap', wordBreak: 'break-all' }}>
                <code>{props.trainingLog}</code>
              </pre>
            </ScrollArea>
          </Paper>
        </Grid.Col>
      )}
    </Grid>
  );
};

export default MetricsDashboard;
