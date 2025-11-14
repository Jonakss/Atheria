import { Paper, Title, Text, Box } from '@mantine/core';
import { BarChart } from '@mantine/charts';

interface HistogramPanelProps {
  histogramData: any[];
}

const HistogramPanel: React.FC<HistogramPanelProps> = ({ histogramData }) => {
  if (!histogramData || histogramData.length === 0) {
    return (
      <Paper withBorder shadow="sm" p="xs" mt="md">
        <Title order={5}>Activación de Canales</Title>
        <Text size="xs" c="dimmed" ta="center" mt="md">Esperando datos de la simulación...</Text>
      </Paper>
    );
  }

  return (
    <Paper withBorder shadow="sm" p="xs" mt="md">
      <Title order={5}>Activación de Canales</Title>
      <Text size="xs" c="dimmed">Promedio de activación para cada canal del vector de estado.</Text>
      <Box style={{ height: 200, marginTop: '10px' }}>
        <BarChart
          h="100%"
          data={histogramData}
          dataKey="channel"
          series={[{ name: 'activation', color: 'blue.6' }]}
          tickLine="y"
        />
      </Box>
    </Paper>
  );
};

export default HistogramPanel;
