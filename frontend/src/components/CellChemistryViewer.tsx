// frontend/src/components/CellChemistryViewer.tsx
import { useEffect, useRef, useState } from 'react';
import { Box, Paper, Text, Button, NumberInput, Group, Stack, Badge } from '@mantine/core';
import { IconAtom, IconRefresh } from '@tabler/icons-react';
import { useWebSocket } from '../hooks/useWebSocket';

interface CellChemistryData {
    coords: number[][];
    cell_indices: number[][];
    error?: string;
}

export function CellChemistryViewer() {
    const canvasRef = useRef<HTMLCanvasElement>(null);
    const { sendCommand } = useWebSocket();
    const [data, setData] = useState<CellChemistryData | null>(null);
    const [nSamples, setNSamples] = useState(10000);
    const [perplexity, setPerplexity] = useState(30);
    const [nIter, setNIter] = useState(1000);
    const [isAnalyzing, setIsAnalyzing] = useState(false);

    // Escuchar mensajes de análisis
    const { ws } = useWebSocket();
    
    useEffect(() => {
        if (!ws) return;
        
        const handleMessage = (event: MessageEvent) => {
            try {
                const message = JSON.parse(event.data);
                if (message.type === 'analysis_cell_chemistry') {
                    setData(message.payload);
                    setIsAnalyzing(false);
                }
            } catch (e) {
                // Ignorar mensajes que no son del análisis
            }
        };

        ws.addEventListener('message', handleMessage);
        return () => ws.removeEventListener('message', handleMessage);
    }, [ws]);

    // Renderizar gráfico cuando los datos cambien
    useEffect(() => {
        if (!data || !canvasRef.current || data.error) return;

        const canvas = canvasRef.current;
        const ctx = canvas.getContext('2d');
        if (!ctx) return;

        const width = canvas.width;
        const height = canvas.height;
        const padding = 40;

        // Limpiar canvas
        ctx.clearRect(0, 0, width, height);

        // Calcular rango de coordenadas
        const coords = data.coords;
        if (coords.length === 0) return;

        const xs = coords.map(c => c[0]);
        const ys = coords.map(c => c[1]);
        const minX = Math.min(...xs);
        const maxX = Math.max(...xs);
        const minY = Math.min(...ys);
        const maxY = Math.max(...ys);

        const rangeX = maxX - minX || 1;
        const rangeY = maxY - minY || 1;

        // Escalar coordenadas al canvas
        const scaleX = (width - 2 * padding) / rangeX;
        const scaleY = (height - 2 * padding) / rangeY;

        // Dibujar ejes
        ctx.strokeStyle = '#666';
        ctx.lineWidth = 1;
        ctx.beginPath();
        ctx.moveTo(padding, padding);
        ctx.lineTo(padding, height - padding);
        ctx.lineTo(width - padding, height - padding);
        ctx.stroke();

        // Agrupar células por densidad usando K-means simple (para colorear clusters)
        // Por simplicidad, usaremos un enfoque basado en densidad
        const cellColors = coords.map((coord, i) => {
            // Calcular densidad local (número de vecinos cercanos)
            const neighbors = coords.filter((other, j) => {
                if (i === j) return false;
                const dx = coord[0] - other[0];
                const dy = coord[1] - other[1];
                const dist = Math.sqrt(dx * dx + dy * dy);
                return dist < (rangeX + rangeY) / 40; // Radio de vecindad adaptativo
            }).length;

            // Mapear densidad a color (verde para baja densidad, azul para media, rojo para alta)
            const density = Math.min(neighbors / 20, 1); // Normalizar
            if (density < 0.33) {
                return `rgb(0, ${Math.floor(255 * density * 3)}, 0)`; // Verde
            } else if (density < 0.67) {
                const t = (density - 0.33) / 0.34;
                return `rgb(0, ${Math.floor(255 * (1 - t))}, ${Math.floor(255 * t)})`; // Verde -> Azul
            } else {
                const t = (density - 0.67) / 0.33;
                return `rgb(${Math.floor(255 * t)}, 0, ${Math.floor(255 * (1 - t))})`; // Azul -> Rojo
            }
        });

        // Dibujar puntos
        coords.forEach((coord, i) => {
            const x = padding + (coord[0] - minX) * scaleX;
            const y = height - padding - (coord[1] - minY) * scaleY;

            ctx.fillStyle = cellColors[i];
            ctx.beginPath();
            ctx.arc(x, y, 2, 0, Math.PI * 2);
            ctx.fill();
        });

        // Etiquetas de ejes
        ctx.fillStyle = '#aaa';
        ctx.font = '12px sans-serif';
        ctx.textAlign = 'center';
        ctx.fillText('t-SNE Dimensión 1', width / 2, height - 10);
        ctx.save();
        ctx.translate(15, height / 2);
        ctx.rotate(-Math.PI / 2);
        ctx.fillText('t-SNE Dimensión 2', 0, 0);
        ctx.restore();
    }, [data]);

    const handleAnalyze = () => {
        setIsAnalyzing(true);
        setData(null);
        sendCommand('analysis', 'cell_chemistry', {
            n_samples: nSamples,
            perplexity: perplexity,
            n_iter: nIter
        });
    };

    return (
        <Paper p="md" withBorder>
            <Stack gap="md">
                <Group justify="space-between">
                    <Group gap="xs">
                        <IconAtom size={20} />
                        <Text fw={600}>Mapa Químico</Text>
                    </Group>
                </Group>

                <Text size="sm" c="dimmed">
                    Visualiza los "tipos" de células en el estado actual usando t-SNE.
                    Si ves "cúmulos" separados, tu Ley M ha aprendido a agrupar células en tipos distintos.
                </Text>

                <Stack gap="xs">
                    <NumberInput
                        label="Muestras máximas"
                        value={nSamples}
                        onChange={(val) => setNSamples(Number(val) || 10000)}
                        min={1000}
                        max={50000}
                        step={1000}
                        description="Número de células a analizar (para eficiencia)"
                    />
                    <NumberInput
                        label="Perplexity"
                        value={perplexity}
                        onChange={(val) => setPerplexity(Number(val) || 30)}
                        min={5}
                        max={50}
                        description="Balance local/global en t-SNE"
                    />
                    <NumberInput
                        label="Iteraciones"
                        value={nIter}
                        onChange={(val) => setNIter(Number(val) || 1000)}
                        min={250}
                        max={5000}
                        step={250}
                        description="Número de iteraciones de t-SNE"
                    />
                </Stack>

                <Button
                    fullWidth
                    onClick={handleAnalyze}
                    loading={isAnalyzing}
                    leftSection={<IconAtom size={16} />}
                >
                    {isAnalyzing ? 'Analizando...' : 'Analizar Tipos de Células'}
                </Button>

                {data?.error && (
                    <Paper p="sm" bg="red.1" withBorder>
                        <Text size="sm" c="red">{data.error}</Text>
                    </Paper>
                )}

                {data && !data.error && (
                    <Box>
                        <Group gap="xs" mb="xs">
                            <Badge color="green">Células: {data.coords.length}</Badge>
                        </Group>
                        <canvas
                            ref={canvasRef}
                            width={600}
                            height={400}
                            style={{
                                width: '100%',
                                maxWidth: '600px',
                                height: '400px',
                                border: '1px solid #ddd',
                                borderRadius: '4px',
                                backgroundColor: '#1a1a1a'
                            }}
                        />
                        <Text size="xs" c="dimmed" mt="xs">
                            Color por densidad: Verde (baja) → Azul (media) → Rojo (alta)
                        </Text>
                    </Box>
                )}
            </Stack>
        </Paper>
    );
}

