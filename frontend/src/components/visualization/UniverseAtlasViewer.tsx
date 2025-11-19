// frontend/src/components/UniverseAtlasViewer.tsx
import { useEffect, useRef, useState } from 'react';
import { Box, Paper, Text, Button, NumberInput, Group, Stack, Badge, Tooltip } from '@mantine/core';
import { IconChartScatter, IconTrash, IconSettings, IconX } from '@tabler/icons-react';
import { useWebSocket } from '../../hooks/useWebSocket';

interface UniverseAtlasData {
    coords: number[][];
    timesteps: number[];
    metrics?: {
        spread: number;
        density: number;
        n_points: number;
    };
    error?: string;
    n_snapshots?: number;
}

export function UniverseAtlasViewer() {
    const canvasRef = useRef<HTMLCanvasElement>(null);
    const { sendCommand, analysisStatus, analysisType } = useWebSocket();
    const [data, setData] = useState<UniverseAtlasData | null>(null);
    const [compressionDim, setCompressionDim] = useState(64);
    const [perplexity, setPerplexity] = useState(30);
    const [nIter, setNIter] = useState(1000);
    
    // Usar el estado global de análisis
    const isAnalyzing = analysisStatus === 'running' && analysisType === 'universe_atlas';

    // Escuchar mensajes de análisis
    const { ws } = useWebSocket();
    
    useEffect(() => {
        if (!ws) return;
        
        const handleMessage = (event: MessageEvent) => {
            try {
                const message = JSON.parse(event.data);
                if (message.type === 'analysis_universe_atlas') {
                    setData(message.payload);
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
        // Usar reduce en lugar de spread operator para evitar stack overflow con arrays grandes
        const minX = xs.reduce((a, b) => Math.min(a, b), Infinity);
        const maxX = xs.reduce((a, b) => Math.max(a, b), -Infinity);
        const minY = ys.reduce((a, b) => Math.min(a, b), Infinity);
        const maxY = ys.reduce((a, b) => Math.max(a, b), -Infinity);

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

        // Dibujar puntos con gradiente temporal
        const n = coords.length;
        coords.forEach((coord, i) => {
            const x = padding + (coord[0] - minX) * scaleX;
            const y = height - padding - (coord[1] - minY) * scaleY;

            // Color basado en el tiempo (del azul al rojo)
            const t = i / n;
            const r = Math.floor(255 * t);
            const g = Math.floor(255 * (1 - t));
            const b = Math.floor(255 * (1 - Math.abs(t - 0.5) * 2));

            ctx.fillStyle = `rgb(${r}, ${g}, ${b})`;
            ctx.beginPath();
            ctx.arc(x, y, 4, 0, Math.PI * 2);
            ctx.fill();

            // Conectar con el punto anterior (trazar la trayectoria)
            if (i > 0) {
                const prevCoord = coords[i - 1];
                const prevX = padding + (prevCoord[0] - minX) * scaleX;
                const prevY = height - padding - (prevCoord[1] - minY) * scaleY;

                ctx.strokeStyle = `rgba(${r}, ${g}, ${b}, 0.3)`;
                ctx.lineWidth = 1;
                ctx.beginPath();
                ctx.moveTo(prevX, prevY);
                ctx.lineTo(x, y);
                ctx.stroke();
            }
        });

        // Dibujar el primer y último punto de forma destacada
        if (n > 0) {
            const first = coords[0];
            const firstX = padding + (first[0] - minX) * scaleX;
            const firstY = height - padding - (first[1] - minY) * scaleY;

            ctx.fillStyle = '#00ff00'; // Verde para inicio
            ctx.beginPath();
            ctx.arc(firstX, firstY, 8, 0, Math.PI * 2);
            ctx.fill();

            const last = coords[n - 1];
            const lastX = padding + (last[0] - minX) * scaleX;
            const lastY = height - padding - (last[1] - minY) * scaleY;

            ctx.fillStyle = '#ff0000'; // Rojo para fin
            ctx.beginPath();
            ctx.arc(lastX, lastY, 8, 0, Math.PI * 2);
            ctx.fill();
        }

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
        setData(null);
        sendCommand('analysis', 'universe_atlas', {
            compression_dim: compressionDim,
            perplexity: perplexity,
            n_iter: nIter
        });
    };

    const handleCancel = () => {
        sendCommand('analysis', 'cancel', {});
    };

    const handleClearSnapshots = () => {
        sendCommand('analysis', 'clear_snapshots', {});
    };

    return (
        <Paper p="md" withBorder>
            <Stack gap="md">
                <Group justify="space-between">
                    <Group gap="xs">
                        <IconChartScatter size={20} />
                        <Text fw={600}>Atlas del Universo</Text>
                    </Group>
                    <Tooltip label="Elimina todos los snapshots almacenados">
                        <Button
                            size="xs"
                            variant="subtle"
                            color="red"
                            leftSection={<IconTrash size={14} />}
                            onClick={handleClearSnapshots}
                        >
                            Limpiar
                        </Button>
                    </Tooltip>
                </Group>

                <Text size="sm" c="dimmed">
                    Visualiza la evolución temporal del universo usando t-SNE.
                    Si ves "continentes" separados, tu Ley M ha creado múltiples fases distintas.
                </Text>

                <Stack gap="xs">
                    <NumberInput
                        label="Dimensión de compresión"
                        value={compressionDim}
                        onChange={(val) => setCompressionDim(Number(val) || 64)}
                        min={16}
                        max={256}
                        step={16}
                        description="Dimensión del vector comprimido"
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

                <Group gap="xs">
                <Button
                        flex={1}
                    onClick={handleAnalyze}
                    loading={isAnalyzing}
                        disabled={isAnalyzing}
                    leftSection={<IconChartScatter size={16} />}
                >
                    {isAnalyzing ? 'Analizando...' : 'Analizar Evolución Temporal'}
                </Button>
                    {isAnalyzing && (
                        <Button
                            color="red"
                            variant="light"
                            onClick={handleCancel}
                            leftSection={<IconX size={16} />}
                        >
                            Cancelar
                        </Button>
                    )}
                </Group>

                {data?.error && (
                    <Paper p="sm" bg="red.1" withBorder>
                        <Text size="sm" c="red">{data.error}</Text>
                        {data.n_snapshots !== undefined && (
                            <Text size="xs" c="red" mt="xs">
                                Snapshots disponibles: {data.n_snapshots}
                            </Text>
                        )}
                    </Paper>
                )}

                {data?.metrics && (
                    <Group gap="xs">
                        <Badge color="blue">Spread: {data.metrics.spread.toFixed(2)}</Badge>
                        <Badge color="green">Puntos: {data.metrics.n_points}</Badge>
                    </Group>
                )}

                {data && !data.error && (
                    <Box>
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
                            Verde: inicio | Rojo: fin | Gradiente: evolución temporal
                        </Text>
                    </Box>
                )}
            </Stack>
        </Paper>
    );
}

