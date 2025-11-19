// frontend/src/components/Complex3DViewer.tsx
import { useRef, useEffect, useState } from 'react';
import { Paper, Stack, Text, Group, Tooltip, ActionIcon, Slider, Button, Badge, Box, Select, NumberInput, Collapse } from '@mantine/core';
import { useWebSocket } from '../../hooks/useWebSocket';
import { IconCube, IconHelpCircle, IconPlayerPlay, IconPlayerPause, IconInfoCircle } from '@tabler/icons-react';
import * as THREE from 'three';
import { OrbitControls } from 'three/examples/jsm/controls/OrbitControls.js';

interface Complex3DFrame {
    step: number;
    timestamp: string;
    real: number[][];
    imag: number[][];
}

export function Complex3DViewer() {
    const containerRef = useRef<HTMLDivElement>(null);
    const sceneRef = useRef<THREE.Scene | null>(null);
    const rendererRef = useRef<THREE.WebGLRenderer | null>(null);
    const cameraRef = useRef<THREE.PerspectiveCamera | null>(null);
    const animationFrameRef = useRef<number | null>(null);
    const pointsRef = useRef<THREE.Points | null>(null);
    const controlsRef = useRef<OrbitControls | null>(null);
    const { simData } = useWebSocket();
    const [historyFrames, setHistoryFrames] = useState<Complex3DFrame[]>([]);
    const [maxFrames, setMaxFrames] = useState<number>(100);
    const [pointSize, setPointSize] = useState(1.0);
    const [autoRotate, setAutoRotate] = useState(true);
    const [loadedHistoryFrames, setLoadedHistoryFrames] = useState<Complex3DFrame[]>([]);
    const [useLoadedHistory, setUseLoadedHistory] = useState(false);
    const [zAxisMode, setZAxisMode] = useState<'magnitude' | 'channel' | 'density' | 'phase'>('magnitude');
    const [selectedChannel, setSelectedChannel] = useState<number>(0);
    const [showChannelInfo, setShowChannelInfo] = useState(false);
    
    // Escuchar cuando se carga un historial desde archivo
    useEffect(() => {
        const handleHistoryFileLoaded = (event: CustomEvent) => {
            const { frames } = event.detail;
            if (frames && Array.isArray(frames)) {
                // Convertir frames del historial a formato Complex3DFrame
                const complexFrames: Complex3DFrame[] = frames
                    .filter((f: any) => f.complex_3d_data)
                    .map((f: any) => ({
                        step: f.step,
                        timestamp: f.timestamp || new Date().toISOString(),
                        real: f.complex_3d_data.real || [],
                        imag: f.complex_3d_data.imag || []
                    }));
                if (complexFrames.length > 0) {
                    setLoadedHistoryFrames(complexFrames);
                    setUseLoadedHistory(true);
                }
            }
        };
        
        window.addEventListener('history_file_loaded', handleHistoryFileLoaded as EventListener);
        return () => {
            window.removeEventListener('history_file_loaded', handleHistoryFileLoaded as EventListener);
        };
    }, []);
    
    // Acumular frames del historial en tiempo real
    useEffect(() => {
        if (useLoadedHistory) return;
        
        if (simData?.complex_3d_data && simData?.step !== undefined) {
            setHistoryFrames(prev => {
                // Evitar duplicados
                const exists = prev.some(f => f.step === simData.step);
                if (exists) return prev;
                
                const newFrame: Complex3DFrame = {
                    step: simData.step || 0,
                    timestamp: new Date().toISOString(),
                    real: (simData.complex_3d_data as any).real || [],
                    imag: (simData.complex_3d_data as any).imag || []
                };
                
                // Mantener solo los últimos maxFrames frames
                const updated = [...prev, newFrame].slice(-(maxFrames || 100));
                return updated;
            });
        }
    }, [simData?.step, simData?.complex_3d_data, maxFrames, useLoadedHistory]);
    
    // Inicializar Three.js
    useEffect(() => {
        if (!containerRef.current) return;
        
        // Inicializar escena
        const scene = new THREE.Scene();
        scene.background = new THREE.Color(0x0a0a0a);
        sceneRef.current = scene;
        
        // Cámara
        const camera = new THREE.PerspectiveCamera(
            75,
            containerRef.current.clientWidth / containerRef.current.clientHeight,
            0.1,
            1000
        );
        camera.position.set(3, 3, 3);
        camera.lookAt(0, 0, 0);
        cameraRef.current = camera;
        
        // Renderer
        const renderer = new THREE.WebGLRenderer({ antialias: true, alpha: true });
        renderer.setSize(containerRef.current.clientWidth, containerRef.current.clientHeight);
        renderer.setPixelRatio(window.devicePixelRatio);
        containerRef.current.appendChild(renderer.domElement);
        rendererRef.current = renderer;
        
        // Luces mejoradas
        const ambientLight = new THREE.AmbientLight(0xffffff, 0.5);
        scene.add(ambientLight);
        
        // Luz principal
        const directionalLight1 = new THREE.DirectionalLight(0xffffff, 0.9);
        directionalLight1.position.set(5, 5, 5);
        scene.add(directionalLight1);
        
        // Luz de relleno cálida
        const directionalLight2 = new THREE.DirectionalLight(0xffaa88, 0.3);
        directionalLight2.position.set(-5, 3, -5);
        scene.add(directionalLight2);
        
        // Luz desde abajo
        const directionalLight3 = new THREE.DirectionalLight(0x88aaff, 0.2);
        directionalLight3.position.set(0, -5, 0);
        scene.add(directionalLight3);
        
        // Ejes de referencia
        const axesHelper = new THREE.AxesHelper(2);
        scene.add(axesHelper);
        
        // Grid helper mejorado
        const gridHelper = new THREE.GridHelper(10, 20, 0x555555, 0x333333);
        scene.add(gridHelper);
        
        // Plano de referencia sutil
        const planeGeometry = new THREE.PlaneGeometry(10, 10);
        const planeMaterial = new THREE.MeshStandardMaterial({
            color: 0x1a1a1a,
            transparent: true,
            opacity: 0.2,
            side: THREE.DoubleSide
        });
        const plane = new THREE.Mesh(planeGeometry, planeMaterial);
        plane.rotation.x = -Math.PI / 2;
        plane.position.y = -2;
        scene.add(plane);
        
        // Controles de órbita
        const controls = new OrbitControls(camera, renderer.domElement);
        controls.enableDamping = true;
        controls.dampingFactor = 0.05;
        controls.enableZoom = true;
        controls.enablePan = true;
        controls.autoRotate = autoRotate;
        controls.autoRotateSpeed = 1.0;
        controlsRef.current = controls;
        
        // Limpiar al desmontar
        return () => {
            if (containerRef.current && renderer.domElement.parentNode) {
                containerRef.current.removeChild(renderer.domElement);
            }
            renderer.dispose();
        };
    }, []);
    
    // Actualizar visualización cuando cambian los frames
    useEffect(() => {
        if (!sceneRef.current || !cameraRef.current) return;
        
        const frames = useLoadedHistory ? loadedHistoryFrames : historyFrames;
        if (frames.length === 0) {
            // Limpiar puntos existentes
            if (pointsRef.current) {
                sceneRef.current.remove(pointsRef.current);
                pointsRef.current = null;
            }
            return;
        }
        
        try {
            // Limpiar puntos anteriores
            if (pointsRef.current) {
                sceneRef.current.remove(pointsRef.current);
            }
            
            // Crear geometría de puntos
            const geometry = new THREE.BufferGeometry();
            const positions: number[] = [];
            const colors: number[] = [];
            
            // Normalizar valores para el rango de visualización
            let minReal = Infinity, maxReal = -Infinity;
            let minImag = Infinity, maxImag = -Infinity;
            let minZ = Infinity, maxZ = -Infinity;
            
            // Primera pasada: encontrar rangos para X, Y y Z
            frames.forEach((frame) => {
                if (!frame.real || !frame.imag) return;
                const H = frame.real.length;
                const W = frame.real[0]?.length || 0;
                
                for (let i = 0; i < H; i++) {
                    for (let j = 0; j < W; j++) {
                        const real = frame.real[i]?.[j] || 0;
                        const imag = frame.imag[i]?.[j] || 0;
                        minReal = Math.min(minReal, real);
                        maxReal = Math.max(maxReal, real);
                        minImag = Math.min(minImag, imag);
                        maxImag = Math.max(maxImag, imag);
                        
                        // Calcular valor Z según el modo seleccionado
                        let zValue = 0;
                        if (zAxisMode === 'magnitude') {
                            zValue = Math.sqrt(real * real + imag * imag);
                        } else if (zAxisMode === 'density') {
                            zValue = real * real + imag * imag; // |psi|²
                        } else if (zAxisMode === 'phase') {
                            zValue = Math.atan2(imag, real); // Fase en radianes
                        } else if (zAxisMode === 'channel') {
                            // Por ahora, usar magnitud como fallback (necesitaríamos datos de canales individuales)
                            zValue = Math.sqrt(real * real + imag * imag);
                        }
                        minZ = Math.min(minZ, zValue);
                        maxZ = Math.max(maxZ, zValue);
                    }
                }
            });
            
            const rangeReal = maxReal - minReal || 1;
            const rangeImag = maxImag - minImag || 1;
            const rangeZ = maxZ - minZ || 1;
            const scale = 2.0; // Escala para el espacio 3D
            
            // Segunda pasada: crear puntos (solo usar el último frame, no todos los frames)
            const frame = frames[frames.length - 1]; // Usar solo el frame más reciente
            if (frame && frame.real && frame.imag) {
                const H = frame.real.length;
                const W = frame.real[0]?.length || 0;
                
                // Submuestrear para mejor rendimiento (cada 2 píxeles)
                const step = 2;
                
                for (let i = 0; i < H; i += step) {
                    for (let j = 0; j < W; j += step) {
                        const real = frame.real[i]?.[j] || 0;
                        const imag = frame.imag[i]?.[j] || 0;
                        
                        // Calcular valor Z según el modo
                        let zValue = 0;
                        if (zAxisMode === 'magnitude') {
                            zValue = Math.sqrt(real * real + imag * imag);
                        } else if (zAxisMode === 'density') {
                            zValue = real * real + imag * imag;
                        } else if (zAxisMode === 'phase') {
                            zValue = Math.atan2(imag, real);
                        } else if (zAxisMode === 'channel') {
                            zValue = Math.sqrt(real * real + imag * imag);
                        }
                        
                        // Normalizar a [-scale, scale]
                        const x = ((real - minReal) / rangeReal - 0.5) * scale * 2;
                        const y = ((imag - minImag) / rangeImag - 0.5) * scale * 2;
                        const z = ((zValue - minZ) / rangeZ - 0.5) * scale * 2;
                        
                        positions.push(x, y, z);
                        
                        // Color basado en magnitud (más informativo que tiempo)
                        const magnitude = Math.sqrt(real * real + imag * imag);
                        const normalizedMag = Math.min(1, magnitude / (Math.max(maxZ, 1)));
                        // Gradiente: azul (bajo) -> verde -> amarillo -> rojo (alto)
                        colors.push(normalizedMag, normalizedMag * 0.8, 1 - normalizedMag * 0.5);
                    }
                }
            }
            
            if (positions.length === 0) return;
            
            // Usar InstancedMesh para mejor rendimiento con muchos puntos
            const sphereGeometry = new THREE.SphereGeometry(pointSize * 0.02, 8, 8);
            const material = new THREE.MeshStandardMaterial({
                metalness: 0.2,
                roughness: 0.6,
                transparent: true,
                opacity: 0.85
            });
            
            const instancedMesh = new THREE.InstancedMesh(sphereGeometry, material, positions.length / 3);
            
            // Matrices y colores para instancias
            const matrix = new THREE.Matrix4();
            const color = new THREE.Color();
            
            for (let i = 0; i < positions.length; i += 3) {
                const x = positions[i];
                const y = positions[i + 1];
                const z = positions[i + 2];
                const instanceIndex = i / 3;
                
                // Posición
                matrix.makeTranslation(x, y, z);
                instancedMesh.setMatrixAt(instanceIndex, matrix);
                
                // Color basado en tiempo (mejor gradiente)
                const r = colors[i];
                const g = colors[i + 1];
                const b = colors[i + 2];
                // Mejorar gradiente: más suave y vibrante
                color.setRGB(
                    Math.pow(r, 0.8),  // Menos rojo al inicio
                    Math.pow(g, 0.9),  // Más verde
                    Math.pow(b, 0.8)   // Menos azul al final
                );
                instancedMesh.setColorAt(instanceIndex, color);
            }
            
            instancedMesh.instanceMatrix.needsUpdate = true;
            if (instancedMesh.instanceColor) {
                instancedMesh.instanceColor.needsUpdate = true;
            }
            
            sceneRef.current.add(instancedMesh);
            pointsRef.current = instancedMesh as any;
            
        } catch (error) {
            console.error('Error actualizando visualización 3D compleja:', error);
        }
    }, [historyFrames, loadedHistoryFrames, useLoadedHistory, pointSize, zAxisMode, selectedChannel]);
    
    // Actualizar auto-rotación de controles
    useEffect(() => {
        if (controlsRef.current) {
            controlsRef.current.autoRotate = autoRotate;
        }
    }, [autoRotate]);
    
    // Loop de animación
    useEffect(() => {
        if (!rendererRef.current || !sceneRef.current || !cameraRef.current) return;
        
        const animate = () => {
            if (controlsRef.current) {
                controlsRef.current.update();
            }
            
            if (rendererRef.current && sceneRef.current && cameraRef.current) {
                rendererRef.current.render(sceneRef.current, cameraRef.current);
            }
            animationFrameRef.current = requestAnimationFrame(animate);
        };
        
        animate();
        
        return () => {
            if (animationFrameRef.current) {
                cancelAnimationFrame(animationFrameRef.current);
            }
        };
    }, []);
    
    // Manejar resize
    useEffect(() => {
        const handleResize = () => {
            if (!containerRef.current || !rendererRef.current || !cameraRef.current) return;
            const width = containerRef.current.clientWidth;
            const height = containerRef.current.clientHeight;
            cameraRef.current.aspect = width / height;
            cameraRef.current.updateProjectionMatrix();
            rendererRef.current.setSize(width, height);
        };
        
        window.addEventListener('resize', handleResize);
        return () => window.removeEventListener('resize', handleResize);
    }, []);
    
    const frames = useLoadedHistory ? loadedHistoryFrames : historyFrames;
    
    return (
        <Paper p="md" withBorder style={{ backgroundColor: 'var(--mantine-color-dark-7)', height: '100%', display: 'flex', flexDirection: 'column' }}>
            <Stack gap="sm" style={{ flex: 1, minHeight: 0 }}>
                <Group justify="space-between">
                    <Group gap="xs">
                        <IconCube size={18} />
                        <Text size="sm" fw={600}>Espacio Complejo 3D</Text>
                        <Tooltip 
                            label="Visualización 3D del espacio complejo. Eje X = Parte Real, Eje Y = Parte Imaginaria, Eje Z = Tiempo. Cada punto representa un píxel en un momento del tiempo. Útil para ver la evolución del estado cuántico en el plano complejo."
                            multiline
                            w={300}
                            withArrow
                        >
                            <ActionIcon size="xs" variant="subtle" color="gray">
                                <IconHelpCircle size={14} />
                            </ActionIcon>
                        </Tooltip>
                    </Group>
                    <Badge size="sm" variant="light" color="blue">
                        {frames.length} frames
                    </Badge>
                </Group>
                
                <Group grow>
                    <Button
                        variant="light"
                        size="xs"
                        leftSection={autoRotate ? <IconPlayerPause size={14} /> : <IconPlayerPlay size={14} />}
                        onClick={() => setAutoRotate(!autoRotate)}
                    >
                        {autoRotate ? 'Pausar Rotación' : 'Rotar'}
                    </Button>
                    <Button
                        variant="light"
                        size="xs"
                        onClick={() => {
                            setHistoryFrames([]);
                            setLoadedHistoryFrames([]);
                            setUseLoadedHistory(false);
                        }}
                    >
                        Limpiar
                    </Button>
                </Group>
                
                {loadedHistoryFrames.length > 0 && (
                    <Group>
                        <Text size="xs" c="dimmed">
                            Usando historial cargado: {loadedHistoryFrames.length} frames
                        </Text>
                        <Button
                            variant="subtle"
                            size="xs"
                            onClick={() => setUseLoadedHistory(!useLoadedHistory)}
                        >
                            {useLoadedHistory ? 'Cambiar a tiempo real' : 'Usar historial cargado'}
                        </Button>
                    </Group>
                )}
                
                <Box>
                    <Text size="xs" c="dimmed" mb="xs">Tamaño de punto: {pointSize.toFixed(1)}</Text>
                    <Slider
                        value={pointSize}
                        onChange={setPointSize}
                        min={0.5}
                        max={5.0}
                        step={0.1}
                        marks={[
                            { value: 0.5, label: '0.5' },
                            { value: 2.5, label: '2.5' },
                            { value: 5.0, label: '5.0' }
                        ]}
                    />
                </Box>
                
                <Box>
                    <Text size="xs" c="dimmed" mb="xs">Máximo de frames: {maxFrames}</Text>
                    <Slider
                        value={maxFrames}
                        onChange={setMaxFrames}
                        min={10}
                        max={500}
                        step={10}
                        marks={[
                            { value: 10, label: '10' },
                            { value: 100, label: '100' },
                            { value: 250, label: '250' },
                            { value: 500, label: '500' }
                        ]}
                    />
                </Box>
                
                <Select
                    label="Eje Z (Altura)"
                    description="Qué representa el eje Z en la visualización"
                    value={zAxisMode}
                    onChange={(val) => val && setZAxisMode(val as any)}
                    data={[
                        { value: 'magnitude', label: 'Magnitud (|ψ|) - Distancia desde origen' },
                        { value: 'density', label: 'Densidad (|ψ|²) - Probabilidad' },
                        { value: 'phase', label: 'Fase (arg(ψ)) - Ángulo en plano complejo' },
                        { value: 'channel', label: 'Canal específico (requiere datos de canales)' }
                    ]}
                />
                
                {zAxisMode === 'channel' && (
                    <NumberInput
                        label="Canal"
                        description="Índice del canal a visualizar (0 a d_state-1)"
                        value={selectedChannel}
                        onChange={(val) => setSelectedChannel(Number(val) || 0)}
                        min={0}
                        max={7}
                    />
                )}
                
                <Button
                    variant="subtle"
                    size="xs"
                    leftSection={<IconInfoCircle size={14} />}
                    onClick={() => setShowChannelInfo(!showChannelInfo)}
                >
                    {showChannelInfo ? 'Ocultar' : 'Mostrar'} información de canales
                </Button>
                
                <Collapse in={showChannelInfo}>
                    <Paper p="xs" withBorder style={{ backgroundColor: 'var(--mantine-color-dark-6)' }}>
                        <Stack gap={4}>
                            <Text size="xs" fw={600}>Información sobre Canales:</Text>
                            <Text size="xs" c="dimmed">
                                El estado cuántico tiene <strong>d_state</strong> canales (típicamente 8).
                                Cada canal es un número complejo con parte real e imaginaria.
                            </Text>
                            <Text size="xs" c="dimmed">
                                <strong>Eje X (rojo):</strong> Parte Real promedio de todos los canales
                            </Text>
                            <Text size="xs" c="dimmed">
                                <strong>Eje Y (verde):</strong> Parte Imaginaria promedio de todos los canales
                            </Text>
                            <Text size="xs" c="dimmed">
                                <strong>Eje Z (azul):</strong> {zAxisMode === 'magnitude' ? 'Magnitud del estado (|ψ|)' :
                                                               zAxisMode === 'density' ? 'Densidad de probabilidad (|ψ|²)' :
                                                               zAxisMode === 'phase' ? 'Fase del estado (arg(ψ))' :
                                                               `Canal ${selectedChannel}`}
                            </Text>
                            <Text size="xs" c="dimmed" mt="xs">
                                <strong>Nota:</strong> Actualmente se promedian todos los canales. 
                                Para visualizar canales individuales, se necesitarían datos adicionales del backend.
                            </Text>
                        </Stack>
                    </Paper>
                </Collapse>
                
                <div
                    ref={containerRef}
                    style={{
                        width: '100%',
                        height: '100%',
                        minHeight: '600px',
                        border: '1px solid var(--mantine-color-dark-4)',
                        borderRadius: 'var(--mantine-radius-sm)',
                        position: 'relative',
                        overflow: 'hidden',
                        backgroundColor: '#0a0a0a'
                    }}
                />
                
                <Text size="xs" c="dimmed" style={{ fontStyle: 'italic' }}>
                    Eje X (rojo): Parte Real promedio | Eje Y (verde): Parte Imaginaria promedio | 
                    Eje Z (azul): {zAxisMode === 'magnitude' ? 'Magnitud (|ψ|)' :
                                  zAxisMode === 'density' ? 'Densidad (|ψ|²)' :
                                  zAxisMode === 'phase' ? 'Fase (arg(ψ))' :
                                  `Canal ${selectedChannel}`}
                    {frames.length > 0 && ` | Frame actual: ${frames[frames.length - 1]?.step || 0}`}
                </Text>
            </Stack>
        </Paper>
    );
}
