// frontend/src/components/History3DViewer.tsx
import { useRef, useEffect, useState } from 'react';
import { Paper, Stack, Text, Group, Tooltip, ActionIcon, Slider, Button, NumberInput, Badge, Box } from '@mantine/core';
import { useWebSocket } from '../../hooks/useWebSocket';
import { IconCube, IconHelpCircle, IconPlayerPlay, IconPlayerPause } from '@tabler/icons-react';
import * as THREE from 'three';
import { OrbitControls } from 'three/examples/jsm/controls/OrbitControls.js';

interface HistoryFrame {
    step: number;
    timestamp: string;
    map_data: number[][];
}

export function History3DViewer() {
    const containerRef = useRef<HTMLDivElement>(null);
    const sceneRef = useRef<THREE.Scene | null>(null);
    const rendererRef = useRef<THREE.WebGLRenderer | null>(null);
    const cameraRef = useRef<THREE.PerspectiveCamera | null>(null);
    const animationFrameRef = useRef<number | null>(null);
    const meshesRef = useRef<THREE.Mesh[]>([]);
    const controlsRef = useRef<OrbitControls | null>(null);
    const { simData } = useWebSocket();
    const [historyFrames, setHistoryFrames] = useState<HistoryFrame[]>([]);
    const [currentTimeSlice, setCurrentTimeSlice] = useState(0);
    const [maxSlices, setMaxSlices] = useState(50);
    const [opacity, setOpacity] = useState(0.3);
    const [autoRotate, setAutoRotate] = useState(true);
    const [loadedHistoryFrames, setLoadedHistoryFrames] = useState<HistoryFrame[]>([]);
    const [useLoadedHistory, setUseLoadedHistory] = useState(false);
    
    // Escuchar cuando se carga un historial desde archivo
    useEffect(() => {
        const handleHistoryFileLoaded = (event: CustomEvent) => {
            const { frames } = event.detail;
            if (frames && Array.isArray(frames)) {
                setLoadedHistoryFrames(frames);
                setUseLoadedHistory(true);
            }
        };
        
        window.addEventListener('history_file_loaded', handleHistoryFileLoaded as EventListener);
        return () => {
            window.removeEventListener('history_file_loaded', handleHistoryFileLoaded as EventListener);
        };
    }, []);
    
    // Acumular frames del historial en tiempo real (solo si no estamos usando historial cargado)
    useEffect(() => {
        if (useLoadedHistory) return; // No acumular si estamos usando historial cargado
        
        if (simData?.map_data && simData?.step !== undefined) {
            setHistoryFrames(prev => {
                // Evitar duplicados
                const exists = prev.some(f => f.step === simData.step);
                if (exists) return prev;
                
                const newFrame: HistoryFrame = {
                    step: simData.step ?? 0,
                    timestamp: new Date().toISOString(),
                    map_data: simData.map_data as number[][]
                };
                
                // Mantener solo los últimos maxSlices frames
                const updated = [...prev, newFrame].slice(-maxSlices);
                return updated;
            });
        }
    }, [simData?.step, simData?.map_data, maxSlices, useLoadedHistory]);
    
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
        camera.position.set(4, 4, 4);
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
        directionalLight1.position.set(5, 8, 5);
        scene.add(directionalLight1);
        
        // Luz de relleno
        const directionalLight2 = new THREE.DirectionalLight(0x4488ff, 0.4);
        directionalLight2.position.set(-5, 3, -5);
        scene.add(directionalLight2);
        
        // Luz desde abajo para mejor contraste
        const directionalLight3 = new THREE.DirectionalLight(0xffffff, 0.2);
        directionalLight3.position.set(0, -5, 0);
        scene.add(directionalLight3);
        
        // Controles de órbita
        const controls = new OrbitControls(camera, renderer.domElement);
        controls.enableDamping = true;
        controls.dampingFactor = 0.05;
        controls.enableZoom = true;
        controls.enablePan = true;
        controls.autoRotate = autoRotate;
        controls.autoRotateSpeed = 1.0;
        controlsRef.current = controls;
        
        // Ejes de referencia
        const axesHelper = new THREE.AxesHelper(2);
        scene.add(axesHelper);
        
        // Grid helper (plano XY)
        const gridHelper = new THREE.GridHelper(4, 20, 0x444444, 0x222222);
        scene.add(gridHelper);
        
        // El loop de animación se maneja en un useEffect separado
        
        // Manejar resize
        const handleResize = () => {
            if (!containerRef.current || !cameraRef.current || !rendererRef.current) return;
            const width = containerRef.current.clientWidth;
            const height = containerRef.current.clientHeight;
            cameraRef.current.aspect = width / height;
            cameraRef.current.updateProjectionMatrix();
            rendererRef.current.setSize(width, height);
        };
        window.addEventListener('resize', handleResize);
        
        return () => {
            window.removeEventListener('resize', handleResize);
            if (animationFrameRef.current) {
                cancelAnimationFrame(animationFrameRef.current);
            }
            if (rendererRef.current && containerRef.current) {
                try {
                    containerRef.current.removeChild(rendererRef.current.domElement);
                } catch (e) {
                    // Ignorar si ya fue removido
                }
                rendererRef.current.dispose();
            }
        };
    }, []);
    
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
    
    // Actualizar visualización 3D cuando cambian los frames del historial
    useEffect(() => {
        // Usar historial cargado o historial en tiempo real
        const framesToRender = useLoadedHistory ? loadedHistoryFrames : historyFrames;
        
        if (!sceneRef.current || framesToRender.length === 0) return;
        
        // Limpiar meshes anteriores
        meshesRef.current.forEach(mesh => {
            sceneRef.current?.remove(mesh);
            if (mesh.geometry) mesh.geometry.dispose();
            if (mesh.material instanceof THREE.Material) {
                if (Array.isArray(mesh.material)) {
                    mesh.material.forEach(mat => mat.dispose());
                } else {
                    mesh.material.dispose();
                }
            }
        });
        meshesRef.current = [];
        
        try {
            // Obtener dimensiones del primer frame
            const firstFrame = framesToRender[0];
            if (!firstFrame?.map_data || !Array.isArray(firstFrame.map_data)) return;
            
            const H = firstFrame.map_data.length;
            const W = firstFrame.map_data[0]?.length || 0;
            if (W === 0) return;
            
            // Normalizar coordenadas espaciales
            const scaleX = 2 / W;
            const scaleY = 2 / H;
            const scaleZ = 4 / Math.max(framesToRender.length, 1); // Espaciado en Z (tiempo)
            
            // Crear un mesh para cada frame (slice temporal)
            framesToRender.forEach((frame, timeIndex) => {
                if (!frame.map_data || !Array.isArray(frame.map_data)) return;
                
                // Crear geometría de plano para este slice temporal
                const geometry = new THREE.PlaneGeometry(2, 2, W - 1, H - 1);
                const positions = geometry.attributes.position;
                
                // Ajustar posiciones y alturas basadas en map_data
                for (let y = 0; y < H; y++) {
                    for (let x = 0; x < W; x++) {
                        const vertexIndex = y * W + x;
                        if (vertexIndex >= positions.count) continue;
                        
                        const value = frame.map_data[y]?.[x];
                        if (typeof value !== 'number' || isNaN(value)) continue;
                        
                        // Posición X (normalizada)
                        positions.setX(vertexIndex, (x / W) * 2 - 1);
                        // Posición Y (altura basada en valor)
                        positions.setY(vertexIndex, (value - 0.5) * 2);
                        // Posición Z (tiempo - este slice específico)
                        positions.setZ(vertexIndex, (timeIndex - framesToRender.length / 2) * scaleZ);
                    }
                }
                
                geometry.attributes.position.needsUpdate = true;
                geometry.computeVertexNormals();
                
                // Material mejorado con mejor iluminación y colores más suaves
                const timeRatio = timeIndex / Math.max(framesToRender.length - 1, 1);
                // Gradiente más suave: azul -> cyan -> verde -> amarillo
                let hue = 0.5 - timeRatio * 0.3; // 0.5 (cyan) -> 0.2 (verde)
                if (hue < 0) hue += 1; // Wrap around
                
                const material = new THREE.MeshStandardMaterial({
                    color: new THREE.Color().setHSL(hue, 0.8, 0.5 + timeRatio * 0.2),
                    metalness: 0.1,
                    roughness: 0.7,
                    transparent: true,
                    opacity: opacity,
                    side: THREE.DoubleSide,
                    wireframe: false,
                    emissive: new THREE.Color().setHSL(hue, 0.5, 0.1) // Brillo sutil
                });
                
                const mesh = new THREE.Mesh(geometry, material);
                meshesRef.current.push(mesh);
                sceneRef.current?.add(mesh);
            });
            
        } catch (error) {
            console.error('Error actualizando visualización 3D:', error);
        }
    }, [historyFrames, loadedHistoryFrames, useLoadedHistory, opacity]);
    
    return (
        <Paper p="md" withBorder style={{ backgroundColor: 'var(--mantine-color-dark-7)', height: '100%', display: 'flex', flexDirection: 'column' }}>
            <Stack gap="sm" style={{ flex: 1, minHeight: 0 }}>
                <Group justify="space-between">
                    <Group gap="xs">
                        <IconCube size={18} />
                        <Text size="sm" fw={600}>Evolución Temporal 3D</Text>
                        <Tooltip 
                            label="Visualización 3D de la evolución temporal. Cada slice transparente representa un momento en el tiempo. Eje Z = tiempo, altura = valor del estado. Útil para ver patrones temporales, ondas, y movimiento de estructuras."
                            multiline
                            style={{ maxWidth: 300 }}
                            withArrow
                        >
                            <ActionIcon size="xs" variant="subtle" color="gray">
                                <IconHelpCircle size={14} />
                            </ActionIcon>
                        </Tooltip>
                    </Group>
                    <Badge size="sm" variant="light" color="blue">
                        {useLoadedHistory ? loadedHistoryFrames.length : historyFrames.length} frames
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
                    <Text size="xs" c="dimmed" mb="xs">Opacidad de capas: {opacity.toFixed(2)}</Text>
                    <Slider
                        value={opacity}
                        onChange={setOpacity}
                        min={0.1}
                        max={1.0}
                        step={0.05}
                        marks={[
                            { value: 0.1, label: '0.1' },
                            { value: 0.5, label: '0.5' },
                            { value: 1.0, label: '1.0' }
                        ]}
                    />
                </Box>
                
                <Box>
                    <Text size="xs" c="dimmed" mb="xs">Máximo de slices: {maxSlices}</Text>
                    <Slider
                        value={maxSlices}
                        onChange={setMaxSlices}
                        min={10}
                        max={200}
                        step={10}
                        marks={[
                            { value: 10, label: '10' },
                            { value: 50, label: '50' },
                            { value: 100, label: '100' },
                            { value: 200, label: '200' }
                        ]}
                    />
                </Box>
                
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
                    Eje X (rojo): Espacial X, Eje Y (verde): Valor/Altura, Eje Z (azul): Tiempo. 
                    {(useLoadedHistory ? loadedHistoryFrames : historyFrames).length > 0 && 
                        ` Mostrando ${(useLoadedHistory ? loadedHistoryFrames : historyFrames).length} frames apilados.`}
                </Text>
            </Stack>
        </Paper>
    );
}

