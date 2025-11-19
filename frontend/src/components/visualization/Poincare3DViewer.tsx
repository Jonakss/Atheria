// frontend/src/components/Poincare3DViewer.tsx
import { useRef, useEffect, useState } from 'react';
import { Paper, Stack, Text, Group, Tooltip, ActionIcon, Slider, Button, Badge, Box } from '@mantine/core';
import { useWebSocket } from '../hooks/useWebSocket';
import { IconCube, IconHelpCircle, IconPlayerPlay, IconPlayerPause } from '@tabler/icons-react';
import * as THREE from 'three';
import { OrbitControls } from 'three/examples/jsm/controls/OrbitControls.js';

export function Poincare3DViewer() {
    const containerRef = useRef<HTMLDivElement>(null);
    const sceneRef = useRef<THREE.Scene | null>(null);
    const rendererRef = useRef<THREE.WebGLRenderer | null>(null);
    const cameraRef = useRef<THREE.PerspectiveCamera | null>(null);
    const animationFrameRef = useRef<number | null>(null);
    const pointsRef = useRef<THREE.Points | null>(null);
    const controlsRef = useRef<OrbitControls | null>(null);
    const { simData } = useWebSocket();
    const [history, setHistory] = useState<number[][]>([]);
    const [maxPoints, setMaxPoints] = useState(1000);
    const [pointSize, setPointSize] = useState(2.0);
    const [autoRotate, setAutoRotate] = useState(true);
    const [showSphere, setShowSphere] = useState(true);
    
    // Acumular puntos históricos
    useEffect(() => {
        if (simData?.poincare_coords && Array.isArray(simData.poincare_coords)) {
            setHistory(prev => {
                // Agregar nuevos puntos
                const newPoints = simData.poincare_coords!.filter((p: number[]) => 
                    Array.isArray(p) && p.length >= 2
                );
                const updated = [...prev, ...newPoints];
                // Mantener solo los últimos maxPoints
                return updated.slice(-maxPoints);
            });
        }
    }, [simData?.poincare_coords, maxPoints]);
    
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
        const ambientLight = new THREE.AmbientLight(0xffffff, 0.4);
        scene.add(ambientLight);
        
        // Luz principal
        const directionalLight1 = new THREE.DirectionalLight(0xffffff, 0.8);
        directionalLight1.position.set(5, 5, 5);
        directionalLight1.castShadow = false;
        scene.add(directionalLight1);
        
        // Luz de relleno
        const directionalLight2 = new THREE.DirectionalLight(0x88ccff, 0.3);
        directionalLight2.position.set(-5, -3, -5);
        scene.add(directionalLight2);
        
        // Luz puntual para resaltar
        const pointLight = new THREE.PointLight(0xffffff, 0.5, 10);
        pointLight.position.set(0, 0, 0);
        scene.add(pointLight);
        
        // Esfera de referencia (Poincaré sphere)
        const sphereGeometry = new THREE.SphereGeometry(1, 32, 32);
        const sphereMaterial = new THREE.MeshBasicMaterial({
            color: 0x333333,
            wireframe: true,
            transparent: true,
            opacity: 0.3
        });
        const sphere = new THREE.Mesh(sphereGeometry, sphereMaterial);
        sphere.visible = showSphere;
        scene.add(sphere);
        
        // Ejes de referencia
        const axesHelper = new THREE.AxesHelper(1.5);
        scene.add(axesHelper);
        
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
    
    // Actualizar esfera
    useEffect(() => {
        if (!sceneRef.current) return;
        const sphere = sceneRef.current.children.find(child => child instanceof THREE.Mesh && child.geometry instanceof THREE.SphereGeometry);
        if (sphere) {
            sphere.visible = showSphere;
        }
    }, [showSphere]);
    
    // Actualizar auto-rotación
    useEffect(() => {
        if (controlsRef.current) {
            controlsRef.current.autoRotate = autoRotate;
        }
    }, [autoRotate]);
    
    // Actualizar visualización cuando cambian los puntos
    useEffect(() => {
        if (!sceneRef.current) return;
        
        if (history.length === 0) {
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
            
            // Normalizar puntos a la esfera unitaria
            history.forEach((point, index) => {
                if (!Array.isArray(point) || point.length < 2) return;
                
                const x = typeof point[0] === 'number' ? point[0] : 0;
                const y = typeof point[1] === 'number' ? point[1] : 0;
                
                // Calcular z para estar en la esfera (x² + y² + z² = 1)
                const z = Math.sqrt(Math.max(0, 1 - x * x - y * y));
                
                // Validar coordenadas
                if (isNaN(x) || isNaN(y) || isNaN(z) || !isFinite(x) || !isFinite(y) || !isFinite(z)) {
                    return;
                }
                
                positions.push(x, y, z);
                
                // Color basado en tiempo (gradiente)
                const t = index / history.length;
                colors.push(t, 0.5, 1 - t);
            });
            
            if (positions.length === 0) return;
            
            geometry.setAttribute('position', new THREE.Float32BufferAttribute(positions, 3));
            geometry.setAttribute('color', new THREE.Float32BufferAttribute(colors, 3));
            
            // Usar InstancedMesh para mejor rendimiento con muchos puntos
            const sphereGeometry = new THREE.SphereGeometry(pointSize * 0.015, 12, 12);
            const material = new THREE.MeshStandardMaterial({
                metalness: 0.3,
                roughness: 0.4,
                transparent: true,
                opacity: 0.9
            });
            
            const instancedMesh = new THREE.InstancedMesh(sphereGeometry, material, history.length);
            
            // Matrices y colores para instancias
            const matrix = new THREE.Matrix4();
            const color = new THREE.Color();
            
            history.forEach((point, index) => {
                if (!Array.isArray(point) || point.length < 2) return;
                
                const x = typeof point[0] === 'number' ? point[0] : 0;
                const y = typeof point[1] === 'number' ? point[1] : 0;
                const z = Math.sqrt(Math.max(0, 1 - x * x - y * y));
                
                if (isNaN(x) || isNaN(y) || isNaN(z) || !isFinite(x) || !isFinite(y) || !isFinite(z)) {
                    return;
                }
                
                // Posición
                matrix.makeTranslation(x, y, z);
                instancedMesh.setMatrixAt(index, matrix);
                
                // Color basado en tiempo (gradiente más suave)
                const t = index / Math.max(history.length - 1, 1);
                // Gradiente azul -> cyan -> amarillo -> rojo
                if (t < 0.33) {
                    color.setRGB(0, t * 3, 1); // Azul -> Cyan
                } else if (t < 0.66) {
                    const localT = (t - 0.33) / 0.33;
                    color.setRGB(0, 1, 1 - localT); // Cyan -> Amarillo
                } else {
                    const localT = (t - 0.66) / 0.34;
                    color.setRGB(localT, 1 - localT, 0); // Amarillo -> Rojo
                }
                instancedMesh.setColorAt(index, color);
            });
            
            instancedMesh.instanceMatrix.needsUpdate = true;
            if (instancedMesh.instanceColor) {
                instancedMesh.instanceColor.needsUpdate = true;
            }
            
            sceneRef.current.add(instancedMesh);
            pointsRef.current = instancedMesh as any;
            
        } catch (error) {
            console.error('Error actualizando visualización Poincaré 3D:', error);
        }
    }, [history, pointSize]);
    
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
    
    return (
        <Paper p="md" withBorder style={{ backgroundColor: 'var(--mantine-color-dark-7)', height: '100%', display: 'flex', flexDirection: 'column' }}>
            <Stack gap="sm" style={{ flex: 1, minHeight: 0 }}>
                <Group justify="space-between">
                    <Group gap="xs">
                        <IconCube size={18} />
                        <Text size="sm" fw={600}>Poincaré 3D</Text>
                        <Tooltip 
                            label="Visualización 3D del espacio de Poincaré. Los puntos se proyectan sobre una esfera unitaria. Útil para visualizar la estructura del espacio de fase en 3D."
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
                        {history.length} puntos
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
                        onClick={() => setHistory([])}
                    >
                        Limpiar
                    </Button>
                </Group>
                
                <Group>
                    <Button
                        variant={showSphere ? "filled" : "outline"}
                        size="xs"
                        onClick={() => setShowSphere(!showSphere)}
                    >
                        {showSphere ? 'Ocultar Esfera' : 'Mostrar Esfera'}
                    </Button>
                </Group>
                
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
                    <Text size="xs" c="dimmed" mb="xs">Máximo de puntos: {maxPoints}</Text>
                    <Slider
                        value={maxPoints}
                        onChange={setMaxPoints}
                        min={100}
                        max={5000}
                        step={100}
                        marks={[
                            { value: 100, label: '100' },
                            { value: 1000, label: '1000' },
                            { value: 2500, label: '2500' },
                            { value: 5000, label: '5000' }
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
                    Arrastra para rotar, rueda para zoom. Eje X (rojo), Y (verde), Z (azul).
                </Text>
            </Stack>
        </Paper>
    );
}

