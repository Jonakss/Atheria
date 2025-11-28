import React, { useEffect, useRef } from 'react';
import * as THREE from 'three';
import { OrbitControls } from 'three/examples/jsm/controls/OrbitControls.js';

interface HolographicViewerProps {
    data: number[]; // Array plano de magnitudes [mag0, mag1, ...]
    phaseData?: number[]; // Array plano de fases [phase0, phase1, ...] opcional
    width: number;
    height: number;
    threshold?: number; // Umbral para no renderizar vacío
    vizType?: string; // Tipo de visualización (ej: 'poincare')
}

export const HolographicViewer2: React.FC<HolographicViewerProps> = ({ 
    data, 
    phaseData, 
    width, 
    height,
    threshold = 0.05,
    vizType = 'holographic'
}) => {
    const mountRef = useRef<HTMLDivElement>(null);
    const sceneRef = useRef<THREE.Scene | null>(null);
    const cameraRef = useRef<THREE.PerspectiveCamera | null>(null);
    const rendererRef = useRef<THREE.WebGLRenderer | null>(null);
    const pointsRef = useRef<THREE.Points | null>(null);
    const controlsRef = useRef<OrbitControls | null>(null);

    // 1. Inicialización de Three.js
    useEffect(() => {
        if (!mountRef.current) return;

        // Escena
        const scene = new THREE.Scene();
        scene.background = new THREE.Color(0x050505); // Fondo casi negro
        scene.fog = new THREE.FogExp2(0x050505, 0.002); // Niebla para profundidad

        // Cámara
        const camera = new THREE.PerspectiveCamera(60, mountRef.current.clientWidth / mountRef.current.clientHeight, 0.1, 2000);
        camera.position.set(0, -150, 100); // Posición inicial inclinada
        camera.up.set(0, 0, 1); // Z es arriba

        // Renderer
        const renderer = new THREE.WebGLRenderer({ antialias: true, alpha: true });
        renderer.setSize(mountRef.current.clientWidth, mountRef.current.clientHeight);
        renderer.setPixelRatio(window.devicePixelRatio);
        mountRef.current.appendChild(renderer.domElement);

        // Controles de órbita
        const controls = new OrbitControls(camera, renderer.domElement);
        controls.enableDamping = true;
        controls.dampingFactor = 0.05;
        controls.autoRotate = false;
        controls.enableRotate = true; // Permitir rotación manual

        // Referencias
        sceneRef.current = scene;
        cameraRef.current = camera;
        rendererRef.current = renderer;
        controlsRef.current = controls;

        // Capturar ref actual para cleanup seguro
        const currentMount = mountRef.current;

        // Loop de animación
        const animate = () => {
            requestAnimationFrame(animate);
            controls.update();
            renderer.render(scene, camera);
        };
        animate();

        // Handler de resize que solo ajusta el renderer sin cambiar la vista
        const handleResize = () => {
            if (!currentMount || !renderer || !camera) return;
            
            const width = currentMount.clientWidth;
            const height = currentMount.clientHeight;
            
            if (width === 0 || height === 0) return;
            
            // Solo ajustar tamaño del renderer y aspect ratio de la cámara
            // NO cambiar posición de la cámara ni controles (mantiene vista del usuario)
            camera.aspect = width / height;
            camera.updateProjectionMatrix();
            renderer.setSize(width, height);
        };

        // Listener de resize
        window.addEventListener('resize', handleResize);
        
        // También ajustar si el contenedor cambia de tamaño
        const resizeObserver = new ResizeObserver(handleResize);
        if (currentMount) {
            resizeObserver.observe(currentMount);
        }

        // Limpieza
        return () => {
            window.removeEventListener('resize', handleResize);
            resizeObserver.disconnect();
            if (currentMount && renderer.domElement) {
                currentMount.removeChild(renderer.domElement);
            }
            renderer.dispose();
        };
    }, []);

    // 2. Actualización de la Geometría (La Magia de los Vóxeles)
    useEffect(() => {
        if (!sceneRef.current || !data || data.length === 0) return;

        // Si ya existen puntos, eliminarlos para recrearlos
        if (pointsRef.current) {
            sceneRef.current.remove(pointsRef.current);
            pointsRef.current.geometry.dispose();
            (pointsRef.current.material as THREE.Material).dispose();
        }

        const particleCount = width * height;
        
        // Atributos para el shader
        const indices: number[] = [];
        const magnitudes: number[] = [];
        const phases: number[] = [];

        // Llenar buffers
        for (let i = 0; i < particleCount; i++) {
            const magnitude = data[i];
            
            // Optimización: Skip puntos con muy poca energía
            if (magnitude > threshold) {
                indices.push(i);
                magnitudes.push(magnitude);
                phases.push(phaseData ? phaseData[i] : 0);
            }
        }

        const geometry = new THREE.BufferGeometry();
        // Usamos 'position' para pasar el índice (x) y magnitud (y) para ahorrar atributos? 
        // Mejor usar atributos explícitos para claridad.
        // Pasamos índice como atributo float para calcular UV en vertex shader
        geometry.setAttribute('particleIndex', new THREE.Float32BufferAttribute(indices, 1));
        geometry.setAttribute('magnitude', new THREE.Float32BufferAttribute(magnitudes, 1));
        geometry.setAttribute('phase', new THREE.Float32BufferAttribute(phases, 1));
        
        // Dummy position attribute needed for Three.js to render count correctly?
        // Actually, drawRange or explicit count in geometry is needed if no position.
        // But Points usually expects position. Let's pass zeros and displace in shader.
        const zeros = new Float32Array(indices.length * 3);
        geometry.setAttribute('position', new THREE.BufferAttribute(zeros, 3));

        // Uniforms
        const uniforms = {
            uWidth: { value: width },
            uHeight: { value: height },
            uTime: { value: 0 },
            uIsPoincare: { value: vizType === 'poincare' || vizType === 'poincare_3d' },
            uScale: { value: 100.0 } // Radio del disco
        };

        // Vertex Shader: Proyección Hiperbólica
        const vertexShader = `
            attribute float particleIndex;
            attribute float magnitude;
            attribute float phase;
            
            uniform float uWidth;
            uniform float uHeight;
            uniform float uScale;
            uniform bool uIsPoincare;
            
            varying vec3 vColor;
            varying float vAlpha;

            // Función para convertir HSL a RGB
            vec3 hsl2rgb(vec3 c) {
                vec3 rgb = clamp(abs(mod(c.x*6.0+vec3(0.0,4.0,2.0),6.0)-3.0)-1.0, 0.0, 1.0);
                return c.z + c.y * (rgb-0.5)*(1.0-abs(2.0*c.z-1.0));
            }

            void main() {
                // 1. Calcular coordenadas UV normalizadas [-1, 1]
                float u = (mod(particleIndex, uWidth) / uWidth) * 2.0 - 1.0;
                float v = (floor(particleIndex / uWidth) / uHeight) * 2.0 - 1.0;
                
                vec3 pos;
                
                if (uIsPoincare) {
                    // 2. Mapeo Cuadrado -> Disco de Poincaré
                    // x = u * sqrt(1 - v^2/2)
                    // y = v * sqrt(1 - u^2/2)
                    float diskX = u * sqrt(1.0 - (v * v) / 2.0);
                    float diskY = v * sqrt(1.0 - (u * u) / 2.0);
                    
                    // 3. Scale-Radius Duality (Visualización)
                    // En AdS/CFT, el radio r representa la escala de energía.
                    // r -> 1 (Borde) es UV (Alta energía/Detalle)
                    // r -> 0 (Centro) es IR (Baja energía/Bulk)
                    
                    // Aquí usamos la magnitud para desplazar en Z (hacia el usuario)
                    // y expandir radialmente para enfatizar la estructura
                    
                    pos = vec3(diskX * uScale, diskY * uScale, magnitude * 50.0);
                } else {
                    // Cartesiano estándar
                    pos = vec3(u * uWidth * 0.5, v * uHeight * 0.5, magnitude * 50.0);
                }

                vec4 mvPosition = modelViewMatrix * vec4(pos, 1.0);
                gl_Position = projectionMatrix * mvPosition;

                // 4. Color basado en Fase (o Magnitud si no hay fase)
                // HSL: Hue = Phase, Sat = 1.0, Light = ajustado para evitar blanco
                float hue = (phase + 3.14159) / (2.0 * 3.14159);
                vec3 color = hsl2rgb(vec3(hue, 1.0, 0.4 + magnitude * 0.3)); // Reducido de 0.5 + mag*0.5
                
                // Si no hay fase (phase == 0 para todos), usar gradiente azul-cian
                if (phase == 0.0) {
                     color = hsl2rgb(vec3(0.6 - magnitude * 0.2, 1.0, 0.25 + magnitude * 0.4)); // Reducido de 0.3 + mag
                }
                
                vColor = color;
                
                // 5. Size Attenuation
                gl_PointSize = max(2.0, magnitude * 10.0) * (300.0 / -mvPosition.z);
                vAlpha = min(1.0, magnitude * 2.0);
            }
        `;

        const fragmentShader = `
            varying vec3 vColor;
            varying float vAlpha;
            
            void main() {
                // Círculo suave
                vec2 coord = gl_PointCoord - vec2(0.5);
                float dist = length(coord);
                
                if (dist > 0.5) discard;
                
                float alpha = 1.0 - smoothstep(0.4, 0.5, dist);
                gl_FragColor = vec4(vColor, alpha * vAlpha);
            }
        `;

        const material = new THREE.ShaderMaterial({
            uniforms,
            vertexShader,
            fragmentShader,
            transparent: true,
            blending: THREE.AdditiveBlending,
            depthWrite: false
        });

        const points = new THREE.Points(geometry, material);
        sceneRef.current.add(points);
        pointsRef.current = points;

    }, [data, phaseData, width, height, threshold, vizType]);

    return (
        <div className="relative w-full h-full">
            <div ref={mountRef} style={{ width: '100%', height: '100%', minHeight: '400px' }} />
            <div className="absolute top-4 left-4 pointer-events-none">
                <div className="px-2 py-1 bg-purple-500/20 border border-purple-500/30 rounded text-xs font-mono text-purple-200 backdrop-blur-sm">
                    Holographic Viewer 2.0 (AdS/CFT)
                </div>
            </div>
        </div>
    );
};

export default HolographicViewer2;