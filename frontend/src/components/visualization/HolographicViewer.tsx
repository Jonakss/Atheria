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

const HolographicViewer: React.FC<HolographicViewerProps> = ({ 
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

        // Si ya existen puntos, eliminarlos para recrearlos (o actualizarlos)
        if (pointsRef.current) {
            sceneRef.current.remove(pointsRef.current);
            pointsRef.current.geometry.dispose();
            (pointsRef.current.material as THREE.Material).dispose();
        }

        const particleCount = width * height;
        const positions: number[] = [];
        const colors: number[] = [];
        const sizes: number[] = [];

        const colorHelper = new THREE.Color();
        const isPoincare = vizType === 'poincare' || vizType === 'poincare_3d';

        // Recorrer el grid 2D y convertirlo a partículas 3D
        for (let i = 0; i < particleCount; i++) {
            const magnitude = data[i];

            // "Sparse Rendering": Solo dibujar si hay energía
            if (magnitude > threshold) {
                // Coordenadas normalizadas [-1, 1]
                let u = ((i % width) / width) * 2 - 1;
                let v = (Math.floor(i / width) / height) * 2 - 1;
                
                let x, y, z;

                if (isPoincare) {
                    // Mapeo Cuadrado -> Disco (Mapping Square to Disk)
                    // x' = u * sqrt(1 - v^2/2)
                    // y' = v * sqrt(1 - u^2/2)
                    const diskX = u * Math.sqrt(1 - (v * v) / 2);
                    const diskY = v * Math.sqrt(1 - (u * u) / 2);

                    // Escalar al tamaño del mundo (ej: 100 unidades de radio)
                    const R = 100;
                    x = diskX * R;
                    y = diskY * R;
                    
                    // Z sigue siendo la magnitud, pero quizás deformada
                    // En Poincaré, el borde es infinito, así que reducimos Z cerca del borde?
                    // Por ahora mantenemos Z = magnitud * 50
                    z = magnitude * 50;
                } else {
                    // Cartesiano estándar
                    x = u * (width / 2); // Escalar a dimensiones originales aprox
                    y = v * (height / 2);
                    z = magnitude * 50; 
                }

                positions.push(x, y, z);

                // Color basado en la Fase (o azul cian por defecto)
                if (phaseData) {
                    // Mapear fase (-PI a PI) a HSL
                    const hue = (phaseData[i] + Math.PI) / (2 * Math.PI);
                    colorHelper.setHSL(hue, 1.0, 0.5);
                } else {
                    // Gradiente azul basado en altura
                    colorHelper.setHSL(0.6, 1.0, Math.min(0.3 + magnitude, 1.0));
                }
                
                colors.push(colorHelper.r, colorHelper.g, colorHelper.b);
                
                // Tamaño basado en magnitud
                // Aumentamos el multiplicador para que sean visibles con el shader
                sizes.push(Math.max(2.0, magnitude * 8.0)); 
            }
        }

        const geometry = new THREE.BufferGeometry();
        geometry.setAttribute('position', new THREE.Float32BufferAttribute(positions, 3));
        geometry.setAttribute('color', new THREE.Float32BufferAttribute(colors, 3));
        geometry.setAttribute('size', new THREE.Float32BufferAttribute(sizes, 1));

        // Shader para partículas circulares con tamaño variable
        const vertexShader = `
            attribute float size;
            varying vec3 vColor;
            void main() {
                vColor = color;
                vec4 mvPosition = modelViewMatrix * vec4(position, 1.0);
                // Size attenuation: escala el tamaño según la distancia a la cámara
                gl_PointSize = size * (300.0 / -mvPosition.z);
                gl_Position = projectionMatrix * mvPosition;
            }
        `;

        const fragmentShader = `
            varying vec3 vColor;
            void main() {
                // Convertir punto cuadrado en círculo suave
                vec2 coord = gl_PointCoord - vec2(0.5);
                float dist = length(coord);
                
                if (dist > 0.5) discard;
                
                // Borde suave (antialiasing manual)
                float alpha = 1.0 - smoothstep(0.4, 0.5, dist);
                
                gl_FragColor = vec4(vColor, alpha);
            }
        `;

        const material = new THREE.ShaderMaterial({
            uniforms: {},
            vertexShader,
            fragmentShader,
            transparent: true,
            blending: THREE.AdditiveBlending,
            depthWrite: false // Importante para transparencia correcta en partículas
        });

        const points = new THREE.Points(geometry, material);
        sceneRef.current.add(points);
        pointsRef.current = points;

    }, [data, phaseData, width, height, threshold, vizType]);

    return <div ref={mountRef} style={{ width: '100%', height: '100%', minHeight: '400px' }} />;
};

export default HolographicViewer;