import React, { useEffect, useRef } from 'react';
import * as THREE from 'three';
import { OrbitControls } from 'three/examples/jsm/controls/OrbitControls.js';
import { useExperimentStore } from '../../store/experimentStore';

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

    // Quantum Ripple Effect Ref
    const rippleMeshRef = useRef<THREE.Mesh | null>(null);
    const { isQuantumInjecting } = useExperimentStore();

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

    // 1.5 Quantum Ripple Effect Initialization
    useEffect(() => {
        if (!sceneRef.current) return;

        // Create a large plane or sphere for the ripple effect
        // Using a billboard plane facing the camera or a sphere surrounding the scene
        const geometry = new THREE.SphereGeometry(200, 32, 32);
        const material = new THREE.ShaderMaterial({
            transparent: true,
            side: THREE.BackSide, // Render inside of sphere
            blending: THREE.AdditiveBlending,
            depthWrite: false,
            uniforms: {
                time: { value: 0 },
                active: { value: 0.0 }
            },
            vertexShader: `
                varying vec2 vUv;
                void main() {
                    vUv = uv;
                    gl_Position = projectionMatrix * modelViewMatrix * vec4(position, 1.0);
                }
            `,
            fragmentShader: `
                uniform float time;
                uniform float active;
                varying vec2 vUv;

                void main() {
                    if (active < 0.01) discard;

                    // Simple radial ripple based on time
                    float dist = length(vUv - 0.5) * 2.0;
                    float ripple = sin(dist * 20.0 - time * 10.0) * 0.5 + 0.5;
                    float fade = 1.0 - dist;

                    vec3 color = vec3(0.6, 0.2, 1.0); // Purple
                    gl_FragColor = vec4(color, ripple * fade * active * 0.3);
                }
            `
        });

        const rippleMesh = new THREE.Mesh(geometry, material);
        sceneRef.current.add(rippleMesh);
        rippleMeshRef.current = rippleMesh;

        return () => {
            if (sceneRef.current && rippleMesh) {
                sceneRef.current.remove(rippleMesh);
                rippleMesh.geometry.dispose();
                rippleMesh.material.dispose();
            }
        };
    }, []);

    // Update Ripple State
    useEffect(() => {
        if (rippleMeshRef.current) {
            const material = rippleMeshRef.current.material as THREE.ShaderMaterial;
            // We'll animate this in the loop ideally, but for now we set a target
            // To make it smooth, we rely on the requestAnimationFrame loop to update 'time'
            // But we need to update 'active' uniform.

            // For simplicity, we just set active to 1.0 when injecting, but we need an animation loop to handle fade out/in properly
            // or just bind it to the state directly.

            // Let's hook into the existing animate loop?
            // The existing animate loop is closed over in the first useEffect.
            // We can use a ref to communicate.
        }
    }, [isQuantumInjecting]);

    // Independent Animation Loop for Ripple (since the main loop is closed)
    // Actually, we can just start a temporary loop or rely on the main one if we could access it.
    // But since we can't easily modify the main loop without full rewrite, let's add a secondary updater
    // OR: Modify the main loop in the first useEffect to read a ref.

    // Better approach: Use a ref to store the state, and let the main loop read it.
    const quantumStateRef = useRef({ isInjecting: false, startTime: 0 });

    useEffect(() => {
        if (isQuantumInjecting) {
            quantumStateRef.current.isInjecting = true;
            quantumStateRef.current.startTime = performance.now();
        } else {
             // We don't immediately set false to allow fade out, but store handles timeout.
             // But visual fade out might be longer.
             quantumStateRef.current.isInjecting = false;
        }
    }, [isQuantumInjecting]);

    // We need to inject the update logic into the main loop.
    // Since I can't easily merge into the big block, I'll use a separate useFrame equivalent.
    // React allows multiple requestAnimationFrames.
    useEffect(() => {
        let frameId: number;
        const animateRipple = () => {
            if (rippleMeshRef.current) {
                const material = rippleMeshRef.current.material as THREE.ShaderMaterial;
                const now = performance.now();
                material.uniforms.time.value = now * 0.001;

                // Smooth transition
                const target = quantumStateRef.current.isInjecting ? 1.0 : 0.0;
                // Linear interpolation for 'active'
                const current = material.uniforms.active.value;
                material.uniforms.active.value += (target - current) * 0.1;
            }
            frameId = requestAnimationFrame(animateRipple);
        };
        animateRipple();
        return () => cancelAnimationFrame(frameId);
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
                const u = ((i % width) / width) * 2 - 1;
                const v = (Math.floor(i / width) / height) * 2 - 1;
                
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
                    // Gradiente azul basado en altura - Luminosidad reducida para evitar blanco
                    colorHelper.setHSL(0.6, 1.0, Math.min(0.2 + magnitude * 0.3, 0.6));
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