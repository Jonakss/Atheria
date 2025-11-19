import React, { useRef, useEffect, useMemo } from 'react';
import * as THREE from 'three';
import { OrbitControls } from 'three/examples/jsm/controls/OrbitControls';

interface HolographicViewerProps {
    data: number[]; // Array plano de magnitudes [mag0, mag1, ...]
    phaseData?: number[]; // Array plano de fases [phase0, phase1, ...] opcional
    width: number;
    height: number;
    threshold?: number; // Umbral para no renderizar vacío
}

const HolographicViewer: React.FC<HolographicViewerProps> = ({ 
    data, 
    phaseData, 
    width, 
    height,
    threshold = 0.05 
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
        controls.autoRotate = true;
        controls.autoRotateSpeed = 0.5;

        // Referencias
        sceneRef.current = scene;
        cameraRef.current = camera;
        rendererRef.current = renderer;
        controlsRef.current = controls;

        // Loop de animación
        const animate = () => {
            requestAnimationFrame(animate);
            controls.update();
            renderer.render(scene, camera);
        };
        animate();

        // Limpieza
        return () => {
            if (mountRef.current && renderer.domElement) {
                mountRef.current.removeChild(renderer.domElement);
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

        // Recorrer el grid 2D y convertirlo a partículas 3D
        for (let i = 0; i < particleCount; i++) {
            const magnitude = data[i];

            // "Sparse Rendering": Solo dibujar si hay energía
            if (magnitude > threshold) {
                const x = (i % width) - width / 2;
                const y = Math.floor(i / width) - height / 2;
                
                // Z es la magnitud (Heightmap)
                // Multiplicamos por 50 para exagerar la altura y que se vea 3D
                const z = magnitude * 50; 

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
                sizes.push(magnitude * 2); 
            }
        }

        const geometry = new THREE.BufferGeometry();
        geometry.setAttribute('position', new THREE.Float32BufferAttribute(positions, 3));
        geometry.setAttribute('color', new THREE.Float32BufferAttribute(colors, 3));
        // Nota: Para tamaños variables por partícula se requiere un shader custom, 
        // aquí usamos tamaño fijo escalado por la cámara para simplicidad.

        const material = new THREE.PointsMaterial({
            size: 2,
            vertexColors: true,
            transparent: true,
            opacity: 0.8,
            sizeAttenuation: true,
            blending: THREE.AdditiveBlending // Efecto "brillante"
        });

        const points = new THREE.Points(geometry, material);
        sceneRef.current.add(points);
        pointsRef.current = points;

    }, [data, phaseData, width, height, threshold]);

    return <div ref={mountRef} style={{ width: '100%', height: '100%', minHeight: '400px' }} />;
};

export default HolographicViewer;