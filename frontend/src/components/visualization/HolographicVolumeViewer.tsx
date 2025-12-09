import React, { useEffect, useRef } from 'react';
import * as THREE from 'three';
import { OrbitControls } from 'three/examples/jsm/controls/OrbitControls.js';

interface HolographicVolumeViewerProps {
    volumeData: number[]; // Flat array [D * H * W] representing the 3D bulk
    depth: number; // D dimension
    width: number; // W dimension
    height: number; // H dimension
    threshold?: number; // Minimum intensity to render
}

export const HolographicVolumeViewer: React.FC<HolographicVolumeViewerProps> = ({ 
    volumeData,
    depth,
    width, 
    height,
    threshold = 0.01
}) => {
    const mountRef = useRef<HTMLDivElement>(null);
    const sceneRef = useRef<THREE.Scene | null>(null);
    const cameraRef = useRef<THREE.PerspectiveCamera | null>(null);
    const rendererRef = useRef<THREE.WebGLRenderer | null>(null);
    const pointsRef = useRef<THREE.Points | null>(null);
    const controlsRef = useRef<OrbitControls | null>(null);

    // Initialize Three.js scene
    useEffect(() => {
        if (!mountRef.current) return;

        const scene = new THREE.Scene();
        scene.background = new THREE.Color(0x0a0a0a);
        scene.fog = new THREE.FogExp2(0x0a0a0a, 0.001);

        const camera = new THREE.PerspectiveCamera(
            60, 
            mountRef.current.clientWidth / mountRef.current.clientHeight, 
            0.1, 
            3000
        );
        camera.position.set(150, 150, 150);
        camera.up.set(0, 0, 1);

        const renderer = new THREE.WebGLRenderer({ antialias: true, alpha: true });
        renderer.setSize(mountRef.current.clientWidth, mountRef.current.clientHeight);
        renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
        mountRef.current.appendChild(renderer.domElement);

        const controls = new OrbitControls(camera, renderer.domElement);
        controls.enableDamping = true;
        controls.dampingFactor = 0.05;
        controls.autoRotate = false;

        sceneRef.current = scene;
        cameraRef.current = camera;
        rendererRef.current = renderer;
        controlsRef.current = controls;

        const currentMount = mountRef.current;

        const animate = () => {
            requestAnimationFrame(animate);
            controls.update();
            renderer.render(scene, camera);
        };
        animate();

        const handleResize = () => {
            if (!currentMount || !renderer || !camera) return;
            
            const w = currentMount.clientWidth;
            const h = currentMount.clientHeight;
            
            if (w === 0 || h === 0) return;
            
            camera.aspect = w / h;
            camera.updateProjectionMatrix();
            renderer.setSize(w, h);
        };

        window.addEventListener('resize', handleResize);
        const resizeObserver = new ResizeObserver(handleResize);
        if (currentMount) {
            resizeObserver.observe(currentMount);
        }

        return () => {
            window.removeEventListener('resize', handleResize);
            resizeObserver.disconnect();
            if (currentMount && renderer.domElement) {
                currentMount.removeChild(renderer.domElement);
            }
            renderer.dispose();
        };
    }, []);

    // Update volume geometry
    useEffect(() => {
        if (!sceneRef.current || !volumeData || volumeData.length === 0) return;

        if (pointsRef.current) {
            sceneRef.current.remove(pointsRef.current);
            pointsRef.current.geometry.dispose();
            (pointsRef.current.material as THREE.Material).dispose();
        }

        const positions: number[] = [];
        const colors: number[] = [];
        const sizes: number[] = [];

        const colorHelper = new THREE.Color();

        // Iterate through the volume
        for (let d = 0; d < depth; d++) {
            for (let h = 0; h < height; h++) {
                for (let w = 0; w < width; w++) {
                    const idx = d * (height * width) + h * width + w;
                    const intensity = volumeData[idx];

                    if (intensity > threshold) {
                        // Map to 3D coordinates
                        // Center the volume around origin
                        const x = (w - width / 2) * 2;
                        const y = (h - height / 2) * 2;
                        const z = (d - depth / 2) * 2;

                        positions.push(x, y, z);

                        // Color gradient: Blue (boundary/surface) to Red (deep bulk)
                        // d=0 is the boundary (blue), d=depth-1 is deep bulk (red)
                        const depthRatio = d / (depth - 1);
                        colorHelper.setHSL(0.6 - depthRatio * 0.6, 1.0, 0.3 + intensity * 0.3);

                        colors.push(colorHelper.r, colorHelper.g, colorHelper.b);

                        // Size based on intensity and depth
                        // Deeper layers should appear larger (coarse-grained)
                        const baseSize = 2.0 + depthRatio * 3.0;
                        sizes.push(baseSize * (0.5 + intensity * 1.5));
                    }
                }
            }
        }

        const geometry = new THREE.BufferGeometry();
        geometry.setAttribute('position', new THREE.Float32BufferAttribute(positions, 3));
        geometry.setAttribute('color', new THREE.Float32BufferAttribute(colors, 3));
        geometry.setAttribute('pSize', new THREE.Float32BufferAttribute(sizes, 1));

        const vertexShader = `
            attribute float pSize;
            varying vec3 vColor;
            void main() {
                vColor = color;
                vec4 mvPosition = modelViewMatrix * vec4(position, 1.0);
                gl_PointSize = pSize * (400.0 / -mvPosition.z);
                gl_Position = projectionMatrix * mvPosition;
            }
        `;

        const fragmentShader = `
            precision mediump float;
            varying vec3 vColor;
            void main() {
                vec2 coord = gl_PointCoord - vec2(0.5);
                float dist = length(coord);
                
                if (dist > 0.5) discard;
                
                float alpha = 1.0 - smoothstep(0.35, 0.5, dist);
                gl_FragColor = vec4(vColor, alpha * 0.8);
            }
        `;

        const material = new THREE.ShaderMaterial({
            uniforms: {},
            vertexShader,
            fragmentShader,
            transparent: true,
            blending: THREE.AdditiveBlending,
            depthWrite: false
        });

        const points = new THREE.Points(geometry, material);
        sceneRef.current.add(points);
        pointsRef.current = points;

    }, [volumeData, depth, width, height, threshold]);

    return (
        <div className="relative w-full h-full">
            <div ref={mountRef} style={{ width: '100%', height: '100%', minHeight: '400px' }} />
            <div className="absolute top-4 left-4 pointer-events-none">
                <div className="px-3 py-1.5 bg-blue-500/20 border border-blue-500/30 rounded text-xs font-mono text-blue-200 backdrop-blur-sm">
                    Holographic Volume (Bulk 3D)
                </div>
            </div>
        </div>
    );
};

export default HolographicVolumeViewer;
