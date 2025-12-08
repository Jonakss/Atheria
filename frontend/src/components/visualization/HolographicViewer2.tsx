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
    channels?: number; // Número de canales (1 para mono, 3 para RGB)
    shape?: number[]; // [D, H, W] para volumétrico
    // Gateway Props
    binaryMode?: boolean;
    binaryThreshold?: number;
    binaryColor?: string;
}

export const HolographicViewer2: React.FC<HolographicViewerProps> = ({ 
    data, 
    phaseData, 
    width, 
    height,
    shape,
    threshold = 0.05,
    vizType = 'holographic',
    channels = 1,
    binaryMode = false,
    binaryThreshold = 0.5,
    binaryColor = '#FFFFFF'
}) => {
    const mountRef = useRef<HTMLDivElement>(null);
    const sceneRef = useRef<THREE.Scene | null>(null);
    // ... refs ...
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

        // Handler de resize
        const handleResize = () => {
            if (!currentMount || !renderer || !camera) return;
            const width = currentMount.clientWidth;
            const height = currentMount.clientHeight;
            if (width === 0 || height === 0) return;
            camera.aspect = width / height;
            camera.updateProjectionMatrix();
            renderer.setSize(width, height);
        };

        window.addEventListener('resize', handleResize);
        const resizeObserver = new ResizeObserver(handleResize);
        if (currentMount) resizeObserver.observe(currentMount);

        return () => {
            window.removeEventListener('resize', handleResize);
            resizeObserver.disconnect();
            if (currentMount && renderer.domElement) currentMount.removeChild(renderer.domElement);
            renderer.dispose();
        };
    }, []);

    // 2. Actualización de la Geometría (La Magia de los Vóxeles)
    useEffect(() => {
        if (!sceneRef.current || !data || data.length === 0) return;

        if (pointsRef.current) {
            sceneRef.current.remove(pointsRef.current);
            pointsRef.current.geometry.dispose();
            (pointsRef.current.material as THREE.Material).dispose();
        }

        // Determine dimensions based on shape or fallback to width/height props
        let volDepth = 1;
        let volHeight = height;
        let volWidth = width;
        let isVolumetric = false;

        if (shape && shape.length === 3) {
            volDepth = shape[0];
            volHeight = shape[1];
            volWidth = shape[2];
            isVolumetric = true;
        }

        const particleCount = volDepth * volHeight * volWidth;
        
        // Atributos para el shader
        const indices: number[] = [];
        const magnitudes: number[] = [];
        const phases: number[] = [];
        const colors: number[] = []; 

        // Llenar buffers
        for (let i = 0; i < particleCount; i++) {
            let magnitude = 0;
            let r = 0, g = 0, b = 0;

            if (channels === 3) {
                 const idx = i * 3;
                 if (idx + 2 < data.length) {
                     r = data[idx];
                     g = data[idx + 1];
                     b = data[idx + 2];
                     magnitude = (r + g + b) / 3.0; 
                 }
            } else {
                 if (i < data.length) magnitude = data[i];
            }
            
            if (magnitude > threshold) {
                indices.push(i);
                magnitudes.push(magnitude);
                phases.push(phaseData ? phaseData[i] : 0);
                if (channels === 3) colors.push(r, g, b);
            }
        }

        const geometry = new THREE.BufferGeometry();
        geometry.setAttribute('particleIndex', new THREE.Float32BufferAttribute(indices, 1));
        geometry.setAttribute('magnitude', new THREE.Float32BufferAttribute(magnitudes, 1));
        geometry.setAttribute('phase', new THREE.Float32BufferAttribute(phases, 1));
        
        if (channels === 3) {
            geometry.setAttribute('color', new THREE.Float32BufferAttribute(colors, 3));
        }
        
        const zeros = new Float32Array(indices.length * 3);
        geometry.setAttribute('position', new THREE.BufferAttribute(zeros, 3));

        // Uniforms
        const uniforms = {
            uWidth: { value: volWidth },
            uHeight: { value: volHeight },
            uDepth: { value: volDepth },
            uIsVolumetric: { value: isVolumetric },
            uTime: { value: 0 },
            uIsPoincare: { value: vizType === 'poincare' || vizType === 'poincare_3d' },
            uScale: { value: 100.0 }, 
            uUseColorAttribute: { value: channels === 3 },
            uBinaryMode: { value: binaryMode },
            uBinaryThreshold: { value: binaryThreshold },
            uBinaryColor: { value: new THREE.Color(binaryColor) }
        };

        // Vertex Shader
        const vertexShader = `
            attribute float particleIndex;
            attribute float magnitude;
            attribute float phase;
            attribute vec3 color;
            
            uniform float uWidth;
            uniform float uHeight;
            uniform float uDepth;
            uniform float uScale;
            uniform bool uIsPoincare;
            uniform bool uIsVolumetric;
            uniform bool uUseColorAttribute;
            uniform bool uBinaryMode;
            uniform float uBinaryThreshold;
            uniform vec3 uBinaryColor;
            
            varying vec3 vColor;
            varying float vAlpha;

            vec3 hsl2rgb(vec3 c) {
                vec3 rgb = clamp(abs(mod(c.x*6.0+vec3(0.0,4.0,2.0),6.0)-3.0)-1.0, 0.0, 1.0);
                return c.z + c.y * (rgb-0.5)*(1.0-abs(2.0*c.z-1.0));
            }

            void main() {
                vec3 pos;

                if (uIsVolumetric) {
                    // Volumétrico: Index -> (x, y, z)
                    float sliceSize = uWidth * uHeight;
                    float zIndex = floor(particleIndex / sliceSize);
                    float rem = mod(particleIndex, sliceSize);
                    float yIndex = floor(rem / uWidth);
                    float xIndex = mod(rem, uWidth);

                    // Normalize [-1, 1]
                    float u = (xIndex / uWidth) * 2.0 - 1.0;
                    float v = (yIndex / uHeight) * 2.0 - 1.0;
                    
                    // Cone Geometry: Scale layers down as Z increases (Renormalization flow)
                    // Z=0 is Boundary (Wide), Z=Depth is Bulk (Narrow)
                    float zNorm = zIndex / max(1.0, uDepth - 1.0); // 0 to 1
                    float scale = 1.0 - (zNorm * 0.5); // Shrink by 50% at deepest

                    // Spread layers in Z
                    float zPos = zIndex * 15.0; // Distance between layers

                    pos = vec3(u * uWidth * 0.5 * scale, v * uHeight * 0.5 * scale, zPos);
                    
                } else {
                    // 2D Logic
                    float u = (mod(particleIndex, uWidth) / uWidth) * 2.0 - 1.0;
                    float v = (floor(particleIndex / uWidth) / uHeight) * 2.0 - 1.0;
                    
                    if (uIsPoincare) {
                        float diskX = u * sqrt(1.0 - (v * v) / 2.0);
                        float diskY = v * sqrt(1.0 - (u * u) / 2.0);
                        pos = vec3(diskX * uScale, diskY * uScale, magnitude * 50.0);
                    } else {
                        pos = vec3(u * uWidth * 0.5, v * uHeight * 0.5, magnitude * 50.0);
                    }
                }

                vec4 mvPosition = modelViewMatrix * vec4(pos, 1.0);
                gl_Position = projectionMatrix * mvPosition;

                // Color Logic
                if (uBinaryMode) {
                   if (magnitude > uBinaryThreshold) {
                       vColor = uUseColorAttribute ? color : uBinaryColor;
                       vAlpha = 1.0;
                   } else {
                       vColor = vec3(0.0);
                       vAlpha = 0.0;
                   }
                } else {
                    if (uUseColorAttribute) {
                        vColor = color;
                    } else {
                        float hue = (phase + 3.14159) / (2.0 * 3.14159);
                        vec3 hslColor = hsl2rgb(vec3(hue, 1.0, 0.2 + magnitude * 0.2)); 
                        if (phase == 0.0) hslColor = hsl2rgb(vec3(0.6 - magnitude * 0.2, 1.0, 0.15 + magnitude * 0.25)); 
                        vColor = hslColor;
                    }
                    vAlpha = min(1.0, magnitude * 2.0);
                
                // 5. Size Attenuation
                // Base size * Magnitude * Perspective
                float perspectiveSize = (300.0 / -mvPosition.z);
                // Clamp max perspective scaling to avoid huge near-camera blobs
                perspectiveSize = min(perspectiveSize, 50.0); 
                
                float finalSize = max(2.0, magnitude * 8.0) * perspectiveSize;
                gl_PointSize = clamp(finalSize, 2.0, 60.0); // Hard clamp to hardware limits/sanity
            }
        `;

        const fragmentShader = `
            varying vec3 vColor;
            varying float vAlpha;
            
            void main() {
                if (vAlpha <= 0.001) discard;
                vec2 coord = gl_PointCoord - vec2(0.5);
                if (length(coord) > 0.5) discard;
                float alpha = 1.0 - smoothstep(0.4, 0.5, length(coord));
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

    }, [data, phaseData, width, height, shape, threshold, vizType, channels, binaryMode, binaryThreshold, binaryColor]);

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