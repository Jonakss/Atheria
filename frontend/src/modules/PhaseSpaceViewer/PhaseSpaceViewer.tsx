/* eslint-disable react/no-unknown-property */
import { OrbitControls } from '@react-three/drei';
import { Canvas } from '@react-three/fiber';
/// <reference types="@react-three/fiber" />
import React, { useEffect, useMemo, useRef, useState } from 'react';
import * as THREE from 'three';
import { useWebSocket } from '../../hooks/useWebSocket';

interface PointData {
    x: number;
    y: number;
    z: number;
    cluster: number;
    color: string;
    orig_x: number;
    orig_y: number;
}

interface AnalysisResult {
    method: string;
    points: PointData[];
    centroids: number[][];
    metrics: any;
}

const PointsCloud = ({ data }: { data: PointData[] }) => {
    const meshRef = useRef<THREE.InstancedMesh>(null);
    const tempObject = useMemo(() => new THREE.Object3D(), []);

    useEffect(() => {
        if (!meshRef.current) return;
        
        // Update instances
        data.forEach((point, i) => {
            tempObject.position.set(point.x * 5, point.y * 5, point.z * 5); // Scale up for visibility
            tempObject.updateMatrix();
            meshRef.current!.setMatrixAt(i, tempObject.matrix);
            meshRef.current!.setColorAt(i, new THREE.Color(point.color));
        });
        
        meshRef.current.instanceMatrix.needsUpdate = true;
        if (meshRef.current.instanceColor) meshRef.current.instanceColor.needsUpdate = true;
    }, [data, tempObject]);

    return (
        <instancedMesh ref={meshRef} args={[undefined, undefined, data.length]}>
            <sphereGeometry args={[0.05, 8, 8]} />
            <meshStandardMaterial />
        </instancedMesh>
    );
};

const PhaseSpaceViewer: React.FC = () => {
    const { sendCommand } = useWebSocket();
    const [analysisData, setAnalysisData] = useState<AnalysisResult | null>(null);
    const [isLoading, setIsLoading] = useState(false);

    useEffect(() => {
        const handleResult = (event: Event) => {
            const customEvent = event as CustomEvent;
            setAnalysisData(customEvent.detail);
            setIsLoading(false);
        };

        window.addEventListener('analysis_result_received', handleResult);
        return () => {
            window.removeEventListener('analysis_result_received', handleResult);
        };
    }, []);

    const handleAnalyze = () => {
        setIsLoading(true);
        sendCommand('inference', 'analyze_snapshot', {});
    };

    return (
        <div className="flex flex-col h-full w-full bg-slate-900 text-white p-4 rounded-lg shadow-xl">
            <div className="flex justify-between items-center mb-4">
                <h2 className="text-xl font-bold bg-gradient-to-r from-blue-400 to-purple-500 bg-clip-text text-transparent">
                    Phase Space Topology
                </h2>
                <div className="flex gap-2">
                    <button
                        onClick={handleAnalyze}
                        disabled={isLoading}
                        className={`px-4 py-2 rounded font-semibold transition-all ${
                            isLoading 
                                ? 'bg-gray-600 cursor-not-allowed' 
                                : 'bg-blue-600 hover:bg-blue-500 shadow-lg hover:shadow-blue-500/50'
                        }`}
                    >
                        {isLoading ? 'Analyzing...' : 'Deep Analyze (UMAP)'}
                    </button>
                </div>
            </div>

            <div className="flex-1 relative bg-black/50 rounded-lg overflow-hidden border border-slate-700">
                {analysisData ? (
                    <Canvas camera={{ position: [0, 0, 10], fov: 60 }}>
                        <ambientLight intensity={0.5} />
                        <pointLight position={[10, 10, 10]} />
                        <PointsCloud data={analysisData.points} />

                        <OrbitControls />
                        <gridHelper args={[20, 20, 0x444444, 0x222222]} />
                        <axesHelper args={[5]} />
                    </Canvas>
                ) : (
                    <div className="absolute inset-0 flex items-center justify-center text-slate-400">
                        <div className="text-center">
                            <p className="mb-2 text-lg">No analysis data available</p>
                            <p className="text-sm opacity-70">Click &quot;Deep Analyze&quot; to map the current quantum state topology</p>
                        </div>
                    </div>
                )}
                
                {analysisData && (
                    <div className="absolute bottom-4 left-4 bg-black/80 p-3 rounded border border-slate-700 text-xs font-mono">
                        <p>Method: <span className="text-green-400">{analysisData.method}</span></p>
                        <p>Points: {analysisData.points.length}</p>
                        <p>Clusters: {analysisData.centroids.length}</p>
                        {analysisData.metrics.trustworthiness && (
                            <p>Trustworthiness: {analysisData.metrics.trustworthiness}</p>
                        )}
                    </div>
                )}
            </div>
        </div>
    );
};

export default PhaseSpaceViewer;
