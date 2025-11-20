// frontend/src/components/ui/LoadingSkeleton.tsx
import Skeleton, { SkeletonTheme } from 'react-loading-skeleton';
import 'react-loading-skeleton/dist/skeleton.css';

interface LoadingSkeletonProps {
    count?: number;
    height?: number | string;
    width?: number | string;
    circle?: boolean;
    className?: string;
}

export function LoadingSkeleton({ 
    count = 1, 
    height = '1rem', 
    width = '100%', 
    circle = false,
    className 
}: LoadingSkeletonProps) {
    // Design System: Usar colores oscuros siempre (Atheria usa dark theme)
    const baseColor = '#080808'; // Design System: #080808
    const highlightColor = '#0a0a0a'; // Design System: #0a0a0a

    return (
        <SkeletonTheme baseColor={baseColor} highlightColor={highlightColor}>
            <Skeleton
                count={count}
                height={height}
                width={width}
                circle={circle}
                className={className}
                duration={1.2}
            />
        </SkeletonTheme>
    );
}

// Componentes predefinidos útiles
export function CardSkeleton() {
    return (
        <div style={{ padding: '1rem' }}>
            <div style={{ marginBottom: '0.5rem' }}>
                <LoadingSkeleton height={24} width="60%" />
            </div>
            <div style={{ marginBottom: '0.25rem' }}>
                <LoadingSkeleton count={3} height={16} />
            </div>
        </div>
    );
}

export function ListSkeleton({ count = 5 }: { count?: number }) {
    return (
        <div>
            {Array.from({ length: count }).map((_, i) => (
                <div key={i} style={{ padding: '0.5rem 0', display: 'flex', gap: '1rem' }}>
                    <LoadingSkeleton circle width={40} height={40} />
                    <div style={{ flex: 1 }}>
                        <div style={{ marginBottom: '0.5rem' }}>
                            <LoadingSkeleton height={16} width="80%" />
                        </div>
                        <LoadingSkeleton height={12} width="60%" />
                    </div>
                </div>
            ))}
        </div>
    );
}

export function TableSkeleton({ rows = 5, columns = 4 }: { rows?: number; columns?: number }) {
    return (
        <div>
            {/* Header */}
            <div style={{ display: 'flex', gap: '1rem', padding: '0.5rem 0', borderBottom: '1px solid rgba(255, 255, 255, 0.1)' }}>
                {Array.from({ length: columns }).map((_, i) => (
                    <LoadingSkeleton key={i} height={16} width="100%" />
                ))}
            </div>
            {/* Rows */}
            {Array.from({ length: rows }).map((_, rowIndex) => (
                <div key={rowIndex} style={{ display: 'flex', gap: '1rem', padding: '0.5rem 0' }}>
                    {Array.from({ length: columns }).map((_, colIndex) => (
                        <LoadingSkeleton key={colIndex} height={14} width="100%" />
                    ))}
                </div>
            ))}
        </div>
    );
}

