// frontend/src/components/ErrorBoundary.tsx
import { AlertTriangle } from 'lucide-react';
import { Component, ErrorInfo, ReactNode } from 'react';

interface Props {
    children: ReactNode;
}

interface State {
    hasError: boolean;
    error: Error | null;
    errorInfo: ErrorInfo | null;
}

export class ErrorBoundary extends Component<Props, State> {
    constructor(props: Props) {
        super(props);
        this.state = {
            hasError: false,
            error: null,
            errorInfo: null,
        };
    }

    static getDerivedStateFromError(error: Error): State {
        return {
            hasError: true,
            error,
            errorInfo: null,
        };
    }

    componentDidCatch(error: Error, errorInfo: ErrorInfo) {
        console.error('Error capturado por ErrorBoundary:', error, errorInfo);
        this.setState({
            error,
            errorInfo,
        });
    }

    handleReset = () => {
        this.setState({
            hasError: false,
            error: null,
            errorInfo: null,
        });
    };

    render() {
        if (this.state.hasError) {
            return (
                <div className="p-8 min-h-screen flex items-center justify-center bg-[#020202]">
                    <div className="max-w-2xl w-full bg-[#0a0a0a]/90 backdrop-blur-md border border-red-500/30 rounded-lg shadow-lg p-6">
                        {/* Header */}
                        <div className="flex items-center gap-3 mb-4">
                            <div className="w-10 h-10 rounded-full bg-red-500/10 flex items-center justify-center border border-red-500/30">
                                <AlertTriangle size={20} className="text-red-500" />
                            </div>
                            <div>
                                <h2 className="text-lg font-bold text-red-400">Error en la aplicación</h2>
                                <p className="text-sm text-gray-400">Ha ocurrido un error inesperado</p>
                            </div>
                        </div>

                        {/* Mensaje */}
                        <div className="mb-6">
                            <p className="text-sm text-gray-300">
                                Por favor, recarga la página o contacta al desarrollador si el problema persiste.
                            </p>
                        </div>

                        {/* Error Details */}
                        {this.state.error && (
                            <div className="mb-4 p-3 bg-red-500/5 border border-red-500/20 rounded">
                                <p className="text-xs font-bold text-red-400 mb-2 uppercase">Error:</p>
                                <pre className="text-xs text-gray-300 font-mono overflow-x-auto whitespace-pre-wrap break-words">
                                    {this.state.error.toString()}
                                </pre>
                            </div>
                        )}

                        {/* Stack Trace */}
                        {this.state.errorInfo && (
                            <div className="mb-6 p-3 bg-gray-800/50 border border-white/10 rounded max-h-48 overflow-auto">
                                <p className="text-xs font-bold text-gray-400 mb-2 uppercase">Stack trace:</p>
                                <pre className="text-xs text-gray-400 font-mono overflow-x-auto whitespace-pre-wrap break-words">
                                    {this.state.errorInfo.componentStack}
                                </pre>
                            </div>
                        )}

                        {/* Botón Reintentar */}
                        <button
                            onClick={this.handleReset}
                            className="w-full px-4 py-2 bg-red-500/10 hover:bg-red-500/20 border border-red-500/30 text-red-400 text-sm font-bold rounded transition-all"
                        >
                            Reintentar
                        </button>
                    </div>
                </div>
            );
        }

        return this.props.children;
    }
}

