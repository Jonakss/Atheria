// frontend/src/components/ErrorBoundary.tsx
import React, { Component, ErrorInfo, ReactNode } from 'react';
import { Alert, Button, Stack, Text, Code } from '@mantine/core';
import { IconAlertTriangle } from '@tabler/icons-react';

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
                <div style={{ 
                    padding: '2rem', 
                    minHeight: '100vh', 
                    display: 'flex', 
                    alignItems: 'center', 
                    justifyContent: 'center',
                    backgroundColor: 'var(--mantine-color-dark-8)'
                }}>
                    <Alert
                        icon={<IconAlertTriangle size={16} />}
                        title="Error en la aplicación"
                        color="red"
                        style={{ maxWidth: 600 }}
                    >
                        <Stack gap="md">
                            <Text size="sm">
                                Ha ocurrido un error inesperado. Por favor, recarga la página o contacta al desarrollador.
                            </Text>
                            
                            {this.state.error && (
                                <div>
                                    <Text size="xs" fw={700} mb="xs">Error:</Text>
                                    <Code block style={{ fontSize: '0.75rem' }}>
                                        {this.state.error.toString()}
                                    </Code>
                                </div>
                            )}
                            
                            {this.state.errorInfo && (
                                <div>
                                    <Text size="xs" fw={700} mb="xs">Stack trace:</Text>
                                    <Code block style={{ fontSize: '0.75rem', maxHeight: 200, overflow: 'auto' }}>
                                        {this.state.errorInfo.componentStack}
                                    </Code>
                                </div>
                            )}
                            
                            <Button onClick={this.handleReset} variant="light" color="red">
                                Reintentar
                            </Button>
                        </Stack>
                    </Alert>
                </div>
            );
        }

        return this.props.children;
    }
}

