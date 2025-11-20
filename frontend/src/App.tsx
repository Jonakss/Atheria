// frontend/src/App.tsx
import { ErrorBoundary } from './components/ui/ErrorBoundary';
import { DashboardLayout } from './modules/Dashboard/layouts/DashboardLayout';

function App() {
    return (
        // Error Boundary para capturar errores y evitar pantalla gris
        // El MantineProvider ya está en main.tsx con el tema personalizado
        <ErrorBoundary>
            <DashboardLayout />
        </ErrorBoundary>
    );
}

export default App;