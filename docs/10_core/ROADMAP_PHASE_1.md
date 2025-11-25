Roadmap Fase 1: El Despertar del Vacío

Objetivo: Implementar el motor disperso y lograr la primera estructura estable en un universo infinito.

Tareas Prioritarias

[ ] Integración de Ruido (Physics):

Modificar src/trainer.py para usar src/physics/noise.py.

Entrenar un modelo nuevo (UNET_UNITARY) bajo condiciones de ruido IonQ alto.

[ ] Visualización 3D (Frontend):

Implementar HolographicViewer.tsx usando Three.js.

Conectar el WebSocket para recibir viewport_tensor en lugar de map_data plano.

[ ] Motor Disperso (Engine):

Finalizar src/engines/harmonic_engine.py.

Crear un script de prueba tests/test_infinite_universe.py que inyecte una "Semilla de Génesis" y la deje correr.

[ ] Detección de Épocas (Analysis):

Conectar EpochDetector al dashboard del frontend para ver una barra de progreso de "Evolución del Universo".