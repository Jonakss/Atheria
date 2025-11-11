const vizOptions = [
  {
    label: 'Análisis de Grid',
    options: [
      { value: 'density', label: 'Densidad' },
      { value: 'state_change', label: 'Magnitud del Cambio' },
      { value: 'channels', label: 'Canales RGB' },
      { value: 'aggregate_phase', label: 'Fase Agregada' },
      { value: 'fft', label: 'Transformada de Fourier 2D' },
    ],
  },
  {
    label: 'Análisis Temporal y Estadístico',
    options: [
      { value: 'spacetime_slice', label: 'Diagrama Espacio-Tiempo' },
      { value: 'spacetime_cube', label: 'Cubo Espacio-Tiempo' },
      { value: 'poincare', label: 'Gráfico de Poincaré' },
      { value: 'density_histogram', label: 'Histograma de Densidad' },
    ],
  },
];

export default vizOptions;