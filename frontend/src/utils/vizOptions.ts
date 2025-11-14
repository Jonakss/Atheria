// frontend/src/utils/vizOptions.ts

export const modelOptions = [
    { value: 'UNET', label: 'UNet' },
    { value: 'SNN_UNET', label: 'SNN UNet' },
    { value: 'DEEP_QCA', label: 'Deep QCA' },
    { value: 'MLP', label: 'MLP' },
    { value: 'UNET_UNITARY', label: 'UNet Unitary' },
];

export const vizOptions = [
    { value: 'density', label: 'Densidad' },
    { value: 'phase', label: 'Fase' },
    { value: 'poincare', label: 'Gráfico de Poincaré' },
];
