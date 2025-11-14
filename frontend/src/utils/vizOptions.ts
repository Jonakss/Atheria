// frontend/src/utils/vizOptions.ts

export const modelOptions = [
    { value: 'UNET', label: 'UNet' },
    { value: 'SNN_UNET', label: 'SNN UNet' },
    { value: 'DEEP_QCA', label: 'Deep QCA' },
    { value: 'MLP', label: 'MLP' },
    { value: 'UNET_UNITARY', label: 'UNet Unitary' },
];

export const vizOptions = [
    { value: 'density_map', label: 'Mapa de Densidad' },
    { value: 'channels_map', label: 'Mapa de Canales RGB' },
    { value: 'phase_map', label: 'Mapa de Fase' },
    { value: 'change_map', label: 'Mapa de Cambio' },
];
