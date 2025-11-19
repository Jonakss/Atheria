// frontend/src/theme.ts
import { createTheme, MantineColorsTuple } from '@mantine/core';

// Función auxiliar para convertir hex a rgba
function hexToRgba(hex: string, alpha: number): string {
  const r = parseInt(hex.slice(1, 3), 16);
  const g = parseInt(hex.slice(3, 5), 16);
  const b = parseInt(hex.slice(5, 7), 16);
  return `rgba(${r}, ${g}, ${b}, ${alpha})`;
}

// Paleta de colores personalizada para Atheria (tonos azules y púrpuras para un look futurista)
const aetheriaBlue: MantineColorsTuple = [
  '#e6f0ff', // 0: lightest
  '#b3d4ff',
  '#80b8ff',
  '#4d9cff',
  '#1a80ff', // 4: default
  '#0066e6',
  '#004db3',
  '#003380',
  '#001a4d',
  '#00001a'  // 9: darkest
];

const aetheriaPurple: MantineColorsTuple = [
  '#f3e5ff',
  '#e1b3ff',
  '#cf80ff',
  '#bd4dff',
  '#ab1aff',
  '#9900e6',
  '#7700b3',
  '#550080',
  '#33004d',
  '#11001a'
];

export const theme = createTheme({
  primaryColor: 'blue',
  colors: {
    blue: aetheriaBlue,
    purple: aetheriaPurple,
  },
  fontFamily: 'Inter, -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif',
  headings: {
    fontFamily: 'Inter, -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif',
    fontWeight: '600',
    sizes: {
      h1: { fontSize: '2.5rem', lineHeight: '1.2' },
      h2: { fontSize: '2rem', lineHeight: '1.3' },
      h3: { fontSize: '1.5rem', lineHeight: '1.4' },
      h4: { fontSize: '1.25rem', lineHeight: '1.5' },
      h5: { fontSize: '1.125rem', lineHeight: '1.5' },
      h6: { fontSize: '1rem', lineHeight: '1.5' },
    },
  },
  defaultRadius: 'md',
  shadows: {
    xs: '0 1px 3px rgba(0, 0, 0, 0.12), 0 1px 2px rgba(0, 0, 0, 0.24)',
    sm: '0 2px 4px rgba(0, 0, 0, 0.12), 0 2px 3px rgba(0, 0, 0, 0.24)',
    md: '0 4px 8px rgba(0, 0, 0, 0.15), 0 3px 6px rgba(0, 0, 0, 0.20)',
    lg: '0 8px 16px rgba(0, 0, 0, 0.15), 0 5px 10px rgba(0, 0, 0, 0.20)',
    xl: '0 12px 24px rgba(0, 0, 0, 0.20), 0 8px 16px rgba(0, 0, 0, 0.25)',
  },
  spacing: {
    xs: '0.5rem',
    sm: '0.75rem',
    md: '1rem',
    lg: '1.5rem',
    xl: '2rem',
  },
  other: {
    transitionDuration: '200ms',
    transitionTimingFunction: 'ease-in-out',
  },
  // Componentes personalizados con hover effects y transiciones
  components: {
    Button: {
      styles: (theme) => ({
        root: {
          transition: 'all 0.2s ease-in-out',
          '&:hover': {
            transform: 'translateY(-1px)',
            boxShadow: theme.shadows.md,
          },
          '&:active': {
            transform: 'translateY(0)',
          },
        },
      }),
      defaultProps: {
        radius: 'md',
      },
    },
    ActionIcon: {
      styles: (theme) => ({
        root: {
          transition: 'all 0.2s ease-in-out',
          '&:hover': {
            transform: 'scale(1.1)',
            backgroundColor: theme.colors.blue[7],
          },
        },
      }),
    },
    Card: {
      styles: (theme) => ({
        root: {
          transition: 'all 0.3s ease-in-out',
          '&:hover': {
            transform: 'translateY(-2px)',
            boxShadow: theme.shadows.lg,
            borderColor: theme.colors.blue[6],
          },
        },
      }),
    },
    Paper: {
      styles: (theme) => ({
        root: {
          transition: 'all 0.3s ease-in-out',
          '&:hover': {
            boxShadow: theme.shadows.md,
          },
        },
      }),
    },
    NavLink: {
      styles: (theme) => ({
        root: {
          transition: 'all 0.2s ease-in-out',
          borderRadius: theme.radius.md,
          '&:hover': {
            backgroundColor: theme.colors.blue[8],
            transform: 'translateX(4px)',
          },
          '&[data-active="true"]': {
            backgroundColor: theme.colors.blue[7],
            borderLeft: `3px solid ${theme.colors.blue[4]}`,
          },
        },
        label: {
          fontWeight: 500,
        },
      }),
    },
    Badge: {
      styles: (theme) => ({
        root: {
          transition: 'all 0.2s ease-in-out',
          '&:hover': {
            transform: 'scale(1.05)',
          },
        },
      }),
    },
    Tabs: {
      styles: (theme) => ({
        tab: {
          transition: 'all 0.2s ease-in-out',
          '&:hover': {
            backgroundColor: theme.colors.blue[8],
            color: theme.colors.blue[2],
          },
          '&[data-active="true"]': {
            borderColor: theme.colors.blue[6],
            backgroundColor: hexToRgba(theme.colors.blue[6], 0.1),
          },
        },
      }),
    },
    Select: {
      styles: (theme) => ({
        input: {
          transition: 'all 0.2s ease-in-out',
          '&:hover': {
            borderColor: theme.colors.blue[6],
          },
          '&:focus': {
            borderColor: theme.colors.blue[5],
            boxShadow: `0 0 0 2px ${hexToRgba(theme.colors.blue[5], 0.2)}`,
          },
        },
      }),
    },
    NumberInput: {
      styles: (theme) => ({
        input: {
          transition: 'all 0.2s ease-in-out',
          '&:hover': {
            borderColor: theme.colors.blue[6],
          },
          '&:focus': {
            borderColor: theme.colors.blue[5],
            boxShadow: `0 0 0 2px ${hexToRgba(theme.colors.blue[5], 0.2)}`,
          },
        },
      }),
    },
    Switch: {
      styles: (theme) => ({
        track: {
          transition: 'all 0.2s ease-in-out',
          '&:hover': {
            borderColor: theme.colors.blue[6],
          },
        },
      }),
    },
  },
});

