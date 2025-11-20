/** @type {import('tailwindcss').Config} */
export default {
  content: [
    "./index.html",
    "./src/**/*.{js,ts,jsx,tsx}",
  ],
  theme: {
    extend: {
      colors: {
        // Paleta Teal & Rosa/Rojo (inspirada en infografía moderna)
        teal: {
          // Teal (verde azulado) - color principal
          50: '#f0fdfa',
          100: '#ccfbf1',
          200: '#99f6e4',
          300: '#5eead4',
          400: '#2dd4bf',
          500: '#14b8a6', // Teal principal
          600: '#0d9488',
          700: '#0f766e',
          800: '#115e59',
          900: '#134e4a',
          950: '#042f2e',
        },
        pink: {
          // Rosa/Rojo - color secundario para gradientes
          50: '#fdf2f8',
          100: '#fce7f3',
          200: '#fbcfe8',
          300: '#f9a8d4',
          400: '#f472b6',
          500: '#ec4899', // Rosa principal
          600: '#db2777',
          700: '#be185d',
          800: '#9f1239',
          900: '#831843',
          950: '#500724',
        },
        // Colores neutros oscuros (fondo)
        dark: {
          50: '#fafafa',
          100: '#f5f5f5',
          200: '#e5e5e5',
          300: '#d4d4d4',
          400: '#a3a3a3',
          500: '#737373',
          600: '#525252',
          700: '#404040',
          800: '#262626',
          900: '#171717',
          950: '#0a0a0a',
          980: '#080808', // Más oscuro
          990: '#050505', // Casi negro
          bg: '#020202', // Fondo principal
        },
      },
      // Efectos de brillo/glow
      boxShadow: {
        'glow-teal': '0 0 10px rgba(20, 184, 166, 0.4), 0 0 20px rgba(20, 184, 166, 0.2)',
        'glow-pink': '0 0 10px rgba(236, 72, 153, 0.4), 0 0 20px rgba(236, 72, 153, 0.2)',
        'glow-gradient': '0 0 20px rgba(20, 184, 166, 0.3), 0 0 40px rgba(236, 72, 153, 0.2)',
      },
      // Gradientes
      backgroundImage: {
        'gradient-teal-pink': 'linear-gradient(135deg, rgba(20, 184, 166, 0.2) 0%, rgba(236, 72, 153, 0.2) 100%)',
        'gradient-teal-pink-strong': 'linear-gradient(135deg, rgba(20, 184, 166, 0.4) 0%, rgba(236, 72, 153, 0.4) 100%)',
      },
    },
  },
  plugins: [],
}
