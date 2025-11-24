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
        // Colores "Deep Space" - más pulidos, con un ligero tinte azulado/pizarra en lugar de negro puro
        dark: {
          50: '#f8fafc',
          100: '#f1f5f9',
          200: '#e2e8f0',
          300: '#cbd5e1',
          400: '#94a3b8',
          500: '#64748b',
          600: '#475569',
          700: '#334155',
          800: '#1e293b',
          900: '#0f172a',  // Dark slate
          950: '#0b0f19',  // Mucho más rico que #0a0a0a
          980: '#05070a',  // Casi negro, pero con tinte
          990: '#030406',  // Muy oscuro
          bg: '#040507',   // Fondo principal - Ligeramente más claro y azulado que #020202
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
        'gradient-deep-space': 'linear-gradient(to bottom, #0b0f19 0%, #040507 100%)', // Nuevo gradiente de fondo
      },
    },
  },
  plugins: [],
}
