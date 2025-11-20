import { defineConfig } from 'vite';
import react from '@vitejs/plugin-react';
import { readFileSync } from 'fs';
import { join } from 'path';

// Leer versión del package.json
const packageJson = JSON.parse(readFileSync(join(__dirname, 'package.json'), 'utf-8'));

// https://vitejs.dev/config/
export default defineConfig({
  plugins: [react()],
  define: {
    'import.meta.env.APP_VERSION': JSON.stringify(packageJson.version),
  },
});

