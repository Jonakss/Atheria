import react from "@vitejs/plugin-react";
import { readFileSync } from "fs";
import { join } from "path";
import { defineConfig } from "vite";

// Leer versión del package.json
const packageJson = JSON.parse(
  readFileSync(join(__dirname, "package.json"), "utf-8")
);

// https://vitejs.dev/config/
export default defineConfig({
  plugins: [react()],
  base: "/Atheria/", // Base path para GitHub Pages
  define: {
    "import.meta.env.APP_VERSION": JSON.stringify(packageJson.version),
  },
  server: {
    port: 3000,
    proxy: {
      "/ws": {
        target: "ws://localhost:8000",
        ws: true,
        changeOrigin: true,
      },
      "/api": {
        target: "http://localhost:8000",
        changeOrigin: true,
      },
    },
  },
});
