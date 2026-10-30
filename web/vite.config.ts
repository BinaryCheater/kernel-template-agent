import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

// The KTA API listens on the control host ([api] in config/kta.toml); the dev server proxies
// /api to it so the browser talks to one origin.
const API = process.env.KTA_API ?? "http://127.0.0.1:7310";

export default defineConfig({
  plugins: [react()],
  server: {
    proxy: { "/api": { target: API, changeOrigin: false, rewrite: (p) => p.replace(/^\/api/, "") } },
  },
});
