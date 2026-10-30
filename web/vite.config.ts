import { defineConfig, type Plugin } from "vite";
import react from "@vitejs/plugin-react";
import { parse } from "yaml";

// Phase A only: the orchestrator's machines.yaml (D96) is compiled to JSON at build time.
// Phase B serves the same file through GET /machines and this plugin goes away.
function yamlAsJson(): Plugin {
  return {
    name: "kta-yaml-as-json",
    transform(code, id) {
      if (!id.endsWith(".yaml")) return null;
      return { code: `export default ${JSON.stringify(parse(code))};`, map: null };
    },
  };
}

export default defineConfig({
  plugins: [react(), yamlAsJson()],
  server: { fs: { allow: [".."] } },
});
