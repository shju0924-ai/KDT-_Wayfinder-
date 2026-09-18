import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

// /api 요청은 FastAPI(8000)로 프록시 — CORS 걱정 없이 개발
export default defineConfig({
  plugins: [react()],
  server: {
    proxy: {
      "/api": "http://localhost:8000",
    },
  },
});
