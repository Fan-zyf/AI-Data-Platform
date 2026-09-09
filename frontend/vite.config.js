import { defineConfig, loadEnv } from 'vite'
import react from '@vitejs/plugin-react'

// https://vite.dev/config/
export default defineConfig(({ mode }) => {
  const env = loadEnv(mode, process.cwd(), '')

  return {
    plugins: [react()],
    server: {
      port: 5173,
      proxy: {
        // 前端 /api/** 请求代理到后端 FastAPI，解决开发环境跨域
        // 注意：目标地址使用 127.0.0.1，避免 localhost 被解析为 IPv6(::1)
        // 而后端只监听 IPv4 时产生连接失败（502）
        '/api': {
          target: env.VITE_PROXY_TARGET || 'http://127.0.0.1:8000',
          changeOrigin: true,
        },
      },
    },
  }
})
