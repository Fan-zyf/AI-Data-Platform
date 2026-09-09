import { useEffect, useState } from 'react'
import './App.css'
import UploadPanel from './upload/UploadPanel.jsx'

const STATUS_META = {
  checking: {
    label: '正在检测后端连接…',
    className: 'status-checking',
    dotClass: 'dot-checking',
  },
  connected: {
    label: '后端服务已连接',
    className: 'status-connected',
    dotClass: 'dot-ok',
  },
  disconnected: {
    label: '后端服务未连接',
    className: 'status-disconnected',
    dotClass: 'dot-error',
  },
}

function SystemStatus() {
  const [state, setState] = useState('checking')
  const [message, setMessage] = useState('')

  useEffect(() => {
    let cancelled = false

    fetch('/api/health')
      .then((res) => {
        if (!res.ok) throw new Error(`HTTP ${res.status}`)
        return res.json()
      })
      .then((data) => {
        if (cancelled) return
        setMessage(data.message ?? '')
        setState(data.status === 'ok' ? 'connected' : 'disconnected')
      })
      .catch(() => {
        if (!cancelled) setState('disconnected')
      })

    return () => {
      cancelled = true
    }
  }, [])

  const meta = STATUS_META[state]

  return (
    <section className="status-card" aria-live="polite">
      <h2 className="status-title">系统状态</h2>
      <div className={`status-row ${meta.className}`}>
        <span className={`dot ${meta.dotClass}`} />
        <span className="status-label">{meta.label}</span>
      </div>
      {message && <p className="status-message">{message}</p>}
      <p className="status-hint">
        提示：请先启动后端服务
        （<code>uvicorn app.main:app --reload</code>），再刷新本页面即可看到连接成功。
      </p>
    </section>
  )
}

function App() {
  return (
    <div className="page">
      <main className="content">
        <header className="hero">
          <span className="badge">v0.2.0 · Data Upload</span>
          <h1 className="title">AI Data Intelligence Platform</h1>
          <p className="subtitle">智能数据分析与预测平台</p>
          <p className="intro">
            当前已支持 CSV / Excel 数据上传与基础画像，后续将逐步支持数据质量检测、
            EDA、特征工程、机器学习建模、模型解释与 LLM 分析。
          </p>
          <SystemStatus />
        </header>

        <UploadPanel />
      </main>
    </div>
  )
}

export default App
