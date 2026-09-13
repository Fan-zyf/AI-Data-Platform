/**
 * v0.7 AI Data Analyst Agent · 主面板
 *
 * 依赖：AgentChat（消息渲染）+ ToolTrace（工具执行轨迹）。
 * 设计目标：
 * - 1 个输入框 + 1 个数据集选择器（localStorage 共享最近一次上传的 dataset_id）
 * - 1 个「开始分析」按钮
 * - 1 个实验 ID 可选输入（未提供则只走 dataset / eda 工具）
 * - 输出：消息列表 + 最新一次调用的工具 trace
 */
import { useEffect, useMemo, useState } from 'react'
import AgentChat from './AgentChat.jsx'
import ToolTrace from './ToolTrace.jsx'

const QUESTION_PRESETS = [
  '这个数据集有什么问题？',
  '模型表现如何？哪些特征最重要？',
  '为什么这个模型效果不好？',
  '请基于 SHAP 给我改进建议。',
  '总结当前数据与模型的整体状况。',
]

const STORAGE_KEY_DATASET = 'ai_data_platform:last_dataset_id'
const STORAGE_KEY_EXPERIMENT = 'ai_data_platform:last_experiment_id'

function readStorage(key) {
  if (typeof window === 'undefined') return ''
  try {
    return window.localStorage.getItem(key) || ''
  } catch (_) {
    return ''
  }
}

function writeStorage(key, value) {
  if (typeof window === 'undefined') return
  try {
    if (value) window.localStorage.setItem(key, value)
    else window.localStorage.removeItem(key)
  } catch (_) {
    /* ignore */
  }
}

export default function AgentPanel({ healthOk }) {
  const [datasetId, setDatasetId] = useState(() => readStorage(STORAGE_KEY_DATASET))
  const [experimentId, setExperimentId] = useState(() => readStorage(STORAGE_KEY_EXPERIMENT))
  const [question, setQuestion] = useState('为什么这个模型效果不好？哪些特征影响最大？')
  const [messages, setMessages] = useState([])
  const [lastResult, setLastResult] = useState(null)
  const [error, setError] = useState('')
  const [loading, setLoading] = useState(false)

  useEffect(() => {
    writeStorage(STORAGE_KEY_DATASET, datasetId.trim())
  }, [datasetId])
  useEffect(() => {
    writeStorage(STORAGE_KEY_EXPERIMENT, experimentId.trim())
  }, [experimentId])

  const ready = useMemo(() => healthOk && datasetId.trim().length > 0, [healthOk, datasetId])

  const handleSubmit = async (event) => {
    event?.preventDefault?.()
    if (!ready || loading) return
    const q = question.trim()
    const ds = datasetId.trim()
    const ex = experimentId.trim()
    if (!q) {
      setError('请输入问题。')
      return
    }
    setError('')
    setLoading(true)
    const userMsg = { role: 'user', text: q, at: new Date().toISOString() }
    setMessages((prev) => [...prev, userMsg])
    try {
      const resp = await fetch('/api/agent/analyze', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          question: q,
          dataset_id: ds,
          experiment_id: ex || null,
        }),
      })
      const payload = await resp.json().catch(() => null)
      if (!resp.ok || !payload || payload.success === false) {
        const code = payload?.error?.code || `HTTP ${resp.status}`
        const detail = payload?.error?.message || resp.statusText
        throw new Error(`${code}：${detail}`)
      }
      setLastResult(payload)
      setMessages((prev) => [
        ...prev,
        { role: 'assistant', payload, at: new Date().toISOString() },
      ])
    } catch (err) {
      setError(err?.message || '调用失败')
    } finally {
      setLoading(false)
    }
  }

  const handleClear = () => {
    setMessages([])
    setLastResult(null)
    setError('')
  }

  return (
    <section className="upload-panel" id="agent-panel">
      <h2 className="section-title">
        AI Data Analyst Agent
        <span className="section-tag">v0.7 · Natural Language Q&amp;A</span>
      </h2>
      <p className="section-desc">
        基于工具调用（EDA / ML / SHAP / Dataset / Report）+ LLM 汇总的结构化分析。 无 LLM 配置时自动降级 Mock。
      </p>

      <div className="panel">
        <form className="agent-form" onSubmit={handleSubmit}>
          <div className="agent-form-grid">
            <label className="ml-field">
              <span className="ml-field-label">dataset_id</span>
              <input
                type="text"
                value={datasetId}
                onChange={(e) => setDatasetId(e.target.value)}
                placeholder="上传后自动写入此处"
                spellCheck={false}
              />
            </label>
            <label className="ml-field">
              <span className="ml-field-label">experiment_id（可选）</span>
              <input
                type="text"
                value={experimentId}
                onChange={(e) => setExperimentId(e.target.value)}
                placeholder="训练后自动写入此处"
                spellCheck={false}
              />
            </label>
          </div>
          <label className="ml-field agent-question-field">
            <span className="ml-field-label">问题</span>
            <textarea
              value={question}
              onChange={(e) => setQuestion(e.target.value)}
              rows={3}
              placeholder="例如：为什么这个模型效果不好？哪些特征影响最大？"
            />
          </label>
          <div className="agent-presets">
            <span className="ml-tip">快捷问题：</span>
            {QUESTION_PRESETS.map((preset) => (
              <button
                key={preset}
                type="button"
                className="chip-btn"
                onClick={() => setQuestion(preset)}
              >
                {preset}
              </button>
            ))}
          </div>
          <div className="agent-actions">
            <button
              type="submit"
              className="btn-primary"
              disabled={!ready || loading || !question.trim()}
            >
              {loading ? (
                <>
                  <span className="spinner" />
                  正在调用 Agent...
                </>
              ) : (
                '开始分析'
              )}
            </button>
            <button
              type="button"
              className="btn-ghost"
              onClick={handleClear}
              disabled={loading}
            >
              清空对话
            </button>
            {!healthOk && (
              <span className="ml-tip">等待后端连接成功后即可使用...</span>
            )}
          </div>
        </form>
      </div>

      {error && (
        <div className="error-box" role="alert">
          <strong>调用失败</strong>
          <span>{error}</span>
        </div>
      )}

      {lastResult && (
        <div className="panel">
          <h3 className="panel-title">本次调用的工具</h3>
          <ToolTrace tools={lastResult.tools_used} plan={lastResult.plan} />
          <AgentResultSummary result={lastResult} />
        </div>
      )}

      {messages.length > 0 && (
        <div className="panel">
          <h3 className="panel-title">对话历史</h3>
          <AgentChat messages={messages} />
        </div>
      )}
    </section>
  )
}

function AgentResultSummary({ result }) {
  const insights = result?.insights || []
  const recs = result?.recommendations || []
  if (!insights.length && !recs.length) return null
  return (
    <div className="agent-summary">
      {insights.length > 0 && (
        <div className="agent-summary-block">
          <h4 className="agent-summary-title">关键洞察</h4>
          <ul className="insights-list">
            {insights.map((it, i) => (
              <li key={`i-${i}`} className="insight-item">
                <span className="insight-dot" />
                <span>{it}</span>
              </li>
            ))}
          </ul>
        </div>
      )}
      {recs.length > 0 && (
        <div className="agent-summary-block">
          <h4 className="agent-summary-title">可执行建议</h4>
          <ul className="insights-list">
            {recs.map((it, i) => (
              <li key={`r-${i}`} className="insight-item">
                <span className="insight-dot" />
                <span>{it}</span>
              </li>
            ))}
          </ul>
        </div>
      )}
    </div>
  )
}
