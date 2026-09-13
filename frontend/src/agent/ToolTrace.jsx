const TOOL_LABEL = {
  dataset: '读取数据集元信息',
  eda: '执行自动 EDA',
  ml: '读取模型评估',
  shap: '加载 SHAP 解释',
  report: '汇总分析报告',
}

const STATUS_ICON = {
  ok: '✓',
  skipped: '⊘',
  error: '✕',
}

const STATUS_CLASS = {
  ok: 'tool-status-ok',
  skipped: 'tool-status-skip',
  error: 'tool-status-err',
}

/**
 * 工具执行轨迹（不展示真实 chain-of-thought，仅工具调用清单与摘要）。
 */
export default function ToolTrace({ tools, plan }) {
  if (!tools || !tools.length) {
    return <p className="panel-empty">没有调用任何工具。</p>
  }
  return (
    <div className="agent-trace">
      {plan && (
        <div className="agent-trace-plan">
          <span className="ml-tip">计划：</span>
          {(plan.selected || []).map((name) => (
            <span key={`s-${name}`} className="agent-chip agent-chip-selected">
              {TOOL_LABEL[name] || name}
            </span>
          ))}
          {(plan.skipped || []).map((s) => (
            <span
              key={`k-${s.name}`}
              className="agent-chip agent-chip-skipped"
              title={s.reason}
            >
              ⊘ {TOOL_LABEL[s.name] || s.name}
            </span>
          ))}
        </div>
      )}
      <ol className="agent-trace-list">
        {tools.map((t) => (
          <li key={t.name} className={`agent-trace-item ${STATUS_CLASS[t.status] || ''}`}>
            <span className="agent-trace-icon">{STATUS_ICON[t.status] || '?'}</span>
            <div className="agent-trace-text">
              <div className="agent-trace-title">
                {TOOL_LABEL[t.name] || t.name}
                {t.status === 'skipped' && <span className="ml-tip">（跳过）</span>}
                {t.status === 'error' && t.error && (
                  <span className="agent-trace-error"> · {t.error}</span>
                )}
              </div>
              {t.summary && <div className="agent-trace-summary">{t.summary}</div>}
            </div>
          </li>
        ))}
      </ol>
    </div>
  )
}
