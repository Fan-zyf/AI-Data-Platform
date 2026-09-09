import { forwardRef, useImperativeHandle, useState } from 'react'
import MissingChart from './MissingChart.jsx'
import NumericTable from './NumericTable.jsx'
import HistogramPanel from './HistogramPanel.jsx'
import CategoricalPanel from './CategoricalPanel.jsx'
import CorrelationHeatmap from './CorrelationHeatmap.jsx'
import OutlierPanel from './OutlierPanel.jsx'

function MetricCard({ label, value, accent }) {
  return (
    <div className={`metric-card ${accent || ''}`}>
      <span className="metric-value">{value}</span>
      <span className="metric-label">{label}</span>
    </div>
  )
}

function shortId(datasetId) {
  return datasetId && datasetId.length > 8 ? `${datasetId.slice(0, 8)}…` : datasetId
}

function formatExpiry(value) {
  if (!value) return ''
  const date = new Date(value)
  return Number.isNaN(date.getTime()) ? String(value) : date.toLocaleString()
}

function versionLabel(versionId) {
  return !versionId || versionId === 'original' ? '原始数据 (Original)' : `版本 ${shortId(versionId)}`
}

const EDASection = forwardRef(function EDASection(
  { datasetId, datasetName, expiresAt, onCleared, initialVersionId = 'original' },
  ref,
) {
  const [loading, setLoading] = useState(false)
  const [clearing, setClearing] = useState(false)
  const [error, setError] = useState(null)
  const [eda, setEda] = useState(null)
  const [runVersionId, setRunVersionId] = useState(initialVersionId)

  const analyze = async (versionId) => {
    const target = versionId || 'original'
    setLoading(true)
    setError(null)
    try {
      const query = target && target !== 'original' ? `?version_id=${encodeURIComponent(target)}` : ''
      const res = await fetch(`/api/datasets/${datasetId}/eda${query}`)
      const payload = await res.json().catch(() => null)
      if (!res.ok) {
        const message =
          payload?.error?.message ||
          (res.status === 404
            ? '数据集或数据版本不存在/已过期，请重新上传'
            : `EDA 请求失败（HTTP ${res.status}）`)
        throw new Error(message)
      }
      if (!payload || payload.success !== true) {
        throw new Error(payload?.error?.message || '服务器返回了无法识别的 EDA 数据')
      }
      setEda(payload)
      setRunVersionId(target)
    } catch (err) {
      setError(err.message || 'EDA 计算失败，请稍后重试')
    } finally {
      setLoading(false)
    }
  }

  // 对外暴露：供 Data Processing / 版本历史“重新 EDA”调用
  useImperativeHandle(ref, () => ({ analyze }), [datasetId])

  const clearDataset = async () => {
    if (clearing || !window.confirm('确认清除当前数据集会话？服务端的临时文件将被删除。')) return
    setClearing(true)
    setError(null)
    try {
      await onCleared()
    } finally {
      setClearing(false)
    }
  }

  const summary = eda?.summary || null
  const shownLabel = versionLabel(runVersionId)

  return (
    <section className="panel eda-section">
      <div className="eda-header">
        <div className="eda-heading">
          <h3 className="panel-title eda-title">自动 EDA</h3>
          <p className="eda-meta">
            数据集 <code>{datasetName || datasetId}</code> · 会话 ID{' '}
            <code title={datasetId}>{shortId(datasetId)}</code>
            {runVersionId && (
              <>
                {' '}· 当前版本 <code className="chip">{shownLabel}</code>
              </>
            )}
            {expiresAt && (
              <>
                {' '}· 到期时间 <code>{formatExpiry(expiresAt)}</code>
              </>
            )}
          </p>
        </div>
        <div className="eda-actions">
          <button type="button" className="btn-primary" disabled={loading || clearing} onClick={() => analyze(runVersionId)}>
            {loading ? (
              <>
                <span className="spinner" aria-hidden="true" /> 计算中…
              </>
            ) : eda ? (
              '重新运行 EDA'
            ) : (
              '开始自动 EDA'
            )}
          </button>
          <button
            type="button"
            className="btn-danger"
            disabled={loading || clearing}
            onClick={clearDataset}
          >
            {clearing ? '清理中…' : '清除当前数据集'}
          </button>
        </div>
      </div>

      <p className="eda-desc">
        对当前数据版本执行只读统计分析与规则型洞察（描述统计 / 直方图 / 分类 Top-N / 缺失 / IQR
        异常值 / Pearson 相关），结果全部基于真实统计计算，不修改任何原始数据。v0.4 起可对比
        Original 与各清洗版本的 EDA。
      </p>

      {!eda && !error && !loading && (
        <p className="panel-empty">尚未运行 EDA，点击「开始自动 EDA」生成统计与图表。</p>
      )}

      {error && (
        <div className="error-box" role="alert">
          <strong>自动 EDA 失败</strong>
          <span>{error}</span>
        </div>
      )}

      {eda && summary && (
        <>
          <div className="metric-grid">
            <MetricCard label="数值字段" value={summary.numeric_columns} accent="accent-blue" />
            <MetricCard
              label="分类字段"
              value={summary.categorical_columns}
              accent="accent-violet"
            />
            <MetricCard
              label="高缺失字段"
              value={summary.high_missing_columns}
              accent={summary.high_missing_columns > 0 ? 'accent-amber' : 'accent-green'}
            />
            <MetricCard
              label="异常值字段"
              value={summary.columns_with_outliers}
              accent={summary.columns_with_outliers > 0 ? 'accent-amber' : 'accent-green'}
            />
            <MetricCard
              label="强相关对"
              value={summary.strong_correlations}
              accent={summary.strong_correlations > 0 ? 'accent-amber' : 'accent-green'}
            />
          </div>

          <section className="panel eda-inner insights">
            <h4 className="eda-subtitle">规则型洞察</h4>
            {summary.rule_based_insights?.length ? (
              <ul className="insights-list">
                {summary.rule_based_insights.map((text, index) => (
                  <li key={`${index}-${text}`} className="insight-item">
                    <span className="insight-dot" aria-hidden="true" />
                    {text}
                  </li>
                ))}
              </ul>
            ) : (
              <p className="panel-empty">暂无洞察。</p>
            )}
          </section>

          <section className="panel eda-inner">
            <h4 className="eda-subtitle">数值描述统计</h4>
            <NumericTable summaries={eda.numeric_summaries} />
          </section>

          <section className="panel eda-inner">
            <h4 className="eda-subtitle">分布直方图（数值字段）</h4>
            <HistogramPanel distributions={eda.numeric_distributions} />
          </section>

          <section className="panel eda-inner">
            <h4 className="eda-subtitle">缺失值分析</h4>
            <MissingChart missing={eda.missing_analysis} />
          </section>

          <section className="panel eda-inner">
            <h4 className="eda-subtitle">分类字段 Top 取值</h4>
            <CategoricalPanel summaries={eda.categorical_summaries} />
          </section>

          <section className="panel eda-inner">
            <h4 className="eda-subtitle">Pearson 相关矩阵</h4>
            <CorrelationHeatmap correlation={eda.correlation} />
          </section>

          <section className="panel eda-inner">
            <h4 className="eda-subtitle">IQR 异常值检测</h4>
            <OutlierPanel outliers={eda.outlier_analysis} />
          </section>
        </>
      )}
    </section>
  )
})

export default EDASection
