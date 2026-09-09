import { useCallback, useEffect, useMemo, useState } from 'react'
import TransformationBuilder from './TransformationBuilder.jsx'
import VersionHistory from './VersionHistory.jsx'
import VersionCompare from './VersionCompare.jsx'
import DataTable from './DataTable.jsx'
import { stepToOperation, missingParams } from './opDefinitions.js'

function shortId(value) {
  if (!value) return ''
  return value.length > 8 ? `${value.slice(0, 8)}…` : value
}

function Metric({ label, value, accent = '', suffix = '' }) {
  return (
    <div className={`metric-card ${accent}`}>
      <span className="metric-value">
        {value}
        {suffix && <small className="metric-suffix">{suffix}</small>}
      </span>
      <span className="metric-label">{label}</span>
    </div>
  )
}

async function apiFetch(path, options) {
  const res = await fetch(path, options)
  const payload = await res.json().catch(() => null)
  if (!res.ok) {
    throw new Error(payload?.error?.message || `请求失败（HTTP ${res.status}）`)
  }
  return payload
}

function PreviewTabs({ preview }) {
  const [tab, setTab] = useState('before')
  if (!preview) return null
  const active = tab === 'before' ? preview.preview_before : preview.preview_after
  const rowsBefore = preview.rows_before
  const rowsAfter = preview.rows_after
  return (
    <div className="dp-preview">
      <div className="dp-preview-toolbar">
        <div className="dp-preview-delta">
          <span>行：{preview.rows_before} → {preview.rows_after}</span>
          <span>列：{preview.columns_before} → {preview.columns_after}</span>
          <span>缺失：{preview.missing_before} → {preview.missing_after}</span>
          <span>重复：{preview.duplicates_before} → {preview.duplicates_after}</span>
        </div>
        <div className="dp-tabs">
          <button type="button" className={`dp-tab ${tab === 'before' ? 'active' : ''}`} onClick={() => setTab('before')}>
            处理前 Preview Before
          </button>
          <button type="button" className={`dp-tab ${tab === 'after' ? 'active' : ''}`} onClick={() => setTab('after')}>
            处理后 Preview After
          </button>
        </div>
      </div>
      {preview.generated_columns.length > 0 && (
        <p className="dp-warning-line">
          新增字段：{preview.generated_columns.slice(0, 8).map((c) => <code key={c} className="op-chip good">+ {c}</code>)}{' '}
          {preview.generated_columns.length > 8 && <span>…共 {preview.generated_columns.length} 个</span>}
        </p>
      )}
      {preview.removed_columns.length > 0 && (
        <p className="dp-warning-line">
          移除字段：{preview.removed_columns.slice(0, 8).map((c) => <code key={c} className="op-chip bad">− {c}</code>)}
        </p>
      )}
      <div className="dp-warnings">
        {preview.warnings?.map((w, i) => (
          <div key={`${w}-${i}`} className="dp-warning">
            ⚠ {w}
          </div>
        ))}
      </div>
      <div className="dp-table-note">
        {tab === 'before' ? `处理前数据预览（${rowsBefore} 行中显示前 20 行）` : `处理后数据预览（${rowsAfter} 行中显示前 20 行）`}
      </div>
      <DataTable rows={active} maxRows={20} />
    </div>
  )
}

function versionMetricsFor(original, versions, sourceId) {
  if (sourceId === 'original' || !sourceId) return original
  return versions.find((v) => v.version_id === sourceId) || null
}

export default function DataProcessingPanel({ datasetId, onRunEda }) {
  const [original, setOriginal] = useState(null)
  const [versions, setVersions] = useState([])
  const [loadingVersions, setLoadingVersions] = useState(true)
  const [sourceId, setSourceId] = useState('original')
  const [sourceMeta, setSourceMeta] = useState(null) // 选中版本的 metrics（list 里来的）
  const [detail, setDetail] = useState(null) // 选中版本的列画像等
  const [detailLoading, setDetailLoading] = useState(false)
  const [steps, setSteps] = useState([])
  const [preview, setPreview] = useState(null)
  const [applyInfo, setApplyInfo] = useState(null)
  const [busyKey, setBusyKey] = useState('')
  const [error, setError] = useState(null)
  const [notice, setNotice] = useState(null)

  const loadDetail = useCallback(
    async (versionId) => {
      if (!datasetId || !versionId) return
      setDetailLoading(true)
      try {
        const body = await apiFetch(`/api/datasets/${datasetId}/versions/${versionId}`)
        setDetail(body)
      } catch (err) {
        setDetail(null)
      } finally {
        setDetailLoading(false)
      }
    },
    [datasetId],
  )

  const refreshVersions = useCallback(
    async (nextSourceId) => {
      if (!datasetId) return
      setLoadingVersions(true)
      try {
        const body = await apiFetch(`/api/datasets/${datasetId}/versions`)
        setOriginal(body.original)
        setVersions(body.versions)
        const existIds = new Set(['original', ...body.versions.map((v) => v.version_id)])
        const target = nextSourceId && existIds.has(nextSourceId) ? nextSourceId : 'original'
        setSourceId(target)
        const item = target === 'original' ? body.original : body.versions.find((v) => v.version_id === target)
        setSourceMeta(item || null)
        await loadDetail(target)
      } catch (err) {
        setError(err.message || '无法加载数据版本列表')
      } finally {
        setLoadingVersions(false)
      }
    },
    [datasetId, loadDetail],
  )

  useEffect(() => {
    refreshVersions('original')
  }, [refreshVersions])

  const selectSource = async (versionId) => {
    if (!versionId || versionId === sourceId) return
    setSourceId(versionId)
    setError(null)
    setPreview(null)
    setApplyInfo(null)
    const item = versionId === 'original' ? original : versions.find((v) => v.version_id === versionId)
    setSourceMeta(item || null)
    await loadDetail(versionId)
  }

  const columns = useMemo(() => {
    if (!detail?.column_profiles) return []
    return detail.column_profiles.map((p) => ({
      name: p.column_name,
      type: p.inferred_type,
      missing_count: p.missing_count,
      unique_count: p.unique_count,
    }))
  }, [detail])

  const stepIssues = useMemo(() => {
    const issues = {}
    steps.forEach((step) => {
      const msg = missingParams(step)
      if (msg) issues[step.id] = msg
    })
    return issues
  }, [steps])

  const canPreview = steps.length > 0 && Object.keys(stepIssues).length === 0 && !busyKey && !!sourceId
  const canApply = canPreview

  const buildOperations = () => steps.map((step) => stepToOperation(step))

  const runPreview = async () => {
    setError(null)
    setNotice(null)
    setApplyInfo(null)
    setBusyKey('preview')
    try {
      const body = await apiFetch(`/api/datasets/${datasetId}/processing/preview`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ source_version_id: sourceId, operations: buildOperations() }),
      })
      setPreview(body)
    } catch (err) {
      setError(err.message)
      setPreview(null)
    } finally {
      setBusyKey('')
    }
  }

  const runApply = async () => {
    const confirmed = window.confirm(
      `将对版本「${sourceId === 'original' ? 'Original' : shortId(sourceId)}」按顺序执行 ${steps.length} 步操作并创建新版本。\n原始数据不会被修改，确认继续？`,
    )
    if (!confirmed) return
    setError(null)
    setNotice(null)
    setBusyKey('apply')
    try {
      const body = await apiFetch(`/api/datasets/${datasetId}/processing/apply`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ source_version_id: sourceId, operations: buildOperations() }),
      })
      setApplyInfo(body)
      setPreview(null)
      setSteps([])
      await refreshVersions(body.version_id)
      setNotice(`已创建新数据版本 ${shortId(body.version_id)}（父版本：${body.parent_version_id === 'original' ? 'Original' : shortId(body.parent_version_id)}）`)
    } catch (err) {
      setError(err.message)
    } finally {
      setBusyKey('')
    }
  }

  const deleteVersion = async (versionId) => {
    if (!window.confirm(`确认删除版本 ${shortId(versionId)}？此操作不可撤销。`)) return
    setError(null)
    setNotice(null)
    setBusyKey(`del-${versionId}`)
    try {
      await apiFetch(`/api/datasets/${datasetId}/versions/${versionId}`, { method: 'DELETE' })
      const nextSource = sourceId === versionId ? 'original' : sourceId
      await refreshVersions(nextSource)
      setNotice(`版本 ${shortId(versionId)} 已删除。`)
    } catch (err) {
      setError(err.message)
    } finally {
      setBusyKey('')
    }
  }

  const metrics = versionMetricsFor(original, versions, sourceId)
  const sourceLabel = sourceId === 'original' ? 'Original' : `Version ${shortId(sourceId)}`
  const profilesCount = columns.length

  return (
    <section className="panel data-processing-panel">
      <div className="eda-heading">
        <h3 className="panel-title eda-title">数据清洗与特征工程</h3>
        <p className="eda-meta">
          数据集会话 <code>{shortId(datasetId)}</code> · 处理基准：<code className="chip">{sourceLabel}</code>
          {detailLoading && <span className="dp-loading-tag">加载画像中…</span>}
        </p>
      </div>
      <p className="eda-desc">
        不可变数据管道：<strong>Original → Transformation Plan → Preview → Apply → 新数据版本</strong>。
        每一步都以结构化 JSON 提交（不接受任何代码），后端按顺序执行；先预览确认效果，再应用生成 UUID
        新版本，原始数据永远不会被修改。该模块用于通用数据准备与探索性处理。
      </p>

      {error && (
        <div className="error-box" role="alert">
          <strong>操作未完成</strong>
          <span>{error}</span>
        </div>
      )}
      {notice && (
        <div className="success-box" role="status">
          {notice}
        </div>
      )}

      <div className="dp-source-bar">
        <label className="dp-field dp-source-select">
          <span className="dp-field-label">处理基准版本（源）</span>
          <select
            value={sourceId}
            disabled={loadingVersions}
            onChange={(e) => selectSource(e.target.value)}
          >
            {original && <option value="original">Original（{original.rows} 行 / {original.columns} 列）</option>}
            {versions.map((v) => (
              <option key={v.version_id} value={v.version_id}>
                Version {shortId(v.version_id)}（{v.rows} 行 / {v.columns} 列）
              </option>
            ))}
          </select>
        </label>
        <div className="metric-grid dp-source-metrics">
          <Metric label="总行数" value={metrics ? metrics.rows : '—'} accent="accent-blue" />
          <Metric label="总列数" value={metrics ? metrics.columns : '—'} accent="accent-violet" />
          <Metric label="缺失值" value={metrics ? metrics.missing : '—'} accent="accent-amber" />
          <Metric label="重复行" value={metrics ? metrics.duplicates : '—'} accent="accent-green" />
        </div>
      </div>

      <div className="dp-main-grid">
        <div className="dp-main">
          <TransformationBuilder columns={columns} steps={steps} onStepsChange={setSteps} />
          {profilesCount === 0 && !detailLoading && (
            <p className="dp-loading-tag">正在加载字段画像…（字段画像来自版本详情接口）</p>
          )}

          {steps.length > 0 && (
            <div className="dp-actions">
              <button type="button" className="btn-ghost" disabled={!canPreview} onClick={runPreview}>
                {busyKey === 'preview' ? '预览中…' : '预览处理结果'}
              </button>
              <button type="button" className="btn-primary" disabled={!canApply} onClick={runApply}>
                {busyKey === 'apply' ? '创建中…' : `应用并创建新版本（${steps.length} 步）`}
              </button>
            </div>
          )}
          {Object.keys(stepIssues).length > 0 && (
            <p className="dp-hint">请先补全步骤参数（{Object.keys(stepIssues).length} 个步骤待完善）。</p>
          )}

          {preview && <PreviewTabs preview={preview} />}

          {applyInfo && (
            <div className="success-card">
              <h5>已创建新数据版本</h5>
              <div className="metric-grid">
                <Metric label="版本 ID" value={shortId(applyInfo.version_id)} accent="accent-blue" />
                <Metric label="父版本" value={applyInfo.parent_version_id === 'original' ? 'Original' : shortId(applyInfo.parent_version_id)} />
                <Metric label="行数" value={applyInfo.rows} accent="accent-violet" />
                <Metric label="列数" value={applyInfo.columns} accent="accent-violet" />
                <Metric label="操作步数" value={applyInfo.operations_applied?.length || 0} />
              </div>
              <div className="dp-warnings">
                {(applyInfo.warnings || []).map((w, i) => (
                  <div key={`aw-${i}`} className="dp-warning">
                    ⚠ {w}
                  </div>
                ))}
              </div>
              <div className="apply-ops-list">
                <span className="dp-field-label">执行的操作</span>
                {(applyInfo.operations_applied || []).map((op, i) => (
                  <code key={`${op.type}-${i}`} className="op-chip">
                    {i + 1}. {op.type}
                    {op.effect?.generated_columns?.length ? `（生成 ${op.effect.generated_columns.length} 列）` : ''}
                    {op.effect?.removed_rows ? `（删除 ${op.effect.removed_rows} 行）` : ''}
                  </code>
                ))}
              </div>
            </div>
          )}
        </div>

        <aside className="dp-side">
          <VersionHistory
            original={original}
            versions={versions}
            sourceId={sourceId}
            onSelect={selectSource}
            onRunEda={onRunEda}
            onDelete={deleteVersion}
            busyKey={busyKey}
            loading={loadingVersions}
          />
        </aside>
      </div>

      <VersionCompare datasetId={datasetId} original={original} versions={versions} />
    </section>
  )
}
