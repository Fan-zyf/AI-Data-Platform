import { useCallback, useEffect, useMemo, useState } from 'react'
import ExplainabilityPanel from '../explainability/ExplainabilityPanel.jsx'
import ExperimentReport, { MODEL_LABELS } from './ExperimentReport.jsx'

const CLASS_MODELS = ['dummy', 'logistic_regression', 'random_forest']
const REG_MODELS = ['dummy', 'ridge', 'random_forest']

const TASK_OPTIONS = [
  { value: 'auto', label: '自动推断（推荐）' },
  { value: 'classification', label: '分类：预测类别' },
  { value: 'regression', label: '回归：预测数值' },
]

const TYPE_GROUP_META = {
  numeric: { label: '数值特征', accent: 'chip-numeric' },
  boolean: { label: '布尔特征', accent: 'chip-boolean' },
  categorical: { label: '分类特征', accent: 'chip-categorical' },
  datetime: { label: '日期时间（不参与建模）', accent: 'chip-datetime', disabled: true },
  text: { label: '文本（高基数不参与建模）', accent: 'chip-text', disabled: true },
}

function shortId(value) {
  if (!value) return ''
  return value.length > 8 ? `${value.slice(0, 8)}…` : value
}

function fmtDate(value) {
  if (!value) return '—'
  const d = new Date(value)
  return Number.isNaN(d.getTime()) ? String(value) : d.toLocaleString()
}

async function apiFetch(path, options) {
  const res = await fetch(path, options)
  const payload = await res.json().catch(() => null)
  if (!res.ok) {
    const message =
      payload?.error?.message ||
      (Array.isArray(payload?.detail) && payload.detail[0]?.msg
        ? payload.detail[0].msg
        : `请求失败（HTTP ${res.status}）`)
    throw new Error(message)
  }
  return payload
}

function Field({ label, children, hint }) {
  return (
    <label className="ml-field">
      <span className="ml-field-label">{label}</span>
      {children}
      {hint && <span className="ml-tip">{hint}</span>}
    </label>
  )
}

/* ============ 实验历史（侧栏） ============ */

function ExperimentRow({ item, active, onSelect, onDelete, busy }) {
  const primaryLabel = { f1_macro: 'F1', rmse: 'RMSE' }[item.primary_metric] || item.primary_metric
  return (
    <div className={`ml-exp-row ${active ? 'active' : ''}`}>
      <div className="ml-exp-row-main" role="button" tabIndex={0} onClick={() => onSelect(item)}>
        <div className="version-row-title">
          <span className={`version-badge ${item.task_type === 'regression' ? 'derived' : 'original'}`}>
            {item.task_type === 'classification' ? '分类' : '回归'}
          </span>
          <span className="ml-exp-target" title={item.target_column}>
            目标：{item.target_column}
          </span>
          <span className="version-date">{fmtDate(item.created_at)}</span>
        </div>
        <div className="version-metrics">
          <span className="ml-exp-model">{MODEL_LABELS[item.best_model] || item.best_model}</span>
          ·<span>{item.train_rows} 训练</span>·<span>{item.test_rows} 测试</span>·
          <span className={item.beats_baseline ? 'cell-ok' : 'num-warn'}>
            {primaryLabel} {item.primary_score === null || item.primary_score === undefined ? '—' : Number(item.primary_score).toFixed(4)}
          </span>
        </div>
        {(item.warnings || []).length > 0 && (
          <div className="ml-exp-warn">⚠ {item.warnings.length} 条警告</div>
        )}
      </div>
      <div className="version-row-actions">
        <button
          type="button"
          className="btn-small btn-danger ghost"
          disabled={busy === item.experiment_id}
          onClick={() => onDelete(item)}
        >
          {busy === item.experiment_id ? '删除中…' : '删除'}
        </button>
      </div>
    </div>
  )
}

/* ============ 预测结果 ============ */

function probabilityCell(probabilities, index, classLabels) {
  if (!probabilities || !classLabels?.length) return '—'
  const row = probabilities[index] || {}
  return (
    <div className="ml-prob">
      {classLabels.map((label) => {
        const p = Number(row[label] ?? 0)
        return (
          <div key={String(label)} className="ml-prob-bar-row">
            <span className="ml-prob-label">{String(label)}</span>
            <div className="ml-prob-track">
              <div
                className="ml-prob-fill"
                style={{ width: `${Math.max(0, Math.min(100, p * 100))}%` }}
              />
            </div>
            <span className="ml-prob-value">{p === 0 ? '0' : p.toFixed(4)}</span>
          </div>
        )
      })}
    </div>
  )
}

function PredictionResult({ payload, records, features, className }) {
  if (!payload) return null
  const predictions = payload.predictions || []
  const classLabels = payload.class_labels || []
  return (
    <div className={`ml-predict-result ${className || ''}`}>
      <div className="ml-predict-head">
        <strong>预测完成</strong>
        <span className="ml-tip">
          有效记录 {payload.count} 条 · 任务类型 {payload.task_type === 'regression' ? '回归' : '分类'}
          {(payload.warnings || []).length > 0 && ` · ${payload.warnings.length} 条警告`}
        </span>
      </div>
      <WarningBox items={payload.warnings} />
      <div className="table-scroll">
        <table className="data-table ml-predict-table">
          <thead>
            <tr>
              <th className="row-index">#</th>
              {features.map((f) => (
                <th key={f}>{f}</th>
              ))}
              <th>预测值</th>
              {classLabels.length > 1 && <th>各类别概率</th>}
            </tr>
          </thead>
          <tbody>
            {predictions.map((p) => {
              const record = records[p.index] || {}
              return (
                <tr key={p.index}>
                  <td className="row-index">{p.index + 1}</td>
                  {features.map((f) => {
                    const v = record[f]
                    return (
                      <td key={f}>
                        {v === null || v === undefined ? <span className="cell-null">—</span> : String(v)}
                      </td>
                    )
                  })}
                  <td>
                    <strong className="ml-prediction">{String(p.prediction)}</strong>
                  </td>
                  {classLabels.length > 1 && (
                    <td>{probabilityCell(payload.probabilities, p.index, classLabels)}</td>
                  )}
                </tr>
              )
            })}
          </tbody>
        </table>
      </div>
    </div>
  )
}

function WarningBox({ items }) {
  if (!items || items.length === 0) return null
  return (
    <div className="dp-warnings">
      {items.map((w, i) => (
        <div key={`${w}-${i}`} className="dp-warning">
          ⚠ {w}
        </div>
      ))}
    </div>
  )
}

/* ============ 主面板 ============ */

export default function MLPanel({ datasetId }) {
  const [original, setOriginal] = useState(null)
  const [versions, setVersions] = useState([])
  const [loadingVersions, setLoadingVersions] = useState(true)
  const [sourceId, setSourceId] = useState('original')
  const [detail, setDetail] = useState(null)
  const [detailLoading, setDetailLoading] = useState(false)

  // 训练配置
  const [targetColumn, setTargetColumn] = useState('')
  const [taskType, setTaskType] = useState('auto')
  const [featureMode, setFeatureMode] = useState('auto')
  const [selectedFeatures, setSelectedFeatures] = useState([])
  const [candidateModels, setCandidateModels] = useState([])
  const [advancedOpen, setAdvancedOpen] = useState(false)
  const [testSize, setTestSize] = useState('0.2')
  const [cvFolds, setCvFolds] = useState('5')
  const [randomState, setRandomState] = useState('42')
  const [excludeColumns, setExcludeColumns] = useState([])

  // 实验 & 报告
  const [experiments, setExperiments] = useState([])
  const [loadingExp, setLoadingExp] = useState(false)
  const [activeReport, setActiveReport] = useState(null)
  const [activeExperimentId, setActiveExperimentId] = useState(null)
  const [busyDelete, setBusyDelete] = useState('')

  // 训练 / 预测
  const [training, setTraining] = useState(false)
  const [predicting, setPredicting] = useState(false)
  const [predictText, setPredictText] = useState('')
  const [predictPayload, setPredictPayload] = useState(null)
  const [predictRecords, setPredictRecords] = useState([])

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
        setError(err.message || '无法加载版本画像')
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
        await loadDetail(target)
      } catch (err) {
        setError(err.message || '无法加载数据版本列表')
      } finally {
        setLoadingVersions(false)
      }
    },
    [datasetId, loadDetail],
  )

  const loadExperiments = useCallback(async () => {
    if (!datasetId) return
    setLoadingExp(true)
    try {
      const body = await apiFetch(`/api/datasets/${datasetId}/ml/experiments`)
      setExperiments(body.experiments || [])
    } catch (err) {
      setError(err.message || '无法加载实验列表')
    } finally {
      setLoadingExp(false)
    }
  }, [datasetId])

  useEffect(() => {
    refreshVersions('original')
    loadExperiments()
  }, [refreshVersions, loadExperiments])

  const selectSource = async (versionId) => {
    if (!versionId || versionId === sourceId) return
    setSourceId(versionId)
    setError(null)
    setTargetColumn('')
    setSelectedFeatures([])
    setExcludeColumns([])
    await loadDetail(versionId)
  }

  /* ---- 列画像与分组 ---- */
  const columns = useMemo(() => {
    if (!detail?.column_profiles) return []
    return detail.column_profiles.map((p) => ({
      name: p.column_name,
      type: p.inferred_type,
      missing_count: p.missing_count,
      unique_count: p.unique_count,
      non_null_count: p.non_null_count,
    }))
  }, [detail])

  const groupedColumns = useMemo(() => {
    const groups = { numeric: [], boolean: [], categorical: [], datetime: [], text: [] }
    columns.forEach((c) => {
      if (groups[c.type]) groups[c.type].push(c)
      else groups.categorical.push(c)
    })
    return groups
  }, [columns])

  const featureCheckColumns = columns.filter((c) => c.name !== targetColumn)

  /* 供手动选择的目标选项直接使用全部列，但排除已经作为目标的选择 */

  const toggleFeature = (name) => {
    setSelectedFeatures((prev) =>
      prev.includes(name) ? prev.filter((f) => f !== name) : [...prev, name],
    )
  }

  const toggleModel = (key) => {
    setCandidateModels((prev) =>
      prev.includes(key) ? prev.filter((k) => k !== key) : [...prev, key],
    )
  }

  const toggleExclude = (name) => {
    setExcludeColumns((prev) =>
      prev.includes(name) ? prev.filter((f) => f !== name) : [...prev, name],
    )
  }

  const canTrain =
    !!datasetId &&
    !!targetColumn &&
    !training &&
    !loadingVersions &&
    columns.length > 0

  /* ---- 训练 ---- */
  const runTrain = async () => {
    setError(null)
    setNotice(null)
    setTraining(true)
    try {
      const body = {
        source_version_id: sourceId === 'original' ? null : sourceId,
        target_column: targetColumn,
        task_type: taskType,
        feature_columns:
          featureMode === 'manual' && selectedFeatures.length > 0 ? [...selectedFeatures] : null,
        exclude_columns: [...excludeColumns],
        candidate_models: candidateModels.length > 0 ? [...candidateModels] : null,
        test_size: Number(testSize),
        cv_folds: Number(cvFolds),
        random_state: Number(randomState),
      }
      const payload = await apiFetch(`/api/datasets/${datasetId}/ml/train`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(body),
      })
      setActiveReport(payload)
      setActiveExperimentId(payload.experiment_id)
      setPredictText('')
      setPredictPayload(null)
      await loadExperiments()
      setNotice(`训练完成：最佳模型 ${MODEL_LABELS[payload.best_model] || payload.best_model}，实验 ${shortId(payload.experiment_id)} 已保存。`)
    } catch (err) {
      setError(err.message || '训练失败')
    } finally {
      setTraining(false)
    }
  }

  /* ---- 选择实验（加载详情） ---- */
  const selectExperiment = async (item) => {
    if (!item || item.experiment_id === activeExperimentId) return
    setError(null)
    try {
      const detail = await apiFetch(
        `/api/datasets/${datasetId}/ml/experiments/${item.experiment_id}`,
      )
      setActiveReport(detail)
      setActiveExperimentId(item.experiment_id)
      setPredictText('')
      setPredictPayload(null)
    } catch (err) {
      setError(err.message || '无法加载实验详情')
    }
  }

  const deleteExperiment = async (item) => {
    if (!item) return
    if (!window.confirm(`确认删除实验 ${shortId(item.experiment_id)}？其模型文件将一并删除。`)) return
    setBusyDelete(item.experiment_id)
    setError(null)
    try {
      await apiFetch(`/api/datasets/${datasetId}/ml/experiments/${item.experiment_id}`, {
        method: 'DELETE',
      })
      if (activeExperimentId === item.experiment_id) {
        setActiveReport(null)
        setActiveExperimentId(null)
        setPredictPayload(null)
        setPredictText('')
      }
      await loadExperiments()
      setNotice(`实验 ${shortId(item.experiment_id)} 已删除。`)
    } catch (err) {
      setError(err.message || '删除实验失败')
    } finally {
      setBusyDelete('')
    }
  }

  /* ---- 预测 ---- */
  const featuresForPrediction = activeReport?.features || []
  const reportVersionId = activeReport?.source_version_id || 'original'

  const loadSampleRecords = async () => {
    if (!activeReport) return
    setError(null)
    try {
      const body = await apiFetch(`/api/datasets/${datasetId}/versions/${reportVersionId}`)
      const preview = body.preview || []
      const sample = preview.slice(0, 3).map((row) => {
        const record = {}
        featuresForPrediction.forEach((f) => {
          record[f] = row[f] === undefined ? null : row[f]
        })
        return record
      })
      if (!sample.length) {
        setPredictText('')
        setError('源版本没有可用的预览行，请手动粘贴 JSON 记录。')
        return
      }
      setPredictText(JSON.stringify(sample, null, 2))
    } catch (err) {
      setError(err.message || '无法读取源版本预览')
    }
  }

  const runPredict = async () => {
    if (!activeExperimentId) return
    let records
    try {
      records = JSON.parse(predictText)
    } catch {
      setError('预测记录 JSON 格式无效，请检查后重试。')
      return
    }
    if (!Array.isArray(records) || records.length === 0) {
      setError('预测记录必须是至少包含 1 条记录的 JSON 数组。')
      return
    }
    setError(null)
    setPredicting(true)
    try {
      const payload = await apiFetch(
        `/api/datasets/${datasetId}/ml/experiments/${activeExperimentId}/predict`,
        {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ records }),
        },
      )
      setPredictPayload(payload)
      setPredictRecords(records)
    } catch (err) {
      setError(err.message || '预测失败')
      setPredictPayload(null)
    } finally {
      setPredicting(false)
    }
  }

  const primaryModelName = MODEL_LABELS[activeReport?.best_model] || activeReport?.best_model

  return (
    <section className="panel ml-panel">
      <div className="eda-heading">
        <h3 className="panel-title eda-title">机器学习训练与预测</h3>
        <p className="eda-meta">
          数据集会话 <code>{shortId(datasetId)}</code>
          {loadingExp && <span className="dp-loading-tag">实验列表加载中…</span>}
        </p>
      </div>
      <p className="eda-desc">
        <strong>防泄漏数据管线：</strong>选择数据版本 → 指定目标字段与特征 → train/test split → 仅在
        训练集内对候选模型做交叉验证（CV）择优 → 在完整训练集 refit 最佳模型 → 对保留测试集做唯一一次评估 →
        保存实验（模型 + 元数据）→ 基于实验对新数据批量预测。统计型预处理全部内嵌在 Pipeline 中，
        绝不提前在全量数据上拟合。
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

      <div className="dp-main-grid">
        {/* ================= 主栏 ================= */}
        <div className="dp-main">
          {/* ---- 训练配置 ---- */}
          <div className="ml-config">
            <h4 className="eda-subtitle">① 训练配置</h4>

            <div className="dp-source-bar">
              <label className="dp-field dp-source-select">
                <span className="dp-field-label">训练数据版本（源）</span>
                <select
                  value={sourceId}
                  disabled={loadingVersions}
                  onChange={(e) => selectSource(e.target.value)}
                >
                  {original && (
                    <option value="original">Original（{original.rows} 行 / {original.columns} 列）</option>
                  )}
                  {versions.map((v) => (
                    <option key={v.version_id} value={v.version_id}>
                      Version {shortId(v.version_id)}（{v.rows} 行 / {v.columns} 列）
                    </option>
                  ))}
                </select>
              </label>
              <div className="metric-grid dp-source-metrics">
                <MetricMini label="总行数" value={detail ? detail.rows : '—'} accent="accent-blue" />
                <MetricMini label="总列数" value={detail ? detail.columns : '—'} accent="accent-violet" />
                <MetricMini label="缺失值" value={detail?.quality ? detail.quality.total_missing : '—'} accent="accent-amber" />
                <MetricMini label="唯一目标候选" value={columns.length} accent="accent-green" />
              </div>
            </div>
            {detailLoading && <p className="dp-hint">正在加载字段画像…</p>}

            <div className="ml-config-grid">
              <Field label="目标字段（Target）">
                <select
                  value={targetColumn}
                  disabled={loadingVersions || !columns.length}
                  onChange={(e) => {
                    const next = e.target.value
                    setTargetColumn(next)
                    setSelectedFeatures((prev) => prev.filter((f) => f !== next))
                    setExcludeColumns((prev) => prev.filter((f) => f !== next))
                  }}
                >
                  <option value="">请选择要预测的字段…</option>
                  {columns.map((c) => (
                    <option key={c.name} value={c.name}>
                      {c.name}（{c.type}
                      {c.unique_count != null ? ` / ${c.unique_count} 唯一` : ''}）
                    </option>
                  ))}
                </select>
              </Field>

              <Field label="任务类型" hint="auto 时后端根据目标取值自动推断">
                <select value={taskType} onChange={(e) => setTaskType(e.target.value)}>
                  {TASK_OPTIONS.map((o) => (
                    <option key={o.value} value={o.value}>
                      {o.label}
                    </option>
                  ))}
                </select>
              </Field>
            </div>

            {/* 特征选择 */}
            <div className="ml-config-grid">
              <Field label="特征选择模式">
                <select
                  value={featureMode}
                  onChange={(e) => setFeatureMode(e.target.value)}
                >
                  <option value="auto">自动选择（类型 + 基数规则）</option>
                  <option value="manual">手动指定特征</option>
                </select>
              </Field>
              {featureMode === 'manual' && (
                <div className="ml-features">
                  <div className="ml-features-head">
                    <span className="dp-field-label">已选 {selectedFeatures.length} 个特征</span>
                    <button type="button" className="icon-btn" title="全选可用特征" onClick={() => setSelectedFeatures(featureCheckColumns.map((c) => c.name))}>
                      ✓
                    </button>
                    <button type="button" className="icon-btn" title="清空选择" onClick={() => setSelectedFeatures([])}>
                      ✕
                    </button>
                  </div>
                  {featureCheckColumns.length === 0 ? (
                    <p className="dp-hint">该版本暂无可作为特征的字段。</p>
                  ) : (
                    <div className="dp-checkbox-grid ml-feature-grid">
                      {featureCheckColumns.map((c) => {
                        const meta = TYPE_GROUP_META[c.type] || { disabled: false }
                        const disabled = !!meta.disabled
                        const checked = selectedFeatures.includes(c.name)
                        return (
                          <label key={c.name} className={`dp-checkbox ${checked ? 'checked' : ''} ${disabled ? 'dp-checkbox-disabled' : ''}`}>
                            <input
                              type="checkbox"
                              checked={checked}
                              disabled={disabled}
                              onChange={() => toggleFeature(c.name)}
                            />
                            <span className={`chip ${meta.accent || 'chip-other'}`}>{c.name}</span>
                            <span className="ml-tip">{c.unique_count} 唯一值</span>
                          </label>
                        )
                      })}
                    </div>
                  )}
                </div>
              )}
              {featureMode === 'manual' && (
                <p className="ml-tip">
                  提示：日期时间与高基数文本不参与建模；显式选取低信息量字段会得到服务端警告。目标字段不能作为特征。
                </p>
              )}
            </div>

            {/* 候选模型 */}
            <div className="ml-config-block">
              <span className="dp-field-label">
                候选模型（不勾选 = 后端按任务类型使用默认集合）
              </span>
              <div className="ml-model-groups">
                <div className="ml-model-group">
                  <span className="ml-tip">分类模型</span>
                  <div className="dp-checkbox-grid">
                    {CLASS_MODELS.map((key) => {
                      const disabled = taskType === 'regression'
                      const checked = candidateModels.includes(key)
                      return (
                        <label key={key} className={`dp-checkbox ${checked ? 'checked' : ''} ${disabled ? 'dp-checkbox-disabled' : ''}`}>
                          <input
                            type="checkbox"
                            checked={checked}
                            disabled={disabled}
                            onChange={() => toggleModel(key)}
                          />
                          {MODEL_LABELS[key]}
                        </label>
                      )
                    })}
                  </div>
                </div>
                <div className="ml-model-group">
                  <span className="ml-tip">回归模型</span>
                  <div className="dp-checkbox-grid">
                    {REG_MODELS.map((key) => {
                      const disabled = taskType === 'classification'
                      const checked = candidateModels.includes(key)
                      return (
                        <label key={key} className={`dp-checkbox ${checked ? 'checked' : ''} ${disabled ? 'dp-checkbox-disabled' : ''}`}>
                          <input
                            type="checkbox"
                            checked={checked}
                            disabled={disabled}
                            onChange={() => toggleModel(key)}
                          />
                          {MODEL_LABELS[key]}
                        </label>
                      )
                    })}
                  </div>
                </div>
              </div>
            </div>

            {/* 高级参数 */}
            <button
              type="button"
              className="ml-advanced-toggle"
              onClick={() => setAdvancedOpen((open) => !open)}
            >
              {advancedOpen ? '▾' : '▸'} 高级参数（切分比例 / CV 折数 / 随机种子 / 排除列）
            </button>
            {advancedOpen && (
              <div className="ml-advanced">
                <div className="ml-config-grid">
                  <Field label="测试集比例 test_size" hint="默认 0.2，介于 (0,1)">
                    <input
                      type="number"
                      min="0.05"
                      max="0.8"
                      step="0.05"
                      value={testSize}
                      onChange={(e) => setTestSize(e.target.value)}
                    />
                  </Field>
                  <Field label="CV 折数 cv_folds" hint="默认 5，范围 2~20">
                    <input
                      type="number"
                      min="2"
                      max="20"
                      step="1"
                      value={cvFolds}
                      onChange={(e) => setCvFolds(e.target.value)}
                    />
                  </Field>
                  <Field label="随机种子 random_state" hint="默认 42，保证可复现">
                    <input
                      type="number"
                      min="0"
                      max="2147483647"
                      step="1"
                      value={randomState}
                      onChange={(e) => setRandomState(e.target.value)}
                    />
                  </Field>
                </div>
                <Field label="额外排除的特征（不参与自动/手动选择）">
                  <div className="dp-checkbox-grid">
                    {columns
                      .filter((c) => c.name !== targetColumn)
                      .map((c) => (
                        <label
                          key={c.name}
                          className={`dp-checkbox ${excludeColumns.includes(c.name) ? 'checked' : ''}`}
                        >
                          <input
                            type="checkbox"
                            checked={excludeColumns.includes(c.name)}
                            onChange={() => toggleExclude(c.name)}
                          />
                          <span className={`chip ${TYPE_GROUP_META[c.type]?.accent || 'chip-other'}`}>
                            {c.name}
                          </span>
                        </label>
                      ))}
                  </div>
                </Field>
              </div>
            )}

            <div className="dp-actions">
              <button
                type="button"
                className="btn-primary"
                disabled={!canTrain}
                onClick={runTrain}
              >
                {training ? (
                  <>
                    <span className="spinner" aria-hidden="true" /> 训练中…
                  </>
                ) : (
                  `开始训练（目标：${targetColumn || '未选择'}）`
                )}
              </button>
              {!targetColumn && !training && columns.length > 0 && (
                <span className="ml-tip">请先选择目标字段。</span>
              )}
            </div>
          </div>

          {/* ---- 当前报告 ---- */}
          <div className="ml-report-area">
            <div className="ml-report-head">
              <h4 className="eda-subtitle">
                {activeReport ? '实验报告' : '实验报告（尚无）'}
              </h4>
              {activeReport && (
                <span className="ml-tip">
                  目标：{activeReport.target_column} · 最佳模型：
                  <code className="op-chip">{primaryModelName}</code>
                  {activeReport.source_version_id &&
                    activeReport.source_version_id !== 'original' && (
                      <> · 源版本 <code className="op-chip">{shortId(activeReport.source_version_id)}</code></>
                    )}
                </span>
              )}
            </div>
            {activeReport ? (
              <ExperimentReport report={activeReport} />
            ) : (
              <p className="panel-empty">
                完成一次训练，或从右侧「实验历史」中选择已有实验，即可在此查看 CV 对比、
                最终测试评估（混淆矩阵 / ROC / 回归散点与残差）与特征信息。
              </p>
            )}
          </div>

          {/* ---- 预测 ---- */}
          <div className="ml-predict ml-report-area">
            <div className="ml-report-head">
              <h4 className="eda-subtitle">③ 基于实验批量预测</h4>
              {activeExperimentId && (
                <span className="ml-tip">
                  使用实验 <code className="op-chip">{shortId(activeExperimentId)}</code>
                  {activeReport && ` · 需要字段：${featuresForPrediction.length} 个特征`}
                </span>
              )}
            </div>
            {!activeExperimentId ? (
              <p className="panel-empty">训练成功后即可用该实验对新数据预测；预测不会重新训练模型。</p>
            ) : (
              <>
                <div className="ml-config-grid ml-predict-editor">
                  <Field label="待预测记录（JSON 数组，最多 1000 条）">
                    <textarea
                      className="ml-textarea"
                      spellCheck="false"
                      value={predictText}
                      placeholder='[{"age": 34, "plan": "gold", ...}]'
                      onChange={(e) => setPredictText(e.target.value)}
                    />
                  </Field>
                  <div className="ml-predict-help">
                    <div className="dp-actions">
                      <button type="button" className="btn-ghost btn-small" onClick={loadSampleRecords}>
                        从源版本载入示例
                      </button>
                      <button
                        type="button"
                        className="btn-primary"
                        disabled={predicting || !predictText.trim()}
                        onClick={runPredict}
                      >
                        {predicting ? '预测中…' : '运行预测'}
                      </button>
                    </div>
                    <p className="ml-tip">
                      每条记录需包含训练时的全部特征字段（顺序不限），多余字段会被忽略并警告。
                      实验源版本：{reportVersionId === 'original' ? 'Original' : `Version ${shortId(reportVersionId)}`}。
                    </p>
                  </div>
                </div>
                {predictPayload && (
                  <PredictionResult
                    payload={predictPayload}
                    records={predictRecords}
                    features={featuresForPrediction}
                    className="ml-predict-result-space"
                  />
                )}
              </>
            )}
          </div>

          {/* ---- 模型可解释性（v0.6 SHAP）---- */}
          {activeReport && (
            <div className="ml-explainability ml-report-area">
              <ExplainabilityPanel
                datasetId={datasetId}
                experiment={activeReport}
              />
            </div>
          )}
        </div>

        {/* ================= 侧栏：实验历史 ================= */}
        <aside className="dp-side">
          <div className="dp-history">
            <div className="dp-history-head">
              <h4 className="eda-subtitle">② 实验历史</h4>
              <span className="dp-builder-count">共 {experiments.length} 个{loadingExp ? '（加载中…）' : ''}</span>
            </div>
            {experiments.length === 0 && !loadingExp && (
              <p className="panel-empty">尚无训练实验。完成训练后实验会显示在这里，可点选查看详情或删除。</p>
            )}
            {experiments.map((item) => (
              <ExperimentRow
                key={item.experiment_id}
                item={item}
                active={item.experiment_id === activeExperimentId}
                onSelect={selectExperiment}
                onDelete={deleteExperiment}
                busy={busyDelete}
              />
            ))}
            {activeReport && (
              <p className="dp-immutable-note">
                <strong>防泄漏约定：</strong>统计预处理仅随 Pipeline 在训练折 / 训练集内 fit；测试集只参与最终评估。
                若从执行过全量统计预处理的派生版本训练，会在结果中给出 data leakage 警告。
              </p>
            )}
          </div>
        </aside>
      </div>
    </section>
  )
}

function MetricMini({ label, value, accent = '' }) {
  return (
    <div className={`metric-card ${accent}`}>
      <span className="metric-value">{value === null || value === undefined ? '—' : value}</span>
      <span className="metric-label">{label}</span>
    </div>
  )
}
