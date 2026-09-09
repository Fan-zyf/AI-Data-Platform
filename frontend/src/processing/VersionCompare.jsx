import { useMemo, useState } from 'react'

function shortId(value) {
  if (!value) return ''
  return value.length > 8 ? `${value.slice(0, 8)}…` : value
}

function displayName(item) {
  return item.is_original ? 'Original' : `Version ${shortId(item.version_id)}`
}

function DeltaCell({ before, after, reverse = false }) {
  const diff = after - before
  if (diff === 0) return <span className="diff-neutral">±0</span>
  const good = reverse ? diff > 0 : diff < 0
  return (
    <span className={good ? 'diff-good' : 'diff-bad'}>
      {diff > 0 ? '+' : ''}
      {diff}
    </span>
  )
}

function MetricLine({ label, a, b, reverse }) {
  return (
    <div className="compare-row">
      <span className="compare-label">{label}</span>
      <span className="compare-value-a">{a}</span>
      <span className="compare-arrow">→</span>
      <span className="compare-value-b">{b}</span>
      <DeltaCell before={a} after={b} reverse={reverse} />
    </div>
  )
}

export default function VersionCompare({ datasetId, original, versions }) {
  const options = useMemo(() => [original, ...versions].filter(Boolean), [original, versions])
  const [versionA, setVersionA] = useState('original')
  const [versionB, setVersionB] = useState('')
  const [result, setResult] = useState(null)
  const [error, setError] = useState(null)
  const [busy, setBusy] = useState(false)

  const compare = async () => {
    if (!versionA || !versionB || versionA === versionB) {
      setError('请选择两个不同的版本进行对比')
      return
    }
    setBusy(true)
    setError(null)
    try {
      const res = await fetch(`/api/datasets/${datasetId}/versions/compare`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ version_a: versionA, version_b: versionB }),
      })
      const payload = await res.json().catch(() => null)
      if (!res.ok) throw new Error(payload?.error?.message || `对比失败（HTTP ${res.status}）`)
      setResult(payload)
    } catch (err) {
      setError(err.message || '版本对比失败')
    } finally {
      setBusy(false)
    }
  }

  const nameA = options.find((o) => o.version_id === versionA)
  const nameB = options.find((o) => o.version_id === versionB)
  const aLabel = nameA ? displayName(nameA) : '—'
  const bLabel = nameB ? displayName(nameB) : '—'

  return (
    <section className="dp-compare">
      <div className="dp-history-head">
        <h4 className="eda-subtitle">版本对比（Version A vs B）</h4>
      </div>
      <div className="compare-controls">
        <label className="dp-field">
          <span className="dp-field-label">版本 A</span>
          <select value={versionA} onChange={(e) => setVersionA(e.target.value)}>
            {options.map((o) => (
              <option key={o.version_id} value={o.version_id}>
                {displayName(o)}（{o.rows} 行 / {o.columns} 列）
              </option>
            ))}
          </select>
        </label>
        <label className="dp-field">
          <span className="dp-field-label">版本 B</span>
          <select value={versionB} onChange={(e) => setVersionB(e.target.value)}>
            <option value="" disabled>
              请选择版本 B
            </option>
            {options.map((o) => (
              <option key={o.version_id} value={o.version_id}>
                {displayName(o)}（{o.rows} 行 / {o.columns} 列）
              </option>
            ))}
          </select>
        </label>
        <button type="button" className="btn-primary" disabled={busy} onClick={compare}>
          {busy ? '对比中…' : '开始对比'}
        </button>
      </div>

      {error && <div className="error-box">{error}</div>}

      {result && (
        <div className="compare-result">
          <div className="compare-direction">
            对比方向：<code>{aLabel}</code> → <code>{bLabel}</code>（绿色表示改善）
          </div>
          <MetricLine label="行数 Rows" a={result.rows_a} b={result.rows_b} />
          <MetricLine label="列数 Columns" a={result.columns_a} b={result.columns_b} />
          <MetricLine label="缺失 Missing" a={result.missing_a} b={result.missing_b} reverse />
          <MetricLine label="重复 Duplicates" a={result.duplicates_a} b={result.duplicates_b} reverse />

          <div className="compare-columns">
            <div>
              <span className="col-diff-title good">新增字段（B 独有）{result.generated_columns.length}</span>
              {result.generated_columns.length ? (
                <div className="chip-wrap">
                  {result.generated_columns.map((col) => (
                    <code key={col} className="op-chip good">
                      + {col}
                    </code>
                  ))}
                </div>
              ) : (
                <p className="panel-empty">无新增字段</p>
              )}
            </div>
            <div>
              <span className="col-diff-title bad">移除字段（A 独有）{result.removed_columns.length}</span>
              {result.removed_columns.length ? (
                <div className="chip-wrap">
                  {result.removed_columns.map((col) => (
                    <code key={col} className="op-chip bad">
                      − {col}
                    </code>
                  ))}
                </div>
              ) : (
                <p className="panel-empty">无移除字段</p>
              )}
            </div>
          </div>
        </div>
      )}
    </section>
  )
}
