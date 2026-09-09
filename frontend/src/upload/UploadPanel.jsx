import { useRef, useState } from 'react'

const ACCEPT = '.csv,.xlsx,.xls'
const ALLOWED_EXT = ['csv', 'xlsx', 'xls']
const MAX_FILE_MB = 20

function formatBytes(bytes) {
  if (!Number.isFinite(bytes) || bytes < 0) return '-'
  if (bytes < 1024) return `${bytes} B`
  if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`
  return `${(bytes / (1024 * 1024)).toFixed(2)} MB`
}

function getExtension(fileName) {
  const dot = fileName.lastIndexOf('.')
  return dot >= 0 ? fileName.slice(dot + 1).toLowerCase() : ''
}

function extractErrorMessage(status, payload) {
  if (payload && payload.success === false && payload.error?.message) {
    return payload.error.message
  }
  if (payload && Array.isArray(payload.detail) && payload.detail[0]?.msg) {
    return payload.detail[0].msg
  }
  if (status === 413) return '文件大小超过服务器限制（最大 20MB）'
  if (status === 422) return '文件格式或内容无法解析，请检查后重试'
  return `上传失败（HTTP ${status || '网络错误'}）`
}

const TYPE_CHIP_CLASS = {
  numeric: 'chip-numeric',
  categorical: 'chip-categorical',
  datetime: 'chip-datetime',
  boolean: 'chip-boolean',
  text: 'chip-text',
}

function typeChipClass(type) {
  return TYPE_CHIP_CLASS[type] || 'chip-other'
}

function UploadZone({ file, uploading, onFileChange }) {
  const inputRef = useRef(null)
  const [dragging, setDragging] = useState(false)

  const openPicker = () => {
    if (!uploading) inputRef.current?.click()
  }

  const handleDrop = (e) => {
    e.preventDefault()
    setDragging(false)
    const dropped = e.dataTransfer?.files?.[0]
    if (dropped && !uploading) onFileChange(dropped)
  }

  return (
    <div
      className={`upload-zone ${dragging ? 'upload-zone-dragging' : ''} ${uploading ? 'upload-zone-disabled' : ''}`}
      onClick={openPicker}
      onDragOver={(e) => {
        e.preventDefault()
        if (!uploading) setDragging(true)
      }}
      onDragLeave={() => setDragging(false)}
      onDrop={handleDrop}
      role="button"
      tabIndex={0}
      onKeyDown={(e) => {
        if (e.key === 'Enter' || e.key === ' ') openPicker()
      }}
    >
      <input
        ref={inputRef}
        type="file"
        accept={ACCEPT}
        className="upload-input"
        onChange={(e) => {
          const picked = e.target.files?.[0]
          if (picked) onFileChange(picked)
          e.target.value = ''
        }}
      />
      {file ? (
        <div className="file-chip">
          <span className="file-chip-icon">CSV/Excel</span>
          <div className="file-chip-info">
            <strong>{file.name}</strong>
            <span>{formatBytes(file.size)}</span>
          </div>
        </div>
      ) : (
        <div className="upload-placeholder">
          <span className="upload-icon" aria-hidden="true">↑</span>
          <p className="upload-hint">点击选择或拖拽文件到这里</p>
          <p className="upload-sub">支持 .csv / .xlsx / .xls，最大 {MAX_FILE_MB}MB</p>
        </div>
      )}
    </div>
  )
}

function MetricCard({ label, value, accent }) {
  return (
    <div className={`metric-card ${accent || ''}`}>
      <span className="metric-value">{value}</span>
      <span className="metric-label">{label}</span>
    </div>
  )
}

function WarningsPanel({ warnings }) {
  return (
    <section className="panel warnings-panel">
      <h3 className="panel-title">数据质量警告</h3>
      {warnings.length ? (
        <ul className="warnings-list">
          {warnings.map((item, index) => (
            <li key={`${index}-${item}`} className="warning-item">
              <span className="warning-dot" aria-hidden="true" />
              {item}
            </li>
          ))}
        </ul>
      ) : (
        <p className="panel-empty">未发现明显的数据质量问题</p>
      )}
    </section>
  )
}

function FieldsTable({ profiles }) {
  return (
    <section className="panel">
      <h3 className="panel-title">字段信息（{profiles.length} 列）</h3>
      <div className="table-scroll">
        <table className="data-table fields-table">
          <thead>
            <tr>
              <th>字段名</th>
              <th>推断类型</th>
              <th>缺失率</th>
              <th>唯一值</th>
            </tr>
          </thead>
          <tbody>
            {profiles.map((p) => (
              <tr key={p.column_name}>
                <td>
                  <span className="field-name">{p.column_name}</span>
                  <code className="field-dtype">{p.dtype}</code>
                </td>
                <td>
                  <span className={`chip ${typeChipClass(p.inferred_type)}`}>
                    {p.inferred_type}
                  </span>
                </td>
                <td>
                  {p.missing_count > 0 ? (
                    <>
                      {p.missing_percentage.toFixed(1)}%
                      <span className="cell-sub">（{p.missing_count} 个）</span>
                    </>
                  ) : (
                    <span className="cell-ok">0%</span>
                  )}
                </td>
                <td>{p.unique_count}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </section>
  )
}

function PreviewTable({ result }) {
  const { column_profiles: profiles, preview } = result
  const columns =
    profiles.length > 0
      ? profiles.map((p) => p.column_name)
      : preview[0]
        ? Object.keys(preview[0])
        : []

  if (!preview.length) {
    return (
      <section className="panel">
        <h3 className="panel-title">数据预览</h3>
        <p className="panel-empty">该数据集没有可预览的行</p>
      </section>
    )
  }

  return (
    <section className="panel">
      <h3 className="panel-title">数据预览（前 {preview.length} 行）</h3>
      <div className="table-scroll">
        <table className="data-table preview-table">
          <thead>
            <tr>
              <th className="row-index">#</th>
              {columns.map((col) => (
                <th key={col}>{col}</th>
              ))}
            </tr>
          </thead>
          <tbody>
            {preview.map((row, rowIndex) => (
              <tr key={`${rowIndex}-${row[columns[0]] ?? ''}`}>
                <td className="row-index">{rowIndex + 1}</td>
                {columns.map((col) => {
                  const value = row[col]
                  return (
                    <td key={col}>
                      {value === null || value === undefined ? (
                        <span className="cell-null">—</span>
                      ) : (
                        String(value)
                      )}
                    </td>
                  )
                })}
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </section>
  )
}

function ResultView({ result }) {
  const { dataset, quality } = result
  return (
    <div className="result-view">
      <div className="result-header">
        <h2 className="result-title">上传成功</h2>
        <p className="result-meta">
          {dataset.file_name} · {dataset.file_type.toUpperCase()} ·{' '}
          {formatBytes(dataset.file_size)}
        </p>
      </div>

      <div className="metric-grid">
        <MetricCard label="总行数" value={dataset.rows} accent="accent-blue" />
        <MetricCard label="总列数" value={dataset.columns} accent="accent-violet" />
        <MetricCard
          label="缺失值"
          value={quality.total_missing}
          accent={quality.total_missing > 0 ? 'accent-amber' : 'accent-green'}
        />
        <MetricCard
          label="重复行"
          value={quality.duplicate_rows}
          accent={quality.duplicate_rows > 0 ? 'accent-amber' : 'accent-green'}
        />
      </div>

      <WarningsPanel warnings={result.warnings} />
      <FieldsTable profiles={result.column_profiles} />
      <PreviewTable result={result} />
    </div>
  )
}

export default function UploadPanel() {
  const [file, setFile] = useState(null)
  const [uploading, setUploading] = useState(false)
  const [error, setError] = useState(null)
  const [result, setResult] = useState(null)

  const handleFileChange = (nextFile) => {
    const ext = getExtension(nextFile.name)
    if (!ALLOWED_EXT.includes(ext)) {
      setFile(null)
      setResult(null)
      setError(`不支持的文件类型 .${ext || '无'}，仅支持 .csv / .xlsx / .xls`)
      return
    }
    setFile(nextFile)
    setError(null)
    setResult(null)
  }

  const handleUpload = async () => {
    if (!file || uploading) return
    setUploading(true)
    setError(null)
    setResult(null)
    try {
      const formData = new FormData()
      formData.append('file', file)
      const res = await fetch('/api/data/upload', { method: 'POST', body: formData })
      const payload = await res.json().catch(() => null)
      if (!res.ok) {
        setError(extractErrorMessage(res.status, payload))
        return
      }
      if (!payload || payload.success !== true) {
        setError(payload?.error?.message || '服务器返回了无法识别的数据')
        return
      }
      setResult(payload)
    } catch {
      setError('无法连接后端服务，请确认后端已启动（uvicorn app.main:app --reload）')
    } finally {
      setUploading(false)
    }
  }

  return (
    <section className="upload-panel">
      <h2 className="section-title">
        数据上传<span className="section-tag">CSV / Excel</span>
      </h2>
      <p className="section-desc">
        上传本地文件进行只读分析，返回数据画像与质量概览；系统不会修改你的原始数据。
        可用示例：<code>data/sample/demo.csv</code>
      </p>

      <div className="upload-card">
        <UploadZone file={file} uploading={uploading} onFileChange={handleFileChange} />
        <div className="upload-actions">
          <button
            type="button"
            className="btn-primary"
            disabled={!file || uploading}
            onClick={handleUpload}
          >
            {uploading ? (
              <>
                <span className="spinner" aria-hidden="true" /> 分析中…
              </>
            ) : (
              '上传并分析'
            )}
          </button>
          {file && !uploading && (
            <button
              type="button"
              className="btn-ghost"
              onClick={() => {
                setFile(null)
                setResult(null)
                setError(null)
              }}
            >
              清除
            </button>
          )}
        </div>
        {uploading && (
          <p className="upload-state uploading">正在上传并解析文件，请稍候…</p>
        )}
      </div>

      {error && (
        <div className="error-box" role="alert">
          <strong>上传失败</strong>
          <span>{error}</span>
        </div>
      )}

      {result && <ResultView result={result} />}
    </section>
  )
}
