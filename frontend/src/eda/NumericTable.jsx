function formatNum(value) {
  if (value === null || value === undefined) return '—'
  const num = Number(value)
  return Number.isFinite(num) ? String(Math.round(num * 10000) / 10000) : '—'
}

const COLUMNS = [
  ['count', '有效数'],
  ['missing_count', '缺失'],
  ['mean', '均值'],
  ['std', '标准差'],
  ['min', '最小值'],
  ['q1', 'Q1'],
  ['median', '中位数'],
  ['q3', 'Q3'],
  ['max', '最大值'],
  ['range', '极差'],
  ['iqr', 'IQR'],
  ['skewness', '偏度'],
]

/**
 * 数值字段描述统计表（mean / std / 分位数 / IQR / 偏度等）。
 */
export default function NumericTable({ summaries }) {
  if (!summaries || summaries.length === 0) {
    return <p className="panel-empty">未检测到数值字段，无描述统计。</p>
  }

  return (
    <div className="table-scroll">
      <table className="data-table eda-stats-table">
        <thead>
          <tr>
            <th>字段</th>
            {COLUMNS.map(([key, label]) => (
              <th key={key}>{label}</th>
            ))}
          </tr>
        </thead>
        <tbody>
          {summaries.map((row) => (
            <tr key={row.column}>
              <td>
                <span className="field-name">{row.column}</span>
              </td>
              {COLUMNS.map(([key]) => (
                <td key={key}>{formatNum(row[key])}</td>
              ))}
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  )
}
