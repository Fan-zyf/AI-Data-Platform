function formatNum(value) {
  if (value === null || value === undefined) return '—'
  const num = Number(value)
  return Number.isFinite(num) ? String(Math.round(num * 10000) / 10000) : '—'
}

/**
 * IQR 异常值检测结果表。
 */
export default function OutlierPanel({ outliers }) {
  if (!outliers || outliers.length === 0) {
    return <p className="panel-empty">未检测到数值字段，未执行 IQR 异常值检测。</p>
  }

  const detected = outliers.filter((item) => item.outlier_count > 0).length

  return (
    <div>
      <p className="eda-chart-note">
        检测方法：IQR（Q1 - 1.5×IQR / Q3 + 1.5×IQR） ·
        {detected > 0 ? ` 共 ${detected} 个字段检出异常值` : ' 所有数值字段均未检出异常值'}
      </p>
      <div className="table-scroll">
        <table className="data-table">
          <thead>
            <tr>
              <th>字段</th>
              <th>方法</th>
              <th>下界</th>
              <th>上界</th>
              <th>异常值数</th>
              <th>占比</th>
            </tr>
          </thead>
          <tbody>
            {outliers.map((row) => {
              const hasOutliers = row.outlier_count > 0
              return (
                <tr key={row.column}>
                  <td>
                    <span className="field-name">{row.column}</span>
                    {hasOutliers && <span className="warning-dot" aria-hidden="true" />}
                  </td>
                  <td>{row.detection_method}</td>
                  <td>{formatNum(row.lower_bound)}</td>
                  <td>{formatNum(row.upper_bound)}</td>
                  <td className={hasOutliers ? 'cell-warning-text' : 'cell-ok'}>
                    {row.outlier_count}
                  </td>
                  <td>{row.outlier_percentage}%</td>
                </tr>
              )
            })}
          </tbody>
        </table>
      </div>
    </div>
  )
}
