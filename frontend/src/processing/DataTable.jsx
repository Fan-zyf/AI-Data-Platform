function renderValue(value) {
  if (value === null || value === undefined) {
    return <span className="cell-null">空</span>
  }
  if (typeof value === 'object') {
    try {
      return <code>{JSON.stringify(value)}</code>
    } catch {
      return String(value)
    }
  }
  return String(value)
}

export default function DataTable({ title, rows, maxRows = 20, emptyText = '暂无数据' }) {
  if (!rows || rows.length === 0) {
    return (
      <div className="data-table-block">
        {title && <h5 className="data-table-title">{title}</h5>}
        <p className="panel-empty">{emptyText}</p>
      </div>
    )
  }
  const columns = Object.keys(rows[0])
  const shown = rows.slice(0, maxRows)
  return (
    <div className="data-table-block">
      {title && (
        <div className="data-table-title-row">
          <h5 className="data-table-title">{title}</h5>
          {rows.length > shown.length && <span className="table-note">仅显示前 {shown.length} 行</span>}
        </div>
      )}
      <div className="table-scroll">
        <table className="data-table">
          <thead>
            <tr>
              {columns.map((col) => (
                <th key={col}>{col}</th>
              ))}
            </tr>
          </thead>
          <tbody>
            {shown.map((row, rowIndex) => (
              <tr key={`${rowIndex}-${String(row[columns[0]])}`}>
                {columns.map((col) => (
                  <td key={col}>{renderValue(row[col])}</td>
                ))}
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  )
}
