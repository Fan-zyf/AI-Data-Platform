function shortId(value) {
  if (!value) return ''
  return value.length > 8 ? `${value.slice(0, 8)}…` : value
}

function fmtDate(value) {
  if (!value) return '—'
  const d = new Date(value)
  return Number.isNaN(d.getTime()) ? String(value) : d.toLocaleString()
}

function VersionRow({ item, isSource, hasChildren, onSelect, onRunEda, onDelete, busyKey }) {
  return (
    <div className={`version-row ${isSource ? 'source' : ''}`}>
      <div className="version-row-main">
        <div className="version-row-title">
          <span className={`version-badge ${item.is_original ? 'original' : 'derived'}`}>
            {item.is_original ? 'Original' : shortId(item.version_id)}
          </span>
          {isSource && <span className="chip chip-active">处理基准</span>}
          {item.parent_version_id && (
            <span className="version-parent">
              父版本：{item.parent_version_id === 'original' ? 'Original' : shortId(item.parent_version_id)}
            </span>
          )}
          <span className="version-date">{fmtDate(item.created_at)}</span>
        </div>
        <div className="version-metrics">
          <span>{item.rows} 行</span>·<span>{item.columns} 列</span>·
          <span className={item.missing > 0 ? 'num-warn' : ''}>{item.missing} 缺失</span>·
          <span className={item.duplicates > 0 ? 'num-warn' : ''}>{item.duplicates} 重复</span>
        </div>
        {item.operations_summary?.length > 0 && (
          <div className="version-ops">
            {item.operations_summary.slice(0, 4).map((text, i) => (
              <code key={`${text}-${i}`} className="op-chip">
                {text}
              </code>
            ))}
            {item.operations_summary.length > 4 && (
              <span className="op-more">+{item.operations_summary.length - 4} 步</span>
            )}
          </div>
        )}
      </div>
      <div className="version-row-actions">
        <button type="button" className="btn-ghost btn-small" disabled={isSource} onClick={() => onSelect(item.version_id)}>
          {isSource ? '已选择' : '设为处理基准'}
        </button>
        <button
          type="button"
          className="btn-ghost btn-small"
          disabled={busyKey === `eda-${item.version_id}`}
          onClick={() => onRunEda(item.version_id)}
        >
          {busyKey === `eda-${item.version_id}` ? '运行中…' : '查看该版本 EDA'}
        </button>
        {!item.is_original && (
          <button
            type="button"
            className="btn-small btn-danger ghost"
            disabled={hasChildren || busyKey === `del-${item.version_id}`}
            title={hasChildren ? '该版本存在子版本，需先删除其子版本' : '删除该派生版本'}
            onClick={() => onDelete(item.version_id)}
          >
            {busyKey === `del-${item.version_id}` ? '删除中…' : '删除'}
          </button>
        )}
      </div>
    </div>
  )
}

export default function VersionHistory({
  original,
  versions,
  sourceId,
  onSelect,
  onRunEda,
  onDelete,
  busyKey,
  loading,
}) {
  const childrenOf = {}
  versions.forEach((item) => {
    if (item.parent_version_id) {
      childrenOf[item.parent_version_id] = (childrenOf[item.parent_version_id] || 0) + 1
    }
  })

  return (
    <section className="dp-history">
      <div className="dp-history-head">
        <h4 className="eda-subtitle">数据版本历史</h4>
        <span className="dp-builder-count">
          共 {(original ? 1 : 0) + versions.length} 个版本{loading ? '（加载中…）' : ''}
        </span>
      </div>
      {original && (
        <VersionRow
          item={original}
          isSource={sourceId === original.version_id}
          hasChildren={(childrenOf[original.version_id] || 0) > 0}
          onSelect={onSelect}
          onRunEda={onRunEda}
          onDelete={onDelete}
          busyKey={busyKey}
        />
      )}
      {versions.length === 0 ? (
        <p className="panel-empty">尚无派生版本。构建 Transformation Plan 并点击「应用并创建新版本」后，版本会显示在这里。</p>
      ) : (
        versions.map((item) => (
          <VersionRow
            key={item.version_id}
            item={item}
            isSource={sourceId === item.version_id}
            hasChildren={(childrenOf[item.version_id] || 0) > 0}
            onSelect={onSelect}
            onRunEda={onRunEda}
            onDelete={onDelete}
            busyKey={busyKey}
          />
        ))
      )}
      <p className="dp-immutable-note">
        <strong>不可变设计：</strong>原始数据永远只读，所有处理都会产生新的派生版本（UUID），可随时对比与回退。
      </p>
    </section>
  )
}
