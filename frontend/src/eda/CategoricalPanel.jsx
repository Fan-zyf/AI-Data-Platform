import EChart from './EChart.jsx'
import {
  PALETTE,
  TOOLTIP_STYLE,
  baseGrid,
  valueAxisStyle,
  categoryAxisStyle,
  truncateLabel,
} from './chartTheme.js'

function categoryChart(summary, colorIndex) {
  const values = summary.top_values || []
  if (values.length === 0) {
    return <p className="panel-empty">该字段没有可统计的非空取值。</p>
  }

  const labels = values.map((item) => truncateLabel(item.value, 12))
  const rotateLabels = values.length > 5
  const option = {
    tooltip: {
      trigger: 'axis',
      ...TOOLTIP_STYLE,
      axisPointer: { type: 'shadow' },
      formatter(params) {
        const p = params[0]
        const meta = values[p.dataIndex]
        return [
          `<b>${summary.column}</b>`,
          `取值：${meta.value}`,
          `频数：${meta.count}（占非缺失 ${meta.percentage}%）`,
        ].join('<br/>')
      },
    },
    grid: baseGrid({ bottom: rotateLabels ? 52 : 16 }),
    xAxis: categoryAxisStyle({
      data: labels,
      axisLabel: rotateLabels ? { rotate: 30, color: '#93a1c4' } : undefined,
    }),
    yAxis: valueAxisStyle(),
    series: [
      {
        type: 'bar',
        data: values.map((item) => item.count),
        barMaxWidth: 34,
        barCategoryGap: '25%',
        itemStyle: { color: PALETTE[colorIndex % PALETTE.length], borderRadius: [3, 3, 0, 0] },
      },
    ],
  }

  return <EChart option={option} height={300} />
}

/**
 * 每个分类（/布尔/低基数文本）字段的 Top 取值频次柱状图。
 */
export default function CategoricalPanel({ summaries }) {
  if (!summaries || summaries.length === 0) {
    return <p className="panel-empty">未检测到适合分类分析的字段（分类 / 布尔 / 低基数文本）。</p>
  }

  return (
    <div className="eda-cat-grid">
      {summaries.map((summary, index) => (
        <div className="eda-cat-card" key={summary.column}>
          <div className="eda-cat-head">
            <strong>{summary.column}</strong>
            <span className="eda-cat-meta">
              <span className="chip chip-categorical">{summary.inferred_type}</span>
              <span>{summary.unique_count} 个取值</span>
              {summary.missing_count > 0 && <span>缺失 {summary.missing_count}</span>}
            </span>
          </div>
          {categoryChart(summary, index)}
        </div>
      ))}
    </div>
  )
}
