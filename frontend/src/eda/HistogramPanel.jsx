import { useMemo, useState } from 'react'
import EChart from './EChart.jsx'
import { TOOLTIP_STYLE, baseGrid, valueAxisStyle, categoryAxisStyle, truncateLabel } from './chartTheme.js'

/**
 * 数值变量直方图（支持切换字段）。
 */
export default function HistogramPanel({ distributions }) {
  const items = distributions || []
  const [selected, setSelected] = useState('')

  const current = useMemo(() => {
    if (items.length === 0) return null
    const target = items.find((item) => item.column === selected) || items[0]
    return target
  }, [items, selected])

  if (!current) {
    return <p className="panel-empty">未检测到数值字段，无直方图数据。</p>
  }

  const labels = []
  for (let i = 0; i < current.counts.length; i += 1) {
    labels.push(`${current.bin_edges[i]} ~ ${current.bin_edges[i + 1]}`)
  }

  const rotateLabels = labels.length > 8
  const option = {
    color: ['#7d9dff'],
    tooltip: {
      trigger: 'axis',
      ...TOOLTIP_STYLE,
      axisPointer: { type: 'shadow' },
      formatter(params) {
        const p = params[0]
        return [`<b>${current.column}</b>`, `区间 ${p.name}`, `频数：${p.value}`].join('<br/>')
      },
    },
    grid: baseGrid({ bottom: rotateLabels ? 46 : 12 }),
    xAxis: categoryAxisStyle({
      name: current.column,
      nameLocation: 'middle',
      nameGap: rotateLabels ? 40 : 30,
      nameTextStyle: { color: '#93a1c4' },
      data: labels,
      axisLabel: rotateLabels ? { rotate: 30, color: '#93a1c4' } : undefined,
    }),
    yAxis: valueAxisStyle({ name: '频数', nameTextStyle: { color: '#93a1c4' } }),
    series: [
      {
        type: 'bar',
        data: current.counts,
        barCategoryGap: '15%',
        itemStyle: { borderRadius: [3, 3, 0, 0] },
      },
    ],
  }

  return (
    <div className="eda-chart-block">
      {items.length > 1 && (
        <div className="eda-chart-toolbar">
          <label className="eda-select-label" htmlFor="histogram-select">
            数值字段
          </label>
          <select
            id="histogram-select"
            className="eda-select"
            value={current.column}
            onChange={(event) => setSelected(event.target.value)}
          >
            {items.map((item) => (
              <option key={item.column} value={item.column}>
                {truncateLabel(item.column, 40)}
              </option>
            ))}
          </select>
        </div>
      )}
      <EChart option={option} height={300} />
    </div>
  )
}
