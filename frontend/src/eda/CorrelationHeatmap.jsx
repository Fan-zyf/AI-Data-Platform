import EChart from './EChart.jsx'
import { TOOLTIP_STYLE, truncateLabel } from './chartTheme.js'

/**
 * 数值变量 Pearson 相关矩阵热力图。
 */
export default function CorrelationHeatmap({ correlation }) {
  const columns = correlation?.columns || []
  const matrix = correlation?.matrix || []

  if (columns.length < 2) {
    return <p className="panel-empty">可计算的数值字段不足两个，无法生成相关矩阵。</p>
  }

  const data = []
  for (let i = 0; i < columns.length; i += 1) {
    for (let j = 0; j < columns.length; j += 1) {
      data.push([j, i, matrix[i][j]])
    }
  }

  const rotateLabels = columns.length > 6
  const option = {
    tooltip: {
      position: 'top',
      ...TOOLTIP_STYLE,
      formatter(params) {
        const [j, i, value] = params.value
        const text = value === null || value === undefined ? '无法计算' : Number(value).toFixed(4)
        return `<b>${columns[i]}</b> 与 <b>${columns[j]}</b><br/>Pearson r = ${text}`
      },
    },
    grid: { left: 12, right: 24, top: 12, bottom: rotateLabels ? 96 : 72, containLabel: true },
    xAxis: {
      type: 'category',
      data: columns.map((col) => truncateLabel(col, 14)),
      axisTick: { show: false },
      axisLine: { lineStyle: { color: 'rgba(255,255,255,0.16)' } },
      axisLabel: rotateLabels ? { rotate: 45, color: '#93a1c4' } : { color: '#93a1c4' },
      splitArea: { show: true, areaStyle: { color: ['rgba(255,255,255,0.02)', 'rgba(255,255,255,0.05)'] } },
    },
    yAxis: {
      type: 'category',
      data: columns.map((col) => truncateLabel(col, 14)),
      axisTick: { show: false },
      axisLine: { lineStyle: { color: 'rgba(255,255,255,0.16)' } },
      axisLabel: { color: '#93a1c4' },
      splitArea: { show: true, areaStyle: { color: ['rgba(255,255,255,0.02)', 'rgba(255,255,255,0.05)'] } },
    },
    visualMap: {
      min: -1,
      max: 1,
      calculable: false,
      orient: 'horizontal',
      left: 'center',
      bottom: 8,
      itemWidth: 12,
      itemHeight: 120,
      textStyle: { color: '#93a1c4' },
      inRange: { color: ['#60a5fa', '#1f2940', '#f87171'] },
    },
    series: [
      {
        type: 'heatmap',
        data,
        label: {
          show: true,
          color: '#dbe3f5',
          fontSize: 11,
          formatter(params) {
            const value = params.value[2]
            return value === null || value === undefined ? '' : Number(value).toFixed(2)
          },
        },
        emphasis: { itemStyle: { shadowBlur: 8, shadowColor: 'rgba(0,0,0,0.5)' } },
      },
    ],
  }

  return <EChart option={option} height={Math.max(300, columns.length * 40 + 130)} />
}
