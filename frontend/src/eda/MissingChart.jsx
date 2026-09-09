import EChart from './EChart.jsx'
import { TOOLTIP_STYLE, baseGrid, valueAxisStyle, categoryAxisStyle } from './chartTheme.js'

/**
 * 各字段缺失率柱状图（后端已按缺失率降序返回）。
 */
export default function MissingChart({ missing }) {
  const { by_column: byColumn, total_missing, overall_missing_percentage } = missing || {}

  if (!byColumn || byColumn.length === 0) {
    return <p className="panel-empty">未发现缺失值，所有字段均完整。</p>
  }

  const option = {
    tooltip: {
      trigger: 'axis',
      ...TOOLTIP_STYLE,
      formatter(params) {
        const item = params[0]
        const col = byColumn[item.dataIndex]
        return [
          `<b>${col.column}</b>`,
          `缺失率：${col.missing_percentage}%`,
          `缺失数：${col.missing_count} 个`,
        ].join('<br/>')
      },
    },
    grid: baseGrid(),
    xAxis: categoryAxisStyle({
      name: '缺失率（%）',
      nameLocation: 'middle',
      nameGap: 28,
      nameTextStyle: { color: '#93a1c4' },
    }),
    yAxis: valueAxisStyle({
      data: byColumn.map((col) => col.column),
      type: 'category',
      inverse: true,
    }),
    series: [
      {
        type: 'bar',
        data: byColumn.map((col) => ({
          value: col.missing_percentage,
          itemStyle: { color: col.missing_percentage >= 20 ? '#fbbf24' : '#7d9dff', borderRadius: [0, 4, 4, 0] },
        })),
        barMaxWidth: 26,
        label: {
          show: true,
          position: 'right',
          color: '#dbe3f5',
          fontSize: 11,
          formatter: ({ value }) => `${value}%`,
        },
      },
    ],
  }

  return (
    <div className="eda-chart-block">
      <p className="eda-chart-note">
        整体缺失率 {overall_missing_percentage}%（共 {total_missing} 个缺失单元格）
      </p>
      <EChart option={option} height={Math.max(180, byColumn.length * 46 + 80)} />
    </div>
  )
}
