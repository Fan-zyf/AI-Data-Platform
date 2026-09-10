import { useMemo } from 'react'
import EChart from '../eda/EChart.jsx'
import {
  AXIS_COLOR,
  AXIS_LINE,
  PALETTE,
  SPLIT_LINE,
  TOOLTIP_STYLE,
  baseGrid,
  categoryAxisStyle,
  truncateLabel,
  valueAxisStyle,
} from '../eda/chartTheme.js'

/**
 * SHAP Summary（散点图，类 beeswarm 风格）：
 * - x 轴：SHAP value（对分类是 log-odds 偏移方向，对回归是目标变量贡献方向）；
 * - y 轴：特征；
 * - 颜色：原始特征值（红=高、蓝=低；当特征取值大多相同时退化为单色）。
 */
export default function SHAPSummaryChart({ summary }) {
  const option = useMemo(() => {
    if (!summary || !summary.feature_names?.length) return null
    const featureNames = summary.feature_names
    const values = summary.values || []
    const rawValues = summary.raw_values || summary.values || []
    const shapValues = summary.shap_values || []
    if (!shapValues.length) return null

    // 准备 series：每个 feature 一条散点 series，颜色由 raw_value 决定
    const series = featureNames.map((fname, fi) => {
      const data = []
      for (let i = 0; i < shapValues.length; i += 1) {
        const shap = Number(shapValues[i][fi] || 0)
        const raw = rawValues[i]?.[fi]
        data.push({
          value: [shap, truncateLabel(fname, 28)],
          rawValue: raw,
          sampleIndex: summary.sample_indices?.[i] ?? i,
        })
      }
      return {
        name: fname,
        type: 'scatter',
        data,
        symbolSize: 7,
        itemStyle: {
          color: (params) => colorForRawValue(params.data.rawValue, rawValues.map((row) => row[fi])),
          opacity: 0.85,
          borderColor: 'rgba(255,255,255,0.4)',
          borderWidth: 0.4,
        },
        emphasis: { focus: 'series' },
      }
    })

    return {
      tooltip: {
        trigger: 'item',
        ...TOOLTIP_STYLE,
        formatter: (params) => {
          const { data, seriesName } = params
          const raw = data.rawValue === undefined || data.rawValue === null ? '—' : formatRaw(data.rawValue)
          return `<b>${seriesName}</b><br/>SHAP: <b>${Number(data.value[0]).toFixed(4)}</b><br/>原始值: <b>${raw}</b><br/>样本 #${data.sampleIndex}`
        },
      },
      grid: { ...baseGrid({ left: 12, right: 16, top: 16, bottom: 32 }), containLabel: true },
      xAxis: {
        type: 'value',
        name: 'SHAP value',
        nameLocation: 'middle',
        nameGap: 22,
        nameTextStyle: { color: AXIS_COLOR, fontSize: 11 },
        ...valueAxisStyle(),
      },
      yAxis: {
        type: 'category',
        data: featureNames.map((n) => truncateLabel(n, 28)),
        ...categoryAxisStyle({ axisLabel: { color: AXIS_COLOR, fontSize: 11 } }),
      },
      visualMap: {
        show: true,
        min: 0,
        max: 1,
        dimension: 0, // 占位（实际颜色由 itemStyle.color 计算）
        inRange: { color: ['#67e8f9', '#7d9dff', '#fbbf24', '#f87171'] },
        textStyle: { color: AXIS_COLOR, fontSize: 10 },
        orient: 'horizontal',
        left: 'center',
        bottom: 4,
        calculable: false,
      },
      series,
    }
  }, [summary])

  if (!option) {
    return <p className="panel-empty">暂无 SHAP summary 数据。</p>
  }
  return (
    <div className="shap-chart-wrap">
      <div className="shap-chart-head">
        <h5>SHAP Summary（散点：每行样本的贡献分布）</h5>
        <span className="ml-tip">
          {summary.feature_names.length} 个 post-preprocessing 特征 · {summary.sample_indices?.length || summary.values.length} 条样本
        </span>
      </div>
      <EChart
        option={option}
        height={Math.max(320, summary.feature_names.length * 32 + 80)}
      />
    </div>
  )
}

function formatRaw(v) {
  if (v === null || v === undefined) return '—'
  if (typeof v === 'number') return Number.isInteger(v) ? String(v) : v.toFixed(3)
  return String(v)
}

function colorForRawValue(value, allValues) {
  if (value === null || value === undefined || value === '') {
    return 'rgba(125, 157, 255, 0.6)'
  }
  // 数值特征：按分位数映射颜色
  const numericAll = allValues.filter((v) => typeof v === 'number' && Number.isFinite(v))
  if (numericAll.length >= allValues.length * 0.7) {
    const min = Math.min(...numericAll)
    const max = Math.max(...numericAll)
    if (!Number.isFinite(min) || !Number.isFinite(max) || min === max) {
      return '#7d9dff'
    }
    const t = Math.max(0, Math.min(1, (Number(value) - min) / (max - min)))
    // 蓝->紫->橙->红
    if (t < 0.33) return `rgba(103, 232, 249, ${0.55 + t * 0.4})`
    if (t < 0.66) return `rgba(125, 157, 255, ${0.55 + t * 0.4})`
    return `rgba(251, 191, 36, ${0.55 + t * 0.4})`
  }
  // 类别特征：按出现频次给色
  const counts = new Map()
  allValues.forEach((v) => counts.set(v, (counts.get(v) || 0) + 1))
  const labels = Array.from(counts.keys()).sort()
  const idx = labels.indexOf(value)
  if (idx < 0) return '#7d9dff'
  return PALETTE[idx % PALETTE.length]
}
