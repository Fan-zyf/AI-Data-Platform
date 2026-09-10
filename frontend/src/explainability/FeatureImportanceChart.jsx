import { useMemo } from 'react'
import EChart from '../eda/EChart.jsx'
import {
  AXIS_COLOR,
  PALETTE,
  TOOLTIP_STYLE,
  baseGrid,
  categoryAxisStyle,
  truncateLabel,
  valueAxisStyle,
} from '../eda/chartTheme.js'

/**
 * 全局特征重要性（barh）：
 * - 来自后端 mean |SHAP value|，按降序展示；
 * - 对分类任务统一以「正类 (class_label_used)」的 SHAP 值为基准；
 * - 对回归任务为模型对目标变量的 SHAP 贡献绝对值均值。
 */
export default function FeatureImportanceChart({ global, classLabelUsed, taskType }) {
  const option = useMemo(() => {
    if (!global || !global.feature_names?.length) return null
    const pairs = global.feature_names
      .map((name, i) => ({ name, value: Number(global.importance[i] || 0) }))
      .sort((a, b) => b.value - a.value)
    const categories = pairs.map((p) => truncateLabel(p.name, 28))
    const series = pairs.map((p) => Number(p.value.toFixed(6)))
    return {
      tooltip: {
        trigger: 'axis',
        axisPointer: { type: 'shadow' },
        ...TOOLTIP_STYLE,
        formatter: (params) => {
          const item = params[0]
          const p = pairs[item.dataIndex]
          return `<b>${p.name}</b><br/>mean |SHAP|: <b>${p.value.toFixed(6)}</b><br/>${classLabelUsed ? `正类：${classLabelUsed}` : '回归目标'}`
        },
      },
      grid: { ...baseGrid({ left: 12, right: 24, top: 16, bottom: 8 }), containLabel: true },
      xAxis: {
        type: 'value',
        ...valueAxisStyle({ name: 'mean |SHAP|', nameTextStyle: { color: AXIS_COLOR, fontSize: 11 } }),
      },
      yAxis: {
        type: 'category',
        data: categories,
        inverse: true,
        ...categoryAxisStyle({ axisLabel: { color: AXIS_COLOR, fontSize: 11 } }),
      },
      series: [
        {
          type: 'bar',
          data: series,
          itemStyle: {
            color: (params) => PALETTE[params.dataIndex % PALETTE.length],
            borderRadius: [0, 4, 4, 0],
          },
          label: {
            show: true,
            position: 'right',
            color: AXIS_COLOR,
            fontSize: 10,
            formatter: (p) => Number(p.value).toFixed(4),
          },
        },
      ],
    }
  }, [global, classLabelUsed, taskType])

  if (!option) {
    return <p className="panel-empty">暂无全局特征重要性数据。</p>
  }
  return (
    <div className="shap-chart-wrap">
      <div className="shap-chart-head">
        <h5>全局特征重要性（Global Feature Importance）</h5>
        <span className="ml-tip">
          基于 {global.n_rows_used} 条样本的 mean |SHAP|{taskType === 'classification' && classLabelUsed ? `（正类 ${classLabelUsed}）` : ''}
        </span>
      </div>
      <EChart
        option={option}
        height={Math.max(240, global.feature_names.length * 26 + 60)}
      />
    </div>
  )
}
