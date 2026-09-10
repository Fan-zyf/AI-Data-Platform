import { useMemo } from 'react'
import EChart from '../eda/EChart.jsx'
import {
  AXIS_COLOR,
  AXIS_LINE,
  TOOLTIP_STYLE,
  baseGrid,
  categoryAxisStyle,
  truncateLabel,
  valueAxisStyle,
} from '../eda/chartTheme.js'

const POS_COLOR = '#f87171'
const NEG_COLOR = '#7d9dff'

/**
 * 单样本 SHAP 解释：
 * - 默认展示后端返回的 1~3 条 sample；
 * - 每条样本显示：预测、base_value 偏移、按 |SHAP| 排序的 top 特征贡献横向条形图。
 */
export default function SampleExplanation({ samples, classLabelUsed, taskType }) {
  if (!samples || !samples.length) {
    return <p className="panel-empty">暂无单点解释数据。</p>
  }
  return (
    <div className="shap-chart-wrap">
      <div className="shap-chart-head">
        <h5>单点预测解释（Single Sample SHAP）</h5>
        <span className="ml-tip">
          {taskType === 'classification' && classLabelUsed ? `正类：${classLabelUsed}` : '回归目标'}
        </span>
      </div>
      <div className="shap-sample-grid">
        {samples.map((sample) => (
          <SampleCard
            key={sample.index}
            sample={sample}
            classLabelUsed={classLabelUsed}
            taskType={taskType}
          />
        ))}
      </div>
    </div>
  )
}

function SampleCard({ sample, classLabelUsed, taskType }) {
  const option = useMemo(() => {
    const contribs = (sample.contributions || [])
      .map((c) => ({
        feature: c.feature,
        value: c.value,
        shap: Number(c.shap || 0),
      }))
      .sort((a, b) => Math.abs(b.shap) - Math.abs(a.shap))
      .slice(0, 15)
    if (!contribs.length) return null
    const categories = contribs.map((c) => truncateLabel(c.feature, 26))
    const data = contribs.map((c) => ({
      value: Number(c.shap.toFixed(6)),
      raw: c.value,
    }))
    return {
      tooltip: {
        trigger: 'axis',
        axisPointer: { type: 'shadow' },
        ...TOOLTIP_STYLE,
        formatter: (params) => {
          const item = params[0]
          const c = contribs[item.dataIndex]
          return `<b>${c.feature}</b><br/>SHAP: <b>${c.shap.toFixed(4)}</b><br/>原始值: <b>${formatValue(c.value)}</b>`
        },
      },
      grid: { ...baseGrid({ left: 12, right: 24, top: 16, bottom: 8 }), containLabel: true },
      xAxis: valueAxisStyle({ name: 'SHAP value', nameTextStyle: { color: AXIS_COLOR, fontSize: 11 } }),
      yAxis: {
        type: 'category',
        data: categories,
        inverse: true,
        ...categoryAxisStyle({ axisLabel: { color: AXIS_COLOR, fontSize: 11 } }),
      },
      series: [
        {
          type: 'bar',
          stack: 'shap',
          data: data.map((d) => d.value),
          itemStyle: {
            color: (params) => (params.value >= 0 ? POS_COLOR : NEG_COLOR),
            borderRadius: [0, 3, 3, 0],
          },
          label: {
            show: true,
            position: 'right',
            color: AXIS_COLOR,
            fontSize: 10,
            formatter: (p) => Number(p.value).toFixed(3),
          },
        },
      ],
    }
  }, [sample])

  const prediction = sample.prediction || {}
  return (
    <div className="shap-sample-card">
      <div className="shap-sample-head">
        <span className="shap-sample-id">样本 #{sample.index + 1}</span>
        <span className="shap-sample-pred">
          {prediction.kind === 'classification' ? (
            <>
              预测：<strong>{String(prediction.class_label ?? '—')}</strong>
              {prediction.probability !== undefined && prediction.probability !== null && (
                <span className="ml-tip">（概率 {(prediction.probability * 100).toFixed(1)}%）</span>
              )}
            </>
          ) : prediction.kind === 'regression' ? (
            <>
              预测值：<strong>{formatValue(prediction.value)}</strong>
            </>
          ) : (
            <>预测：—</>
          )}
        </span>
      </div>
      {sample.base_value_delta !== undefined && sample.base_value_delta !== null && (
        <div className="shap-sample-base">
          SHAP 之和（对 base value 的偏移）：<b>{Number(sample.base_value_delta).toFixed(4)}</b>
        </div>
      )}
      {option ? (
        <EChart option={option} height={Math.max(220, (sample.contributions?.length || 0) * 20 + 60)} />
      ) : (
        <p className="panel-empty">该样本没有可显示的特征贡献。</p>
      )}
    </div>
  )
}

function formatValue(v) {
  if (v === null || v === undefined) return '—'
  if (typeof v === 'number') return Number.isInteger(v) ? String(v) : v.toFixed(3)
  return String(v)
}
