import EChart from '../eda/EChart.jsx'
import DataTable from '../processing/DataTable.jsx'
import {
  TOOLTIP_STYLE,
  baseGrid,
  valueAxisStyle,
  categoryAxisStyle,
  truncateLabel,
} from '../eda/chartTheme.js'

export const MODEL_LABELS = {
  dummy: 'Dummy Baseline',
  logistic_regression: 'Logistic Regression',
  random_forest: 'Random Forest',
  ridge: 'Ridge',
}

const METRIC_LABELS = {
  accuracy: 'Accuracy',
  balanced_accuracy: 'Balanced Acc',
  precision_macro: 'Precision (macro)',
  recall_macro: 'Recall (macro)',
  f1_macro: 'F1 (macro)',
  mae: 'MAE',
  rmse: 'RMSE',
  r2: 'R²',
}

const TASK_LABELS = { classification: '分类', regression: '回归' }

function fmtNum(value, digits = 4) {
  if (value === null || value === undefined || value === '') return '—'
  const n = Number(value)
  if (!Number.isFinite(n)) return '—'
  return n.toFixed(digits)
}

function shortId(value) {
  if (!value) return ''
  return value.length > 8 ? `${value.slice(0, 8)}…` : value
}

function fmtDate(value) {
  if (!value) return '—'
  const d = new Date(value)
  return Number.isNaN(d.getTime()) ? String(value) : d.toLocaleString()
}

function MetricCard({ label, value, accent = '', sub }) {
  return (
    <div className={`metric-card ${accent}`}>
      <span className="metric-value">{value}</span>
      <span className="metric-label">{label}</span>
      {sub && <span className="metric-label cell-sub">{sub}</span>}
    </div>
  )
}

/* 模型展示名兜底 */
function modelName(key) {
  return MODEL_LABELS[key] || key
}

/* 单条警告（复用 dp-warning 视觉） */
function WarningLines({ items }) {
  if (!items || items.length === 0) return null
  return (
    <div className="dp-warnings">
      {items.map((w, i) => (
        <div key={`${w}-${i}`} className="dp-warning">
          ⚠ {w}
        </div>
      ))}
    </div>
  )
}

/* ============ 分类可视化 ============ */

function ConfusionHeatmap({ cm }) {
  const labels = cm?.labels || []
  const matrix = cm?.matrix || []
  if (!labels.length || !matrix.length) {
    return <p className="panel-empty">暂无混淆矩阵数据。</p>
  }
  const data = []
  for (let i = 0; i < labels.length; i += 1) {
    for (let j = 0; j < labels.length; j += 1) {
      data.push([j, i, matrix[i][j]])
    }
  }
  const rotate = labels.some((l) => String(l).length > 6)
  const cellMax = Math.max(1, ...matrix.flat().map(Number))
  const option = {
    tooltip: {
      position: 'top',
      ...TOOLTIP_STYLE,
      formatter(params) {
        const [j, i, value] = params.value
        return `<b>真实 ${labels[i]}</b> · <b>预测 ${labels[j]}</b><br/>样本数：${value}`
      },
    },
    grid: { left: 12, right: 28, top: 12, bottom: rotate ? 92 : 64, containLabel: true },
    xAxis: {
      type: 'category',
      name: '预测值',
      data: labels.map((l) => truncateLabel(String(l), 12)),
      ...categoryAxisStyle({ splitArea: { show: true, areaStyle: { color: ['rgba(255,255,255,0.02)', 'rgba(255,255,255,0.05)'] } } }),
      axisLabel: rotate ? { rotate: 35, color: '#93a1c4' } : { color: '#93a1c4' },
    },
    yAxis: {
      type: 'category',
      name: '真实值',
      data: labels.map((l) => truncateLabel(String(l), 12)),
      ...categoryAxisStyle({ splitArea: { show: true, areaStyle: { color: ['rgba(255,255,255,0.02)', 'rgba(255,255,255,0.05)'] } } }),
    },
    visualMap: {
      min: 0,
      max: cellMax,
      calculable: false,
      orient: 'horizontal',
      left: 'center',
      bottom: 2,
      itemWidth: 12,
      itemHeight: 110,
      textStyle: { color: '#93a1c4' },
      inRange: { color: ['#1f2940', '#3f6ef2', '#34d399'] },
    },
    series: [
      {
        type: 'heatmap',
        data,
        label: {
          show: true,
          color: '#dbe3f5',
          fontSize: 11,
          formatter: (p) => p.value[2],
        },
        emphasis: { itemStyle: { shadowBlur: 8, shadowColor: 'rgba(0,0,0,0.5)' } },
      },
    ],
  }
  return <EChart option={option} height={Math.max(240, labels.length * 52 + 130)} />
}

function RocChart({ roc }) {
  if (!roc || !roc.fpr?.length) {
    return <p className="panel-empty">该分类模型不支持概率输出或未生成 ROC 曲线。</p>
  }
  const points = roc.fpr.map((fpr, i) => [fpr, roc.tpr[i]])
  const option = {
    tooltip: {
      trigger: 'axis',
      ...TOOLTIP_STYLE,
      formatter(params) {
        return params.map((p) => `${p.marker}${p.seriesName}<br/>(${fmtNum(p.value[0])}, ${fmtNum(p.value[1])})`).join('<br/>')
      },
    },
    legend: { top: 0, textStyle: { color: '#93a1c4' }, data: [`ROC（AUC = ${fmtNum(roc.auc)}）`, '随机基线'] },
    grid: baseGrid(),
    xAxis: {
      type: 'value',
      min: 0,
      max: 1,
      name: 'False Positive Rate',
      ...valueAxisStyle(),
    },
    yAxis: {
      type: 'value',
      min: 0,
      max: 1,
      name: 'True Positive Rate',
      ...valueAxisStyle(),
    },
    series: [
      {
        name: `ROC（AUC = ${fmtNum(roc.auc)}）`,
        type: 'line',
        showSymbol: false,
        smooth: true,
        lineStyle: { width: 2 },
        areaStyle: { color: 'rgba(79,140,255,0.12)' },
        data: points,
      },
      {
        name: '随机基线',
        type: 'line',
        showSymbol: false,
        lineStyle: { type: 'dashed', color: '#7f8db3', width: 1 },
        data: [[0, 0], [1, 1]],
      },
    ],
  }
  return (
    <div className="ml-chart-block">
      <p className="eda-chart-note">正类（Positive）：{String(roc.positive_class ?? '')}</p>
      <EChart option={option} height={260} />
    </div>
  )
}

/* ============ 回归可视化 ============ */

function histogram(values, bins = 18) {
  const finite = values.filter((v) => Number.isFinite(v))
  if (!finite.length) return { keys: [], counts: [] }
  const min = Math.min(...finite)
  const max = Math.max(...finite)
  if (max === min) return { keys: [`${fmtNum(min)}`], counts: [finite.length] }
  const width = (max - min) / bins
  const counts = new Array(bins).fill(0)
  finite.forEach((v) => {
    let idx = Math.floor((v - min) / width)
    if (idx >= bins) idx = bins - 1
    counts[idx] += 1
  })
  const keys = Array.from({ length: bins }, (_, i) => `${fmtNum(min + i * width, 2)}`)
  return { keys, counts }
}

function RegressionScatter({ scatter }) {
  if (!scatter || scatter.length === 0) {
    return <p className="panel-empty">暂无散点数据。</p>
  }
  const actuals = scatter.map((s) => Number(s.actual)).filter(Number.isFinite)
  const lo = Math.min(...actuals)
  const hi = Math.max(...actuals)
  const option = {
    tooltip: {
      trigger: 'item',
      ...TOOLTIP_STYLE,
      formatter(p) {
        return `实际值：${fmtNum(p.value[0], 3)}<br/>预测值：${fmtNum(p.value[1], 3)}`
      },
    },
    legend: { top: 0, textStyle: { color: '#93a1c4' } },
    grid: baseGrid(),
    xAxis: { type: 'value', name: '实际值', ...valueAxisStyle() },
    yAxis: { type: 'value', name: '预测值', ...valueAxisStyle() },
    series: [
      {
        name: '实际 vs 预测',
        type: 'scatter',
        symbolSize: 8,
        itemStyle: { color: 'rgba(79,140,255,0.85)' },
        data: scatter.map((s) => [Number(s.actual), Number(s.predicted)]),
      },
      {
        name: '理想对角线',
        type: 'line',
        showSymbol: false,
        lineStyle: { type: 'dashed', color: '#7f8db3', width: 1 },
        data: [[lo, lo], [hi, hi]],
      },
    ],
  }
  return <EChart option={option} height={280} />
}

function ResidualHistogram({ scatter }) {
  if (!scatter || scatter.length === 0) {
    return <p className="panel-empty">暂无残差数据。</p>
  }
  const residuals = scatter.map((s) => Number(s.predicted) - Number(s.actual)).filter(Number.isFinite)
  const { keys, counts } = histogram(residuals)
  if (!keys.length) return <p className="panel-empty">残差全部相同或为空，无法分箱。</p>
  const option = {
    tooltip: {
      trigger: 'axis',
      ...TOOLTIP_STYLE,
      formatter(params) {
        const p = params[0]
        return `残差区间 ${p.name}～<br/>样本数：${p.value}`
      },
    },
    grid: baseGrid(),
    xAxis: {
      type: 'category',
      data: keys,
      name: '残差（预测 - 实际）',
      ...categoryAxisStyle({ axisLabel: { color: '#93a1c4', rotate: keys.length > 10 ? 45 : 0 } }),
    },
    yAxis: { type: 'value', name: '样本数', ...valueAxisStyle() },
    series: [
      {
        type: 'bar',
        barWidth: '70%',
        itemStyle: { color: '#b29bff' },
        data: counts,
      },
    ],
  }
  return <EChart option={option} height={250} />
}

/* ============ 主体报告 ============ */

export default function ExperimentReport({ report }) {
  if (!report) return null
  const taskType = report.task_type
  const isClassification = taskType === 'classification'
  const test = report.test || {}
  const testMetrics = test.metrics || {}
  const cv = report.cv || {}
  const metricName = METRIC_LABELS[report.primary_metric] || report.primary_metric
  const testSummary = report.test_score_summary || {}
  const excluded = report.excluded_columns || []
  const ranking = report.cv_ranking || cv.ranking || []

  // CV 对比表：按主指标排名展示各候选模型的均值
  const cvResults = cv.results || {}
  const firstModel = Object.keys(cvResults)[0]
  const metricKeys = firstModel ? Object.keys(cvResults[firstModel].metrics || {}) : []
  const orderedModels = ranking.length
    ? ranking.map((r) => r.model)
    : Object.keys(cvResults)

  return (
    <div className="ml-report">
      {report.potential_data_leakage && (
        <div className="dp-warning">
          ⚠ 该实验基于执行过全量统计型预处理的派生版本训练，评估指标可能偏乐观（详见 warnings）。
        </div>
      )}
      <WarningLines items={report.warnings} />

      <div className="metric-grid">
        <MetricCard label="实验 ID" value={shortId(report.experiment_id)} accent="accent-blue" />
        <MetricCard label="任务类型" value={TASK_LABELS[taskType] || taskType} />
        <MetricCard label="最佳模型" value={modelName(report.best_model)} accent="accent-violet" />
        <MetricCard
          label={report.beats_baseline ? '优于 Dummy 基线' : '未优于基线'}
          value={report.beats_baseline ? '是' : '否'}
          accent={report.beats_baseline ? 'accent-green' : 'accent-amber'}
        />
        <MetricCard
          label={`Test ${metricName}`}
          value={fmtNum(testSummary.score)}
          accent="accent-blue"
          sub={`CV 均值 ${fmtNum(report.best_cv_primary_mean)}`}
        />
      </div>

      <div className="ml-chip-row">
        <code className="op-chip">目标字段：{report.target_column}</code>
        <code className="op-chip">源版本：{report.source_version_id === 'original' ? 'Original' : shortId(report.source_version_id)}</code>
        <code className="op-chip">训练 {report.train_rows} 行 / 测试 {report.test_rows} 行</code>
        <code className="op-chip">CV 折数：{report.effective_cv_folds}（请求 {report.requested_cv_folds}）</code>
        <code className="op-chip">预处理 fit 范围：{report.preprocessing_fit_scope || 'train_only'}</code>
        <code className="op-chip">random_state={report.random_state}</code>
        {report.created_at && <code className="op-chip">{fmtDate(report.created_at)}</code>}
      </div>

      <p className="eda-chart-note">
        任务判定说明：{report.task_type_reason || '—'}
        {report.dropped_missing_target_rows ? `；因目标缺失剔除 ${report.dropped_missing_target_rows} 行` : ''}
      </p>

      {/* CV 模型对比 */}
      <section className="ml-report-block">
        <h4 className="eda-subtitle">训练集内交叉验证 · 候选模型对比</h4>
        {orderedModels.length ? (
          <div className="table-scroll">
            <table className="data-table">
              <thead>
                <tr>
                  <th>候选模型</th>
                  {metricKeys.map((m) => (
                    <th key={m}>{METRIC_LABELS[m] || m}</th>
                  ))}
                  <th>主指标排名</th>
                </tr>
              </thead>
              <tbody>
                {orderedModels.map((key, rankIndex) => {
                  const detail = cvResults[key]
                  if (!detail) return null
                  const rankLabel =
                    report.best_model === key
                      ? `最佳`
                      : `#${rankIndex + 1}`
                  return (
                    <tr key={key} className={report.best_model === key ? 'ml-row-best' : ''}>
                      <td>
                        <span className="field-name">{modelName(key)}</span>
                        {report.best_model === key && <span className="chip chip-active">best</span>}
                      </td>
                      {metricKeys.map((m) => (
                        <td key={m}>
                          {fmtNum(detail.metrics?.[m]?.mean)}
                          <span className="cell-sub">±{fmtNum(detail.metrics?.[m]?.std)}</span>
                        </td>
                      ))}
                      <td>{rankLabel}</td>
                    </tr>
                  )
                })}
              </tbody>
            </table>
          </div>
        ) : (
          <p className="panel-empty">无 CV 结果。</p>
        )}
        {cv.model_errors && Object.keys(cv.model_errors).length > 0 && (
          <div className="dp-warnings">
            {Object.entries(cv.model_errors).map(([key, err]) => (
              <div key={key} className="dp-warning">
                ⚠ 候选模型 {modelName(key)} 在 CV 中训练失败（已跳过）：{err}
              </div>
            ))}
          </div>
        )}
      </section>

      {/* 特征信息 */}
      <section className="ml-report-block">
        <h4 className="eda-subtitle">特征与样本信息</h4>
        <div className="ml-chip-row">
          <code className="op-chip">特征共 {report.features?.length || 0} 列</code>
          {(report.numeric_features || []).map((c) => (
            <code key={c} className="op-chip good">数值 · {c}</code>
          ))}
          {(report.categorical_features || []).map((c) => (
            <code key={c} className="op-chip">分类 · {c}</code>
          ))}
        </div>
        {excluded.length > 0 && (
          <p className="ml-tip">
            已排除字段（不参与训练）：
            {excluded.map((item) => (
              <span key={item.column} className="op-chip bad">
                − {item.column}
                {item.reason ? `（${item.reason}）` : ''}
              </span>
            ))}
          </p>
        )}
        <p className="ml-tip">
          编码后特征（feature_names_out）：{report.feature_names_out?.length ?? 0} 维 · 数据指纹
          <code className="dataset-id-text"> {shortId(report.dataset_sha256)}</code> · sklearn
          {report.sklearn_version ? ` ${report.sklearn_version}` : ''}
        </p>
      </section>

      {/* 测试集评估 */}
      <section className="ml-report-block">
        <h4 className="eda-subtitle">
          保留测试集最终评估（{isClassification ? '分类' : '回归'} · untouched Test Set）
        </h4>
        <div className="table-scroll">
          <table className="data-table">
            <tbody>
              {Object.entries(testMetrics).map(([key, value]) => (
                <tr key={key}>
                  <td className="ml-kv-key">{METRIC_LABELS[key] || key}</td>
                  <td>{fmtNum(value)}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>

        {isClassification ? (
          <>
            {test.per_class_metrics?.length > 0 && (
              <DataTable
                title="各类别指标"
                rows={test.per_class_metrics.map((c) => ({
                  类别: String(c.class),
                  精确率: fmtNum(c.precision),
                  召回率: fmtNum(c.recall),
                  F1: fmtNum(c.f1),
                  样本数: c.support,
                }))}
              />
            )}
            <div className="ml-chart-row">
              <div className="ml-chart-block">
                <h5 className="data-table-title">混淆矩阵</h5>
                <ConfusionHeatmap cm={test.confusion_matrix} />
              </div>
              <div className="ml-chart-block">
                <h5 className="data-table-title">ROC 曲线</h5>
                <RocChart roc={test.roc} />
              </div>
            </div>
          </>
        ) : (
          <>
            <div className="ml-tip">
              残差统计：均值 {fmtNum(test.residuals?.mean)} · 标准差 {fmtNum(test.residuals?.std)} ·
              范围 [{fmtNum(test.residuals?.min)}, {fmtNum(test.residuals?.max)}]
            </div>
            <div className="ml-chart-row">
              <div className="ml-chart-block">
                <h5 className="data-table-title">实际 vs 预测（抽样散点）</h5>
                <RegressionScatter scatter={test.scatter} />
              </div>
              <div className="ml-chart-block">
                <h5 className="data-table-title">残差分布（抽样）</h5>
                <ResidualHistogram scatter={test.scatter} />
              </div>
            </div>
          </>
        )}
      </section>
    </div>
  )
}
