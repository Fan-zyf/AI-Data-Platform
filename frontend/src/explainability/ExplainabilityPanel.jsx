import { useCallback, useEffect, useState } from 'react'
import FeatureImportanceChart from './FeatureImportanceChart.jsx'
import SampleExplanation from './SampleExplanation.jsx'
import SHAPSummaryChart from './SHAPSummaryChart.jsx'

const MODEL_LABELS = {
  dummy: 'Dummy Baseline',
  logistic_regression: 'Logistic Regression',
  random_forest: 'Random Forest',
  ridge: 'Ridge',
}

const EXPLAINER_LABELS = {
  tree: 'Tree SHAP（精确）',
  linear: 'Linear SHAP（interventional）',
  kernel: 'Kernel SHAP（兜底）',
}

/**
 * 模型可解释性（SHAP）面板：
 * - 选中某个 Experiment 后才会显示；
 * - 提供「生成 / 重新生成 SHAP 解释」按钮；
 * - 解释完成后展示：全局特征重要性 + SHAP summary + 单点解释；
 * - 错误以结构化错误条形式呈现。
 */
export default function ExplainabilityPanel({ datasetId, experiment }) {
  const [explanation, setExplanation] = useState(null)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState(null)
  const [notice, setNotice] = useState(null)
  const [maxSummaryRows, setMaxSummaryRows] = useState('')
  const [sampleIndicesText, setSampleIndicesText] = useState('')

  // 切换到不同实验时，强制重置所有 SHAP 状态
  useEffect(() => {
    setExplanation(null)
    setError(null)
    setNotice(null)
    setMaxSummaryRows('')
    setSampleIndicesText('')
  }, [experiment?.experiment_id])

  const fetchExplanation = useCallback(
    async ({ regenerate = false, body = null } = {}) => {
      if (!experiment?.experiment_id) return
      setLoading(true)
      setError(null)
      setNotice(null)
      try {
        let payload = body
        if (!payload) {
          payload = {}
          if (regenerate) payload.regenerate = true
          if (maxSummaryRows) {
            const parsed = Number(maxSummaryRows)
            if (Number.isFinite(parsed) && parsed > 0) payload.max_summary_rows = parsed
          }
          if (sampleIndicesText.trim()) {
            const arr = sampleIndicesText
              .split(',')
              .map((s) => Number(s.trim()))
              .filter((n) => Number.isInteger(n))
            if (arr.length) payload.sample_indices = arr
          }
        }
        const res = await fetch(
          `/api/ml/experiments/${experiment.experiment_id}/explain`,
          {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(payload),
          },
        )
        const data = await res.json().catch(() => null)
        if (!res.ok) {
          setError(data?.error?.message || `请求失败（HTTP ${res.status}）`)
          return
        }
        if (!data || data.success !== true) {
          setError(data?.error?.message || '服务器返回了无法识别的数据')
          return
        }
        setExplanation(data)
        if (regenerate) {
          setNotice(`已为实验 ${shortId(data.experiment_id)} 重新生成 SHAP 解释。`)
        } else if (data.cached) {
          setNotice('已加载缓存的解释结果（重新生成请勾选「强制重算」）。')
        } else {
          setNotice(`已为实验 ${shortId(data.experiment_id)} 生成 ${EXPLAINER_LABELS[data.explainer_type] || data.explainer_type} 解释。`)
        }
      } catch (err) {
        setError(err?.message || '无法连接后端服务')
      } finally {
        setLoading(false)
      }
    },
    [experiment?.experiment_id, maxSummaryRows, sampleIndicesText],
  )

  if (!experiment) {
    return null
  }

  const modelName = MODEL_LABELS[experiment.best_model] || experiment.best_model
  const taskLabel = experiment.task_type === 'regression' ? '回归' : '分类'

  return (
    <section className="panel explainability-panel">
      <div className="eda-heading">
        <h3 className="panel-title eda-title">模型可解释性（SHAP）</h3>
        <p className="eda-meta">
          实验 <code>{shortId(experiment.experiment_id)}</code> · {taskLabel} ·{' '}
          {modelName}
          {experiment.explainability?.status === 'computed' && (
            <span className="ml-tip"> · 已生成解释（{EXPLAINER_LABELS[experiment.explainability.explainer_type] || experiment.explainability.explainer_type}）</span>
          )}
        </p>
      </div>
      <p className="eda-desc">
        基于 v0.5 训练完成的 Pipeline，使用 SHAP（Tree / Linear / Kernel 解释器自动路由）
        对源版本数据进行全局特征重要性 + Summary + 单点预测解释。
        解释结果持久化到 <code>runtime/datasets/&lt;id&gt;/ml/experiments/&lt;exp&gt;/shap_result.json</code>。
      </p>

      {error && (
        <div className="error-box" role="alert">
          <strong>SHAP 生成失败</strong>
          <span>{error}</span>
        </div>
      )}
      {notice && (
        <div className="success-box" role="status">
          {notice}
        </div>
      )}

      <div className="shap-config">
        <div className="shap-config-grid">
          <label className="ml-field">
            <span className="ml-field-label">Summary 采样行数（可选）</span>
            <input
              type="number"
              min="1"
              max="1000"
              placeholder={`默认 ${experiment.explainability?.n_rows_used || 200}`}
              value={maxSummaryRows}
              onChange={(e) => setMaxSummaryRows(e.target.value)}
            />
            <span className="ml-tip">从源版本随机采样，1~1000。</span>
          </label>
          <label className="ml-field">
            <span className="ml-field-label">单点解释样本索引（可选）</span>
            <input
              type="text"
              placeholder="例如：0, 5, 12"
              value={sampleIndicesText}
              onChange={(e) => setSampleIndicesText(e.target.value)}
            />
            <span className="ml-tip">相对于采样行；最多 20 个，整数。</span>
          </label>
        </div>
        <div className="dp-actions">
          <button
            type="button"
            className="btn-primary"
            disabled={loading}
            onClick={() => fetchExplanation({ regenerate: false })}
          >
            {loading
              ? '计算中…'
              : explanation
                ? '加载 / 刷新解释'
                : '生成 SHAP 解释'}
          </button>
          {explanation && (
            <button
              type="button"
              className="btn-ghost btn-small"
              disabled={loading}
              onClick={() => fetchExplanation({ regenerate: true })}
            >
              强制重算
            </button>
          )}
        </div>
      </div>

      {explanation && (
        <div className="shap-result">
          <div className="shap-result-meta">
            <span className="ml-tip">
              解释器：<b>{EXPLAINER_LABELS[explanation.explainer_type] || explanation.explainer_type}</b>
              {' · '}模型：<b>{explanation.model_class}</b>
              {' · '}使用样本数：<b>{explanation.global?.n_rows_used}</b>
              {' · '}post-preprocessing 特征数：<b>{explanation.n_features_post_preprocessing}</b>
              {explanation.cached && (
                <span className="ml-tip"> · 来自缓存（{shortId(explanation.created_at)}）</span>
              )}
            </span>
          </div>

          {(explanation.warnings || []).length > 0 && (
            <div className="dp-warnings">
              {explanation.warnings.map((w, i) => (
                <div key={`${w}-${i}`} className="dp-warning">
                  ⚠ {w}
                </div>
              ))}
            </div>
          )}

          <FeatureImportanceChart
            global={explanation.global}
            classLabelUsed={explanation.class_label_used}
            taskType={explanation.task_type}
          />
          <SHAPSummaryChart summary={explanation.summary} />
          <SampleExplanation
            samples={explanation.samples}
            classLabelUsed={explanation.class_label_used}
            taskType={explanation.task_type}
          />
        </div>
      )}

      {!explanation && !loading && (
        <p className="panel-empty">
          点击「生成 SHAP 解释」即可计算全局特征重要性、SHAP summary 与若干单点预测解释。
          对树模型使用 TreeExplainer（精确），线性模型使用 LinearExplainer，Dummy 等其他模型自动走 KernelExplainer 兜底。
        </p>
      )}
    </section>
  )
}

function shortId(value) {
  if (!value) return ''
  if (typeof value === 'string' && value.length > 19) return value.slice(0, 19)
  return String(value)
}
