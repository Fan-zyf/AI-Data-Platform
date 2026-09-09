import { useState } from 'react'
import { OP_DEFS, OP_GROUPS, FILL_STRATEGIES, FEATURE_OPTIONS, createStep, describeStep, missingParams } from './opDefinitions.js'

const TYPE_TEXT = {
  numeric: '数值',
  categorical: '分类',
  text: '文本',
  datetime: '日期时间',
  boolean: '布尔',
}

function columnOptions(columns, filter) {
  if (!columns || columns.length === 0) return []
  const list = filter ? columns.filter((col) => filter.includes(col.type)) : columns
  return list
}

function ParamSelect({ label, value, options, onChange, placeholder = '请选择…', disabled }) {
  return (
    <label className="dp-field">
      <span className="dp-field-label">{label}</span>
      <select
        value={value || ''}
        disabled={disabled || options.length === 0}
        onChange={(e) => onChange(e.target.value)}
      >
        <option value="" disabled>
          {options.length === 0 ? '无可用字段' : placeholder}
        </option>
        {options.map((opt) => (
          <option key={opt.value} value={opt.value}>
            {opt.label}
          </option>
        ))}
      </select>
      {disabled && <em className="dp-field-hint">当前版本没有符合条件的字段</em>}
    </label>
  )
}

function renderStepForm({ step, columns, onChange }) {
  const numericCols = columnOptions(columns, ['numeric'])
  const catCols = columnOptions(columns, ['categorical', 'boolean', 'text'])
  const dateCols = columnOptions(columns, ['datetime'])
  const anyCols = columnOptions(columns, null)
  const colItem = (col) => ({
    value: col.name,
    label: `${col.name}（${TYPE_TEXT[col.type] || col.type}${col.missing_count ? `，缺失 ${col.missing_count}` : ''}）`,
  })

  const pickers = {
    fill_missing: (
      <>
        <ParamSelect label="字段" value={step.column} options={anyCols.map(colItem)} onChange={(v) => onChange('column', v)} />
        <label className="dp-field">
          <span className="dp-field-label">填充策略</span>
          <select value={step.strategy} onChange={(e) => onChange('strategy', e.target.value)}>
            {FILL_STRATEGIES.map((s) => (
              <option key={s.value} value={s.value}>
                {s.label}
              </option>
            ))}
          </select>
        </label>
        {step.strategy === 'constant' && (
          <label className="dp-field">
            <span className="dp-field-label">填充值 constant</span>
            <input
              type="text"
              value={step.fillValue}
              onChange={(e) => onChange('fillValue', e.target.value)}
              placeholder="例如 0 或 “未知”"
            />
          </label>
        )}
      </>
    ),
    drop_columns: (
      <label className="dp-field">
        <span className="dp-field-label">删除字段（可多选）</span>
        <select
          multiple
          value={step.columns}
          onChange={(e) => {
            const values = Array.from(e.target.selectedOptions, (o) => o.value)
            onChange('columns', values)
          }}
        >
          {anyCols.map((col) => (
            <option key={col.name} value={col.name}>
              {col.name}
            </option>
          ))}
        </select>
        <em className="dp-field-hint">按住 Ctrl/⌘ 多选；系统禁止删除全部字段</em>
      </label>
    ),
    convert_type: (
      <>
        <ParamSelect label="字段" value={step.column} options={anyCols.map(colItem)} onChange={(v) => onChange('column', v)} />
        <label className="dp-field">
          <span className="dp-field-label">转换目标类型</span>
          <select value={step.targetType} onChange={(e) => onChange('targetType', e.target.value)}>
            <option value="numeric">numeric 数值</option>
            <option value="string">string 文本</option>
            <option value="datetime">datetime 日期时间</option>
            <option value="boolean">boolean 布尔</option>
          </select>
        </label>
      </>
    ),
    remove_outliers: (
      <>
        <ParamSelect label="数值字段" value={step.column} options={numericCols.map(colItem)} onChange={(v) => onChange('column', v)} />
        <label className="dp-field">
          <span className="dp-field-label">处理方法</span>
          <select value={step.action} onChange={(e) => onChange('action', e.target.value)}>
            <option value="clip">clip 裁剪到 IQR 边界</option>
            <option value="remove_rows">remove_rows 删除异常行</option>
          </select>
        </label>
      </>
    ),
    text_transform: (
      <>
        <ParamSelect label="文本字段" value={step.column} options={catCols.map(colItem)} onChange={(v) => onChange('column', v)} />
        <label className="dp-field">
          <span className="dp-field-label">处理动作</span>
          <select value={step.action} onChange={(e) => onChange('action', e.target.value)}>
            <option value="strip">strip 去除首尾空白</option>
            <option value="lowercase">lowercase 转小写</option>
            <option value="uppercase">uppercase 转大写</option>
          </select>
        </label>
      </>
    ),
    date_features: (
      <>
        <ParamSelect label="日期字段" value={step.column} options={dateCols.map(colItem)} onChange={(v) => onChange('column', v)} />
        <fieldset className="dp-features">
          <legend>派生特征（多选）</legend>
          <div className="dp-checkbox-grid">
            {FEATURE_OPTIONS.map((f) => {
              const active = step.features.includes(f.value)
              return (
                <label key={f.value} className={`dp-checkbox ${active ? 'checked' : ''}`}>
                  <input
                    type="checkbox"
                    checked={active}
                    onChange={(e) => {
                      const next = e.target.checked
                        ? [...step.features, f.value]
                        : step.features.filter((x) => x !== f.value)
                      onChange('features', next)
                    }}
                  />
                  {f.label}
                </label>
              )
            })}
          </div>
        </fieldset>
        <em className="dp-field-hint">新字段命名：{step.column ? `${step.column}_year` : '<字段>_year'}，冲突时会返回错误</em>
      </>
    ),
    one_hot_encode: (
      <>
        <ParamSelect label="分类字段" value={step.column} options={catCols.map(colItem)} onChange={(v) => onChange('column', v)} />
        <label className="dp-checkbox">
          <input
            type="checkbox"
            checked={step.allowHighCardinality}
            onChange={(e) => onChange('allowHighCardinality', e.target.checked)}
          />
          我已确认允许高基数字段（超过 50 个取值时仍执行）
        </label>
      </>
    ),
    scale_numeric: (
      <>
        <ParamSelect label="数值字段" value={step.column} options={numericCols.map(colItem)} onChange={(v) => onChange('column', v)} />
        <label className="dp-field">
          <span className="dp-field-label">缩放方法</span>
          <select value={step.scaleMethod} onChange={(e) => onChange('scaleMethod', e.target.value)}>
            <option value="standardization">standardization 标准化 (x-μ)/σ</option>
            <option value="min_max">min-max 归一化 (x-min)/(max-min)</option>
          </select>
        </label>
      </>
    ),
    drop_duplicates: null,
  }

  return pickers[step.type] || null
}

function StepCard({ step, index, total, columns, onChange, onRemove, onMove }) {
  const issue = missingParams(step)
  return (
    <div className={`dp-step-card ${issue ? 'invalid' : ''}`}>
      <div className="dp-step-head">
        <span className="dp-step-index">Step {index + 1}</span>
        <strong className="dp-step-title">{OP_DEFS[step.type]?.label || step.type}</strong>
        <div className="dp-step-actions">
          <button type="button" className="icon-btn" title="上移" disabled={index === 0} onClick={() => onMove(index, -1)}>
            ↑
          </button>
          <button type="button" className="icon-btn" title="下移" disabled={index === total - 1} onClick={() => onMove(index, 1)}>
            ↓
          </button>
          <button type="button" className="icon-btn danger" title="删除该步骤" onClick={() => onRemove(step.id)}>
            ✕
          </button>
        </div>
      </div>
      <p className="dp-step-desc">{OP_DEFS[step.type]?.desc}</p>
      <div className="dp-step-form">{renderStepForm({ step, columns, onChange })}</div>
      <div className="dp-step-foot">
        <code>{describeStep(step)}</code>
        {issue && <span className="dp-step-issue">{issue}</span>}
      </div>
    </div>
  )
}

export default function TransformationBuilder({ columns, steps, onStepsChange }) {
  const [openGroup, setOpenGroup] = useState(null)

  const addStep = (type) => {
    const step = createStep(type)
    if (!step) return
    // 自动填充第一个可用字段，减少操作
    const def = OP_DEFS[type]
    const numeric = columns.filter((c) => c.type === 'numeric')
    const cat = columns.filter((c) => ['categorical', 'text', 'boolean'].includes(c.type))
    const dt = columns.filter((c) => c.type === 'datetime')
    if (type === 'remove_outliers' || type === 'scale_numeric') step.column = numeric[0]?.name || ''
    else if (type === 'date_features') step.column = dt[0]?.name || ''
    else if (['text_transform', 'one_hot_encode'].includes(type)) step.column = cat[0]?.name || ''
    else if (['fill_missing', 'convert_type'].includes(type)) step.column = columns[0]?.name || ''
    onStepsChange([...steps, step])
  }

  const updateStep = (id, key, value) => {
    onStepsChange(steps.map((s) => (s.id === id ? { ...s, [key]: value } : s)))
  }

  const removeStep = (id) => {
    onStepsChange(steps.filter((s) => s.id !== id))
  }

  const moveStep = (index, dir) => {
    const target = index + dir
    if (target < 0 || target >= steps.length) return
    const next = [...steps]
    ;[next[index], next[target]] = [next[target], next[index]]
    onStepsChange(next)
  }

  return (
    <div className="dp-builder">
      <div className="dp-builder-head">
        <h4 className="eda-subtitle">Transformation Builder（步骤按顺序执行）</h4>
        <div className="dp-builder-tools">
          <span className="dp-builder-count">{steps.length} 个步骤</span>
          {steps.length > 0 && (
            <button type="button" className="btn-ghost btn-small" onClick={() => onStepsChange([])}>
              清空
            </button>
          )}
        </div>
      </div>

      {columns.length > 0 && (
        <div className="dp-add-steps">
          {OP_GROUPS.map((group) => (
            <details key={group.label} className="dp-group" open={openGroup === group.label} onToggle={(e) => setOpenGroup(e.target.open ? group.label : null)}>
              <summary className="dp-group-summary">{group.label}</summary>
              <div className="dp-group-buttons">
                {group.types.map((type) => (
                  <button key={type} type="button" className="chip-btn" onClick={() => addStep(type)}>
                    + {OP_DEFS[type].label}
                  </button>
                ))}
              </div>
            </details>
          ))}
        </div>
      )}

      {steps.length === 0 ? (
        <p className="panel-empty">点击上方「添加处理步骤」构建数据清洗 / 特征工程流程。</p>
      ) : (
        <div className="dp-steps">
          {steps.map((step, index) => (
            <StepCard
              key={step.id}
              step={step}
              index={index}
              total={steps.length}
              columns={columns}
              onChange={(key, value) => updateStep(step.id, key, value)}
              onRemove={removeStep}
              onMove={moveStep}
            />
          ))}
        </div>
      )}
    </div>
  )
}
