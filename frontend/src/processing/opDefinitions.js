// v0.4 Transformation Plan 操作定义（与后端 app/models/processing.py 保持一致）
export const OP_DEFS = {
  drop_duplicates: { type: 'drop_duplicates', label: '删除重复行', group: '数据清洗', desc: '删除完全重复的行，仅保留首次出现' },
  fill_missing: { type: 'fill_missing', label: '缺失值处理', group: '数据清洗', desc: '按策略处理选定字段的缺失值' },
  drop_columns: { type: 'drop_columns', label: '删除字段', group: '数据清洗', desc: '从数据集中删除选定字段（不能删除全部字段）' },
  convert_type: { type: 'convert_type', label: '类型转换', group: '数据清洗', desc: '安全转换字段类型（string/numeric/datetime/boolean）' },
  remove_outliers: { type: 'remove_outliers', label: 'IQR 异常值处理', group: '数据清洗', desc: '基于 IQR 边界裁剪(clip)或删除异常行(remove_rows)' },
  text_transform: { type: 'text_transform', label: '文本处理', group: '数据清洗', desc: '对文本字段执行确定性的 strip / lowercase / uppercase' },
  date_features: { type: 'date_features', label: '日期特征工程', group: '特征工程', desc: '由日期字段派生 year/month/day/day_of_week/quarter' },
  one_hot_encode: { type: 'one_hot_encode', label: 'One-Hot 编码', group: '特征工程', desc: '分类字段转为一组 0/1 指示字段（高基数需显式确认）' },
  scale_numeric: { type: 'scale_numeric', label: '数值缩放', group: '特征工程', desc: 'standardization(z-score) 或 min-max 归一化' },
}

export const OP_ORDER = Object.keys(OP_DEFS)

export const OP_GROUPS = [
  { label: '数据清洗', types: OP_ORDER.filter((t) => OP_DEFS[t].group === '数据清洗') },
  { label: '特征工程', types: OP_ORDER.filter((t) => OP_DEFS[t].group === '特征工程') },
]

export const FILL_STRATEGIES = [
  { value: 'mean', label: '平均值 mean', numeric: true },
  { value: 'median', label: '中位数 median', numeric: true },
  { value: 'constant', label: '固定值 constant', numeric: true, other: true },
  { value: 'mode', label: '众数 mode', other: true },
  { value: 'drop_rows', label: '删除含缺失的行', numeric: true, other: true },
  { value: 'drop_columns', label: '直接删除字段', numeric: true, other: true },
]

export const FEATURE_OPTIONS = [
  { value: 'year', label: 'year 年' },
  { value: 'month', label: 'month 月' },
  { value: 'day', label: 'day 日' },
  { value: 'day_of_week', label: 'day_of_week 星期' },
  { value: 'quarter', label: 'quarter 季度' },
]

export function createStep(type) {
  if (!OP_DEFS[type]) return null
  const step = {
    id: `${type}-${Date.now()}-${Math.random().toString(36).slice(2, 7)}`,
    type,
    column: '',
    columns: [],
    strategy: '',
    fillValue: '',
    targetType: '',
    action: '',
    scaleMethod: '',
    features: [],
    allowHighCardinality: false,
  }
  if (type === 'fill_missing') step.strategy = 'median'
  if (type === 'convert_type') step.targetType = 'numeric'
  if (type === 'remove_outliers') step.action = 'clip'
  if (type === 'text_transform') step.action = 'strip'
  if (type === 'scale_numeric') step.scaleMethod = 'standardization'
  if (type === 'date_features') step.features = ['year', 'month']
  return step
}

export function missingParams(step) {
  if (step.type === 'drop_duplicates') return null
  if (step.type === 'fill_missing' && !step.strategy) return '请选择填充策略'
  if (step.type === 'date_features' && step.features.length === 0) return '请至少选择一个日期特征'
  if (step.type === 'drop_columns' && step.columns.length === 0) return '请选择要删除的字段'
  if (step.type !== 'drop_columns' && !step.column) return '请选择一个字段'
  return null
}

function valuePart(step, detail) {
  return detail ? ` ${detail}` : ''
}

export function describeStep(step) {
  const def = OP_DEFS[step.type]
  if (!def) return step.type
  let detail = ''
  const single = step.column || (step.columns.length === 1 ? step.columns[0] : '')
  switch (step.type) {
    case 'fill_missing':
      detail = `字段 ${single} · ${step.strategy || '?'}${step.strategy === 'constant' && step.fillValue !== '' ? ` = ${step.fillValue}` : ''}`
      break
    case 'drop_columns':
      detail = step.columns.length ? `共 ${step.columns.length} 个字段` : ''
      break
    case 'convert_type':
      detail = `字段 ${single} → ${step.targetType || '?'}`
      break
    case 'remove_outliers':
      detail = `字段 ${single} · ${step.action === 'clip' ? '裁剪边界' : '删除异常行'}`
      break
    case 'text_transform':
      detail = `字段 ${single} · ${step.action || '?'}`
      break
    case 'date_features':
      detail = `字段 ${single} · ${step.features.length ? step.features.join('/') : '?'}`
      break
    case 'one_hot_encode':
      detail = `字段 ${single}${step.allowHighCardinality ? '（已确认高基数）' : ''}`
      break
    case 'scale_numeric':
      detail = `字段 ${single} · ${step.scaleMethod === 'standardization' ? '标准化' : 'Min-Max'}`
      break
    default:
      break
  }
  return `${def.label}${valuePart(step, detail)}`
}

// 将前端 step 转为后端 TransformationOperation JSON
export function stepToOperation(step) {
  const base = { type: step.type }
  if (step.type === 'drop_columns') {
    base.columns = step.columns
    return base
  }
  if (step.type === 'drop_duplicates') return base
  if (step.type === 'fill_missing') {
    base.columns = [step.column]
    base.strategy = step.strategy
    if (step.strategy === 'constant') {
      const parsed = Number(step.fillValue)
      base.fill_value = step.fillValue !== '' && !Number.isNaN(parsed) ? parsed : step.fillValue
    }
    return base
  }
  if (step.type === 'convert_type') {
    base.column = step.column
    base.target_type = step.targetType
    return base
  }
  if (step.type === 'remove_outliers') {
    base.column = step.column
    base.method = 'IQR'
    base.action = step.action
    return base
  }
  if (step.type === 'text_transform') {
    base.columns = [step.column]
    base.action = step.action
    return base
  }
  if (step.type === 'date_features') {
    base.column = step.column
    base.features = step.features
    return base
  }
  if (step.type === 'one_hot_encode') {
    base.columns = [step.column]
    base.allow_high_cardinality = step.allowHighCardinality
    return base
  }
  if (step.type === 'scale_numeric') {
    base.columns = [step.column]
    base.scale_method = step.scaleMethod
    return base
  }
  return base
}
