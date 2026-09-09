// ECharts 在暗色面板上的统一配色与坐标轴样式

export const PALETTE = [
  '#7d9dff',
  '#b29bff',
  '#34d399',
  '#fbbf24',
  '#f87171',
  '#67e8f9',
  '#f472b6',
  '#a3e635',
]

export const AXIS_COLOR = '#93a1c4'
export const AXIS_LINE = 'rgba(255,255,255,0.16)'
export const SPLIT_LINE = 'rgba(255,255,255,0.07)'
export const TOOLTIP_BG = 'rgba(17, 23, 51, 0.96)'
export const TOOLTIP_BORDER = 'rgba(125, 157, 255, 0.5)'

export const TOOLTIP_STYLE = {
  backgroundColor: TOOLTIP_BG,
  borderColor: TOOLTIP_BORDER,
  borderWidth: 1,
  textStyle: { color: '#e6edfb', fontSize: 12 },
  extraCssText: 'border-radius:8px;box-shadow:0 6px 20px rgba(0,0,0,0.35);',
}

export function baseGrid(extra = {}) {
  return { left: 12, right: 16, top: 32, bottom: 8, containLabel: true, ...extra }
}

export function valueAxisStyle(extra = {}) {
  return {
    axisLine: { lineStyle: { color: AXIS_LINE } },
    axisLabel: { color: AXIS_COLOR },
    splitLine: { lineStyle: { color: SPLIT_LINE } },
    ...extra,
  }
}

export function categoryAxisStyle(extra = {}) {
  return {
    axisLine: { lineStyle: { color: AXIS_LINE } },
    axisTick: { show: false },
    axisLabel: { color: AXIS_COLOR },
    ...extra,
  }
}

export function truncateLabel(text, max = 12) {
  const value = String(text ?? '')
  return value.length > max ? `${value.slice(0, max)}…` : value
}
