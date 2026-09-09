import { useEffect, useRef } from 'react'
import * as echarts from 'echarts'

/**
 * 轻量 ECharts 封装：负责实例生命周期与自适应尺寸，
 * 图表配置完全由父组件通过 option 传入。
 */
export default function EChart({ option, height = 280, className = '' }) {
  const containerRef = useRef(null)
  const chartRef = useRef(null)

  useEffect(() => {
    const el = containerRef.current
    if (!el) return

    const chart = echarts.init(el)
    chartRef.current = chart

    const observer = new ResizeObserver(() => chart.resize())
    observer.observe(el)

    return () => {
      observer.disconnect()
      chart.dispose()
      chartRef.current = null
    }
  }, [])

  useEffect(() => {
    if (chartRef.current && option) {
      chartRef.current.setOption(option, true)
    }
  }, [option])

  return <div ref={containerRef} className={`echart ${className}`} style={{ height }} />
}
