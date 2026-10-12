import { useState } from 'react'
import { f3 } from '../format'

/**
 * Horizontal dot plot with ± std whiskers, an optional shaded band and reference line.
 * rows: [{label, mean, std, highlight?}]. Shapes differ for highlighted rows so identity is not colour-only.
 */
export default function DotPlot({ rows, domain, reference = 0.5, band = null, referenceLabel = 'chance', xLabel }) {
  const [hover, setHover] = useState(null)
  const rowH = 30, left = 340, right = 24, top = 8, bottom = 44, width = 900
  const height = top + bottom + rows.length * rowH
  const x = (v) => left + ((v - domain[0]) / (domain[1] - domain[0])) * (width - left - right)
  const ticks = Array.from({ length: 5 }, (_, i) => domain[0] + ((domain[1] - domain[0]) * i) / 4)
  return (
    <svg viewBox={`0 0 ${width} ${height}`} role="img" aria-label={xLabel} style={{ width: '100%', height: 'auto' }}>
      {band && <rect x={x(band[0])} y={top} width={x(band[1]) - x(band[0])} height={rows.length * rowH} fill="var(--grid)" opacity="0.7" />}
      {ticks.map((t) => (
        <g key={t}>
          <line x1={x(t)} x2={x(t)} y1={top} y2={top + rows.length * rowH} stroke="var(--grid)" />
          <text x={x(t)} y={height - 24} textAnchor="middle" fontSize="12" fill="var(--ink2)">{t.toFixed(2)}</text>
        </g>
      ))}
      <line x1={x(reference)} x2={x(reference)} y1={top} y2={top + rows.length * rowH} stroke="var(--ink2)" strokeWidth="1.5" />
      <text x={x(reference)} y={top - 0} dy="-0" fontSize="11" fill="var(--ink2)" textAnchor="middle" />
      {rows.map((r, i) => {
        const y = top + i * rowH + rowH / 2
        const col = r.highlight ? 'var(--s2)' : 'var(--s1)'
        const lo = r.mean - (r.std || 0), hi = r.mean + (r.std || 0)
        return (
          <g key={r.label} onMouseEnter={() => setHover(i)} onMouseLeave={() => setHover(null)} onFocus={() => setHover(i)} onBlur={() => setHover(null)} tabIndex={0}>
            <rect x={0} y={y - rowH / 2} width={width} height={rowH} fill="transparent" />
            <text x={left - 10} y={y + 4} textAnchor="end" fontSize="13" fill="var(--ink)">{r.label}</text>
            <line x1={x(lo)} x2={x(hi)} y1={y} y2={y} stroke={col} strokeWidth="2" />
            <line x1={x(lo)} x2={x(lo)} y1={y - 5} y2={y + 5} stroke={col} strokeWidth="2" />
            <line x1={x(hi)} x2={x(hi)} y1={y - 5} y2={y + 5} stroke={col} strokeWidth="2" />
            {r.highlight
              ? <rect x={x(r.mean) - 5} y={y - 5} width="10" height="10" fill={col} stroke="var(--surface)" strokeWidth="2" />
              : <circle cx={x(r.mean)} cy={y} r="5" fill={col} stroke="var(--surface)" strokeWidth="2" />}
            {hover === i && (
              <text x={Math.min(x(hi) + 10, width - 150)} y={y - 8} fontSize="12" fill="var(--ink)" fontWeight="600">
                {f3(r.mean)} ± {f3(r.std)}
              </text>
            )}
          </g>
        )
      })}
      <text x={(left + width - right) / 2} y={height - 4} textAnchor="middle" fontSize="12" fill="var(--ink2)">{xLabel}</text>
      <text x={x(reference)} y={height - 24 + 14} textAnchor="middle" fontSize="11" fill="var(--ink2)" />
    </svg>
  )
}
