import { ResponsiveContainer } from 'recharts'

export const COLORS = ['var(--s1)', 'var(--s2)', 'var(--s3)', 'var(--s4)']

const ICONS = {
  home: 'M3 11l9-8 9 8M5 10v10h5v-6h4v6h5V10',
  shield: 'M12 3l8 3v6c0 5-3.5 8-8 9-4.5-1-8-4-8-9V6l8-3zM9 12l2 2 4-4',
  users: 'M16 20v-1a4 4 0 00-4-4H7a4 4 0 00-4 4v1M9.5 11a3.5 3.5 0 100-7 3.5 3.5 0 000 7zM21 20v-1a4 4 0 00-3-3.9M16 4.1a3.5 3.5 0 010 6.8',
  network: 'M12 5a2 2 0 100-4 2 2 0 000 4zM5 21a2 2 0 100-4 2 2 0 000 4zM19 21a2 2 0 100-4 2 2 0 000 4zM12 5v6M12 11l-6 6M12 11l6 6',
  upload: 'M12 16V4M7 9l5-5 5 5M4 20h16',
  download: 'M12 4v12M7 11l5 5 5-5M4 20h16',
  database: 'M4 6c0-1.7 3.6-3 8-3s8 1.3 8 3-3.6 3-8 3-8-1.3-8-3zM4 6v6c0 1.7 3.6 3 8 3s8-1.3 8-3V6M4 12v6c0 1.7 3.6 3 8 3s8-1.3 8-3v-6',
  chart: 'M4 20V10M10 20V4M16 20v-8M22 20H2',
  activity: 'M3 12h4l3-8 4 16 3-8h4',
  lock: 'M6 11V8a6 6 0 1112 0v3M5 11h14v10H5z',
  key: 'M15 7a4 4 0 11-3.9 4.9L3 20v-3l2-2v-2h2v-2h2.1A4 4 0 0115 7z',
  cloud: 'M7 18a5 5 0 01-.5-10A6 6 0 0118 9a4.5 4.5 0 01-.5 9H7z',
  check: 'M5 13l4 4L19 7',
  alert: 'M12 3l10 18H2L12 3zM12 10v5M12 18h.01',
  eye: 'M2 12s4-7 10-7 10 7 10 7-4 7-10 7S2 12 2 12zM12 15a3 3 0 100-6 3 3 0 000 6z',
  flask: 'M9 3h6M10 3v6L4 19a2 2 0 001.8 3h12.4A2 2 0 0020 19l-6-10V3',
  gauge: 'M12 21a9 9 0 110-18 9 9 0 010 18zM12 12l4-4',
  layers: 'M12 3l9 5-9 5-9-5 9-5zM3 13l9 5 9-5M3 17.5l9 5 9-5',
  bolt: 'M13 2L4 14h7l-1 8 9-12h-7l1-8z',
  logout: 'M9 21H5a2 2 0 01-2-2V5a2 2 0 012-2h4M16 17l5-5-5-5M21 12H9',
  login: 'M15 3h4a2 2 0 012 2v14a2 2 0 01-2 2h-4M10 17l5-5-5-5M15 12H3',
  scale: 'M12 3v18M5 7h14M5 7l-3 7a3.5 3.5 0 006 0L5 7zM19 7l-3 7a3.5 3.5 0 006 0l-3-7z',
}

export function Icon({ name, size = 18, stroke = 1.8 }) {
  return (
    <svg width={size} height={size} viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth={stroke}
         strokeLinecap="round" strokeLinejoin="round" aria-hidden="true"><path d={ICONS[name] || ICONS.home} /></svg>
  )
}

export function Spinner() { return <span className="spinner" aria-hidden="true" /> }

export function Loading({ what = 'data' }) {
  return <div className="loading" role="status"><Spinner />Loading {what}…</div>
}

export function ErrorBox({ error, retry }) {
  return (
    <div className="err" role="alert">
      <b>Could not load data.</b> {error?.message}
      {error?.status === 503 && <div>The results are not generated yet.</div>}
      {retry && <div className="row"><button className="btn secondary sm" onClick={retry}>Retry</button></div>}
    </div>
  )
}

export function EmptyState({ children = 'Nothing to show yet.' }) {
  return <div className="empty">{children}</div>
}

/** Renders loading / error / empty states around a data-dependent child. */
export function Async({ q, what, children }) {
  if (q.loading && !q.data) return <Loading what={what} />
  if (q.error && !q.data) return <ErrorBox error={q.error} retry={q.retry} />
  if (!q.data) return <EmptyState />
  return children(q.data)
}

export function Badge({ tone = '', dot = false, children, title }) {
  return <span className={`badge ${tone} ${dot ? 'dot' : ''}`} title={title}>{children}</span>
}

export const RISK_TONE = { low: 'good', medium: 'warn', high: 'crit' }
export function RiskBadge({ risk }) {
  if (!risk) return <Badge>n/a</Badge>
  return (
    <Badge tone={RISK_TONE[risk.level]} dot title={risk.factors.map((f) => `+${f.points} ${f.label}`).join('\n')}>
      {risk.score} · {risk.level}
    </Badge>
  )
}

export function Meter({ value, max = 100, tone = 'good' }) {
  return <div className="meter" role="img" aria-label={`${value} of ${max}`}><i style={{ width: `${Math.min(100, (value / max) * 100)}%` }} className={tone} /></div>
}

export function StatCard({ label, value, hint, icon, tone = '' }) {
  return (
    <div className={`card stat ${tone}`}>
      {icon && <span className="ic"><Icon name={icon} size={18} /></span>}
      <div className="v">{value}</div>
      <div className="l">{label}</div>
      {hint && <div className="l small muted">{hint}</div>}
    </div>
  )
}

export function DataTable({ columns, rows, caption, onRowClick }) {
  if (!rows?.length) return <EmptyState />
  return (
    <div className="scroll">
      <table>
        {caption && <caption>{caption}</caption>}
        <thead>
          <tr>{columns.map((c) => <th key={c.key} className={c.num ? 'num' : ''}>{c.label}</th>)}</tr>
        </thead>
        <tbody>
          {rows.map((r, i) => (
            <tr key={i} className={onRowClick ? 'clickable' : ''} onClick={onRowClick ? () => onRowClick(r) : undefined}>
              {columns.map((c) => <td key={c.key} className={c.num ? 'num' : ''}>{c.render ? c.render(r) : r[c.key]}</td>)}
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  )
}

export function ChartCard({ title, children, height = 280, legend }) {
  return (
    <div className="card">
      <h4>{title}</h4>
      {legend && (
        <div className="legend">
          {legend.map((l, i) => <span key={l} style={{ '--c': COLORS[i] }}>{l}</span>)}
        </div>
      )}
      <div style={{ width: '100%', height }}>
        <ResponsiveContainer>{children}</ResponsiveContainer>
      </div>
    </div>
  )
}

/** Hover/focus explanation of a technical term. */
const GLOSSARY = {
  epsilon: 'ε (epsilon): the privacy loss bound. Smaller ε = stronger differential privacy. Here it is record-level, at δ = 1e-5.',
  delta: 'δ (delta): the small probability that the ε guarantee fails. Must be below 1 / (number of records).',
  fedavg: 'FedAvg: the server averages client model weights, each weighted by its number of training samples.',
  auc: 'ROC-AUC: probability a random positive is ranked above a random negative. 0.5 = chance, 1.0 = perfect.',
  recall: 'Recall (sensitivity): share of real cancer cases that the model catches.',
  noniid: 'Non-IID: hospitals have different data distributions (here: different sizes and cancer prevalence).',
}
export function Tip({ term, children }) {
  return <span className="tip" tabIndex={0} title={GLOSSARY[term]}>{children}</span>
}

export function PageHeader({ title, eyebrow, children }) {
  return (
    <div className="page-head">
      {eyebrow && <div className="eyebrow">{eyebrow}</div>}
      <h2>{title}</h2>
      {children && <p className="lede">{children}</p>}
    </div>
  )
}

export function CodeBlock({ children }) {
  return <pre className="code">{children}</pre>
}

export function timeAgo(iso) {
  if (!iso) return 'n/a'
  const s = Math.max(0, (Date.now() - new Date(iso).getTime()) / 1000)
  if (s < 60) return 'just now'
  if (s < 3600) return `${Math.floor(s / 60)} min ago`
  if (s < 86400) return `${Math.floor(s / 3600)} h ago`
  return `${Math.floor(s / 86400)} d ago`
}
