import { ResponsiveContainer } from 'recharts'

export const COLORS = ['var(--s1)', 'var(--s2)', 'var(--s3)', 'var(--s4)']

export function Loading({ what = 'data' }) {
  return <div className="loading" role="status">Loading {what}…</div>
}

export function ErrorBox({ error, retry }) {
  return (
    <div className="err" role="alert">
      <b>Could not load data.</b> {error?.message}
      {error?.status === 503 && <div>The results are not generated yet.</div>}
      {retry && <div className="row"><button className="btn secondary" onClick={retry}>Retry</button></div>}
    </div>
  )
}

export function EmptyState({ children = 'Nothing to show yet.' }) {
  return <div className="empty">{children}</div>
}

/** Renders loading / error / empty states around a data-dependent child. */
export function Async({ q, what, children }) {
  if (q.loading) return <Loading what={what} />
  if (q.error) return <ErrorBox error={q.error} retry={q.retry} />
  if (!q.data) return <EmptyState />
  return children(q.data)
}

export function StatCard({ label, value, hint }) {
  return (
    <div className="card stat">
      <div className="v">{value}</div>
      <div className="l">{label}</div>
      {hint && <div className="l">{hint}</div>}
    </div>
  )
}

export function DataTable({ columns, rows, caption }) {
  if (!rows?.length) return <EmptyState />
  return (
    <div className="scroll">
      <table>
        {caption && <caption className="l" style={{ textAlign: 'left', color: 'var(--ink2)' }}>{caption}</caption>}
        <thead>
          <tr>{columns.map((c) => <th key={c.key} className={c.num ? 'num' : ''}>{c.label}</th>)}</tr>
        </thead>
        <tbody>
          {rows.map((r, i) => (
            <tr key={i}>
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

export function PageHeader({ title, children }) {
  return (
    <>
      <h2>{title}</h2>
      {children && <p className="lede">{children}</p>}
    </>
  )
}
