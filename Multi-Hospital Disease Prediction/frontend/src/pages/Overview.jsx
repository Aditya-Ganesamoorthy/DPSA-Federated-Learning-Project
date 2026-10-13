import { Fragment, useEffect, useState } from 'react'
import { Bar, BarChart, CartesianGrid, Cell, Legend, Pie, PieChart, ReferenceLine, Tooltip, XAxis, YAxis } from 'recharts'
import { useApi } from '../api'
import { Async, Badge, ChartCard, EmptyState, Icon, Meter, PageHeader, RISK_TONE, RiskBadge, StatCard, timeAgo } from '../components/ui'
import { f3, pct } from '../format'

const SEV_TONE = { high: 'crit', medium: 'warn', low: 'info', info: '' }
const pretty = (s) => s.replace(/_/g, ' ')

function RiskRing({ risk, size = 112 }) {
  const r = 46, c = 2 * Math.PI * r
  const col = { low: 'var(--good)', medium: 'var(--gold)', high: 'var(--crit)' }[risk.level]
  return (
    <div className="ring-wrap" style={{ width: size, height: size }}>
      <svg viewBox="0 0 112 112" width={size} height={size} role="img" aria-label={`Risk score ${risk.score} (${risk.level})`}>
        <circle cx="56" cy="56" r={r} fill="none" stroke="var(--surface3)" strokeWidth="11" />
        <circle cx="56" cy="56" r={r} fill="none" stroke={col} strokeWidth="11" strokeLinecap="round" strokeDasharray={c}
                strokeDashoffset={c * (1 - risk.score / 100)} transform="rotate(-90 56 56)" style={{ transition: 'stroke-dashoffset 1s cubic-bezier(.2,.8,.2,1)' }} />
      </svg>
      <div className="center"><div>{risk.score}<small>{risk.level} risk</small></div></div>
    </div>
  )
}

// ------------------------------------------------------------------------------------------------ security dashboard
function RoundsTable({ rounds }) {
  const [open, setOpen] = useState(null)
  if (!rounds.length) return <EmptyState>No federated round has finished yet. Rounds appear here as soon as hospitals contribute updates.</EmptyState>
  const shown = [...rounds].reverse()
  return (
    <div className="scroll card" style={{ padding: 0 }}>
      <table>
        <thead><tr>
          <th>Round</th><th>When</th><th className="num">Hospitals</th><th className="num">Received</th><th className="num">Rejected</th>
          <th>Differential privacy</th><th>Raw data sent</th><th>Aggregation</th><th>Risk</th></tr></thead>
        <tbody>
          {shown.map((r) => (
            <Fragment key={r.version}>
              <tr className="clickable" onClick={() => setOpen(open === r.version ? null : r.version)} tabIndex={0}
                  onKeyDown={(e) => (e.key === 'Enter' || e.key === ' ') && setOpen(open === r.version ? null : r.version)} aria-expanded={open === r.version}>
                <td><b>v{r.version}</b></td><td>{timeAgo(r.time)}</td>
                <td className="num">{r.hospitals}</td><td className="num">{r.received}</td>
                <td className="num">{r.rejected ? <Badge tone="crit">{r.rejected}</Badge> : 0}</td>
                <td>{r.dp_enabled === 0 ? <Badge tone="warn">disabled</Badge> : r.dp_enabled === r.accepted ? <Badge tone="good">enabled</Badge> : <Badge tone="info">{r.dp_enabled}/{r.accepted} enabled</Badge>}</td>
                <td><Badge tone="good">{r.raw_rows_transmitted} rows</Badge></td>
                <td className="small">{r.aggregator}</td>
                <td><RiskBadge risk={r.risk} /></td>
              </tr>
              {open === r.version && (
                <tr className="detail"><td colSpan={9}>
                  <div className="grid two" style={{ margin: 0 }}>
                    <div><h4>Why this risk score</h4>
                      {r.risk.factors.length ? r.risk.factors.map((f) => <div key={f.label} className="small">+{f.points} · {f.label}</div>) : <div className="small">No risk factors.</div>}</div>
                    <div><h4>Round details</h4>
                      <div className="small">Update payload: {r.payload_kb} KB total (weights only).</div>
                      <div className="small">Rejected: {r.rejected ? Object.entries(r.rejected_by_reason).map(([k, v]) => `${v} × ${k}`).join(', ') : 'none'}.</div>
                      <div className="small">Test AUC of the new model: {f3(r.test_auc)} (0.5 = chance).</div></div>
                  </div>
                </td></tr>
              )}
            </Fragment>
          ))}
        </tbody>
      </table>
    </div>
  )
}

function SecurityDashboard({ d, onRefresh }) {
  const last = d.rounds[d.rounds.length - 1]
  const live = d.round_in_progress?.risk
  const risk = live || last?.risk
  const t = d.totals
  const priv = d.privacy
  const dpTone = { enabled: 'good', partial: 'info', disabled: 'warn', none: '' }[priv.status]
  return (
    <>
      <div className="grid">
        <StatCard icon="users" label="Registered hospitals" value={d.network.registered} hint={`${d.network.participated} have contributed`} />
        <StatCard icon="network" label="Reported this round" value={`${d.network.active_now}/${d.network.needed_per_round}`} hint="hospitals needed to close the round" />
        <StatCard icon="layers" label="Global model version" value={`v${d.model.version}`} hint={d.model.aggregator === 'median' ? 'median aggregation' : 'FedAvg aggregation'} />
        <StatCard icon="download" label="Updates received" value={t.received} hint="all rounds" />
        <StatCard icon="check" tone="good" label="Updates accepted" value={t.accepted} />
        <StatCard icon="alert" tone={t.rejected ? 'crit' : ''} label="Updates rejected" value={t.rejected}
                  hint={t.rejected ? Object.entries(t.rejected_by_reason).map(([k, v]) => `${v} ${k}`).join(', ') : 'none so far'} />
      </div>

      <div className="grid three">
        <div className="card soft">
          <h4>Does raw patient data leave a hospital?</h4>
          <div style={{ display: 'flex', alignItems: 'center', gap: 14 }}>
            <span className="badge good" style={{ fontSize: 22, padding: '8px 20px', fontFamily: 'var(--serif)' }}>NO</span>
            <div><b>{d.data_egress.raw_rows_transmitted} patient rows</b> transmitted<div className="small muted">in every round recorded</div></div>
          </div>
          <dl className="kv" style={{ marginTop: 14 }}>
            <dt>Leaves</dt><dd style={{ fontWeight: 500 }}>{d.data_egress.leaves_hospital}</dd>
            <dt>Stays local</dt><dd style={{ fontWeight: 500 }}>{d.data_egress.stays_local}</dd>
            <dt>Avg update</dt><dd>{d.data_egress.avg_update_kb != null ? `${d.data_egress.avg_update_kb} KB` : 'n/a yet'}</dd>
          </dl>
          <div className="small muted" style={{ marginTop: 10 }}>{d.data_egress.guarantees.map((g) => <div key={g}>• {g}</div>)}</div>
        </div>

        <div className="card soft">
          <h4>Privacy protection status</h4>
          <div style={{ display: 'flex', alignItems: 'center', gap: 12 }}>
            <Badge tone={dpTone} dot><span style={{ fontSize: 15 }}>Differential privacy {priv.status === 'none' ? 'not used yet' : priv.status}</span></Badge>
          </div>
          <p className="small" style={{ margin: '10px 0' }}>{priv.text}</p>
          <dl className="kv">
            <dt>Rounds with DP</dt><dd>{priv.dp_rounds} of {priv.rounds}</dd>
            <dt>Largest ε (δ = 1e-5)</dt><dd>{priv.max_epsilon != null ? priv.max_epsilon : 'n/a'}</dd>
          </dl>
          <div className="small muted" style={{ marginTop: 10 }}>DP is applied on each hospital's own machine. The server recomputes ε from the declared settings across rounds but cannot verify the noise was added.</div>
        </div>

        <div className="card soft" style={{ display: 'flex', gap: 18, alignItems: 'center' }}>
          {risk ? <RiskRing risk={risk} /> : <div className="empty" style={{ width: 112, padding: 10 }}>No round yet</div>}
          <div>
            <h4>{live ? 'Round in progress' : 'Latest round'} risk</h4>
            {risk ? (risk.factors.length ? risk.factors.slice(0, 4).map((f) => <div key={f.label} className="small">+{f.points} {f.label}</div>) : <div className="small">No risk factors.</div>)
              : <div className="small muted">Scores appear after the first update arrives.</div>}
          </div>
        </div>
      </div>

      <h3>Security of each training round</h3>
      <RoundsTable rounds={d.rounds} />
      {d.rounds.length > 0 && (
        <ChartCard title="Risk score by round (lower is safer)" height={210}>
          <BarChart data={d.rounds.map((r) => ({ round: `v${r.version}`, score: r.risk.score, level: r.risk.level }))}>
            <CartesianGrid stroke="var(--grid)" vertical={false} />
            <XAxis dataKey="round" /><YAxis domain={[0, 100]} />
            <Tooltip formatter={(v, n, p) => [`${v} (${p.payload.level})`, 'risk score']} />
            <ReferenceLine y={30} stroke="var(--good)" strokeDasharray="4 4" /><ReferenceLine y={60} stroke="var(--crit)" strokeDasharray="4 4" />
            <Bar dataKey="score" radius={[6, 6, 0, 0]}>
              {d.rounds.map((r) => <Cell key={r.version} fill={{ low: 'var(--good)', medium: 'var(--gold)', high: 'var(--crit)' }[r.risk.level]} />)}
            </Bar>
          </BarChart>
        </ChartCard>
      )}

      <div className="grid two">
        <div className="card">
          <h4>Security controls</h4>
          <div style={{ display: 'grid', gap: 10 }}>
            {d.controls.map((c) => (
              <div key={c.name} style={{ display: 'flex', gap: 10, alignItems: 'flex-start' }}>
                <Badge tone={c.status === 'on' ? 'good' : c.status === 'off' ? 'warn' : 'info'} dot>{c.status}</Badge>
                <div><b className="small">{c.name}</b><div className="small muted">{c.detail}</div></div>
              </div>
            ))}
          </div>
        </div>
        <div className="card">
          <div className="row" style={{ justifyContent: 'space-between', margin: 0 }}><h4 style={{ margin: 0 }}>Detected attacks</h4>
            <button className="btn ghost sm" onClick={onRefresh}>Refresh</button></div>
          {d.attacks.length ? d.attacks.map((a) => (
            <div key={a.label} style={{ display: 'flex', gap: 10, alignItems: 'center', margin: '10px 0' }}>
              <Badge tone={SEV_TONE[a.severity]} dot>{a.severity}</Badge><b className="small">{a.label}</b>
              <span className="small muted" style={{ marginLeft: 'auto' }}>{a.count}× · last {timeAgo(a.last_time)}</span>
            </div>
          )) : <div className="note ok" style={{ margin: '12px 0' }}>No attacks detected since the server started.</div>}
          <h4 style={{ marginTop: 18 }}>Security events</h4>
          <div className="events timeline">
            {d.events.length ? d.events.map((e) => (
              <div key={e.id} className={`tl-item ${e.severity}`}>
                <b>{pretty(e.type)}</b> {e.actor && e.actor !== 'unknown' && <span className="muted">· {e.actor}</span>}
                <div className="small muted">{e.detail} · {timeAgo(e.time)}</div>
              </div>
            )) : <div className="small muted">No events yet.</div>}
          </div>
          <div className="small muted">Events are kept since the server last started; account names are masked.</div>
        </div>
      </div>
    </>
  )
}

// ------------------------------------------------------------------------------------------------ demo
function Demo() {
  const [seed, setSeed] = useState(1)
  const d = useApi(`/security/demo?seed=${seed}`)
  const [grow, setGrow] = useState(false)
  useEffect(() => { setGrow(false); const t = setTimeout(() => setGrow(true), 80); return () => clearTimeout(t) }, [d.data])
  return (
    <Async q={d} what="demo">
      {(x) => {
        const max = Math.max(...x.attacks.flatMap((a) => [a.ordinary.model_shift, a.protected.model_shift]), 1e-9)
        return (
          <>
            <div className="row">
              <button className="btn" onClick={() => setSeed((s) => s + 1)}><Icon name="bolt" size={16} />Run a new session</button>
              <span className="small muted">Same hospitals and same attacker in both sessions (seed {seed}).</span>
            </div>
            <div className="compare" role="table" aria-label="Ordinary versus protected federated learning">
              <div className="h">Security control</div><div className="h ord">Ordinary federated learning</div><div className="h pro">Protected session</div>
              {x.table.map((r) => (
                <Fragment key={r.control}>
                  <div>{r.control}</div><div className="ord-c">{r.ordinary}</div><div className={`pro-c ${r.better ? 'win' : ''}`}>{r.protected}</div>
                </Fragment>
              ))}
            </div>
            <div className="grid two">
              <div className="card bars">
                <h4>How far each attacker moved the global model</h4>
                {x.attacks.map((a) => (
                  <div key={a.key} style={{ marginBottom: 14 }}>
                    <div className="small"><b>{a.name}</b>, update size {a.update_norm}</div>
                    {[['Ordinary', a.ordinary, 'ord'], ['Protected', a.protected, 'pro']].map(([label, v, cls]) => (
                      <div className="bar-row" key={label}>
                        <span>{label}</span>
                        <div className="bar-track"><div className={`bar-fill ${cls}`} style={{ width: grow ? `${Math.max(2, (v.model_shift / max) * 100)}%` : '0%' }} /></div>
                        <span className="mono">{v.model_shift.toFixed(2)}</span>
                      </div>
                    ))}
                    <div className="small muted">{a.protected.attacker_accepted ? `Passed the size guard, neutralised by ${a.protected.stage}.` : `Blocked by the ${a.protected.stage}.`}</div>
                  </div>
                ))}
              </div>
              <div className="card" style={{ display: 'flex', justifyContent: 'space-around', alignItems: 'center', gap: 16, flexWrap: 'wrap' }}>
                {[['Ordinary', x.risk.ordinary], ['Protected', x.risk.protected]].map(([label, r]) => (
                  <div key={label} style={{ textAlign: 'center' }}><RiskRing risk={r} /><div style={{ marginTop: 8 }}><b>{label}</b></div>
                    <Badge tone={RISK_TONE[r.level]}>{r.level} risk</Badge></div>
                ))}
              </div>
            </div>
            <p className="small muted">{x.method}</p>
          </>
        )
      }}
    </Async>
  )
}

// ------------------------------------------------------------------------------------------------ EDA
function Heatmap({ m }) {
  const n = m.names.length
  const color = (v) => {
    const a = Math.min(1, Math.abs(v))
    return v >= 0 ? `rgba(168,105,15,${0.12 + a * 0.88})` : `rgba(47,93,138,${0.12 + a * 0.88})`
  }
  return (
    <div className="card">
      <h4>Feature correlation matrix</h4>
      <div className="heat" style={{ gridTemplateColumns: `repeat(${n}, 1fr)` }} role="img" aria-label="Correlation heatmap of all features and the cancer label">
        {m.values.flatMap((row, i) => row.map((v, j) => (
          <div key={`${i}-${j}`} title={`${pretty(m.names[i])} vs ${pretty(m.names[j])}: ${v.toFixed(3)}`} style={{ background: i === j ? '#e9dcb9' : color(v) }} />)))}
      </div>
      <div className="small muted" style={{ marginTop: 8 }}>Brown = positive, blue = negative. Hover a cell. Off-diagonal cells are almost white: features are uncorrelated with each other and with the label (last row/column).</div>
    </div>
  )
}

function Eda() {
  const eda = useApi('/results/eda')
  const cl = useApi('/clients')
  return (
    <Async q={eda} what="dataset analysis">
      {(e) => {
        const feats = e.features.map((f) => ({ name: pretty(f.name), cancer: +(f.prevalence_positive * 100).toFixed(1), none: +(f.prevalence_negative * 100).toFixed(1),
          corr: +f.correlation_with_cancer.toFixed(4) }))
        const age = e.age.histogram.bins.map((b, i) => ({ bin: b, cancer: e.age.histogram.positive[i], none: e.age.histogram.negative[i] }))
        const pie = [{ name: 'Lung cancer', value: e.positives }, { name: 'No lung cancer', value: e.negatives }]
        return (
          <>
            <div className="grid">
              <StatCard icon="database" label="Patient records" value={e.rows.toLocaleString()} hint="after removing duplicates" />
              <StatCard icon="activity" label="Cancer prevalence" value={pct(e.prevalence)} hint={`${e.positives} of ${e.rows}`} />
              <StatCard icon="layers" label="Input features" value={e.n_features} hint="gender, age, 13 symptoms/habits" />
              <StatCard icon="gauge" tone="warn" label="Strongest feature link" value={`|r| = ${e.max_abs_correlation.toFixed(3)}`} hint="correlation with the label" />
            </div>
            <div className="note">
              <b>What the data says.</b> No single feature is meaningfully related to the diagnosis (the strongest correlation is {e.max_abs_correlation.toFixed(3)}), the age profile is almost identical for both groups
              (mean {e.age.mean_positive.toFixed(1)} vs {e.age.mean_negative.toFixed(1)} years) and the classes are balanced. A model therefore cannot do much better than chance on this dataset, which is exactly what the
              experiments show; the project demonstrates the privacy-preserving <i>process</i>, not a diagnostic tool.
            </div>
            <div className="grid two">
              <ChartCard title="Class balance" height={260}>
                <PieChart>
                  <Pie data={pie} dataKey="value" nameKey="name" innerRadius={62} outerRadius={96} paddingAngle={3} label={(p) => `${(p.percent * 100).toFixed(1)}%`}>
                    <Cell fill="var(--s1)" /><Cell fill="var(--s2)" />
                  </Pie>
                  <Tooltip /><Legend />
                </PieChart>
              </ChartCard>
              <ChartCard title="Age distribution by diagnosis (patients per age band)" height={260}>
                <BarChart data={age}><CartesianGrid stroke="var(--grid)" vertical={false} /><XAxis dataKey="bin" /><YAxis /><Tooltip /><Legend />
                  <Bar dataKey="cancer" name="Lung cancer" fill="var(--s1)" radius={[5, 5, 0, 0]} /><Bar dataKey="none" name="No lung cancer" fill="var(--s2)" radius={[5, 5, 0, 0]} /></BarChart>
              </ChartCard>
            </div>
            <div className="grid two">
              <ChartCard title="Symptom/habit rate by diagnosis (% of patients answering yes)" height={420}>
                <BarChart data={feats.filter((f) => f.name !== 'gender')} layout="vertical" margin={{ left: 70 }}>
                  <CartesianGrid stroke="var(--grid)" horizontal={false} /><XAxis type="number" domain={[0, 100]} unit="%" /><YAxis type="category" dataKey="name" width={150} tick={{ fontSize: 11 }} interval={0} />
                  <Tooltip formatter={(v) => `${v}%`} /><Legend />
                  <Bar dataKey="cancer" name="Lung cancer" fill="var(--s1)" radius={[0, 4, 4, 0]} /><Bar dataKey="none" name="No lung cancer" fill="var(--s2)" radius={[0, 4, 4, 0]} />
                </BarChart>
              </ChartCard>
              <ChartCard title="Correlation of each feature with lung cancer (0 = unrelated)" height={420}>
                <BarChart data={feats.filter((f) => f.name !== 'gender')} layout="vertical" margin={{ left: 70 }}>
                  <CartesianGrid stroke="var(--grid)" horizontal={false} /><XAxis type="number" domain={[-0.15, 0.15]} /><YAxis type="category" dataKey="name" width={150} tick={{ fontSize: 11 }} interval={0} />
                  <Tooltip formatter={(v) => f3(v)} /><ReferenceLine x={0} stroke="var(--ink2)" />
                  <Bar dataKey="corr" name="correlation" radius={[0, 4, 4, 0]}>{feats.filter((f) => f.name !== 'gender').map((f) => <Cell key={f.name} fill={f.corr >= 0 ? 'var(--s1)' : 'var(--s2)'} />)}</Bar>
                </BarChart>
              </ChartCard>
            </div>
            <div className="grid two">
              <Heatmap m={e.correlation_matrix} />
              <Async q={cl} what="hospitals">
                {(c) => (
                  <ChartCard title="The three simulated hospitals: size and cancer rate (non-IID)" height={280}>
                    <BarChart data={c.noniid.map((h, i) => ({ name: `Hospital ${i + 1}`, rows: h.rows, prevalence: +(h.prevalence * 100).toFixed(1) }))}>
                      <CartesianGrid stroke="var(--grid)" vertical={false} /><XAxis dataKey="name" />
                      <YAxis yAxisId="l" /><YAxis yAxisId="r" orientation="right" unit="%" domain={[0, 100]} /><Tooltip /><Legend />
                      <Bar yAxisId="l" dataKey="rows" name="Training rows" fill="var(--s1)" radius={[5, 5, 0, 0]} />
                      <Bar yAxisId="r" dataKey="prevalence" name="Cancer rate %" fill="var(--s3)" radius={[5, 5, 0, 0]} />
                    </BarChart>
                  </ChartCard>
                )}
              </Async>
            </div>
          </>
        )
      }}
    </Async>
  )
}

export default function Overview() {
  const q = useApi('/security/overview')
  return (
    <>
      <div className="card hero" style={{ marginBottom: 6 }}>
        <div className="eyebrow">Federated Health Network</div>
        <h2>Security &amp; Privacy Analysis Dashboard</h2>
        <p className="lede" style={{ marginBottom: 0 }}>
          Hospitals train a shared lung-cancer model without ever sharing patient records. This dashboard audits every training round:
          who took part, what was received and rejected, whether raw data left a hospital, how privacy was protected and which attacks were detected.
        </p>
      </div>
      <Async q={q} what="security dashboard">{(d) => <SecurityDashboard d={d} onRefresh={q.retry} />}</Async>

      <h3>Demo: ordinary federated learning vs a protected session</h3>
      <p className="lede" style={{ marginTop: 0 }}>The same four hospitals and the same attacker, trained two ways. See what each setup lets through.</p>
      <Demo />

      <h3>Exploratory data analysis</h3>
      <Eda />
    </>
  )
}
