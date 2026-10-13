import { Fragment, useEffect, useState } from 'react'
import { CartesianGrid, Line, LineChart, ReferenceLine, Tooltip, XAxis, YAxis } from 'recharts'
import { request, useApi, useAuth } from '../api'
import FlowAnimation from '../components/FlowAnimation'
import { Async, Badge, ChartCard, CodeBlock, DataTable, EmptyState, Icon, PageHeader, RiskBadge, StatCard, timeAgo } from '../components/ui'
import { f3 } from '../format'

const HF_FALLBACK = 'Adhi-gg029/lung-cancer-federated-learning'

function usePolling(q, ms) {
  useEffect(() => {
    const id = setInterval(() => { if (!document.hidden) q.retry() }, ms)
    return () => clearInterval(id)
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [ms])
}

function RoundPanel({ s, network, onChange }) {
  const { user } = useAuth()
  const [dl, setDl] = useState({ busy: false, err: null })
  const [admin, setAdmin] = useState({ busy: false, msg: null, err: false })
  const repo = s.hf_repo || HF_FALLBACK
  const need = s.min_clients
  const others = (network || []).map((h) => ({ name: h.username, pending: s.pending_hospitals.includes(h.username) }))
  const phase = s.n_pending > 0 ? 'waiting' : 'idle'
  const pub = s.last_publish?.status
  const hfStatus = !s.hf_repo ? 'not configured' : pub === 'ok' ? 'ok' : pub === 'error' ? 'error' : 'pending'

  const download = async () => {
    setDl({ busy: true, err: null })
    try {
      const res = await request('/fl/model', { raw: true })
      const blob = await res.blob()
      const a = document.createElement('a')
      a.href = URL.createObjectURL(blob)
      a.download = `global_v${res.headers.get('X-Model-Version')}.npz`
      a.click()
      URL.revokeObjectURL(a.href)
      setDl({ busy: false, err: null })
    } catch (e) { setDl({ busy: false, err: e.message }) }
  }
  const aggregate = async () => {
    setAdmin({ busy: true, msg: null, err: false })
    try {
      const r = await request('/admin/aggregate', { method: 'POST' })
      setAdmin({ busy: false, msg: `Aggregated: global model is now v${r.new_version}`, err: false })
      onChange()
    } catch (e) { setAdmin({ busy: false, msg: e.message, err: true }) }
  }

  return (
    <div className="card hero">
      <FlowAnimation phase={phase} version={s.version} pending={s.n_pending} needed={need} hfStatus={hfStatus} others={others} youLabel={user?.username || 'Hospital'} />
      <hr style={{ border: 0, borderTop: '1px solid #efd88e', margin: '14px 0 18px' }} />
      <div className="grid two" style={{ margin: 0, alignItems: 'start' }}>
        <div>
          <div className="eyebrow">Round building model v{s.version + 1}</div>
          <div style={{ display: 'flex', gap: 14, alignItems: 'baseline' }}>
            <span style={{ font: '750 40px var(--sans)', letterSpacing: '-1px', color: 'var(--brown)' }}>{s.n_pending}<span className="muted" style={{ fontSize: 22 }}> / {need}</span></span>
            <span className="muted">hospitals have reported</span>
          </div>
          <div className="meter" style={{ margin: '10px 0' }}><i style={{ width: `${Math.min(100, (s.n_pending / need) * 100)}%`, background: 'linear-gradient(90deg, var(--gold2), var(--gold-d))' }} /></div>
          <div className="small" style={{ minHeight: 22 }}>{s.pending_hospitals.length
            ? <>Waiting on: {s.pending_hospitals.map((h) => <Badge key={h} tone="info">{h}</Badge>)}</> : 'No updates yet. The first hospital to train starts the round.'}</div>
          <div className="row" style={{ marginTop: 16 }}>
            <button className="btn" onClick={download} disabled={dl.busy}><Icon name="download" size={16} />{dl.busy ? 'Downloading…' : `Download model v${s.version}`}</button>
            <a className="btn secondary" href={`https://huggingface.co/${repo}`} target="_blank" rel="noreferrer"><Icon name="cloud" size={16} />Open on Hugging Face</a>
            {user?.role === 'admin' && <button className="btn secondary" onClick={aggregate} disabled={admin.busy || s.n_pending === 0}>Aggregate pending now</button>}
          </div>
          {dl.err && <div className="err-msg" role="alert">{dl.err}</div>}
          {admin.msg && <div className={admin.err ? 'err-msg' : 'small'} role="status">{admin.msg}</div>}
        </div>
        <dl className="kv" style={{ margin: 0 }}>
          <dt>Aggregation</dt><dd>{s.aggregator === 'median' ? 'Coordinate-wise median (≥ 3 hospitals)' : 'FedAvg (weighted by rows)'}</dd>
          <dt>Update size limit</dt><dd>{s.max_update_norm} (L2 distance)</dd>
          <dt>Round risk so far</dt><dd><RiskBadge risk={s.round.risk} /></dd>
          <dt>Rejected this round</dt><dd>{Object.keys(s.round.rejected).length ? Object.entries(s.round.rejected).map(([k, v]) => `${v} ${k}`).join(', ') : 'none'}</dd>
          <dt>Hugging Face</dt><dd>{!s.hf_repo ? <Badge>not configured</Badge> : <Badge tone={pub === 'ok' ? 'good' : pub === 'error' ? 'crit' : ''} dot>{pub === 'ok' ? `v${s.last_publish.version} published` : pub === 'error' ? 'last publish failed' : 'waiting for first publish'}</Badge>}</dd>
        </dl>
      </div>
    </div>
  )
}

function Network({ rows, me, isAdmin, onChange }) {
  const [busy, setBusy] = useState(null)
  const [err, setErr] = useState(null)
  const act = async (h, action) => {
    if (action === 'remove' && !window.confirm(`Remove ${h.name}? Its account and any pending update are deleted.`)) return
    setBusy(h.username); setErr(null)
    try {
      await request(action === 'remove' ? `/admin/hospitals/${h.username}` : `/admin/hospitals/${h.username}/${action}`, { method: action === 'remove' ? 'DELETE' : 'POST' })
      onChange()
    } catch (e) { setErr(e.message) } finally { setBusy(null) }
  }
  if (!rows.length) return <EmptyState>No hospital has registered yet. Be the first: use “Register hospital”.</EmptyState>
  return (
    <>
      <DataTable rows={rows} columns={[
        { key: 'name', label: 'Hospital', render: (h) => <><b>{h.name}</b>{h.username === me && <> <Badge tone="info">you</Badge></>}
          <div className="small muted">{h.parent ? `sub-hospital of ${h.parent}` : 'independent'} · @{h.username}</div></> },
        { key: 'joined', label: 'Joined', render: (h) => timeAgo(h.joined) },
        { key: 'status', label: 'Status', render: (h) => h.status === 'suspended' ? <Badge tone="crit" dot>suspended</Badge> : h.pending ? <Badge tone="info" dot>reported this round</Badge> : <Badge tone="good" dot>active</Badge> },
        { key: 'accepted', label: 'Updates accepted', num: true },
        { key: 'rejected', label: 'Rejected', num: true, render: (h) => (h.rejected ? <Badge tone="crit">{h.rejected}</Badge> : 0) },
        { key: 'dp', label: 'Privacy (DP)', render: (h) => (h.dp_updates ? <Badge tone="good">{h.dp_updates} update(s), ε ≈ {h.epsilon}</Badge> : <span className="muted small">not used</span>) },
        { key: 'last', label: 'Last model', render: (h) => (h.last_version != null ? `v${h.last_version}` : '-') },
        ...(isAdmin ? [{ key: 'admin', label: 'Admin', render: (h) => (
          <span style={{ display: 'inline-flex', gap: 6 }}>
            {h.status === 'suspended'
              ? <button className="btn sm secondary" disabled={busy === h.username} onClick={() => act(h, 'restore')}>Restore</button>
              : <button className="btn sm secondary" disabled={busy === h.username} onClick={() => act(h, 'suspend')}>Suspend</button>}
            <button className="btn sm danger" disabled={busy === h.username} onClick={() => act(h, 'remove')}>Remove</button>
          </span>) }] : []),
      ]} />
      {err && <div className="err-msg" role="alert">{err}</div>}
    </>
  )
}

function History({ history }) {
  const [open, setOpen] = useState(null)
  const rows = [...history].reverse()
  return (
    <>
      <div className="scroll card" style={{ padding: 0 }}>
        <table>
          <thead><tr><th>Version</th><th>Built</th><th>Hospitals</th><th className="num">Training rows</th><th className="num">Test AUC</th><th className="num">Test accuracy</th><th>Round risk</th></tr></thead>
          <tbody>
            {rows.map((r) => (
              <Fragment key={r.version}>
                <tr className={r.security ? 'clickable' : ''} onClick={() => r.security && setOpen(open === r.version ? null : r.version)}>
                  <td><b>v{r.version}</b></td><td>{r.version === 0 ? 'initial primary model' : timeAgo(r.time)}</td>
                  <td>{r.participants.length ? r.participants.map((p) => <Badge key={p}>{p}</Badge>) : <span className="muted small">3 simulated hospitals</span>}</td>
                  <td className="num">{r.total_examples || '-'}</td><td className="num">{f3(r.test_metrics?.auc)}</td><td className="num">{f3(r.test_metrics?.accuracy)}</td>
                  <td>{r.security ? <RiskBadge risk={r.security.risk} /> : <span className="muted small">n/a</span>}</td>
                </tr>
                {open === r.version && r.security && (
                  <tr className="detail"><td colSpan={7} className="small">
                    {r.note}. Received {r.security.received}, accepted {r.security.accepted}, rejected {r.security.rejected}. Differential privacy in {r.security.dp_enabled} of {r.security.accepted} updates.
                    Raw patient rows transmitted: {r.security.egress.raw_rows_transmitted}.
                  </td></tr>
                )}
              </Fragment>
            ))}
          </tbody>
        </table>
      </div>
      {history.length > 1 && (
        <ChartCard title="Global model test AUC by version (0.5 = chance)" height={220}>
          <LineChart data={history.map((r) => ({ v: `v${r.version}`, auc: r.test_metrics?.auc }))}>
            <CartesianGrid stroke="var(--grid)" /><XAxis dataKey="v" /><YAxis domain={[0.4, 0.7]} tickFormatter={f3} /><Tooltip formatter={(v) => f3(v)} />
            <ReferenceLine y={0.5} stroke="var(--ink2)" strokeDasharray="4 4" />
            <Line dataKey="auc" stroke="var(--s1)" strokeWidth={3} dot={{ r: 4 }} />
          </LineChart>
        </ChartCard>
      )}
      <p className="small muted">Metrics use the shared test file bundled with the server. On this dataset every model sits near chance (AUC ≈ 0.5), so versions will not visibly improve.</p>
    </>
  )
}

export default function Federation() {
  const { user } = useAuth()
  const st = useApi('/fl/status')
  const hist = useApi('/fl/history')
  const net = useApi('/fl/network')
  usePolling(st, 6000)
  const refresh = () => { st.retry(); hist.retry(); net.retry() }
  const origin = window.location.origin
  const mine = net.data?.hospitals.find((h) => h.username === user?.username)

  return (
    <>
      <PageHeader eyebrow="Live network" title="Live Federation">
        The server holds the primary global model. Hospitals download it, train on their own data, and send back only weights. When enough hospitals
        have reported, the server averages them into the next version and publishes it.
      </PageHeader>

      <Async q={st} what="federation status">{(s) => <RoundPanel s={s} network={net.data?.hospitals} onChange={refresh} />}</Async>

      {mine && (
        <div className="grid" style={{ marginTop: 16 }}>
          <StatCard icon="check" tone="good" label="Your accepted updates" value={mine.accepted} />
          <StatCard icon="alert" tone={mine.rejected ? 'crit' : ''} label="Your rejected updates" value={mine.rejected} />
          <StatCard icon="shield" label="Your privacy budget" value={mine.epsilon != null ? `ε ≈ ${mine.epsilon}` : 'no DP yet'} hint={`${mine.dp_updates} DP update(s)`} />
          <StatCard icon="layers" label="Your last model" value={mine.last_version != null ? `v${mine.last_version}` : 'none'} />
        </div>
      )}

      <h3>Hospitals in the network</h3>
      <Async q={net} what="network">{(n) => <Network rows={n.hospitals} me={user?.username} isAdmin={user?.role === 'admin'} onChange={refresh} />}</Async>

      <h3>Global model versions</h3>
      <Async q={hist} what="round history">{(h) => <History history={h.history} />}</Async>

      <h3>How a hospital takes part</h3>
      <div className="steps">
        <div className="step"><b>Register</b><span>Create your hospital (or sub-hospital) account on the Sign in page.</span></div>
        <div className="step"><b>Train</b><span>Open “Train and Contribute”, drop in a CSV batch and train in your browser.</span></div>
        <div className="step"><b>Update</b><span>Only weights are sent. The server builds the next version and updates Hugging Face.</span></div>
      </div>
      <div className="card" style={{ marginTop: 16 }}>
        <h4>Prefer the command line?</h4>
        <CodeBlock>{`set FL_SERVER=${origin}
set FL_USERNAME=<your-username>
set FL_PASSWORD=<your-password>
python -m client.cli register --hospital-name "My Hospital"      (once)
python -m client.cli run --data my_hospital.csv --epochs 2 --sigma 1.0`}</CodeBlock>
        <div className="small muted">The same protocol: download, train locally, upload weights. <span className="mono">--sigma</span> turns on differential privacy and <span className="mono">--source hf</span> downloads from Hugging Face.</div>
      </div>
    </>
  )
}
