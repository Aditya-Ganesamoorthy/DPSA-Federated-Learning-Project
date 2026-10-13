import { useState } from 'react'
import { CartesianGrid, Line, LineChart, ReferenceLine, Tooltip, XAxis, YAxis } from 'recharts'
import { request, useApi, useAuth } from '../api'
import { Async, ChartCard, DataTable, PageHeader, StatCard } from '../components/ui'
import { f3 } from '../format'

const HF_REPO = 'Adhi-gg029/lung-cancer-federated-learning'

function Admin({ onChange }) {
  const [name, setName] = useState('')
  const [created, setCreated] = useState(null)
  const [msg, setMsg] = useState(null)
  const [busy, setBusy] = useState(false)
  const run = async (fn) => {
    setBusy(true); setMsg(null)
    try { await fn() } catch (e) { setMsg({ err: true, text: e.message }) } finally { setBusy(false) }
  }
  return (
    <div className="card" style={{ marginTop: 12 }}>
      <h4>Administrator tools</h4>
      <form className="row" onSubmit={(e) => { e.preventDefault(); run(async () => {
        setCreated(await request('/admin/hospitals', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ name }) }))
      }) }}>
        <label className="f" style={{ minWidth: 260 }}>New hospital name
          <input value={name} onChange={(e) => setName(e.target.value)} placeholder="e.g. City General Hospital"
                 style={{ padding: 8, border: '1px solid var(--line)', borderRadius: 6, background: 'var(--surface)', color: 'var(--ink)' }} />
        </label>
        <button className="btn" disabled={busy || name.trim().length < 3}>Create credentials</button>
        <button type="button" className="btn secondary" disabled={busy} onClick={() => run(async () => {
          const r = await request('/admin/aggregate', { method: 'POST' })
          setMsg({ text: `Aggregated: new global model v${r.new_version}` }); onChange()
        })}>Aggregate pending updates now</button>
      </form>
      {created && (
        <div className="note ok" role="status">
          Credentials for <b>{created.hospital_id}</b>: username <span className="mono">{created.username}</span>, password{' '}
          <span className="mono">{created.password}</span>. Share them securely; the password is derived from the server key and can be re-issued.
        </div>
      )}
      {msg && <p role={msg.err ? 'alert' : 'status'} style={{ color: msg.err ? 'var(--crit)' : 'inherit' }}>{msg.text}</p>}
    </div>
  )
}

export default function Federation() {
  const { user } = useAuth()
  const st = useApi('/fl/status')
  const hist = useApi('/fl/history')
  const [dl, setDl] = useState({ busy: false, err: null })
  const origin = window.location.origin

  const download = async () => {
    setDl({ busy: true, err: null })
    try {
      const res = await request('/fl/model', { raw: true })
      const blob = await res.blob()
      const v = res.headers.get('X-Model-Version')
      const a = document.createElement('a')
      a.href = URL.createObjectURL(blob)
      a.download = `global_v${v}.npz`
      a.click()
      URL.revokeObjectURL(a.href)
      setDl({ busy: false, err: null })
    } catch (e) { setDl({ busy: false, err: e.message }) }
  }

  return (
    <>
      <PageHeader title="Federated learning: live server">
        This server holds the primary global model. Hospitals download it, train on their own data (which never leaves the hospital),
        and upload only the new weights. When enough hospitals have reported, the server averages them (FedAvg) into the next version.
      </PageHeader>
      <Async q={st} what="federation status">
        {(s) => (
          <>
            <div className="grid">
              <StatCard label="Global model version" value={`v${s.version}`} />
              <StatCard label="Hospitals needed per round" value={s.min_clients} hint={`aggregation: ${s.aggregator}`} />
              <StatCard label="Updates waiting this round" value={s.n_pending} hint={s.pending_hospitals.join(', ') || 'none yet'} />
              <StatCard label="Model parameters" value={s.n_params} />
              <StatCard label="Hugging Face sync" value={s.last_publish?.status ?? 'n/a'} hint={s.hf_repo || 'not configured'} />
            </div>
            <div className="row">
              <button className="btn" onClick={download} disabled={dl.busy}>{dl.busy ? 'Downloading…' : `Download global model v${s.version}`}</button>
              <a href={`https://huggingface.co/${s.hf_repo || HF_REPO}`} target="_blank" rel="noreferrer">Model on Hugging Face</a>
            </div>
            {dl.err && <p role="alert" style={{ color: 'var(--crit)' }}>{dl.err}</p>}
          </>
        )}
      </Async>

      <h3>Round history</h3>
      <Async q={hist} what="round history">
        {(h) => (
          <>
            <DataTable rows={[...h.history].reverse()} columns={[
              { key: 'version', label: 'Version', num: true },
              { key: 'time', label: 'Time (UTC)' },
              { key: 'participants', label: 'Hospitals', render: (r) => r.participants.join(', ') || '—' },
              { key: 'total_examples', label: 'Training rows', num: true },
              { key: 'auc', label: 'Test AUC', num: true, render: (r) => f3(r.test_metrics?.auc) },
              { key: 'accuracy', label: 'Test accuracy', num: true, render: (r) => f3(r.test_metrics?.accuracy) },
            ]} />
            {h.history.length > 1 && (
              <ChartCard title="Global model test AUC by version (0.5 = chance)" height={240}>
                <LineChart data={h.history.map((r) => ({ v: r.version, auc: r.test_metrics?.auc }))}>
                  <CartesianGrid stroke="var(--grid)" />
                  <XAxis dataKey="v" label={{ value: 'Version', position: 'insideBottom', offset: -2 }} />
                  <YAxis domain={[0.4, 0.7]} tickFormatter={f3} /><Tooltip formatter={(v) => f3(v)} />
                  <ReferenceLine y={0.5} stroke="var(--ink2)" strokeDasharray="4 4" />
                  <Line dataKey="auc" stroke="var(--s1)" strokeWidth={2} dot={{ r: 4 }} />
                </LineChart>
              </ChartCard>
            )}
            <p className="lede">Metrics come from the shared test file bundled with the server. On this dataset all models sit near chance (AUC about 0.5), so versions will not visibly improve.</p>
          </>
        )}
      </Async>

      {user?.role === 'admin' && <Admin onChange={() => { st.retry(); hist.retry() }} />}

      <h3>How a hospital takes part</h3>
      <div className="card">
        <ol>
          <li>Get credentials from the administrator and install the client (<span className="mono">pip install -r client/requirements.txt</span>).</li>
          <li>Run this on the hospital's machine (your CSV never leaves it):
            <pre className="mono" style={{ whiteSpace: 'pre-wrap' }}>{`set FL_SERVER=${origin}
set FL_USERNAME=<your-hospital-id>
set FL_PASSWORD=<your-password>
python -m client.cli run --data my_hospital.csv --epochs 2`}</pre>
          </li>
          <li>The client downloads the current global model, trains locally, and uploads the weights and your row count. Add <span className="mono">--sigma 1.0</span> for local differential privacy, or <span className="mono">--source hf</span> to download from Hugging Face.</li>
          <li>When {st.data?.min_clients ?? 'enough'} hospitals have reported, the server publishes the next global model version.</li>
        </ol>
      </div>
    </>
  )
}
