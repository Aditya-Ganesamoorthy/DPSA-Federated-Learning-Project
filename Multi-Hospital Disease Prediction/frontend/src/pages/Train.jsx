import { useCallback, useEffect, useRef, useState } from 'react'
import { CartesianGrid, Line, LineChart, Tooltip, XAxis, YAxis } from 'recharts'
import { ApiError, request, useApi, useAuth } from '../api'
import FlowAnimation from '../components/FlowAnimation'
import { Async, Badge, ChartCard, Icon, PageHeader, StatCard } from '../components/ui'
import { f3 } from '../format'
import { evaluate, flatDistance, prepareDataset, split, toParams, toPlain, trainLocal } from '../lib/trainer'

const sleep = (ms) => new Promise((r) => setTimeout(r, ms))
const SIGMAS = [0.8, 1.0, 1.5, 2.0, 3.0]
const json = (body) => ({ method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(body) })

function Metrics({ label, m }) {
  if (!m) return <div><b>{label}</b><div className="small muted">not enough rows for a hold-out check</div></div>
  return (
    <div><b>{label}</b>
      <dl className="kv" style={{ marginTop: 6 }}><dt>AUC</dt><dd>{f3(m.auc)}</dd><dt>Recall</dt><dd>{f3(m.recall)}</dd><dt>Accuracy</dt><dd>{f3(m.accuracy)}</dd></dl>
    </div>
  )
}

function Trainer({ spec }) {
  const { user } = useAuth()
  const status = useApi('/fl/status')
  const [data, setData] = useState(null)
  const [fileInfo, setFileInfo] = useState(null)
  const [parseErr, setParseErr] = useState(null)
  const [over, setOver] = useState(false)
  const [epochs, setEpochs] = useState(spec.training.local_epochs)
  const [dpOn, setDpOn] = useState(false)
  const [sigma, setSigma] = useState(1.0)
  const [eps, setEps] = useState(null)
  const [run, setRun] = useState({ phase: 'idle', progress: 0, loss: null, curve: [], result: null, error: null, newVersion: null, hf: 'not configured', waiting: null })
  const cancelRef = useRef(false)
  const fileRef = useRef(null)
  const isAdmin = user?.role === 'admin'
  const busy = ['downloading', 'training', 'uploading', 'waiting', 'aggregating', 'publishing'].includes(run.phase)

  useEffect(() => () => { cancelRef.current = true }, [])

  const holdFraction = data && data.n >= 100 ? 0.2 : 0
  const nTrain = data ? data.n - Math.floor(data.n * holdFraction) : 0
  const steps = nTrain ? epochs * Math.ceil(nTrain / spec.training.batch_size) : 0
  const q = nTrain ? Math.min(1, spec.training.batch_size / nTrain) : 0

  // epsilon preview for the planned DP run (alone and cumulative with everything this hospital already released)
  useEffect(() => {
    setEps(null)
    if (!dpOn || !nTrain) return undefined
    let live = true
    request('/fl/privacy-preview', json({ sigma, sample_rate: q, steps })).then((r) => live && setEps(r)).catch(() => live && setEps(null))
    return () => { live = false }
  }, [dpOn, sigma, q, steps, nTrain])

  const loadFile = useCallback(async (file) => {
    setParseErr(null); setData(null); setFileInfo(null)
    setRun((r) => ({ ...r, phase: 'idle', result: null, error: null, curve: [] }))
    if (!file) return
    if (file.size > spec.limits.max_file_mb * 1024 * 1024) { setParseErr(`File is larger than ${spec.limits.max_file_mb} MB.`); return }
    try {
      const d = prepareDataset(await file.text(), spec)
      if (d.n < spec.limits.min_rows) throw new Error(`Only ${d.n} usable rows. At least ${spec.limits.min_rows} are needed to train.`)
      if (d.positives === 0 || d.positives === d.n) throw new Error('All rows have the same diagnosis. Both cancer and no-cancer cases are needed.')
      setData(d); setFileInfo({ name: file.name, size: file.size })
    } catch (e) { setParseErr(e.message) }
  }, [spec])

  const pollStatus = async (until, maxMs) => {
    const t0 = Date.now()
    let st = null
    while (!cancelRef.current && Date.now() - t0 < maxMs) {
      try { st = await request('/fl/status') } catch { /* keep trying */ }
      if (st && until(st)) return st
      await sleep(2500)
    }
    return st
  }

  const start = async () => {
    cancelRef.current = false
    const set = (p) => setRun((r) => ({ ...r, ...p }))
    set({ phase: 'downloading', progress: 0, loss: null, curve: [], result: null, error: null, newVersion: null, waiting: null })
    try {
      const model = await request('/fl/model-json')
      await sleep(700)
      const params = toParams(model.weights)
      const start0 = toParams(model.weights)
      const { train, hold } = split(data, holdFraction, 1)
      const canHold = hold.n >= 20 && hold.y.some((v) => v > 0.5) && hold.y.some((v) => v < 0.5)
      const before = canHold ? evaluate(params, hold.X, hold.y, hold.n, hold.nIn) : null
      set({ phase: 'training' })
      const curve = []
      const tr = await trainLocal({
        params, X: train.X, y: train.y, n: train.n, nIn: train.nIn, epochs, lr: spec.training.lr, momentum: spec.training.momentum,
        batchSize: spec.training.batch_size, dp: dpOn ? { sigma, clip: spec.training.clip_norm } : null,
        shouldCancel: () => cancelRef.current,
        onProgress: (p) => {
          if (p.epochDone) curve.push({ epoch: p.epoch, loss: +p.loss.toFixed(4) })
          set({ progress: p.step / p.totalSteps, loss: p.loss, curve: [...curve] })
        },
      })
      const after = canHold ? evaluate(params, hold.X, hold.y, hold.n, hold.nIn) : null
      const dist = flatDistance(params, start0)
      set({ phase: 'uploading', progress: 1 })
      await sleep(600)
      const privacy = dpOn ? { dp_enabled: true, sigma, sample_rate: tr.sampleRate, steps: tr.steps } : { dp_enabled: false }
      let out
      try {
        out = await request('/fl/update-json', json({ base_version: model.version, num_examples: train.n, weights: toPlain(params), privacy }))
      } catch (e) {
        set({ phase: 'error', error: e.status === 409 ? 'Another hospital finished a round while you were training, so your update is out of date. Press the button again to retrain from the newest model.' : e.message })
        return
      }
      const result = { before, after, dist, rows: train.n, hold: hold.n, epochs, dp: dpOn, sigma, accepted: out.accepted, pending: out.pending, needed: out.needed, version: model.version }
      set({ result })
      let newVersion = out.new_version
      if (!out.aggregated) {
        set({ phase: 'waiting', waiting: { pending: out.pending.length, needed: out.needed + out.pending.length } })
        const st = await pollStatus((s) => s.version > model.version, 10 * 60 * 1000)
        if (cancelRef.current) return
        if (!st || st.version <= model.version) {
          set({ phase: 'idle', result: { ...result, stillWaiting: true } })
          return
        }
        newVersion = st.version
      }
      set({ phase: 'aggregating', newVersion, waiting: null })
      await sleep(1300)
      let hf = 'not configured'
      let st = await request('/fl/status').catch(() => null)
      if (st?.hf_repo) {
        set({ phase: 'publishing', hf: 'pending' })
        st = await pollStatus((s) => s.last_publish?.version >= newVersion && ['ok', 'error'].includes(s.last_publish?.status), 60000)
        hf = st?.last_publish?.version >= newVersion ? st.last_publish.status : 'pending'
      }
      set({ phase: 'done', newVersion, hf, result: { ...result, newVersion, hfStatus: hf, aggregated: true } })
      status.retry()
    } catch (e) {
      if (e.message === 'cancelled') { set({ phase: 'idle' }); return }
      set({ phase: 'error', error: e instanceof ApiError ? e.message : `Training failed: ${e.message}` })
    }
  }

  const r = run.result
  const st = status.data
  return (
    <>
      <div className="steps" style={{ margin: '14px 0' }}>
        {[['Load your batch', 'A CSV of your patients, read in your browser'], ['Choose protection', 'Optional differential privacy'], ['Train locally', 'Your machine does the learning'], ['Contribute', 'Only weights are sent; the model updates']].map(([t, d], i) => (
          <div key={t} className={`step ${i === 0 && data ? 'done' : ''} ${(i === 0 && !data) || (i === 2 && busy) ? 'active' : ''} ${i === 3 && run.phase === 'done' ? 'done' : ''}`}><b>{t}</b><span>{d}</span></div>
        ))}
      </div>

      <div className="card" style={{ marginBottom: 16 }}>
        <FlowAnimation phase={run.phase} progress={run.progress} loss={run.loss} version={st?.version} newVersion={run.newVersion}
          pending={run.waiting?.pending ?? st?.n_pending ?? 0} needed={run.waiting?.needed ?? st?.min_clients ?? 2} hfStatus={run.hf}
          youLabel={user?.username || 'Your hospital'} others={[]} />
        {run.phase === 'training' && <div className="gauge" style={{ marginTop: 8 }} role="progressbar" aria-valuenow={Math.round(run.progress * 100)} aria-valuemin={0} aria-valuemax={100}><i style={{ width: `${run.progress * 100}%` }} /></div>}
      </div>

      <div className="grid two" style={{ alignItems: 'start' }}>
        <div className="card">
          <h4>1. Your batch of data</h4>
          <div className={`dropzone ${over ? 'over' : ''}`} onClick={() => !busy && fileRef.current?.click()} role="button" tabIndex={0}
               onKeyDown={(e) => (e.key === 'Enter' || e.key === ' ') && !busy && fileRef.current?.click()}
               onDragOver={(e) => { e.preventDefault(); setOver(true) }} onDragLeave={() => setOver(false)}
               onDrop={(e) => { e.preventDefault(); setOver(false); if (!busy) loadFile(e.dataTransfer.files?.[0]) }}>
            <Icon name="upload" size={30} />
            <div className="big">{fileInfo ? fileInfo.name : 'Drop a CSV here or click to choose'}</div>
            <div className="small muted">{fileInfo ? `${(fileInfo.size / 1024).toFixed(1)} KB, read locally` : 'Stays on this computer. Nothing is uploaded.'}</div>
            <input ref={fileRef} type="file" accept=".csv,text/csv" hidden onChange={(e) => loadFile(e.target.files?.[0])} disabled={busy} aria-label="Choose a CSV file" />
          </div>
          {parseErr && <div className="err" role="alert" style={{ marginTop: 12 }}><b>Cannot use this file.</b> {parseErr}</div>}
          {data && (
            <div className="grid" style={{ margin: '14px 0 0' }}>
              <StatCard label="Usable rows" value={data.n.toLocaleString()} />
              <StatCard label="Cancer cases" value={`${((data.positives / data.n) * 100).toFixed(1)}%`} hint={`${data.positives} of ${data.n}`} />
              <StatCard label="Layout" value={data.layout} hint={data.duplicatesDropped ? `${data.duplicatesDropped} duplicates removed` : 'no duplicates'} />
            </div>
          )}
          {data?.issues.length > 0 && (
            <div className="note" style={{ marginTop: 12 }}><b>{data.skipped} row(s) skipped.</b>
              {data.issues.slice(0, 4).map((i) => <div key={i} className="small">{i}</div>)}{data.issues.length > 4 && <div className="small">…and {data.issues.length - 4} more</div>}</div>
          )}
          <details style={{ marginTop: 12 }}>
            <summary className="small" style={{ cursor: 'pointer' }}>Expected columns</summary>
            <div className="small muted" style={{ marginTop: 6 }}>
              <span className="mono">{[...spec.features, spec.target].join(', ')}</span><br />
              Either the raw survey layout (gender M/F, age in years, symptoms coded 1 = no / 2 = yes, {spec.target} YES/NO) or the encoded layout (all 0/1, age ÷ 100).
            </div>
          </details>
        </div>

        <div className="card">
          <h4>2. Training and privacy</h4>
          <label className="f">Local epochs: <b>{epochs}</b>
            <input type="range" min="1" max="8" value={epochs} onChange={(e) => setEpochs(+e.target.value)} disabled={busy} aria-label="Local epochs" />
          </label>
          <div style={{ display: 'flex', gap: 12, alignItems: 'center', margin: '18px 0 6px' }}>
            <label className="switch"><input type="checkbox" checked={dpOn} onChange={(e) => setDpOn(e.target.checked)} disabled={busy} aria-label="Differential privacy" /><span /></label>
            <div><b>Differential privacy</b> <Badge tone={dpOn ? 'good' : 'warn'}>{dpOn ? 'enabled' : 'disabled'}</Badge>
              <div className="small muted">Clips each patient's influence and adds noise before anything leaves your machine.</div></div>
          </div>
          {dpOn && (
            <div style={{ marginTop: 10 }}>
              <label className="f">Noise level σ (more noise = stronger privacy, lower accuracy)
                <select value={sigma} onChange={(e) => setSigma(+e.target.value)} disabled={busy}>{SIGMAS.map((s) => <option key={s} value={s}>σ = {s}</option>)}</select>
              </label>
              <div className="note info" style={{ marginTop: 10 }}>
                {eps ? <>This run: <b>ε ≈ {eps.this_update}</b>. Cumulative for your hospital after this run: <b>ε ≈ {eps.cumulative_after}</b> (δ = {eps.delta}).</> : data ? 'Computing the privacy budget…' : 'Load a file to see the privacy budget (ε).'}
                <div className="small muted">The server recomputes ε from these settings and adds it up across all your rounds. It cannot verify that noise was applied.</div>
              </div>
            </div>
          )}
          <div className="row" style={{ marginTop: 18 }}>
            <button className="btn" onClick={start} disabled={!data || busy || isAdmin}><Icon name="bolt" size={16} />{busy ? 'Running…' : 'Train and contribute'}</button>
            {busy && run.phase === 'waiting' && <button className="btn secondary" onClick={() => { cancelRef.current = true }}>Stop waiting</button>}
          </div>
          <div className="small muted">Trains on {nTrain || '…'} rows{holdFraction ? ` and keeps ${data.n - nTrain} aside to measure before/after` : ''}. Global model now: {st ? `v${st.version}` : '…'}.</div>
        </div>
      </div>

      {isAdmin && <div className="note info">You are signed in as the administrator. Administrators manage the network but do not contribute data: sign in with a hospital account to train.</div>}

      {run.error && <div className="err" role="alert" style={{ marginTop: 16 }}><b>Could not finish.</b> {run.error}</div>}

      {(run.curve.length > 0 || r) && (
        <div className="grid two" style={{ marginTop: 16 }}>
          {run.curve.length > 0 && (
            <div>
            <ChartCard title="Local training loss by epoch" height={220}>
              <LineChart data={run.curve}><CartesianGrid stroke="var(--grid)" /><XAxis dataKey="epoch" /><YAxis domain={['auto', 'auto']} /><Tooltip />
                <Line dataKey="loss" stroke="var(--s1)" strokeWidth={3} dot={{ r: 4 }} isAnimationActive /></LineChart>
            </ChartCard>
            <div className="small muted" style={{ marginTop: 8 }}>On this dataset the diagnosis is almost unrelated to the features, so the loss hovers around 0.69 (a coin flip) instead of falling steadily.</div>
            </div>
          )}
          {r && (
            <div className="card">
              <h4>Result of this round</h4>
              <div className="grid" style={{ margin: '0 0 10px', gridTemplateColumns: '1fr 1fr' }}>
                <Metrics label="Global model (before)" m={r.before} /><Metrics label="After your training" m={r.after} />
              </div>
              <div className="small muted">Measured on {r.hold} rows you kept aside. This dataset has almost no learnable signal, so metrics hover around chance (AUC 0.5).</div>
              <hr style={{ border: 0, borderTop: '1px solid var(--line)', margin: '14px 0' }} />
              <dl className="kv">
                <dt>Update accepted</dt><dd><Badge tone="good">yes</Badge></dd>
                <dt>Sent to the server</dt><dd>1,057 weights + row count ({r.rows}); 0 patient rows</dd>
                <dt>Change vs global model</dt><dd>{r.dist.toFixed(3)} (limit {st?.max_update_norm ?? 20})</dd>
                <dt>Differential privacy</dt><dd>{r.dp ? `enabled (σ = ${r.sigma})` : 'disabled'}</dd>
                {r.stillWaiting ? <><dt>Round</dt><dd>Waiting for other hospitals (checked for 10 min)</dd></> : r.aggregated ? <>
                  <dt>New global model</dt><dd><Badge tone="good">v{r.newVersion}</Badge></dd>
                  <dt>Hugging Face</dt><dd><Badge tone={r.hfStatus === 'ok' ? 'good' : r.hfStatus === 'error' ? 'crit' : ''}>{r.hfStatus === 'ok' ? 'updated' : r.hfStatus === 'error' ? 'sync failed' : r.hfStatus}</Badge></dd></> : null}
              </dl>
              {!r.aggregated && !r.stillWaiting && <div className="note" style={{ marginTop: 10 }}>Update accepted. Waiting for {r.needed} more hospital(s); the model is built and published as soon as they report.</div>}
            </div>
          )}
        </div>
      )}
    </>
  )
}

export default function Train() {
  const spec = useApi('/features')
  return (
    <>
      <PageHeader eyebrow="Contribute to the global model" title="Train and Contribute">
        Upload a batch of your hospital's records. The model trains right here in your browser, so patient data never leaves your machine. Only the
        improved weights are sent; when enough hospitals have reported, the server averages them into a new global model and updates it on Hugging Face.
      </PageHeader>
      <Async q={spec} what="training setup">{(s) => <Trainer spec={s} />}</Async>
    </>
  )
}
