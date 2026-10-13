import { useState } from 'react'
import { ApiError, request, useApi } from '../api'
import { Async, PageHeader } from '../components/ui'
import { f3 } from '../format'

const label = (n) => n.replace(/_/g, ' ').replace(/^./, (c) => c.toUpperCase())

function RiskGauge({ p, thr }) {
  return (
    <div>
      <div className="gauge" role="img" aria-label={`Predicted probability ${f3(p)}, threshold ${thr}`}>
        <i style={{ width: `${p * 100}%` }} /><b style={{ left: `${thr * 100}%` }} title="decision threshold" />
      </div>
      <div className="l" style={{ color: 'var(--ink2)', fontSize: 13 }}>Probability {f3(p)} · threshold {thr} (black tick)</div>
    </div>
  )
}

function Form({ spec }) {
  const init = Object.fromEntries(spec.fields.map((f) => [f.name, f.type === 'bool' ? false : f.type === 'enum' ? f.options[0] : '']))
  const [vals, setVals] = useState({ ...init, age: 55 })
  const [model, setModel] = useState('global')
  const [res, setRes] = useState(null)
  const [err, setErr] = useState(null)
  const [fieldErr, setFieldErr] = useState({})
  const [busy, setBusy] = useState(false)

  const clientErrors = () => {
    const e = {}
    const age = spec.fields.find((f) => f.name === 'age')
    if (!Number.isInteger(Number(vals.age)) || vals.age === '' || vals.age < age.min || vals.age > age.max)
      e.age = `Age must be a whole number between ${age.min} and ${age.max}`
    return e
  }

  const submit = async (ev) => {
    ev.preventDefault()
    const ce = clientErrors()
    setFieldErr(ce); setErr(null); setRes(null)
    if (Object.keys(ce).length) return
    setBusy(true)
    try {
      const patient = { ...vals, age: Number(vals.age) }
      setRes(await request('/predict', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ model, patient }) }))
    } catch (e) {
      if (e instanceof ApiError && e.status === 422) {
        setFieldErr(Object.fromEntries(e.fields.map((f) => [f.field.replace(/^patient\./, ''), f.message])))
      } else setErr(e)
    } finally { setBusy(false) }
  }

  return (
    <form onSubmit={submit} noValidate>
      <div className="note"><b>Educational demo, not medical advice.</b> Every model here is near chance on this dataset (AUC ≈ 0.5), so the output has no clinical meaning.</div>
      <div className="card">
        <label className="f" style={{ maxWidth: 260, marginBottom: 12 }}>Model
          <select value={model} onChange={(e) => setModel(e.target.value)}>
            {spec.models.map((m) => <option key={m} value={m}>{m === 'global' ? 'global (live, latest version)' : m.replace('_', ' + ')}</option>)}
          </select>
        </label>
        <div className="fields">
          {spec.fields.map((f) => (
            f.type === 'bool' ? (
              <label className="f check" key={f.name}>
                <input type="checkbox" checked={vals[f.name]} onChange={(e) => setVals({ ...vals, [f.name]: e.target.checked })} />
                {label(f.name)}
                {fieldErr[f.name] && <span className="err-msg">{fieldErr[f.name]}</span>}
              </label>
            ) : (
              <label className="f" key={f.name}>{label(f.name)}
                {f.type === 'enum' ? (
                  <select value={vals[f.name]} onChange={(e) => setVals({ ...vals, [f.name]: e.target.value })}>
                    {f.options.map((o) => <option key={o} value={o}>{o === 'M' ? 'Male' : 'Female'}</option>)}
                  </select>
                ) : (
                  <input type="number" min={f.min} max={f.max} value={vals[f.name]} aria-invalid={!!fieldErr[f.name]}
                         onChange={(e) => setVals({ ...vals, [f.name]: e.target.value })} />
                )}
                {fieldErr[f.name] && <span className="err-msg" role="alert">{fieldErr[f.name]}</span>}
              </label>
            )
          ))}
        </div>
        <div className="row"><button className="btn" disabled={busy}>{busy ? 'Predicting…' : 'Predict'}</button></div>
      </div>
      {err && <div className="err" role="alert"><b>Error.</b> {err.message}</div>}
      {res && (
        <div className="card" style={{ marginTop: 12 }} role="status">
          <h4>Result ({res.model_used.replace('_', ' + ')}{res.model_used === 'global' ? ` v${res.model_version}` : ''})</h4>
          <RiskGauge p={res.probability} thr={res.threshold} />
          <p>Predicted label: <b>{res.label === 1 ? 'cancer (positive)' : 'no cancer (negative)'}</b></p>
          <p className="lede">{res.disclaimer} {res.model_note}</p>
        </div>
      )}
    </form>
  )
}

export default function Predict() {
  const q = useApi('/features')
  return (
    <>
      <PageHeader title="Predict">Enter the 15 survey features. The server applies the same preprocessing as training.</PageHeader>
      <Async q={q} what="form">{(spec) => <Form spec={spec} />}</Async>
    </>
  )
}
