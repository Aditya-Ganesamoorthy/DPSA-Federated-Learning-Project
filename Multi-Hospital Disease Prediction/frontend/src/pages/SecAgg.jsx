import { useEffect, useState } from 'react'
import { request, ApiError } from '../api'
import { DataTable, ErrorBox, Loading, PageHeader } from '../components/ui'

const nums = (a) => a.map((v) => (Number.isInteger(v) ? v : v.toFixed(2))).join(', ')

export default function SecAgg() {
  const [seed, setSeed] = useState(1)
  const [drop, setDrop] = useState(null)
  const [tick, setTick] = useState(0)
  const [state, setState] = useState({ loading: true, data: null, error: null })

  useEffect(() => {
    let live = true
    setState((s) => ({ ...s, loading: true, error: null }))
    request(`/secagg/demo?seed=${seed}${drop === null ? '' : `&drop=${drop}`}`)
      .then((data) => live && setState({ loading: false, data, error: null }))
      .catch((error) => live && setState({ loading: false, data: null, error: error instanceof ApiError ? error : new ApiError(String(error), 0) }))
    return () => { live = false }
  }, [seed, drop, tick])

  const d = state.data
  return (
    <>
      <PageHeader title="Secure aggregation (live toy demo)">
        Three hospitals each hold a 4-number update. Every pair shares a seed; the lower-indexed hospital adds the pair's mask, the higher-indexed one subtracts it
        (integer arithmetic mod 2³²). The server sees only masked vectors, yet their sum equals the true weighted average.
      </PageHeader>
      <div className="row">
        <button className="btn" onClick={() => { setSeed((s) => s + 1); setDrop(null) }}>New random updates</button>
        <button className="btn secondary" onClick={() => setDrop(drop === null ? 1 : null)} aria-pressed={drop !== null}>
          {drop === null ? 'Drop hospital 2' : 'Restore hospital 2'}
        </button>
        <span className="mono">seed {seed}</span>
      </div>
      {state.loading && !d && <Loading what="demo" />}
      {state.error && <ErrorBox error={state.error} retry={() => setTick((t) => t + 1)} />}
      {d && (
        <>
          <DataTable
            rows={d.counts.map((n, i) => ({
              h: `Hospital ${i + 1}${d.dropped_client === i ? ' (dropped)' : ''}`, n, w: d.weights[i].toFixed(2),
              u: nums(d.updates[i]), m: d.masked[i].join(', '),
            }))}
            columns={[
              { key: 'h', label: 'Hospital' }, { key: 'n', label: 'Samples', num: true }, { key: 'w', label: 'Weight n/N', num: true },
              { key: 'u', label: 'True update (private)', render: (r) => <span className="mono">{r.u}</span> },
              { key: 'm', label: 'What the server receives (masked)', render: (r) => <span className="mono">{r.m}</span> },
            ]}
          />
          <div className="grid" style={{ gridTemplateColumns: 'repeat(auto-fit, minmax(300px, 1fr))' }}>
            <div className="card"><h4>Server's aggregate</h4><div className="mono">[{nums(d.server_aggregate)}]</div></div>
            <div className="card"><h4>Plain FedAvg (never computed by the server)</h4><div className="mono">[{nums(d.plain_fedavg)}]</div></div>
          </div>
          <div className={`note ${d.masks_cancelled ? 'ok' : ''}`} role="status">
            {d.masks_cancelled ? '✓ Masks cancelled.' : '✗ Masks did NOT cancel.'} Max absolute error: <b>{d.max_abs_error.toExponential(2)}</b>. {d.explanation}
          </div>
          <div className="note">
            <b>Honest limits.</b> This is a simulation: pair seeds are shared directly, not derived by key agreement. It hides individual updates from the server but does not
            protect the final model from inference, which is why DP is combined with it.
          </div>
        </>
      )}
    </>
  )
}
