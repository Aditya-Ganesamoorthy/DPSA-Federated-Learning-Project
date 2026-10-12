import { useState } from 'react'
import { CartesianGrid, ErrorBar, Line, LineChart, ReferenceArea, ReferenceLine, Scatter, ScatterChart, Tooltip, XAxis, YAxis, Legend } from 'recharts'
import { useApi } from '../api'
import { Async, ChartCard, DataTable, PageHeader, Tip } from '../components/ui'
import { f3, pm } from '../format'

const METRICS = [['auc', 'ROC-AUC'], ['recall', 'Recall'], ['accuracy', 'Accuracy'], ['f1', 'F1'], ['specificity', 'Specificity']]

function Roc() {
  const [model, setModel] = useState('federated')
  const q = useApi(`/results/roc?model=${model}`)
  return (
    <>
      <div className="seg row" role="group" aria-label="ROC model">
        {['centralized', 'federated', 'federated_dp'].map((m) => (
          <button key={m} aria-pressed={model === m} onClick={() => setModel(m)}>{m.replace('_', ' + ')}</button>
        ))}
      </div>
      <Async q={q} what="ROC">
        {(d) => (
          <ChartCard title="ROC curve (seed 0). Diagonal = chance" height={300}>
            <LineChart data={d.fpr.map((x, i) => ({ fpr: x, tpr: d.tpr[i], chance: x }))}>
              <CartesianGrid stroke="var(--grid)" />
              <XAxis type="number" dataKey="fpr" domain={[0, 1]} label={{ value: 'False positive rate', position: 'insideBottom', offset: -2 }} />
              <YAxis domain={[0, 1]} />
              <Tooltip formatter={(v) => f3(v)} />
              <Line dataKey="chance" stroke="var(--ink2)" strokeDasharray="4 4" dot={false} name="chance" />
              <Line dataKey="tpr" stroke="var(--s1)" strokeWidth={2} dot={false} name="TPR" />
              <Legend />
            </LineChart>
          </ChartCard>
        )}
      </Async>
    </>
  )
}

export default function Comparison() {
  const [metric, setMetric] = useState('auc')
  const q = useApi('/results/comparison')
  const s = useApi('/results/sanity')
  return (
    <>
      <PageHeader title="Centralized vs federated comparison">
        Every method uses the same architecture, preprocessing, optimiser and test file ({'≈'}600 rows). Mean ± std over seeds.
        <Tip term="auc"> ROC-AUC</Tip> and <Tip term="recall">recall</Tip> lead because a missed cancer is the costly error.
      </PageHeader>
      <div className="seg row" role="group" aria-label="Metric">
        {METRICS.map(([k, l]) => <button key={k} aria-pressed={metric === k} onClick={() => setMetric(k)}>{l}</button>)}
      </div>
      <Async q={q} what="comparison">
        {(d) => {
          const band = d.chance_band_auc
          const rows = d.methods
          const inBand = rows.every((m) => m.metrics.auc.mean !== null && Math.abs(m.metrics.auc.mean - 0.5) <= band + (m.metrics.auc.std || 0))
          return (
            <>
              <div className={`note ${inBand ? '' : 'ok'}`}>
                {inBand
                  ? <>Every method's AUC lies inside the 95% chance band (0.5 ± {f3(band)}, n = {d.test_rows}). There is <b>no meaningful accuracy difference</b> between centralized, federated and federated + DP learning, because the data holds no learnable signal. Do not read the small differences as wins or losses.</>
                  : <>Some methods fall outside the 95% chance band (0.5 ± {f3(band)}).</>}
              </div>
              <DataTable
                rows={rows}
                columns={[
                  { key: 'label', label: 'Method' },
                  ...[['auc', 'AUC'], ['recall', 'Recall'], ['accuracy', 'Accuracy'], ['f1', 'F1']].map(([k, l]) => ({
                    key: k, label: l, num: true,
                    render: (r) => <span style={k === metric ? { fontWeight: 700 } : undefined}>{pm(r.metrics[k])}</span>,
                  })),
                ]}
              />
              <p className="lede" style={{ marginTop: 8 }}>
                Centralized training uses a matched budget of {d.matched_budget_epochs} epochs (= 20 rounds × 2 local epochs).
                "Local-only" is each hospital training alone on its own rows.
              </p>
              <ChartCard title={`${METRICS.find((m) => m[0] === metric)[1]} by method (± std)`} height={Math.max(260, rows.length * 34)}>
                <ScatterChart margin={{ left: 150 }}>
                  <CartesianGrid stroke="var(--grid)" />
                  {metric === 'auc' && <ReferenceArea x1={0.5 - band} x2={0.5 + band} fill="var(--grid)" fillOpacity={0.6} />}
                  <ReferenceLine x={0.5} stroke="var(--ink2)" />
                  <XAxis type="number" dataKey="v" domain={[0.4, 0.6]} tickFormatter={f3} />
                  <YAxis type="category" dataKey="label" width={150} interval={0} tick={{ fontSize: 12 }} />
                  <Tooltip formatter={(v) => f3(v)} />
                  <Scatter data={rows.map((m) => ({ label: m.label, v: m.metrics[metric].mean, sd: m.metrics[metric].std }))} fill="var(--s1)">
                    <ErrorBar dataKey="sd" direction="x" stroke="var(--s1)" />
                  </Scatter>
                </ScatterChart>
              </ChartCard>
            </>
          )
        }}
      </Async>

      <h3>ROC</h3>
      <Roc />

      <h3>Pipeline validation on synthetic labels</h3>
      <Async q={s} what="sanity check">
        {(d) => (
          <>
            <div className="note ok">
              {d.note} This is <b>not</b> the real dataset: the real feature rows get a planted-signal label so we can check
              that every method <i>can</i> learn. If they do, near-chance results on the real data come from the data, not from a pipeline bug.
            </div>
            <DataTable
              rows={Object.entries(d.methods).map(([k, v]) => ({ k, ...v }))}
              columns={[{ key: 'k', label: 'Method' }, { key: 'auc', label: 'AUC', num: true, render: (r) => pm(r.auc) },
                { key: 'recall', label: 'Recall', num: true, render: (r) => pm(r.recall) },
                { key: 'accuracy', label: 'Accuracy', num: true, render: (r) => pm(r.accuracy) }]}
            />
          </>
        )}
      </Async>
    </>
  )
}
