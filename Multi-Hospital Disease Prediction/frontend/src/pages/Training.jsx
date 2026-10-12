import { useState } from 'react'
import { CartesianGrid, Legend, Line, LineChart, ReferenceLine, Tooltip, XAxis, YAxis } from 'recharts'
import { useApi } from '../api'
import { Async, ChartCard, PageHeader, Tip } from '../components/ui'
import { f3 } from '../format'

const METRICS = [['auc', 'ROC-AUC'], ['recall', 'Recall'], ['accuracy', 'Accuracy'], ['loss', 'Loss']]

function Curves({ scenario, metric }) {
  const q = useApi(`/results/rounds?scenario=${scenario}`)
  return (
    <Async q={q} what="training curves">
      {(d) => {
        const data = d.rounds.map((r, i) => ({
          round: r, mean: d.test[metric].mean[i], lo: d.test[metric].mean[i] - d.test[metric].std[i],
          hi: d.test[metric].mean[i] + d.test[metric].std[i],
        }))
        const clients = Object.keys(d.per_client)
        const cdata = d.rounds.slice(1).map((r, i) => ({
          round: r, ...Object.fromEntries(clients.map((h) => [h, d.per_client[h][metric === 'loss' ? 'auc' : metric]?.[i + 1]])),
        }))
        const chance = metric === 'auc' || metric === 'accuracy' || metric === 'recall' ? 0.5 : null
        return (
          <div className="grid" style={{ gridTemplateColumns: 'repeat(auto-fit, minmax(320px, 1fr))' }}>
            <ChartCard title={`Global model on the shared test set (mean of ${d.n_seeds} seeds)`}>
              <LineChart data={data}>
                <CartesianGrid stroke="var(--grid)" />
                <XAxis dataKey="round" label={{ value: 'Round', position: 'insideBottom', offset: -2 }} />
                <YAxis domain={metric === 'loss' ? ['auto', 'auto'] : [0.3, 0.7]} tickFormatter={f3} />
                <Tooltip formatter={(v) => f3(v)} />
                {chance && <ReferenceLine y={chance} stroke="var(--ink2)" strokeDasharray="4 4" label={{ value: 'chance', fill: 'var(--ink2)', fontSize: 12 }} />}
                <Line dataKey="mean" name="Global model" stroke="var(--s1)" strokeWidth={2} dot={{ r: 3 }} />
                <Line dataKey="lo" name="mean − std" stroke="var(--s1)" strokeOpacity={0.35} dot={false} strokeDasharray="2 3" />
                <Line dataKey="hi" name="mean + std" stroke="var(--s1)" strokeOpacity={0.35} dot={false} strokeDasharray="2 3" />
                <Legend />
              </LineChart>
            </ChartCard>
            <ChartCard title={`Global model on each hospital's own data (${metric === 'loss' ? 'AUC' : METRICS.find((m) => m[0] === metric)[1]})`}>
              <LineChart data={cdata}>
                <CartesianGrid stroke="var(--grid)" />
                <XAxis dataKey="round" /><YAxis domain={['auto', 'auto']} tickFormatter={f3} />
                <Tooltip formatter={(v) => f3(v)} />
                {clients.map((h, i) => (
                  <Line key={h} dataKey={h} stroke={['var(--s1)', 'var(--s2)', 'var(--s3)'][i]} strokeWidth={2}
                        strokeDasharray={['0', '6 3', '2 2'][i]} dot={{ r: 2 }} />
                ))}
                <Legend />
              </LineChart>
            </ChartCard>
          </div>
        )
      }}
    </Async>
  )
}

export default function Training() {
  const [scenario, setScenario] = useState('noniid')
  const [metric, setMetric] = useState('auc')
  const list = useApi('/results/rounds?scenario=iid') // cheap availability probe
  return (
    <>
      <PageHeader title="Federated training">
        Each round the three hospitals train locally and the server averages their weights (<Tip term="fedavg">FedAvg</Tip>).
        Curves are the global model's test metrics after every round. A flat line at chance is the expected behaviour on this dataset.
      </PageHeader>
      <div className="row">
        <div className="seg" role="group" aria-label="Scenario">
          {[['noniid', 'Non-IID'], ['iid', 'IID']].map(([k, l]) => (
            <button key={k} aria-pressed={scenario === k} onClick={() => setScenario(k)}>{l}</button>
          ))}
        </div>
        <div className="seg" role="group" aria-label="Metric">
          {METRICS.map(([k, l]) => (
            <button key={k} aria-pressed={metric === k} onClick={() => setMetric(k)}>{l}</button>
          ))}
        </div>
      </div>
      {list.error ? <Async q={list}>{() => null}</Async> : <Curves scenario={scenario} metric={metric} />}
    </>
  )
}
