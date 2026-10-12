import { CartesianGrid, ErrorBar, Legend, Line, LineChart, ReferenceLine, Scatter, ScatterChart, Tooltip, XAxis, YAxis } from 'recharts'
import { useApi } from '../api'
import { Async, ChartCard, DataTable, PageHeader, Tip } from '../components/ui'
import { eps, f3, pm } from '../format'

const THREATS = [
  ['Raw data exposure', 'Central server / breach', 'FL: data never leaves the hospital', 'Updates still leak information'],
  ['Membership inference', 'Server or model user', 'DP-SGD', 'Needs small ε; utility cost'],
  ['Update / gradient inversion', 'Honest-but-curious server', 'Secure aggregation + DP', 'Simulated SecAgg is not cryptographic'],
  ['Model poisoning', 'Compromised hospital', 'Discussed only (not mitigated)', 'Needs robust aggregation'],
  ['Client dropout', 'Network', 'SecAgg failure demo', 'Real protocol needs secret sharing'],
]

export default function Privacy() {
  const pc = useApi('/results/privacy-curve')
  const at = useApi('/results/attack')
  return (
    <>
      <PageHeader title="Privacy & Security">
        Differential privacy here is record-level DP-SGD (per-sample clipping C = 1, Gaussian noise σ·C) with a persistent accountant across
        rounds. It protects against the server and anyone who sees shared updates, not against a compromised hospital.
      </PageHeader>

      <h3><Tip term="epsilon">ε</Tip> vs utility</h3>
      <Async q={pc} what="privacy curve">
        {(d) => {
          const pts = d.points
          const dp = pts.filter((p) => p.epsilon !== null)
          const base = pts.find((p) => p.epsilon === null)
          const series = (k) => dp.map((p) => ({ eps: p.epsilon, v: p.metrics[k].mean, sd: p.metrics[k].std, sigma: p.sigma }))
          return (
            <>
              <DataTable
                rows={pts}
                columns={[
                  { key: 'sigma', label: 'Noise σ', num: true, render: (r) => (r.sigma === 0 ? '0 (no DP)' : r.sigma) },
                  { key: 'epsilon', label: 'ε (max over hospitals)', num: true, render: (r) => eps(r.epsilon) },
                  { key: 'auc', label: 'AUC', num: true, render: (r) => pm(r.metrics.auc) },
                  { key: 'recall', label: 'Recall', num: true, render: (r) => pm(r.metrics.recall) },
                  { key: 'accuracy', label: 'Accuracy', num: true, render: (r) => pm(r.metrics.accuracy) },
                ]}
                caption={`δ = ${d.delta}, ${d.rounds} rounds × ${d.local_epochs} local epochs. "No DP" means ε = ∞.`}
              />
              <div className="grid" style={{ gridTemplateColumns: 'repeat(auto-fit, minmax(320px, 1fr))' }}>
                {[['auc', 'AUC'], ['recall', 'Recall']].map(([k, l]) => (
                  <ChartCard key={k} title={`${l} vs ε (labels show σ; dashed = no DP)`}>
                    <ScatterChart>
                      <CartesianGrid stroke="var(--grid)" />
                      <XAxis type="number" dataKey="eps" name="ε" scale="log" domain={['auto', 'auto']} label={{ value: 'ε (log scale)', position: 'insideBottom', offset: -2 }} />
                      <YAxis type="number" dataKey="v" domain={[0.3, 0.7]} tickFormatter={f3} name={l} />
                      <Tooltip formatter={(v) => f3(v)} />
                      {base && <ReferenceLine y={base.metrics[k].mean} stroke="var(--s2)" strokeDasharray="5 4" />}
                      <ReferenceLine y={0.5} stroke="var(--ink2)" />
                      <Scatter data={series(k)} fill="var(--s1)" line={{ stroke: 'var(--s1)', strokeWidth: 2 }}
                               label={{ dataKey: 'sigma', position: 'top', fill: 'var(--ink2)', fontSize: 11 }}>
                        <ErrorBar dataKey="sd" direction="y" stroke="var(--s1)" />
                      </Scatter>
                    </ScatterChart>
                  </ChartCard>
                ))}
              </div>
              <div className="note">
                Utility does not fall as ε shrinks because there is no signal to lose: every setting is at chance. The cost of
                DP on real signal is shown on the planted-signal data (Comparison page). Larger σ gives smaller <Tip term="epsilon">ε</Tip> (stronger privacy).
              </div>
            </>
          )
        }}
      </Async>

      <h3>Membership-inference attack</h3>
      <Async q={at} what="attack results">
        {(d) => {
          const rows = [...d.models, d.positive_control, d.negative_control || d.untrained_control]
          return (
            <>
              <p className="lede">Loss-threshold attack. Members are training records; non-members come from a dedicated untouched hold-out; sets are balanced so AUC 0.5 = chance.</p>
              <ChartCard title="Attack AUC by model (0.5 = chance)" height={Math.max(260, rows.length * 34)}>
                <ScatterChart margin={{ left: 190 }}>
                  <CartesianGrid stroke="var(--grid)" />
                  <ReferenceLine x={0.5} stroke="var(--ink2)" label={{ value: 'chance', fill: 'var(--ink2)', fontSize: 12 }} />
                  <XAxis type="number" dataKey="v" domain={[0.4, 1]} tickFormatter={f3} />
                  <YAxis type="category" dataKey="label" width={190} interval={0} tick={{ fontSize: 12 }} />
                  <Tooltip formatter={(v) => f3(v)} />
                  <Scatter data={rows.map((r) => ({ label: r.label, v: r.auc.mean, sd: r.auc.std }))} fill="var(--s1)">
                    <ErrorBar dataKey="sd" direction="x" stroke="var(--s1)" />
                  </Scatter>
                </ScatterChart>
              </ChartCard>
              <DataTable rows={rows} columns={[
                { key: 'label', label: 'Target' },
                { key: 'auc', label: 'Attack AUC', num: true, render: (r) => pm(r.auc) },
                { key: 'best_accuracy', label: 'Best-threshold accuracy', num: true, render: (r) => f3(r.best_accuracy) },
              ]} />
              <div className="note ok">
                The positive control (a deliberately overfit large MLP on 150 rows) shows the attack code works. Real models stay near 0.5 because a small,
                lightly-overfit model on near-random labels leaks little through its loss. That is a result, not proof of safety: stronger attacks exist.
              </div>
            </>
          )
        }}
      </Async>

      <h3>Threat model</h3>
      <DataTable rows={THREATS.map(([t, w, m, r]) => ({ t, w, m, r }))} columns={[
        { key: 't', label: 'Threat' }, { key: 'w', label: 'Who' }, { key: 'm', label: 'Mitigation here' }, { key: 'r', label: 'Residual risk' }]} />
    </>
  )
}
