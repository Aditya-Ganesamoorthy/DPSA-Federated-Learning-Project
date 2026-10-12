import { Bar, BarChart, CartesianGrid, Cell, LabelList, Tooltip, XAxis, YAxis } from 'recharts'
import { useApi } from '../api'
import { Async, ChartCard, DataTable, PageHeader, StatCard } from '../components/ui'
import { f2, f3, mb } from '../format'

export default function Communication() {
  const q = useApi('/results/communication')
  return (
    <>
      <PageHeader title="Communication efficiency">
        Each round every hospital downloads and uploads the full model. We count both directions, for all clients.
      </PageHeader>
      <Async q={q} what="communication results">
        {(d) => {
          const e2 = d.by_local_epochs.find((e) => e.local_epochs === 2) || d.by_local_epochs[0]
          const bars = [...d.by_local_epochs.map((e) => ({ name: `FL, E=${e.local_epochs}`, mb: e.total_mb, fl: true })),
            { name: 'Raw data, uploaded once (CSV)', mb: d.raw_data.csv_mb, fl: false }]
          return (
            <>
              <div className="grid">
                <StatCard label="Model parameters" value={d.model_params} hint="float32, 4 bytes each" />
                <StatCard label="Per client per round" value={mb(e2.bytes_per_client_per_round)} hint="upload + download" />
                <StatCard label="All clients per round" value={mb(e2.bytes_per_round_total)} />
                <StatCard label={`Total, ${e2.rounds} rounds`} value={mb(e2.total_bytes)} />
                <StatCard label="Raw data (centralize once)" value={mb(d.raw_data.csv_bytes)} hint={`${d.raw_data.rows} rows as CSV`} />
              </div>
              <div className="note">
                With this tiny dataset and {e2.rounds} rounds, federated learning transfers <b>more</b> bytes than uploading the raw data once
                ({mb(e2.total_bytes)} vs {mb(d.raw_data.csv_bytes)}). Here the benefit of FL is privacy, not bandwidth. The balance flips for large datasets or few rounds.
              </div>
              <ChartCard title="Total data transferred (MB)" height={260}>
                <BarChart data={bars} layout="vertical" margin={{ left: 150, right: 60 }}>
                  <CartesianGrid stroke="var(--grid)" horizontal={false} />
                  <XAxis type="number" /><YAxis type="category" dataKey="name" width={150} />
                  <Tooltip formatter={(v) => `${f2(v)} MB`} />
                  <Bar dataKey="mb" radius={[0, 4, 4, 0]}>
                    {bars.map((b, i) => <Cell key={i} fill={b.fl ? 'var(--s1)' : 'var(--s2)'} />)}
                    <LabelList dataKey="mb" position="right" formatter={(v) => `${f2(v)} MB`} fill="var(--ink2)" />
                  </Bar>
                </BarChart>
              </ChartCard>
              <h3>Effect of local epochs E</h3>
              <DataTable rows={d.by_local_epochs} columns={[
                { key: 'local_epochs', label: 'Local epochs E', num: true },
                { key: 'total_mb', label: 'Total MB', num: true, render: (r) => f2(r.total_mb) },
                { key: 'wall', label: 'Wall s / round', num: true, render: (r) => f3(r.mean_wall_s_per_round) },
                { key: 'conv', label: 'Rounds to loss convergence (10%)', num: true, render: (r) => r.rounds_to_loss_convergence_10pct ?? 'n/a' },
                { key: 'tgt', label: `Rounds to AUC ≥ ${f3(d.target_auc)}`, num: true, render: (r) => r.rounds_to_95pct_of_centralized_auc ?? 'not reached' },
                { key: 'auc', label: 'Final AUC', num: true, render: (r) => f3(r.final_test_auc) },
              ]} />
              <p className="lede" style={{ marginTop: 8 }}>{d.note}</p>
            </>
          )
        }}
      </Async>
    </>
  )
}
