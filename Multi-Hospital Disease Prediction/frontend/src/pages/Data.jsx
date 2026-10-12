import { Bar, BarChart, CartesianGrid, Legend, Tooltip, XAxis, YAxis } from 'recharts'
import { useApi } from '../api'
import { Async, ChartCard, DataTable, PageHeader, StatCard, Tip } from '../components/ui'
import { pct } from '../format'

export default function Data() {
  const ds = useApi('/dataset/summary')
  const cl = useApi('/clients')
  return (
    <>
      <PageHeader title="Data & Hospitals">
        3,000 survey-style records (15 features, binary lung-cancer label) split across three simulated hospitals.
      </PageHeader>
      <Async q={ds} what="dataset summary">
        {(d) => (
          <>
            <div className="grid">
              <StatCard label="Raw rows" value={d.preprocess.raw_rows} />
              <StatCard label="Duplicates removed" value={d.preprocess.duplicates_removed} />
              <StatCard label="Missing values" value={d.preprocess.missing_values} />
              <StatCard label="Positive rate" value={pct(d.preprocess.prevalence)} />
              <StatCard label="Pooled k-anonymity" value={d.k_anonymity_pooled} hint="age band × gender" />
            </div>
            <div className="card">
              <h4>Privacy preprocessing (data minimisation)</h4>
              <table>
                <thead><tr><th>Step</th><th>Before</th><th>After</th></tr></thead>
                <tbody>
                  <tr><td>Direct identifiers</td><td>none present in file</td><td>checked and asserted absent</td></tr>
                  <tr><td>Age</td><td>exact integer ({d.preprocess.age_min}–{d.preprocess.age_max})</td><td>10-year band for the k-anonymity report; model uses age/100</td></tr>
                  <tr><td>Duplicates</td><td>{d.preprocess.raw_rows} rows</td><td>{d.preprocess.clean_rows} rows</td></tr>
                </tbody>
              </table>
              <p className="lede" style={{ marginTop: 8 }}>
                Dropping identifiers does not make data anonymous. The layers are: minimisation, then federated learning, then DP / secure aggregation.
              </p>
            </div>
            <div className="row">Split rows: {Object.entries(d.split_rows).map(([k, v]) => <span key={k} className="card" style={{ padding: '4px 10px' }}>{k}: {v}</span>)}</div>
          </>
        )}
      </Async>

      <h3>Hospitals</h3>
      <Async q={cl} what="hospitals">
        {(c) => (
          <>
            {['noniid', 'iid'].map((scen) => (
              <div key={scen} style={{ marginBottom: 16 }}>
                <h4>{scen === 'iid' ? 'IID control' : 'Non-IID (main scenario)'}</h4>
                <DataTable
                  rows={c[scen]}
                  columns={[
                    { key: 'hospital', label: 'Hospital' },
                    { key: 'rows', label: 'Rows', num: true },
                    { key: 'positives', label: 'Positives', num: true },
                    { key: 'prevalence', label: 'Prevalence', num: true, render: (r) => pct(r.prevalence) },
                    { key: 'k_anonymity', label: 'k-anonymity', num: true },
                  ]}
                />
              </div>
            ))}
            <div className="note">
              Non-IID hospital 3 has k = {c.noniid[2].k_anonymity}: at least one (age band, gender) group there has a single record.
              Reported as-is; this is why hospital data alone is not "anonymous".
            </div>
            <div className="grid" style={{ gridTemplateColumns: 'repeat(auto-fit, minmax(300px, 1fr))' }}>
              <ChartCard title="Training rows per hospital" legend={['IID', 'Non-IID']}>
                <BarChart data={c.iid.map((r, i) => ({ h: `H${i + 1}`, IID: r.rows, 'Non-IID': c.noniid[i].rows }))}>
                  <CartesianGrid stroke="var(--grid)" vertical={false} />
                  <XAxis dataKey="h" /><YAxis /><Tooltip />
                  <Bar dataKey="IID" fill="var(--s1)" radius={[4, 4, 0, 0]} />
                  <Bar dataKey="Non-IID" fill="var(--s2)" radius={[4, 4, 0, 0]} />
                </BarChart>
              </ChartCard>
              <ChartCard title="Cancer prevalence per hospital" legend={['IID', 'Non-IID']}>
                <BarChart data={c.iid.map((r, i) => ({ h: `H${i + 1}`, IID: r.prevalence, 'Non-IID': c.noniid[i].prevalence }))}>
                  <CartesianGrid stroke="var(--grid)" vertical={false} />
                  <XAxis dataKey="h" /><YAxis domain={[0, 1]} tickFormatter={(v) => `${v * 100}%`} />
                  <Tooltip formatter={(v) => pct(v)} />
                  <Bar dataKey="IID" fill="var(--s1)" radius={[4, 4, 0, 0]} />
                  <Bar dataKey="Non-IID" fill="var(--s2)" radius={[4, 4, 0, 0]} />
                </BarChart>
              </ChartCard>
            </div>
            <p className="lede">Hover <Tip term="noniid">non-IID</Tip> for its definition. Quantity skew is 50/30/20; label skew comes from a Dirichlet(1) draw.</p>
          </>
        )}
      </Async>
    </>
  )
}
