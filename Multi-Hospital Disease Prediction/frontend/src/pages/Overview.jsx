import { useApi } from '../api'
import { Async, PageHeader, StatCard } from '../components/ui'
import { f3 } from '../format'

const STEPS = [
  ['1. Data', 'Audit, dedupe, drop identifiers, generalise age to 10-year bands, k-anonymity check.'],
  ['2. Hospitals', 'Three simulated hospitals (IID control and non-IID main scenario). Only client code opens their files.'],
  ['3. Training', 'Centralized baseline, local-only baseline, Flower FedAvg, FedAvg + DP-SGD, FedAvg + secure aggregation.'],
  ['4. Evaluation', 'One shared test set, 5 seeds, mean ± std. Membership-inference attack with a positive control.'],
  ['5. Analysis', 'Privacy-utility curve, communication cost, threat model, honest interpretation.'],
]

export default function Overview() {
  const q = useApi('/meta')
  return (
    <>
      <PageHeader title="Decentralized ML: Federated Learning for Privacy Preservation">
        Traditional ML pools patient records in one place, which creates a privacy risk. Federated learning trains one shared
        model while each hospital keeps its raw data. We add differential privacy and secure aggregation on top, and
        measure what that costs.
      </PageHeader>

      <div className="note">
        <b>Read this first.</b> On this Kaggle dataset the label is almost unpredictable from the features (cross-validated
        logistic regression AUC ≈ 0.50). Every model, centralized or federated, therefore scores near chance. The privacy,
        communication and security analyses are the substance of the work; we do not claim clinical value.
      </div>

      <h3>Pipeline</h3>
      <div className="grid">
        {STEPS.map(([t, d]) => (
          <div className="card" key={t}><h4>{t}</h4>{d}</div>
        ))}
      </div>

      <h3>Why federated?</h3>
      <div className="card">
        <pre className="mono" style={{ margin: 0, whiteSpace: 'pre-wrap' }}>{`Hospital 1 ─┐ local training            ┌─ global model
Hospital 2 ─┼─ model updates only ─▶ Server (FedAvg) ─┤  sent back to all
Hospital 3 ─┘ (raw rows never leave)    └─ w = Σ (n_k / N) · w_k`}</pre>
      </div>

      <h3>Run details</h3>
      <Async q={q} what="run metadata">
        {(m) => (
          <div className="grid">
            <StatCard label="Seeds per experiment" value={m.seeds.length} />
            <StatCard label="FL rounds" value={m.rounds} />
            <StatCard label="Local epochs" value={m.hyperparameters.local_epochs} />
            <StatCard label="DP clip norm C" value={f3(m.hyperparameters.clip_norm)} />
            <StatCard label="Flower" value={m.flwr} hint={`Opacus ${m.opacus} · torch ${m.torch}`} />
          </div>
        )}
      </Async>
    </>
  )
}
