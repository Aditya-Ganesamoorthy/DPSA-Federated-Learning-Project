import { useApi } from '../api'
import { PageHeader } from '../components/ui'
import { eps, f3, mb, pm } from '../format'

function Answer({ n, q, children }) {
  return (
    <div className="card" style={{ marginBottom: 12 }}>
      <h4>Question {n}</h4>
      <p style={{ margin: '0 0 8px', fontWeight: 600 }}>{q}</p>
      {children}
    </div>
  )
}

export default function QA() {
  const cmp = useApi('/results/comparison').data
  const pc = useApi('/results/privacy-curve').data
  const at = useApi('/results/attack').data
  const cm = useApi('/results/communication').data
  const sc = useApi('/results/secagg-check').data
  const m = (id) => cmp?.methods.find((x) => x.id === id)
  const e2 = cm?.by_local_epochs.find((e) => e.local_epochs === 2)
  const dp = pc?.points.filter((p) => p.epsilon !== null)

  return (
    <>
      <PageHeader title="Guiding questions & limitations">
        Answers are backed by the measured results; numbers load from the backend.
      </PageHeader>

      <Answer n={1} q="How does federated learning differ from centralized ML?">
        <p>Centralized ML copies every record to one server and trains there. In FL the data stays in each hospital; only model weights travel. The server sees updates, never rows.</p>
        {cmp && <p>Measured here: centralized MLP AUC {pm(m('centralized_mlp').metrics.auc)}, FedAvg non-IID {pm(m('fl_noniid').metrics.auc)}, IID {pm(m('fl_iid').metrics.auc)}.
          All sit inside the chance band (±{f3(cmp.chance_band_auc)}), so the dataset cannot separate them; the honest conclusion is "no measurable accuracy cost", not "FL is better".</p>}
      </Answer>

      <Answer n={2} q="What are the main challenges (communication, heterogeneity, security)?">
        <ul>
          <li><b>Communication:</b> {e2 ? <>the model has {cm.model_params} parameters; 20 rounds cost {mb(e2.total_bytes)} versus {mb(cm.raw_data.csv_bytes)} to upload the raw data once. FL moved more bytes here.</> : 'loading…'}</li>
          <li><b>Heterogeneity:</b> hospitals differ in size (50/30/20) and prevalence. Weighted FedAvg handles size; label skew can cause client drift (see per-hospital curves on the Training page).</li>
          <li><b>Security:</b> updates still leak (membership inference, inversion), clients can poison, and dropouts break masking. See the Privacy page threat model.</li>
        </ul>
      </Answer>

      <Answer n={3} q="How does model aggregation work?">
        <p>FedAvg: <span className="mono">w_global = Σ_k (n_k / N) · w_k</span>. Worked example with two hospitals, n = 1 and 3, one weight each equal to 1 and 4:
          (1·1 + 3·4) / 4 = <b>3.25</b>. Weighting by n_k stops a tiny hospital from counting as much as a large one. We verified Flower's aggregation against our hand-written 40-line loop (max difference 6e-8).</p>
      </Answer>

      <Answer n={4} q="What techniques enhance privacy and security?">
        <ul>
          <li><b>Differential privacy (DP-SGD):</b> per-sample clipping + Gaussian noise. {dp && <>Across σ = {dp.map((p) => p.sigma).join(', ')} the per-hospital worst-case ε is {dp.map((p) => eps(p.epsilon)).join(', ')} (δ = {pc.delta}, {pc.rounds} rounds). The accountant persists across rounds; resetting it each round would under-report ε.</>}</li>
          <li><b>Secure aggregation:</b> pairwise masks mod 2³² cancel exactly. {sc && <>Integer sum difference: <b>{sc.integer_masked_sum_vs_plain_sum_max_abs_diff}</b>; a dropped client gives an error of {sc.dropout_float_error.toFixed(0)}, so it is detected.</>} It is a simulation, not a cryptographic protocol.</li>
          <li><b>Attack evaluation:</b> {at && <>membership-inference AUC is {pm(at.models[0].auc)} for the centralized model and {pm(at.models[1].auc)} for FedAvg, versus {pm(at.positive_control.auc)} on the deliberately overfit positive control (the attack works) and {pm(at.models[at.models.length - 1].auc)} with DP at σ = {at.models[at.models.length - 1].sigma}. Leakage falls from centralized to FedAvg to FedAvg + DP.</>}</li>
        </ul>
      </Answer>

      <Answer n={5} q="What are real-world applications in healthcare and finance?">
        <ul>
          <li><b>Healthcare:</b> multi-hospital imaging and EHR models (for example the NVIDIA Clara / EXAM study predicting oxygen needs of COVID-19 patients across 20 sites; brain-tumour segmentation by the Intel/UPenn FeTS initiative), where patient data cannot leave each hospital.</li>
          <li><b>Finance:</b> cross-bank fraud and anti-money-laundering detection without sharing customer transactions (for example federated pilots by banks and the UK/US PETs prize challenges); credit scoring across institutions.</li>
          <li><b>Also:</b> keyboard prediction on phones (Google Gboard), which popularised cross-device FL.</li>
        </ul>
      </Answer>

      <h3>Limitations</h3>
      <div className="card">
        <ul>
          <li>The dataset is survey-style/synthetic-looking and the label is essentially unlearnable from the features (CV AUC ≈ 0.50), so accuracy comparisons are uninformative. The pipeline is validated separately on planted-signal labels.</li>
          <li>Secure aggregation is simulated (shared seeds, no key exchange, no dropout recovery).</li>
          <li>DP is record-level against the server; it does not defend against a malicious hospital. Opacus' default noise generator is not a cryptographically secure RNG.</li>
          <li>Flower is run in-process (real clients and strategy, no gRPC/ray simulation); a production deployment would use a real Flower server and TLS.</li>
          <li>Three hospitals and about 1,800 training rows; a tiny model. Not medically validated. Educational demo only.</li>
        </ul>
      </div>
    </>
  )
}
