import { useState } from 'react'
import { useApi } from './api'
import Overview from './pages/Overview'
import Data from './pages/Data'
import Training from './pages/Training'
import Comparison from './pages/Comparison'
import Privacy from './pages/Privacy'
import Communication from './pages/Communication'
import SecAgg from './pages/SecAgg'
import Predict from './pages/Predict'
import QA from './pages/QA'

const PAGES = [
  ['overview', 'Overview', Overview],
  ['data', 'Data & Hospitals', Data],
  ['training', 'Training', Training],
  ['comparison', 'Comparison', Comparison],
  ['privacy', 'Privacy & Security', Privacy],
  ['communication', 'Communication', Communication],
  ['secagg', 'Secure Aggregation', SecAgg],
  ['predict', 'Predict', Predict],
  ['qa', 'Q&A / Report', QA],
]

export default function App() {
  const [page, setPage] = useState(() => PAGES.find(([k]) => k === window.location.hash.slice(1))?.[0] || 'overview')
  const health = useApi('/health')
  const Page = PAGES.find(([k]) => k === page)[2]
  const go = (k) => { setPage(k); window.location.hash = k }
  const offline = !!health.error
  return (
    <>
      {offline && (
        <div className="banner" role="alert">
          Backend offline: {health.error.message}
          <button className="btn secondary" onClick={health.retry}>Retry</button>
        </div>
      )}
      {health.data && !health.data.results_loaded && (
        <div className="banner" role="alert">Results are missing. Run <span className="mono">python -m ml.run_all</span> first.</div>
      )}
      <div className="layout">
        <nav className="nav" aria-label="Pages">
          <h1>Federated lung cancer prediction</h1>
          {PAGES.map(([k, l]) => (
            <button key={k} aria-current={page === k ? 'page' : undefined} onClick={() => go(k)}>{l}</button>
          ))}
        </nav>
        <main className="main"><Page key={page} /></main>
      </div>
    </>
  )
}
