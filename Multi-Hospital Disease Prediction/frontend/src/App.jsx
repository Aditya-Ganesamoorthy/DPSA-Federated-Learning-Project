import { useEffect, useMemo, useState } from 'react'
import { AuthContext, loadSession, logoutRequest, useApi } from './api'
import Overview from './pages/Overview'
import Data from './pages/Data'
import Training from './pages/Training'
import Comparison from './pages/Comparison'
import Privacy from './pages/Privacy'
import Communication from './pages/Communication'
import SecAgg from './pages/SecAgg'
import Federation from './pages/Federation'
import Predict from './pages/Predict'
import QA from './pages/QA'
import Login from './pages/Login'

// auth: true = requires a hospital/admin login
const PAGES = [
  ['overview', 'Overview', Overview],
  ['data', 'Data & Hospitals', Data],
  ['training', 'Training', Training],
  ['comparison', 'Comparison', Comparison],
  ['privacy', 'Privacy & Security', Privacy],
  ['communication', 'Communication', Communication],
  ['secagg', 'Secure Aggregation', SecAgg],
  ['federation', 'Live Federation', Federation, true],
  ['predict', 'Predict', Predict, true],
  ['qa', 'Q&A / Report', QA],
]

export default function App() {
  const [user, setUser] = useState(() => loadSession())
  const [page, setPage] = useState(() => PAGES.find(([k]) => k === window.location.hash.slice(1))?.[0] || 'overview')
  const health = useApi('/health')
  const ctx = useMemo(() => ({ user, setUser, logout: () => { logoutRequest(); setUser(null) } }), [user])

  useEffect(() => {
    const out = () => setUser(null)
    window.addEventListener('fl-logout', out)
    return () => window.removeEventListener('fl-logout', out)
  }, [])

  const entry = PAGES.find(([k]) => k === page) || PAGES[0]
  const Page = entry[2]
  const gated = entry[3] && !user
  const go = (k) => { setPage(k); window.location.hash = k }
  const offline = !!health.error
  return (
    <AuthContext.Provider value={ctx}>
      {offline && (
        <div className="banner" role="alert">
          Server unreachable: {health.error.message}
          <button className="btn secondary" onClick={health.retry}>Retry</button>
        </div>
      )}
      {health.data && !health.data.results_loaded && (
        <div className="banner" role="alert">Results are missing. Run <span className="mono">python -m ml.run_all</span> first.</div>
      )}
      <div className="layout">
        <nav className="nav" aria-label="Pages">
          <h1>Federated lung cancer prediction</h1>
          {PAGES.map(([k, l, , a]) => (
            <button key={k} aria-current={page === k ? 'page' : undefined} onClick={() => go(k)}>{l}{a && !user ? ' 🔒' : ''}</button>
          ))}
          <div style={{ marginTop: 16, padding: '0 10px', color: 'var(--ink2)', fontSize: 13 }}>
            {user ? (
              <>
                Signed in as <b>{user.username}</b> ({user.role})
                <div><button className="btn secondary" style={{ marginTop: 6 }} onClick={ctx.logout}>Log out</button></div>
              </>
            ) : (
              <button className="btn" onClick={() => go('login')}>Hospital login</button>
            )}
          </div>
        </nav>
        <main className="main">
          {page === 'login' || gated
            ? <Login onDone={() => go(entry[3] ? page : 'federation')} />
            : <Page key={page} />}
        </main>
      </div>
    </AuthContext.Provider>
  )
}
