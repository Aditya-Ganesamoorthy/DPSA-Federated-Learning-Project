import { useEffect, useMemo, useState } from 'react'
import { AuthContext, loadSession, logoutRequest, useApi } from './api'
import { Badge, Icon } from './components/ui'
import Overview from './pages/Overview'
import Data from './pages/Data'
import Training from './pages/Training'
import Comparison from './pages/Comparison'
import Privacy from './pages/Privacy'
import Communication from './pages/Communication'
import Federation from './pages/Federation'
import Train from './pages/Train'
import Login from './pages/Login'

// [key, label, component, requiresLogin, icon, group]
const PAGES = [
  ['overview', 'Overview', Overview, false, 'shield', 'Network'],
  ['federation', 'Live Federation', Federation, true, 'network', 'Network'],
  ['train', 'Train & Contribute', Train, true, 'upload', 'Network'],
  ['data', 'Data & Hospitals', Data, false, 'database', 'Research results'],
  ['training', 'Training', Training, false, 'activity', 'Research results'],
  ['comparison', 'Comparison', Comparison, false, 'scale', 'Research results'],
  ['privacy', 'Privacy Experiments', Privacy, false, 'lock', 'Research results'],
  ['communication', 'Communication', Communication, false, 'chart', 'Research results'],
]

export default function App() {
  const [user, setUser] = useState(() => loadSession())
  const [page, setPage] = useState(() => {
    const h = window.location.hash.slice(1)
    return PAGES.some(([k]) => k === h) || h === 'login' || h === 'register' ? h : 'overview'
  })
  const health = useApi('/health')
  const ctx = useMemo(() => ({ user, setUser, logout: () => { logoutRequest(); setUser(null) } }), [user])

  useEffect(() => {
    const out = () => setUser(null)
    window.addEventListener('fl-logout', out)
    return () => window.removeEventListener('fl-logout', out)
  }, [])
  useEffect(() => {
    const onHash = () => { const h = window.location.hash.slice(1); if (PAGES.some(([k]) => k === h) || h === 'login' || h === 'register') setPage(h) }
    window.addEventListener('hashchange', onHash)
    return () => window.removeEventListener('hashchange', onHash)
  }, [])

  const entry = PAGES.find(([k]) => k === page)
  const Page = entry ? entry[2] : Overview
  const gated = entry && entry[3] && !user
  const go = (k) => { setPage(k); window.location.hash = k; window.scrollTo({ top: 0 }) }
  const groups = [...new Set(PAGES.map((p) => p[5]))]
  const authPage = page === 'login' || page === 'register' || gated

  return (
    <AuthContext.Provider value={ctx}>
      <div className="shell">
        <aside className="sidebar">
          <div className="brand">
            <div className="brand-row">
              <div className="brand-mark" aria-hidden="true">
                <svg width="26" height="26" viewBox="0 0 24 24" fill="none" stroke="#4a2d0f" strokeWidth="2.2" strokeLinecap="round" strokeLinejoin="round">
                  <path d="M12 21s-7-4.4-7-10a4 4 0 017-2.6A4 4 0 0119 11c0 5.6-7 10-7 10z" /><path d="M5 13h3l2-3 3 5 2-2h4" />
                </svg>
              </div>
              <div><h1>Federated Health Network</h1><small>Privacy-preserving learning</small></div>
            </div>
          </div>
          <nav className="nav" aria-label="Pages">
            {groups.map((g) => (
              <div key={g}>
                <div className="nav-label">{g}</div>
                {PAGES.filter((p) => p[5] === g).map(([k, label, , lock, icon]) => (
                  <button key={k} aria-current={page === k ? 'page' : undefined} onClick={() => go(k)}>
                    <Icon name={icon} size={18} />{label}{lock && !user && <span className="lock" aria-label="login required">🔒</span>}
                  </button>
                ))}
              </div>
            ))}
          </nav>
          <div className="side-foot">Hospital data never leaves the hospital. Only model weights are shared.</div>
        </aside>

        <div className="content">
          {health.error && (
            <div className="banner" role="alert">Server unreachable: {health.error.message}
              <button className="btn secondary sm" onClick={health.retry}>Retry</button></div>
          )}
          {health.data && !health.data.results_loaded && (
            <div className="banner warn" role="alert">Results are missing. Run <span className="mono">python -m ml.run_all</span> first.</div>
          )}
          <header className="topbar">
            <div className="grow">
              {health.data ? <Badge tone="good" dot>Server online · global model v{health.data.global_model_version}</Badge> : <Badge>Connecting…</Badge>}
            </div>
            {user ? (
              <div className="userchip">
                <span className="avatar" aria-hidden="true">{user.username.slice(0, 1).toUpperCase()}</span>
                <div><b>{user.username}</b><span className="role">{user.role}</span></div>
                <button className="btn ghost sm" onClick={ctx.logout}><Icon name="logout" size={15} />Log out</button>
              </div>
            ) : (
              <>
                <button className="btn secondary sm" onClick={() => go('login')}><Icon name="login" size={15} />Sign in</button>
                <button className="btn sm" onClick={() => go('register')}>Register hospital</button>
              </>
            )}
          </header>
          <main className="main">
            {authPage
              ? <Login key={page} initialMode={page === 'register' ? 'register' : 'login'} onDone={() => go(entry && entry[3] ? entry[0] : 'federation')} />
              : <Page key={page} />}
          </main>
        </div>
      </div>
    </AuthContext.Provider>
  )
}
