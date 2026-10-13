import { useState } from 'react'
import { loginRequest, useAuth } from '../api'
import { PageHeader } from '../components/ui'

export default function Login({ onDone }) {
  const { setUser } = useAuth()
  const [username, setUsername] = useState('')
  const [password, setPassword] = useState('')
  const [err, setErr] = useState(null)
  const [busy, setBusy] = useState(false)

  const submit = async (e) => {
    e.preventDefault()
    setErr(null); setBusy(true)
    try {
      setUser(await loginRequest(username, password))
      onDone?.()
    } catch (ex) { setErr(ex.message) } finally { setBusy(false) }
  }

  return (
    <>
      <PageHeader title="Hospital login">
        Sign in with the credentials issued to your hospital by the administrator. Logged-in hospitals can download the
        global model, take part in federated rounds and use the prediction page.
      </PageHeader>
      <form className="card" style={{ maxWidth: 420 }} onSubmit={submit} noValidate>
        <label className="f">Hospital ID (username)
          <input type="text" value={username} onChange={(e) => setUsername(e.target.value)} autoComplete="username" required
                 style={{ padding: 8, border: '1px solid var(--line)', borderRadius: 6, background: 'var(--surface)', color: 'var(--ink)' }} />
        </label>
        <label className="f" style={{ marginTop: 10 }}>Password
          <input type="password" value={password} onChange={(e) => setPassword(e.target.value)} autoComplete="current-password" required
                 style={{ padding: 8, border: '1px solid var(--line)', borderRadius: 6, background: 'var(--surface)', color: 'var(--ink)' }} />
        </label>
        {err && <p role="alert" style={{ color: 'var(--crit)' }}>{err}</p>}
        <div className="row"><button className="btn" disabled={busy || !username || !password}>{busy ? 'Signing in…' : 'Sign in'}</button></div>
        <p className="lede" style={{ fontSize: 13 }}>No account? Ask the administrator to create one for your hospital. Repeated failed attempts are temporarily locked.</p>
      </form>
    </>
  )
}
