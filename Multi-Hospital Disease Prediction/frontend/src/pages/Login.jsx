import { useState } from 'react'
import { loginRequest, registerRequest, useApi, useAuth } from '../api'
import { Icon, PageHeader } from '../components/ui'

function Field({ label, hint, error, children }) {
  return (
    <label className="f">{label}
      {children}
      {hint && !error && <span className="field-hint">{hint}</span>}
      {error && <span className="err-msg" role="alert">{error}</span>}
    </label>
  )
}

export default function Login({ onDone, initialMode = 'login' }) {
  const { setUser } = useAuth()
  const health = useApi('/health')
  const needsInvite = health.data?.registration === 'invite'
  const [mode, setMode] = useState(initialMode)
  const [f, setF] = useState({ hospitalName: '', parent: '', username: '', password: '', confirm: '', invite: '' })
  const [err, setErr] = useState(null)
  const [fieldErr, setFieldErr] = useState({})
  const [busy, setBusy] = useState(false)
  const set = (k) => (e) => setF({ ...f, [k]: e.target.value })

  const validate = () => {
    const e = {}
    if (mode === 'register') {
      if (f.hospitalName.trim().length < 3) e.hospitalName = 'Enter your hospital name (at least 3 characters)'
      if (!/^[a-z0-9][a-z0-9-]{2,30}[a-z0-9]$/.test(f.username.trim().toLowerCase())) e.username = '4-32 characters: lowercase letters, digits and hyphens'
      if (f.password.length < 8) e.password = 'Use at least 8 characters'
      else if (f.password.toLowerCase() === f.username.trim().toLowerCase()) e.password = 'Password must differ from the username'
      if (f.confirm !== f.password) e.confirm = 'Passwords do not match'
      if (needsInvite && !f.invite.trim()) e.invite = 'This network requires an invite code'
    } else {
      if (!f.username.trim()) e.username = 'Enter your username'
      if (!f.password) e.password = 'Enter your password'
    }
    return e
  }

  const submit = async (ev) => {
    ev.preventDefault()
    const e = validate()
    setFieldErr(e); setErr(null)
    if (Object.keys(e).length) return
    setBusy(true)
    try {
      const s = mode === 'login' ? await loginRequest(f.username.trim(), f.password) : await registerRequest({ ...f, username: f.username.trim() })
      setUser(s)
      onDone?.(s)
    } catch (ex) { setErr(ex.message) } finally { setBusy(false) }
  }

  return (
    <>
      <PageHeader eyebrow="Hospital access" title={mode === 'login' ? 'Sign in to your hospital' : 'Join the network'}>
        {mode === 'login'
          ? 'Sign in to download the global model, train on your own data and contribute to the next version.'
          : 'Hospitals and sub-hospitals register themselves. Your patient data stays with you: only model weights are ever shared.'}
      </PageHeader>
      <div className="grid two" style={{ alignItems: 'start' }}>
        <form className="card" onSubmit={submit} noValidate aria-label={mode === 'login' ? 'Sign in' : 'Register'}>
          <div className="seg" role="tablist" aria-label="Access mode" style={{ marginBottom: 18 }}>
            <button type="button" role="tab" aria-selected={mode === 'login'} onClick={() => { setMode('login'); setErr(null); setFieldErr({}) }}>Sign in</button>
            <button type="button" role="tab" aria-selected={mode === 'register'} onClick={() => { setMode('register'); setErr(null); setFieldErr({}) }}>Register hospital</button>
          </div>
          <div style={{ display: 'grid', gap: 14 }}>
            {mode === 'register' && (
              <>
                <Field label="Hospital name" error={fieldErr.hospitalName}>
                  <input type="text" value={f.hospitalName} onChange={set('hospitalName')} autoComplete="organization" placeholder="e.g. City General Hospital" />
                </Field>
                <Field label="Parent hospital or network (optional)" hint="Fill this in if you are a sub-hospital or branch.">
                  <input type="text" value={f.parent} onChange={set('parent')} placeholder="e.g. State Health Network" />
                </Field>
              </>
            )}
            <Field label="Username" error={fieldErr.username} hint={mode === 'register' ? 'Lowercase letters, digits and hyphens. Shown publicly in the round history; avoid revealing the institution if that matters.' : undefined}>
              <input type="text" value={f.username} onChange={set('username')} autoComplete="username" placeholder="e.g. city-general" />
            </Field>
            <Field label="Password" error={fieldErr.password}>
              <input type="password" value={f.password} onChange={set('password')} autoComplete={mode === 'login' ? 'current-password' : 'new-password'} />
            </Field>
            {mode === 'register' && (
              <Field label="Confirm password" error={fieldErr.confirm}>
                <input type="password" value={f.confirm} onChange={set('confirm')} autoComplete="new-password" />
              </Field>
            )}
            {mode === 'register' && (needsInvite || f.invite) && (
              <Field label="Invite code" error={fieldErr.invite}>
                <input type="text" value={f.invite} onChange={set('invite')} />
              </Field>
            )}
          </div>
          {err && <div className="err" role="alert" style={{ marginTop: 14 }}>{err}</div>}
          <div className="row">
            <button className="btn" disabled={busy}>{busy ? 'Please wait…' : mode === 'login' ? 'Sign in' : 'Create hospital account'}</button>
          </div>
        </form>

        <div className="card soft">
          <h4>Why a login?</h4>
          <div style={{ display: 'grid', gap: 14, marginTop: 6 }}>
            {[['shield', 'Only registered hospitals can send updates', 'Every update is tied to an account, checked and logged.'],
              ['lock', 'Your data never leaves your browser', 'Training happens on your machine. We receive weights, not patients.'],
              ['key', 'Passwords are never stored', 'We keep a salted, server-peppered hash only.'],
              ['eye', 'Everything is auditable', 'The Overview dashboard shows every received and rejected update.']].map(([ic, t, d]) => (
              <div key={t} style={{ display: 'flex', gap: 12 }}>
                <span className="avatar" style={{ borderRadius: 10, flexShrink: 0 }}><Icon name={ic} size={16} /></span>
                <div><b>{t}</b><div className="small muted">{d}</div></div>
              </div>
            ))}
          </div>
          {mode === 'register' && !needsInvite && (
            <p className="small muted" style={{ marginTop: 16 }}>Registration is open on this server. The network protects itself with update checks, optional median aggregation and an audit log; an administrator can suspend any account.</p>
          )}
        </div>
      </div>
    </>
  )
}
