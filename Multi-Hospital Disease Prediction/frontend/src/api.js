import { createContext, useCallback, useContext, useEffect, useState } from 'react'

// Production build is served by the same FastAPI service, so default to the same origin.
const configured = import.meta.env.VITE_API_BASE_URL
export const API_BASE = (configured ?? (import.meta.env.PROD ? '' : 'http://localhost:8000')).replace(/\/$/, '')

export class ApiError extends Error {
  constructor(message, status, fields, body) {
    super(message)
    this.status = status
    this.fields = fields || []
    this.body = body
  }
}

// ---- session (token kept in sessionStorage as a per-tab convenience; the page works without it) ----
const KEY = 'fl_session'
export function loadSession() {
  try { return JSON.parse(sessionStorage.getItem(KEY)) } catch { return null }
}
function saveSession(s) {
  try { s ? sessionStorage.setItem(KEY, JSON.stringify(s)) : sessionStorage.removeItem(KEY) } catch { /* private mode */ }
}
let session = loadSession()

export const AuthContext = createContext({ user: null, login: async () => {}, logout: () => {} })
export const useAuth = () => useContext(AuthContext)

export async function loginRequest(username, password) {
  const data = await request('/auth/login', {
    method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ username, password }),
  })
  session = { token: data.access_token, username: data.username, role: data.role }
  saveSession(session)
  return session
}
export async function registerRequest(form) {
  const data = await request('/auth/register', {
    method: 'POST', headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ hospital_name: form.hospitalName, parent_network: form.parent || '', username: form.username,
      password: form.password, invite_code: form.invite || '' }),
  })
  session = { token: data.access_token, username: data.username, role: data.role }
  saveSession(session)
  return session
}
export function logoutRequest() { session = null; saveSession(null) }

export async function request(path, options = {}) {
  const headers = { ...(options.headers || {}) }
  if (session?.token) headers.Authorization = `Bearer ${session.token}`
  let res
  try {
    res = await fetch(`${API_BASE}/api/v1${path}`, { ...options, headers })
  } catch {
    throw new ApiError('Cannot reach the server. If it was idle, it may be waking up: retry in a minute.', 0)
  }
  if (options.raw && res.ok) return res
  const body = await res.json().catch(() => null)
  if (!res.ok) {
    if (res.status === 401 && session) { logoutRequest(); window.dispatchEvent(new Event('fl-logout')) }
    const e = body?.error
    throw new ApiError(e?.message || `Request failed (${res.status})`, res.status, e?.fields, body)
  }
  return body
}

/** Fetch hook with loading/error state and manual retry. Pass null to skip. */
export function useApi(path) {
  const [state, setState] = useState({ data: null, error: null, loading: !!path })
  const [n, setN] = useState(0)
  useEffect(() => {
    if (!path) { setState({ data: null, error: null, loading: false }); return undefined }
    let live = true
    setState((s) => ({ ...s, loading: true, error: null }))
    request(path)
      .then((data) => live && setState({ data, error: null, loading: false }))
      .catch((error) => live && setState({ data: null, error, loading: false }))
    return () => { live = false }
  }, [path, n])
  const retry = useCallback(() => setN((x) => x + 1), [])
  return { ...state, retry }
}
