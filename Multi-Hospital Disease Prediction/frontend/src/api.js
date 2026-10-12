import { useCallback, useEffect, useState } from 'react'

export const API_BASE = (import.meta.env.VITE_API_BASE_URL || 'http://localhost:8000').replace(/\/$/, '')

export class ApiError extends Error {
  constructor(message, status, fields) {
    super(message)
    this.status = status
    this.fields = fields || []
  }
}

export async function request(path, options) {
  let res
  try {
    res = await fetch(`${API_BASE}/api/v1${path}`, options)
  } catch {
    throw new ApiError('Cannot reach the backend. Is it running?', 0)
  }
  const body = await res.json().catch(() => null)
  if (!res.ok) {
    const e = body?.error
    throw new ApiError(e?.message || `Request failed (${res.status})`, res.status, e?.fields)
  }
  return body
}

/** Fetch hook with loading/error state and manual retry. */
export function useApi(path) {
  const [state, setState] = useState({ data: null, error: null, loading: true })
  const [n, setN] = useState(0)
  useEffect(() => {
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
