export const NA = 'n/a'

export const f3 = (v) => (v === null || v === undefined || Number.isNaN(v) ? NA : Number(v).toFixed(3))
export const f2 = (v) => (v === null || v === undefined || Number.isNaN(v) ? NA : Number(v).toFixed(2))
export const pct = (v) => (v === null || v === undefined || Number.isNaN(v) ? NA : `${(v * 100).toFixed(1)}%`)
export const pm = (m) => (!m || m.mean === null ? NA : `${f3(m.mean)} ± ${f3(m.std)}`)
export const eps = (v) => (v === null || v === undefined ? 'No DP' : Number(v).toFixed(1))
export const mb = (bytes) => (bytes < 1e5 ? `${(bytes / 1e3).toFixed(1)} KB` : `${(bytes / 1e6).toFixed(2)} MB`)
