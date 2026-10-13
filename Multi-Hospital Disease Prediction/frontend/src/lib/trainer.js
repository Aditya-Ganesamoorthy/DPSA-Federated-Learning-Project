// In-browser federated-learning client: CSV preprocessing, a small MLP, local training (optionally DP-SGD) and metrics.
// Everything here runs on the hospital's own machine; the CSV is never uploaded. No dependencies (runs in Node too).
//
// Model layout matches the server / numpy / torch exactly (arrays in torch state_dict order):
//   p0 W0 (32x15)  p1 b0 (32)  p2 W2 (16x32)  p3 b2 (16)  p4 W4 (1x16)  p5 b4 (1)      x -> relu -> relu -> logit

const H1 = 32, H2 = 16

export function shapesFor(nIn) {
  return [[H1, nIn], [H1], [H2, H1], [H2], [1, H2], [1]]
}

export function toParams(flatArrays) {
  return flatArrays.map((a) => Float64Array.from(a))
}

export function toPlain(params) {
  return params.map((p) => Array.from(p, (v) => Math.fround(v)))
}

function cloneZeros(params) {
  return params.map((p) => new Float64Array(p.length))
}

// ---------------------------------------------------------------- forward / backward
function forward(params, x, nIn) {
  const [W0, b0, W2, b2, W4, b4] = params
  const a1 = new Float64Array(H1), h1 = new Float64Array(H1)
  for (let j = 0; j < H1; j++) {
    let s = b0[j]
    const o = j * nIn
    for (let i = 0; i < nIn; i++) s += W0[o + i] * x[i]
    a1[j] = s
    h1[j] = s > 0 ? s : 0
  }
  const a2 = new Float64Array(H2), h2 = new Float64Array(H2)
  for (let j = 0; j < H2; j++) {
    let s = b2[j]
    const o = j * H1
    for (let k = 0; k < H1; k++) s += W2[o + k] * h1[k]
    a2[j] = s
    h2[j] = s > 0 ? s : 0
  }
  let z = b4[0]
  for (let j = 0; j < H2; j++) z += W4[j] * h2[j]
  return { a1, h1, a2, h2, z }
}

const sigmoid = (z) => 1 / (1 + Math.exp(-z))

export function lossFromLogit(z, y) {
  return Math.max(z, 0) - y * z + Math.log1p(Math.exp(-Math.abs(z)))
}

/** Gradient of the BCE-with-logits loss for ONE sample, accumulated into `g` (same layout as params). Returns loss. */
export function accumulateSampleGrad(params, x, y, nIn, g, scale = 1) {
  const [, , W2, , W4] = params
  const f = forward(params, x, nIn)
  const dz = (sigmoid(f.z) - y) * scale
  g[5][0] += dz
  const dh2pre = new Float64Array(H2)
  for (let j = 0; j < H2; j++) {
    g[4][j] += dz * f.h2[j]
    dh2pre[j] = f.a2[j] > 0 ? dz * W4[j] : 0
  }
  const dh1 = new Float64Array(H1)
  for (let j = 0; j < H2; j++) {
    const d = dh2pre[j]
    g[3][j] += d
    const o = j * H1
    for (let k = 0; k < H1; k++) {
      g[2][o + k] += d * f.h1[k]
      dh1[k] += d * W2[o + k]
    }
  }
  for (let k = 0; k < H1; k++) {
    const d = f.a1[k] > 0 ? dh1[k] : 0
    g[1][k] += d
    const o = k * nIn
    for (let i = 0; i < nIn; i++) g[0][o + i] += d * x[i]
  }
  return lossFromLogit(f.z, y)
}

export function sampleGradient(params, x, y, nIn) {
  const g = cloneZeros(params)
  const loss = accumulateSampleGrad(params, x, y, nIn, g)
  return { grad: g, loss }
}

// ---------------------------------------------------------------- metrics
export function predictProba(params, X, n, nIn) {
  const out = new Float64Array(n)
  for (let r = 0; r < n; r++) out[r] = sigmoid(forward(params, X.subarray(r * nIn, (r + 1) * nIn), nIn).z)
  return out
}

/** Rank-based ROC-AUC with tie handling; NaN when only one class is present. */
export function auc(y, p) {
  const n = y.length
  let n1 = 0
  for (let i = 0; i < n; i++) n1 += y[i] > 0.5 ? 1 : 0
  const n0 = n - n1
  if (n1 === 0 || n0 === 0) return NaN
  const order = Array.from({ length: n }, (_, i) => i).sort((a, b) => p[a] - p[b])
  const ranks = new Float64Array(n)
  let i = 0
  while (i < n) {
    let j = i
    while (j + 1 < n && p[order[j + 1]] === p[order[i]]) j++
    const r = (i + j) / 2 + 1
    for (let k = i; k <= j; k++) ranks[order[k]] = r
    i = j + 1
  }
  let sum = 0
  for (let k = 0; k < n; k++) if (y[k] > 0.5) sum += ranks[k]
  return (sum - (n1 * (n1 + 1)) / 2) / (n1 * n0)
}

export function evaluate(params, X, y, n, nIn) {
  const p = predictProba(params, X, n, nIn)
  let correct = 0, tp = 0, pos = 0, loss = 0
  for (let i = 0; i < n; i++) {
    const pred = p[i] >= 0.5 ? 1 : 0
    correct += pred === y[i] ? 1 : 0
    if (y[i] > 0.5) { pos++; tp += pred }
    const pc = Math.min(Math.max(p[i], 1e-7), 1 - 1e-7)
    loss += -(y[i] * Math.log(pc) + (1 - y[i]) * Math.log(1 - pc))
  }
  return { accuracy: correct / n, recall: pos ? tp / pos : 0, auc: auc(y, p), loss: loss / n }
}

// ---------------------------------------------------------------- randomness
export function mulberry32(seed) {
  let a = seed >>> 0
  return () => {
    a = (a + 0x6d2b79f5) >>> 0
    let t = a
    t = Math.imul(t ^ (t >>> 15), t | 1)
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61)
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296
  }
}

/** Uniform (0,1) from the browser's cryptographic RNG: used for DP noise and Poisson sampling by default. */
export function cryptoUniform() {
  const c = globalThis.crypto
  const buf = new Uint32Array(1)
  c.getRandomValues(buf)
  return (buf[0] + 0.5) / 4294967296
}

export function gaussian(rng) {
  const u = rng(), v = rng()
  return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * v)
}

function shuffle(n, rng) {
  const idx = Uint32Array.from({ length: n }, (_, i) => i)
  for (let i = n - 1; i > 0; i--) {
    const j = Math.floor(rng() * (i + 1))
    const t = idx[i]; idx[i] = idx[j]; idx[j] = t
  }
  return idx
}

const tick = () => new Promise((r) => setTimeout(r, 0))

// ---------------------------------------------------------------- local training
/**
 * Train `params` in place on (X, y). Plain minibatch SGD with momentum (same recipe as the research code), or DP-SGD:
 * Poisson sampling at rate q = batch/n, per-sample gradient clipping to `clip`, Gaussian noise sigma*clip on the summed
 * gradient, division by the expected batch size. Returns what the privacy accountant needs.
 */
export async function trainLocal({ params, X, y, n, nIn, epochs, lr = 0.05, momentum = 0.9, batchSize = 32,
  dp = null, rng = null, onProgress = null, shouldCancel = null }) {
  const rand = rng || cryptoUniform
  const velocity = cloneZeros(params)
  const stepsPerEpoch = Math.ceil(n / batchSize)
  const totalSteps = epochs * stepsPerEpoch
  const q = Math.min(1, batchSize / n)
  let step = 0, lastLoss = NaN
  for (let epoch = 1; epoch <= epochs; epoch++) {
    const order = dp ? null : shuffle(n, rand)
    let lossSum = 0, lossCount = 0
    for (let s = 0; s < stepsPerEpoch; s++) {
      if (shouldCancel && shouldCancel()) throw new Error('cancelled')
      const g = cloneZeros(params)
      let denom
      if (dp) {
        denom = q * n                                           // expected batch size (what Opacus divides by)
        for (let i = 0; i < n; i++) {
          if (rand() >= q) continue                              // Poisson sampling
          const gi = cloneZeros(params)
          lossSum += accumulateSampleGrad(params, X.subarray(i * nIn, (i + 1) * nIn), y[i], nIn, gi)
          lossCount++
          let sq = 0
          for (const a of gi) for (let k = 0; k < a.length; k++) sq += a[k] * a[k]
          const scale = Math.min(1, dp.clip / (Math.sqrt(sq) + 1e-6))
          for (let t = 0; t < g.length; t++) for (let k = 0; k < g[t].length; k++) g[t][k] += gi[t][k] * scale
        }
        for (const a of g) for (let k = 0; k < a.length; k++) a[k] += gaussian(rand) * dp.sigma * dp.clip
      } else {
        const lo = s * batchSize, hi = Math.min(n, lo + batchSize)
        denom = hi - lo
        for (let b = lo; b < hi; b++) {
          const i = order[b]
          lossSum += accumulateSampleGrad(params, X.subarray(i * nIn, (i + 1) * nIn), y[i], nIn, g)
          lossCount++
        }
      }
      for (let t = 0; t < params.length; t++) {
        const p = params[t], v = velocity[t], gt = g[t]
        for (let k = 0; k < p.length; k++) {
          v[k] = momentum * v[k] + gt[k] / denom
          p[k] -= lr * v[k]
        }
      }
      step++
      if (step % 3 === 0) {
        if (onProgress) onProgress({ epoch, epochs, step, totalSteps, loss: lossCount ? lossSum / lossCount : lastLoss })
        await tick()
      }
    }
    lastLoss = lossCount ? lossSum / lossCount : lastLoss
    if (onProgress) onProgress({ epoch, epochs, step, totalSteps, loss: lastLoss, epochDone: true })
    await tick()
  }
  return { loss: lastLoss, steps: totalSteps, sampleRate: q }
}

export function flatDistance(a, b) {
  let s = 0
  for (let t = 0; t < a.length; t++) for (let k = 0; k < a[t].length; k++) s += (a[t][k] - b[t][k]) ** 2
  return Math.sqrt(s)
}

// ---------------------------------------------------------------- CSV -> model inputs
export function parseCSV(text) {
  const rows = []
  let row = [], field = '', inQ = false
  const t = text.charCodeAt(0) === 0xfeff ? text.slice(1) : text
  for (let i = 0; i < t.length; i++) {
    const c = t[i]
    if (inQ) {
      if (c === '"') { if (t[i + 1] === '"') { field += '"'; i++ } else inQ = false } else field += c
    } else if (c === '"') inQ = true
    else if (c === ',') { row.push(field); field = '' }
    else if (c === '\n' || c === '\r') {
      if (c === '\r' && t[i + 1] === '\n') i++
      row.push(field); field = ''
      if (row.length > 1 || row[0] !== '') rows.push(row)
      row = []
    } else field += c
  }
  if (field !== '' || row.length) { row.push(field); if (row.length > 1 || row[0] !== '') rows.push(row) }
  return rows
}

const norm = (h) => h.trim().replace(/^﻿/, '').toLowerCase().replace(/\s+/g, '_')

/**
 * Turn CSV text into model inputs using the server's schema (GET /features). Accepts the raw Kaggle layout
 * (GENDER M/F, AGE, survey fields coded 1/2, LUNG_CANCER YES/NO) or the project's encoded layout (numeric 0/1, age/100).
 */
export function prepareDataset(text, spec) {
  const rows = parseCSV(text)
  if (rows.length < 2) throw new Error('The file has no data rows.')
  const header = rows[0].map(norm)
  const need = [...spec.features, spec.target]
  const missing = need.filter((c) => !header.includes(c))
  if (missing.length) throw new Error(`Missing columns: ${missing.join(', ')}. Expected: ${need.join(', ')}`)
  const col = Object.fromEntries(header.map((h, i) => [h, i]))
  const body = rows.slice(1)
  const maxRows = spec.limits.max_rows
  if (body.length > maxRows) throw new Error(`Too many rows (${body.length}); the limit is ${maxRows}.`)
  const rawLayout = Number.isNaN(Number(body[0][col.gender])) // M/F text means the raw Kaggle layout
  const nIn = spec.features.length
  const seen = new Set()
  const X = [], Y = []
  const issues = []
  let dups = 0
  const bin = new Set(spec.binary_features)
  for (let r = 0; r < body.length; r++) {
    const rec = body[r]
    const line = r + 2
    if (rec.length < header.length) { issues.push(`Row ${line}: expected ${header.length} columns, found ${rec.length}`); continue }
    if (rawLayout) {
      const key = need.map((c) => rec[col[c]].trim()).join('|')
      if (seen.has(key)) { dups++; continue }
      seen.add(key)
    }
    const x = new Float64Array(nIn)
    let bad = null
    spec.features.forEach((f, i) => {
      const v = rec[col[f]].trim()
      if (f === 'gender') {
        if (rawLayout) { const m = spec.gender_map[v.toUpperCase()]; if (m === undefined) bad = `gender must be M or F (got "${v}")`; else x[i] = m } else {
          x[i] = Number(v); if (x[i] !== 0 && x[i] !== 1) bad = `gender must be 0 or 1 (got "${v}")` }
      } else if (f === 'age') {
        const a = Number(v)
        if (!Number.isFinite(a)) bad = `age is not a number ("${v}")`
        else if (rawLayout) { if (a < spec.age_range[0] || a > spec.age_range[1]) bad = `age out of range (${a})`; else x[i] = a / spec.age_scale }
        else { if (a < 0 || a > 1.5) bad = `encoded age must be age/100 (got ${a})`; else x[i] = a }
      } else if (bin.has(f)) {
        const c = Number(v)
        if (rawLayout) { if (c !== spec.survey_codes.no && c !== spec.survey_codes.yes) bad = `${f} must be 1 or 2 (got "${v}")`; else x[i] = c === spec.survey_codes.yes ? 1 : 0 }
        else { if (c !== 0 && c !== 1) bad = `${f} must be 0 or 1 (got "${v}")`; else x[i] = c }
      }
    })
    const tv = rec[col[spec.target]].trim()
    let yv
    if (rawLayout) { yv = spec.target_map[tv.toUpperCase()]; if (yv === undefined) bad = bad || `${spec.target} must be YES or NO (got "${tv}")` }
    else { yv = Number(tv); if (yv !== 0 && yv !== 1) bad = bad || `${spec.target} must be 0 or 1 (got "${tv}")` }
    if (bad) { issues.push(`Row ${line}: ${bad}`); continue }
    X.push(x); Y.push(yv)
  }
  const n = X.length
  const flat = new Float64Array(n * nIn)
  X.forEach((x, r) => flat.set(x, r * nIn))
  let positives = 0
  for (const v of Y) positives += v
  return { X: flat, y: Float64Array.from(Y), n, nIn, positives, duplicatesDropped: dups, issues, layout: rawLayout ? 'raw' : 'encoded',
    skipped: body.length - dups - n }
}

/** Deterministic holdout split (seeded) so before/after metrics are on rows the model did not train on. */
export function split(data, holdoutFraction, seed = 1) {
  const rng = mulberry32(seed)
  const idx = shuffle(data.n, rng)
  const nHold = Math.floor(data.n * holdoutFraction)
  const pick = (ids) => {
    const X = new Float64Array(ids.length * data.nIn), y = new Float64Array(ids.length)
    ids.forEach((id, r) => { X.set(data.X.subarray(id * data.nIn, (id + 1) * data.nIn), r * data.nIn); y[r] = data.y[id] })
    return { X, y, n: ids.length, nIn: data.nIn }
  }
  return { train: pick(Array.from(idx.slice(nHold))), hold: pick(Array.from(idx.slice(0, nHold))) }
}
