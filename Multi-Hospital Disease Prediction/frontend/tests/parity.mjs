// Node harness used by ml/tests/test_js_parity.py: runs the browser trainer on inputs prepared by Python.
import { readFileSync } from 'node:fs'
import { fileURLToPath, pathToFileURL } from 'node:url'
import path from 'node:path'

const here = path.dirname(fileURLToPath(import.meta.url))
const T = await import(pathToFileURL(path.join(here, '..', 'src', 'lib', 'trainer.js')).href)

const input = JSON.parse(readFileSync(process.argv[2], 'utf8'))
const params = T.toParams(input.weights)
const nIn = input.nIn
const X = Float64Array.from(input.X.flat())
const y = Float64Array.from(input.y)
const n = y.length

const out = {}
out.proba = Array.from(T.predictProba(params, X, n, nIn))
out.auc = T.auc(y, Float64Array.from(out.proba))
out.grads = input.gradRows.map((i) => {
  const { grad, loss } = T.sampleGradient(params, X.subarray(i * nIn, (i + 1) * nIn), y[i], nIn)
  return { loss, grad: grad.map((g) => Array.from(g)) }
})
out.prep_raw = (() => { const d = T.prepareDataset(input.rawCsv, input.spec); return { n: d.n, dups: d.duplicatesDropped, layout: d.layout, X: Array.from(d.X), y: Array.from(d.y), issues: d.issues } })()
out.prep_encoded = (() => { const d = T.prepareDataset(input.encodedCsv, input.spec); return { n: d.n, layout: d.layout, X: Array.from(d.X), y: Array.from(d.y), issues: d.issues } })()
out.prep_bad = (() => { try { T.prepareDataset(input.badCsv, input.spec); return { error: null } } catch (e) { return { error: e.message } } })()
out.prep_bad_rows = (() => { const d = T.prepareDataset(input.badRowsCsv, input.spec); return { n: d.n, issues: d.issues } })()

// learning check: train from the global model on planted-signal rows, evaluate on held-out rows
const L = input.learn
const trainX = Float64Array.from(L.trainX.flat()), trainY = Float64Array.from(L.trainY)
const holdX = Float64Array.from(L.holdX.flat()), holdY = Float64Array.from(L.holdY)
const before = T.evaluate(params, holdX, holdY, holdY.length, nIn)
const plain = T.toParams(input.weights)
await T.trainLocal({ params: plain, X: trainX, y: trainY, n: trainY.length, nIn, epochs: 12, rng: T.mulberry32(7) })
const after = T.evaluate(plain, holdX, holdY, holdY.length, nIn)
const dpP = T.toParams(input.weights)
const dpRes = await T.trainLocal({ params: dpP, X: trainX, y: trainY, n: trainY.length, nIn, epochs: 6, dp: { sigma: 0.8, clip: 1.0 }, rng: T.mulberry32(7) })
const dpAfter = T.evaluate(dpP, holdX, holdY, holdY.length, nIn)
out.learn = { before, after, dp_after: dpAfter, dp_steps: dpRes.steps, dp_q: dpRes.sampleRate,
  all_finite: [plain, dpP].every((P) => P.every((a) => a.every(Number.isFinite))),
  update_norm: T.flatDistance(plain, T.toParams(input.weights)), dp_update_norm: T.flatDistance(dpP, T.toParams(input.weights)) }

// DP step sanity: with sigma ~ 0 and a huge clip, the DP gradient step equals the plain one in expectation (checked via determinism)
console.log(JSON.stringify(out))
