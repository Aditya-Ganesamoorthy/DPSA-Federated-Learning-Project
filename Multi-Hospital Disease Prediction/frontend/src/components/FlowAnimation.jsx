import { Spinner } from './ui'

export const PHASE_TEXT = {
  idle: 'Ready. Waiting for the next federated round.',
  downloading: 'Downloading the current global model from the server…',
  training: 'Training locally. Patient data never leaves this machine.',
  uploading: 'Uploading ONLY the updated weights (no patient records)…',
  waiting: 'Update received. Waiting for the other hospitals in this round…',
  aggregating: 'Server is averaging the hospital updates (FedAvg)…',
  publishing: 'Publishing the new global model to Hugging Face…',
  done: 'Round complete. The global model has been updated.',
  error: 'Something went wrong. See the message below.',
}
const BUSY = new Set(['downloading', 'training', 'uploading', 'aggregating', 'publishing'])

const DOWN = 'M392 128 C330 86 260 86 206 122'
const UP = 'M206 178 C260 214 330 214 392 172'
const PUB = 'M528 150 L722 150'

function Packet({ path, color = 'var(--gold)', dur = '1.7s', label }) {
  return (
    <g>
      <circle r="8" fill={color} stroke="#fff" strokeWidth="2">
        <animateMotion dur={dur} repeatCount="indefinite" path={path} />
      </circle>
      {label && <text fontSize="10" fontWeight="700" fill="var(--brown)" textAnchor="middle" dy="-13">
        {label}<animateMotion dur={dur} repeatCount="indefinite" path={path} /></text>}
    </g>
  )
}

export default function FlowAnimation({ phase = 'idle', progress = 0, loss = null, version = null, newVersion = null,
  pending = 0, needed = 2, hfStatus = 'not configured', others = [], youLabel = 'Your hospital', compact = false }) {
  const training = phase === 'training'
  const server = phase === 'aggregating' || phase === 'waiting' || phase === 'uploading' || phase === 'downloading'
  const hub = phase === 'publishing' || phase === 'done'
  const ringC = 2 * Math.PI * 46
  const frac = Math.min(1, needed ? pending / needed : 0)
  const trainC = 2 * Math.PI * 58
  const hfTone = hfStatus === 'ok' ? 'var(--good)' : hfStatus === 'error' ? 'var(--crit)' : 'var(--ink3)'
  const shown = others.slice(0, 4)

  return (
    <div>
      <svg className="flow" viewBox="0 0 920 300" role="img" aria-label={`Federated learning animation: ${PHASE_TEXT[phase]}`}
           style={compact ? { maxHeight: 230 } : undefined}>
        <defs>
          <linearGradient id="goldg" x1="0" x2="1" y1="0" y2="1"><stop offset="0" stopColor="#f0c14b" /><stop offset="1" stopColor="#b9780f" /></linearGradient>
          <linearGradient id="brownc" x1="0" x2="1" y1="0" y2="1"><stop offset="0" stopColor="#7a4d18" /><stop offset="1" stopColor="#4a2d0f" /></linearGradient>
          <filter id="soft" x="-20%" y="-20%" width="140%" height="140%"><feDropShadow dx="0" dy="4" stdDeviation="5" floodColor="#5c3a14" floodOpacity=".22" /></filter>
        </defs>

        {/* links */}
        <path d={DOWN} fill="none" stroke="var(--line)" strokeWidth="3" />
        <path d={UP} fill="none" stroke="var(--line)" strokeWidth="3" />
        <path d={PUB} fill="none" stroke="var(--line)" strokeWidth="3" />
        {phase === 'downloading' && <path d={DOWN} fill="none" stroke="var(--gold)" strokeWidth="3" className="dash" />}
        {phase === 'uploading' && <path d={UP} fill="none" stroke="var(--gold)" strokeWidth="3" className="dash" />}
        {phase === 'publishing' && <path d={PUB} fill="none" stroke="var(--gold)" strokeWidth="3" className="dash" />}
        {phase === 'downloading' && <Packet path={DOWN.replace('M392 128 C330 86 260 86 206 122', 'M392 128 C330 86 260 86 206 122')} label="model" />}
        {phase === 'uploading' && <Packet path={UP} label="weights only" />}
        {phase === 'publishing' && <Packet path={PUB} color="var(--s2)" dur="1.4s" label="new version" />}
        <text x="299" y="70" fontSize="11" fill="var(--ink3)" textAnchor="middle">download model</text>
        <text x="299" y="246" fontSize="11" fill="var(--ink3)" textAnchor="middle">upload weights</text>
        <text x="625" y="136" fontSize="11" fill="var(--ink3)" textAnchor="middle">publish</text>

        {/* other hospitals */}
        {shown.map((o, i) => {
          const cy = 40 + i * 74
          return (
            <g key={o.name + i} className="node">
              <path d={`M64 ${cy} C 90 ${cy}, 90 150, 96 150`} fill="none" stroke={o.pending ? 'var(--gold)' : 'var(--line)'} strokeWidth="2" className={o.pending ? 'dash' : ''} />
              <circle cx="40" cy={cy} r="22" fill={o.pending ? '#fff3c9' : '#fff'} stroke={o.pending ? 'var(--gold)' : 'var(--line)'} strokeWidth="2" className={o.pending && phase === 'waiting' ? 'pulse' : ''} />
              <text x="40" y={cy + 4} fontSize="12" fontWeight="700" textAnchor="middle" fill="var(--brown)">{(o.name || '?').slice(0, 2).toUpperCase()}</text>
            </g>
          )
        })}

        {/* your hospital */}
        <g className="node" filter="url(#soft)">
          <circle cx="150" cy="150" r="64" fill="#fffaf0" stroke={training ? 'var(--gold)' : 'var(--line)'} strokeWidth="2" />
          {training && <circle cx="150" cy="150" r="58" fill="none" stroke="var(--line2)" strokeWidth="7" />}
          {training && <circle cx="150" cy="150" r="58" fill="none" stroke="url(#goldg)" strokeWidth="7" strokeLinecap="round"
            strokeDasharray={trainC} strokeDashoffset={trainC * (1 - Math.min(1, Math.max(0.02, progress)))} transform="rotate(-90 150 150)"
            style={{ transition: 'stroke-dashoffset .3s linear' }} />}
          {training && <circle cx="150" cy="150" r="70" fill="none" stroke="var(--gold)" strokeWidth="2" className="glow" />}
          {/* database cylinder */}
          <g transform="translate(150 134)" stroke="var(--brown)" strokeWidth="2.4" fill="#fff3c9">
            <ellipse cx="0" cy="-12" rx="22" ry="8" /><path d="M-22 -12v26c0 4.5 10 8 22 8s22-3.5 22-8v-26" /><path d="M-22 1c0 4.5 10 8 22 8s22-3.5 22-8" fill="none" />
          </g>
          <text x="150" y="190" fontSize="12.5" fontWeight="700" textAnchor="middle" fill="var(--brown)">{youLabel.length > 22 ? youLabel.slice(0, 21) + '…' : youLabel}</text>
          {/* lock badge: data stays here */}
          <g transform="translate(196 104)" className={training ? 'pulse' : ''}>
            <circle r="15" fill="#fff" stroke="var(--good)" strokeWidth="2" />
            <path d="M-5 0v-3a5 5 0 0110 0v3M-7 0h14v9h-14z" fill="var(--good-bg)" stroke="var(--good)" strokeWidth="2" strokeLinejoin="round" />
          </g>
          <text x="150" y="226" fontSize="11" textAnchor="middle" fill={training ? 'var(--good)' : 'var(--ink3)'} fontWeight={training ? 700 : 400}>
            {training ? `data stays local${loss != null ? ` · loss ${loss.toFixed(3)}` : ''}` : 'patient data stays here'}
          </text>
        </g>

        {/* server */}
        <g className="node" filter="url(#soft)">
          <circle cx="460" cy="150" r="66" fill="url(#brownc)" />
          <circle cx="460" cy="150" r="46" fill="none" stroke="rgba(255,243,201,.25)" strokeWidth="7" />
          <circle cx="460" cy="150" r="46" fill="none" stroke="url(#goldg)" strokeWidth="7" strokeLinecap="round" strokeDasharray={ringC}
                  strokeDashoffset={ringC * (1 - frac)} transform="rotate(-90 460 150)" style={{ transition: 'stroke-dashoffset .5s' }} />
          {(phase === 'aggregating') && <circle cx="460" cy="150" r="74" fill="none" stroke="var(--gold2)" strokeWidth="2" className="glow" />}
          <g transform="translate(460 150)" className={phase === 'aggregating' ? 'spin' : ''} fill="#ffeec2">
            <circle r="13" fill="none" stroke="#ffeec2" strokeWidth="3.4" /><circle r="4.5" />
            {[0, 45, 90, 135, 180, 225, 270, 315].map((a) => <rect key={a} x="-3" y="-21" width="6" height="8" rx="1.5" transform={`rotate(${a})`} />)}
          </g>
          <text x="460" y="236" fontSize="12.5" fontWeight="700" textAnchor="middle" fill="var(--brown)">Federated server</text>
          <text x="460" y="252" fontSize="11" textAnchor="middle" fill="var(--ink3)">
            {version != null ? `model v${version}` : 'global model'} · {pending}/{needed} hospitals reported
          </text>
        </g>

        {/* Hugging Face */}
        <g className={`node ${hub || phase === 'idle' || phase === 'waiting' ? '' : 'dim'}`} filter="url(#soft)">
          <circle cx="796" cy="150" r="64" fill="#fffaf0" stroke={hub ? 'var(--s2)' : 'var(--line)'} strokeWidth="2" />
          {phase === 'publishing' && <circle cx="796" cy="150" r="72" fill="none" stroke="var(--s2)" strokeWidth="2" className="glow" />}
          <path d="M770 160a16 16 0 01-1-31 22 22 0 0142-4 15 15 0 01-1 35z" fill="#e5eef7" stroke="var(--s2)" strokeWidth="2.4" strokeLinejoin="round" />
          {phase === 'done' && hfStatus !== 'error' && (
            <g className="ok" transform="translate(822 112)"><circle r="14" fill="var(--good)" /><path d="M-6 0l4 4 8-8" stroke="#fff" strokeWidth="3" fill="none" strokeLinecap="round" strokeLinejoin="round" /></g>
          )}
          <text x="796" y="190" fontSize="12.5" fontWeight="700" textAnchor="middle" fill="var(--brown)">Hugging Face</text>
          <text x="796" y="236" fontSize="11" textAnchor="middle" fill={hfTone} fontWeight="600">
            {newVersion != null && (phase === 'done' || phase === 'publishing') ? `v${newVersion} · ` : ''}{hfStatus === 'ok' ? 'synced' : hfStatus === 'error' ? 'sync failed' : hfStatus === 'pending' ? 'syncing…' : hfStatus}
          </text>
        </g>
      </svg>
      <div className="caption" role="status" aria-live="polite">{BUSY.has(phase) && <Spinner />}{PHASE_TEXT[phase]}</div>
    </div>
  )
}
