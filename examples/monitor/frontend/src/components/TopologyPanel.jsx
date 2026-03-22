import { useState } from 'react'
import { C } from '../theme'

const MAX_PARAM_MB = 50

// ── helpers ───────────────────────────────────────────────────────────────────

function deriveActivity(batches) {
  if (!batches?.length) return {}
  const latest = batches.reduce((a, b) => (b.batch_id > a.batch_id ? b : a), batches[0])
  const result = {}
  for (const [host, stats] of Object.entries(latest.workers || {})) {
    result[host] = { fwd_ms: stats.fwd_ms ?? null, bwd_ms: stats.bwd_ms ?? null }
  }
  return result
}

// ── sub-components ────────────────────────────────────────────────────────────

function MemBar({ mb, color }) {
  if (!mb) return null
  const pct = Math.min(mb / MAX_PARAM_MB * 100, 100)
  return (
    <div style={{ marginTop: 6 }}>
      <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: 2 }}>
        <span style={{ fontSize: 9, color: C.muted, textTransform: 'uppercase', letterSpacing: '.06em' }}>params</span>
        <span style={{ fontSize: 9, color: C.muted, fontFamily: C.mono }}>{mb} MB</span>
      </div>
      <div style={{ height: 4, borderRadius: 2, background: '#21262d' }}>
        <div style={{ height: '100%', width: `${pct}%`, borderRadius: 2, background: color, opacity: .8, transition: 'width .4s' }} />
      </div>
    </div>
  )
}

function LayerPill({ name, color }) {
  return (
    <span style={{
      display: 'inline-block', padding: '1px 6px', borderRadius: 3,
      background: color + '22', color, fontSize: 9, fontFamily: C.mono,
      margin: '1px 2px 1px 0', whiteSpace: 'nowrap',
    }}>
      {name}
    </span>
  )
}

function TimingBadge({ label, ms, color }) {
  return (
    <span style={{
      display: 'inline-flex', alignItems: 'center', gap: 3,
      padding: '1px 5px', borderRadius: 3,
      background: color + '22', fontSize: 9, fontFamily: C.mono,
    }}>
      <span style={{ color: C.muted }}>{label}</span>
      <span style={{ color }}>{ms.toFixed(0)}ms</span>
    </span>
  )
}

function Arrow({ active, fwdActive }) {
  const fwdColor = fwdActive ? C.blue   : C.border
  const bwdColor = fwdActive ? C.border : C.orange
  return (
    <div style={{ padding: '2px 4px', marginLeft: 20 }}>
      {/* forward arrow */}
      <div style={{ display: 'flex', alignItems: 'center', gap: 4, marginBottom: 2 }}>
        <span className={active && fwdActive ? 'active-pulse' : ''}
          style={{ fontSize: 8, color: fwdColor, letterSpacing: '.04em', minWidth: 22, transition: 'color .3s' }}>
          fwd →
        </span>
        <div style={{ flex: 1, height: 1, background: fwdColor, opacity: .7, transition: 'background .3s' }} />
        <svg width="6" height="7" viewBox="0 0 6 7" style={{ flexShrink: 0 }}>
          <polygon points="0,0 6,3.5 0,7" fill={fwdColor} opacity=".9" />
        </svg>
      </div>
      {/* backward arrow */}
      <div style={{ display: 'flex', alignItems: 'center', gap: 4 }}>
        <svg width="6" height="7" viewBox="0 0 6 7" style={{ flexShrink: 0, transform: 'rotate(180deg)' }}>
          <polygon points="0,0 6,3.5 0,7" fill={bwdColor} opacity=".9" />
        </svg>
        <div style={{ flex: 1, height: 1, background: bwdColor, opacity: .7, transition: 'background .3s' }} />
        <span className={active && !fwdActive ? 'active-pulse' : ''}
          style={{ fontSize: 8, color: bwdColor, letterSpacing: '.04em', minWidth: 22, textAlign: 'right', transition: 'color .3s' }}>
          ← bwd
        </span>
      </div>
    </div>
  )
}

function WorkerCard({ worker, index, accent, activity, onSelect, selected }) {
  const [hovered, setHovered] = useState(false)
  const isActive = !!activity
  const showDetail = hovered || selected

  const border     = isActive ? accent : (showDetail ? accent : C.border)
  const stripeOpacity = isActive ? .9 : (showDetail ? .6 : .25)

  return (
    <div
      onClick={() => onSelect && onSelect(worker.hostname)}
      onMouseEnter={() => setHovered(true)}
      onMouseLeave={() => setHovered(false)}
      style={{
        border: `1.5px solid ${border}`,
        borderRadius: 8,
        padding: '10px 12px',
        background: isActive ? accent + '12' : (showDetail ? accent + '0d' : C.surface2),
        cursor: onSelect ? 'pointer' : 'default',
        transition: 'all .2s',
        position: 'relative',
      }}
    >
      {/* top accent stripe — pulses when active */}
      <div
        className={isActive ? 'active-pulse' : ''}
        style={{
          position: 'absolute', top: 0, left: 8, right: 8, height: 2,
          borderRadius: '0 0 2px 2px', background: accent, opacity: stripeOpacity,
        }}
      />

      {/* header: slice index + hostname + GPU badge */}
      <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: 3 }}>
        <div style={{ display: 'flex', alignItems: 'center', gap: 5, overflow: 'hidden' }}>
          <span style={{ fontSize: 9, fontWeight: 700, color: accent, fontFamily: C.mono,
            background: accent + '22', padding: '0px 5px', borderRadius: 3, flexShrink: 0 }}>
            S{index + 1}
          </span>
          <span style={{ fontSize: 11, fontWeight: 700, color: C.text, fontFamily: C.mono,
            overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}>
            {worker.hostname}
          </span>
        </div>
        <span style={{
          fontSize: 9, padding: '1px 6px', borderRadius: 10, flexShrink: 0,
          background: worker.cuda_alloc_mb > 0 ? C.green + '22' : C.muted + '22',
          color: worker.cuda_alloc_mb > 0 ? C.green : C.muted,
          fontWeight: 700, letterSpacing: '.04em',
        }}>
          {worker.cuda_alloc_mb > 0 ? 'GPU' : 'CPU'}
        </span>
      </div>

      {/* role */}
      <div style={{ marginBottom: 5 }}>
        <span style={{ fontSize: 9, color: accent, letterSpacing: '.04em', textTransform: 'uppercase' }}>
          {worker.is_last ? 'last slice · computes loss' : `slice → ${(worker.next || '').split(':')[0] || '…'}`}
        </span>
      </div>

      {/* layer pills — always visible */}
      <div style={{ display: 'flex', flexWrap: 'wrap', marginBottom: 2 }}>
        {worker.layers.map((l, i) => <LayerPill key={i} name={l} color={accent} />)}
      </div>

      {/* timing from most recent batch */}
      {activity && (activity.fwd_ms != null || activity.bwd_ms != null) && (
        <div style={{ display: 'flex', gap: 4, marginTop: 5, flexWrap: 'wrap' }}>
          {activity.fwd_ms != null && <TimingBadge label="fwd" ms={activity.fwd_ms} color={C.blue}   />}
          {activity.bwd_ms != null && <TimingBadge label="bwd" ms={activity.bwd_ms} color={C.orange} />}
        </div>
      )}

      {/* param bar */}
      <MemBar mb={worker.param_mb} color={accent} />

      {/* GPU allocated — on hover/select */}
      {showDetail && worker.cuda_alloc_mb > 0 && (
        <div style={{ marginTop: 4, display: 'flex', justifyContent: 'space-between' }}>
          <span style={{ fontSize: 9, color: C.muted }}>GPU allocated</span>
          <span style={{ fontSize: 9, color: C.green, fontFamily: C.mono }}>{worker.cuda_alloc_mb} MB</span>
        </div>
      )}
    </div>
  )
}

// ── panel ─────────────────────────────────────────────────────────────────────

export default function TopologyPanel({ topology, batches, onSelectWorker }) {
  const [selected, setSelected] = useState(null)

  function handleSelect(hostname) {
    const next = selected === hostname ? null : hostname
    setSelected(next)
    onSelectWorker && onSelectWorker(next)
  }

  const activity = deriveActivity(batches)
  const anyActive = Object.keys(activity).length > 0

  return (
    <div style={{ width: 252, borderRight: `1px solid ${C.border}`, display: 'flex', flexDirection: 'column', flexShrink: 0 }}>
      <div style={{ padding: '10px 14px 8px', fontSize: 10, fontWeight: 700, letterSpacing: '0.1em', textTransform: 'uppercase', color: C.muted, borderBottom: `1px solid ${C.border}`, display: 'flex', alignItems: 'center', justifyContent: 'space-between' }}>
        <span>Topology</span>
        {anyActive && <span className="active-pulse" style={{ fontSize: 8, color: C.green, letterSpacing: '.06em' }}>● LIVE</span>}
      </div>
      <div style={{ flex: 1, overflowY: 'auto', overflowX: 'hidden', padding: '10px 10px 12px' }}>
        {!topology?.length ? (
          <div style={{ color: C.muted, fontSize: 11, textAlign: 'center', marginTop: 24 }}>Waiting for workers…</div>
        ) : (
          <>
            {/* Coordinator */}
            <div style={{ border: `1.5px solid ${C.purple}`, borderRadius: 8, padding: '8px 12px', background: C.surface2, position: 'relative' }}>
              <div style={{ position: 'absolute', top: 0, left: 8, right: 8, height: 2, borderRadius: '0 0 2px 2px', background: C.purple, opacity: .35 }} />
              <div style={{ fontSize: 11, fontWeight: 700, color: C.text }}>Coordinator</div>
              <div style={{ fontSize: 9, color: C.purple, marginTop: 2, textTransform: 'uppercase', letterSpacing: '.04em' }}>drives training loop</div>
            </div>

            {topology.map((w, i) => {
              const accent   = w.is_last ? C.orange : C.blue
              const workerActivity = activity[w.hostname] ?? null
              // fwdActive: worker has fwd but not bwd yet → currently in fwd phase
              const fwdActive = workerActivity && workerActivity.fwd_ms != null && workerActivity.bwd_ms == null
              return (
                <div key={w.hostname}>
                  <Arrow active={anyActive} fwdActive={!fwdActive} />
                  <WorkerCard
                    worker={w}
                    index={i}
                    accent={accent}
                    activity={workerActivity}
                    onSelect={handleSelect}
                    selected={selected === w.hostname}
                  />
                </div>
              )
            })}
          </>
        )}
      </div>
    </div>
  )
}
