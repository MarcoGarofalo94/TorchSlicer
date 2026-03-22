import { useState } from 'react'
import { C } from '../theme'

const MAX_PARAM_MB = 50  // scale for memory bar

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
    <span style={{ display: 'inline-block', padding: '1px 6px', borderRadius: 3, background: color + '22', color, fontSize: 9, fontFamily: C.mono, margin: '1px 2px 1px 0', whiteSpace: 'nowrap' }}>
      {name}
    </span>
  )
}

function Arrow({ color, label }) {
  return (
    <div style={{ display: 'flex', alignItems: 'center', padding: '2px 0', marginLeft: 20 }}>
      <div style={{ display: 'flex', flexDirection: 'column', alignItems: 'center', gap: 2, width: '100%' }}>
        <div style={{ display: 'flex', alignItems: 'center', gap: 4, width: '100%' }}>
          <span style={{ fontSize: 8, color, letterSpacing: '.04em', minWidth: 22 }}>{label}</span>
          <div style={{ flex: 1, height: 1, background: color, opacity: .6 }} />
          <svg width="7" height="8" viewBox="0 0 7 8" style={{ flexShrink: 0 }}>
            <polygon points="0,0 7,4 0,8" fill={color} opacity=".8" />
          </svg>
        </div>
      </div>
    </div>
  )
}

function WorkerCard({ worker, isFirst, accent, onSelect, selected }) {
  const [hovered, setHovered] = useState(false)
  const active = hovered || selected

  const gpuMb = worker.cuda_alloc_mb || 0
  const border = active ? accent : C.border

  return (
    <div
      onClick={() => onSelect && onSelect(worker.hostname)}
      onMouseEnter={() => setHovered(true)}
      onMouseLeave={() => setHovered(false)}
      style={{
        border: `1.5px solid ${border}`,
        borderRadius: 8,
        padding: '10px 12px',
        background: active ? accent + '0d' : C.surface2,
        cursor: onSelect ? 'pointer' : 'default',
        transition: 'all .15s',
        position: 'relative',
      }}
    >
      {/* top accent stripe */}
      <div style={{ position: 'absolute', top: 0, left: 8, right: 8, height: 2, borderRadius: '0 0 2px 2px', background: accent, opacity: active ? .7 : .3 }} />

      {/* header row */}
      <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: 4 }}>
        <span style={{ fontSize: 11, fontWeight: 700, color: C.text, fontFamily: C.mono, overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap', maxWidth: 120 }}>
          {worker.hostname}
        </span>
        <span style={{ fontSize: 9, padding: '1px 6px', borderRadius: 10, background: gpuMb > 0 ? C.green + '22' : C.muted + '22', color: gpuMb > 0 ? C.green : C.muted, fontWeight: 700, letterSpacing: '.04em', flexShrink: 0 }}>
          {gpuMb > 0 ? 'GPU' : 'CPU'}
        </span>
      </div>

      {/* role tag */}
      <div style={{ marginBottom: 6 }}>
        <span style={{ fontSize: 9, color: accent, letterSpacing: '.04em', textTransform: 'uppercase' }}>
          {worker.is_last ? 'last slice · loss' : `slice → ${(worker.next || '').split(':')[0] || '…'}`}
        </span>
      </div>

      {/* layer count + pills (always show count, pills on hover) */}
      <div>
        <span style={{ fontSize: 9, color: C.muted }}>{worker.n_layers} layers</span>
        {active && (
          <div style={{ marginTop: 4, display: 'flex', flexWrap: 'wrap' }}>
            {worker.layers.map((l, i) => <LayerPill key={i} name={l} color={accent} />)}
          </div>
        )}
      </div>

      {/* memory bar (always) */}
      <MemBar mb={worker.param_mb} color={accent} />

      {/* GPU allocated (on hover) */}
      {active && gpuMb > 0 && (
        <div style={{ marginTop: 4, display: 'flex', justifyContent: 'space-between' }}>
          <span style={{ fontSize: 9, color: C.muted }}>GPU allocated</span>
          <span style={{ fontSize: 9, color: C.green, fontFamily: C.mono }}>{gpuMb} MB</span>
        </div>
      )}
    </div>
  )
}

export default function TopologyPanel({ topology, onSelectWorker }) {
  const [selected, setSelected] = useState(null)

  function handleSelect(hostname) {
    const next = selected === hostname ? null : hostname
    setSelected(next)
    onSelectWorker && onSelectWorker(next)
  }

  return (
    <div style={{ width: 244, borderRight: `1px solid ${C.border}`, display: 'flex', flexDirection: 'column', flexShrink: 0 }}>
      <div style={{ padding: '10px 14px 8px', fontSize: 10, fontWeight: 700, letterSpacing: '0.1em', textTransform: 'uppercase', color: C.muted, borderBottom: `1px solid ${C.border}` }}>
        Topology
      </div>
      <div style={{ flex: 1, overflowY: 'auto', overflowX: 'hidden', padding: '10px 10px 12px' }}>
        {!topology?.length ? (
          <div style={{ color: C.muted, fontSize: 11, textAlign: 'center', marginTop: 24 }}>Waiting for workers…</div>
        ) : (
          <>
            {/* Coordinator node */}
            <div style={{ border: `1.5px solid ${C.purple}`, borderRadius: 8, padding: '8px 12px', background: C.surface2, marginBottom: 0 }}>
              <div style={{ position: 'relative' }}>
                <div style={{ position: 'absolute', top: -8, left: 8, right: 8, height: 2, borderRadius: '0 0 2px 2px', background: C.purple, opacity: .4 }} />
              </div>
              <div style={{ fontSize: 11, fontWeight: 700, color: C.text }}>Coordinator</div>
              <div style={{ fontSize: 9, color: C.purple, marginTop: 2, textTransform: 'uppercase', letterSpacing: '.04em' }}>orchestrates training</div>
            </div>

            {/* Workers with arrows */}
            {topology.map((w, i) => {
              const accent = w.is_last ? C.orange : C.blue
              return (
                <div key={w.hostname}>
                  <div style={{ padding: '2px 4px' }}>
                    <Arrow color={C.blue} label="fwd →" />
                    <Arrow color={C.orange} label="← bwd" />
                  </div>
                  <WorkerCard
                    worker={w}
                    isFirst={i === 0}
                    accent={accent}
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
