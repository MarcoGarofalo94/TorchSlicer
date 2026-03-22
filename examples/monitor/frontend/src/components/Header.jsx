import { C } from '../theme'

function StatChip({ label, value, color }) {
  return (
    <div style={{ textAlign: 'center' }}>
      <div style={{ fontSize: 9, color: C.muted, textTransform: 'uppercase', letterSpacing: '.08em' }}>{label}</div>
      <div style={{ fontSize: 13, fontWeight: 700, color, fontFamily: C.mono }}>{value}</div>
    </div>
  )
}

export default function Header({ connected, lastUpdate, epochs, batches }) {
  const lastEpoch = epochs[epochs.length - 1]
  const lastBatch = batches[batches.length - 1]
  return (
    <div style={{ background: C.surface, borderBottom: `1px solid ${C.border}`, padding: '0 20px', height: 46, display: 'flex', alignItems: 'center', justifyContent: 'space-between', flexShrink: 0 }}>
      <img src="/torchslicer_monitor_logo.svg" alt="TorchSlicer Monitor" style={{ height: 28 }}
           onError={e => { e.currentTarget.style.display = 'none' }} />
      <div style={{ display: 'flex', alignItems: 'center', gap: 28 }}>
        {lastEpoch && <StatChip label="epoch"    value={String(lastEpoch.epoch)}           color={C.purple} />}
        {lastEpoch && <StatChip label="avg loss" value={lastEpoch.avg_loss.toFixed(4)}     color={C.yellow} />}
        {batches.length > 0 && <StatChip label="batches" value={String(batches.length)}    color={C.blue}   />}
        {lastBatch && <StatChip label="batch ms" value={`${lastBatch.total_ms}ms`}         color={C.green}  />}
        <div style={{ display: 'flex', alignItems: 'center', gap: 6 }}>
          <div style={{ width: 8, height: 8, borderRadius: '50%', background: connected ? C.green : C.red, boxShadow: connected ? `0 0 8px ${C.green}` : 'none', transition: 'all .4s' }} />
          <span style={{ color: C.muted, fontSize: 11 }}>{connected ? 'live' : 'reconnecting…'}</span>
        </div>
        {lastUpdate && <span style={{ color: C.border, fontSize: 11, fontFamily: C.mono }}>{lastUpdate}</span>}
      </div>
    </div>
  )
}
