import { useState, useMemo } from 'react'
import { C, selectStyle } from '../theme'

export default function BatchesTab({ batches, topology, onSelectBatch }) {
  const hosts  = topology.map(w => w.hostname)
  const short  = h => h.length > 8 ? h.slice(0, 7) + '…' : h
  const epochs = useMemo(() => [...new Set(batches.map(b => b.epoch))].sort((a, b) => a - b), [batches])

  const [filterEpoch, setFilterEpoch] = useState(null)
  const [sortKey, setSortKey]         = useState('batch_id')
  const [sortDir, setSortDir]         = useState(-1)

  function toggleSort(k) {
    if (sortKey === k) setSortDir(d => -d)
    else { setSortKey(k); setSortDir(-1) }
  }

  const rows = useMemo(() => {
    let list = filterEpoch != null ? batches.filter(b => b.epoch === filterEpoch) : [...batches]
    list.sort((a, b) => {
      const av = a[sortKey] ?? 0, bv = b[sortKey] ?? 0
      return sortDir * (av < bv ? -1 : av > bv ? 1 : 0)
    })
    return list
  }, [batches, filterEpoch, sortKey, sortDir])

  const thStyle = (k) => ({
    padding: '5px 10px', textAlign: 'left', fontSize: 10, fontWeight: 500, whiteSpace: 'nowrap',
    color: sortKey === k ? C.blue : C.muted, background: C.surface,
    position: 'sticky', top: 0, zIndex: 2, cursor: 'pointer', userSelect: 'none',
    boxShadow: `0 1px 0 ${C.border}`,
  })

  const TH = ({ k, children }) => (
    <th onClick={() => toggleSort(k)} style={thStyle(k)}>
      {children}{sortKey === k ? (sortDir < 0 ? ' ↓' : ' ↑') : ''}
    </th>
  )

  return (
    <div style={{ flex: 1, display: 'flex', flexDirection: 'column', gap: 10, padding: 12, minHeight: 0, overflow: 'hidden' }}>
      <div style={{ display: 'flex', alignItems: 'center', gap: 10, flexShrink: 0 }}>
        <label style={{ color: C.muted, fontSize: 11 }}>Epoch:</label>
        <select value={filterEpoch ?? ''} onChange={e => setFilterEpoch(e.target.value ? +e.target.value : null)} style={selectStyle}>
          <option value="">All epochs</option>
          {epochs.map(e => <option key={e} value={e}>Epoch {e}</option>)}
        </select>
        <span style={{ color: C.muted, fontSize: 11, marginLeft: 'auto' }}>
          {rows.length} <span style={{ color: C.border }}>batches</span>
          &nbsp;·&nbsp;Click a row to inspect its swimlane
        </span>
      </div>
      <div style={{ flex: 1, overflowY: 'auto', minHeight: 0, border: `1px solid ${C.border}`, borderRadius: 6 }}>
        <table style={{ width: '100%', borderCollapse: 'collapse', fontFamily: C.mono, fontSize: 11 }}>
          <thead>
            <tr>
              <TH k="epoch">Epoch</TH>
              <TH k="batch_id">Batch</TH>
              <TH k="loss">Loss</TH>
              <TH k="total_ms">Total</TH>
              {hosts.map(h => (
                <>
                  <TH key={h + '_fwd'} k={h + '_fwd'}>{short(h)} fwd</TH>
                  <TH key={h + '_bwd'} k={h + '_bwd'}>{short(h)} bwd</TH>
                </>
              ))}
            </tr>
          </thead>
          <tbody>
            {rows.map(b => (
              <tr key={b.batch_id} onClick={() => onSelectBatch(b.batch_id)} style={{ cursor: 'pointer' }}
                onMouseEnter={e => e.currentTarget.style.background = 'rgba(88,166,255,.06)'}
                onMouseLeave={e => e.currentTarget.style.background = 'transparent'}>
                <td style={{ padding: '4px 10px', color: C.purple,  borderBottom: `1px solid rgba(48,54,61,.35)` }}>{b.epoch}</td>
                <td style={{ padding: '4px 10px',                   borderBottom: `1px solid rgba(48,54,61,.35)` }}>{b.batch_id}</td>
                <td style={{ padding: '4px 10px', color: C.yellow,  borderBottom: `1px solid rgba(48,54,61,.35)` }}>{b.loss?.toFixed(4)}</td>
                <td style={{ padding: '4px 10px', color: b.total_ms > 500 ? C.red : C.green, borderBottom: `1px solid rgba(48,54,61,.35)` }}>{b.total_ms}ms</td>
                {hosts.map(h => {
                  const w = b.workers?.[h] || {}
                  return (
                    <>
                      <td key={h + '_f'} style={{ padding: '4px 10px', color: w.fwd_ms > 100 ? C.red : C.green, borderBottom: `1px solid rgba(48,54,61,.35)` }}>{w.fwd_ms != null ? `${w.fwd_ms}ms` : '—'}</td>
                      <td key={h + '_b'} style={{ padding: '4px 10px', color: w.bwd_ms > 100 ? C.red : C.green, borderBottom: `1px solid rgba(48,54,61,.35)` }}>{w.bwd_ms != null ? `${w.bwd_ms}ms` : '—'}</td>
                    </>
                  )
                })}
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  )
}
