import { useState, useEffect, useMemo, useRef } from 'react'
import { C, panel, panelTitle, selectStyle } from '../theme'
import { drawSwimlane } from '../utils/svgDraw'

function Swimlane({ batch, topology }) {
  const svgRef = useRef(null)
  useEffect(() => { if (svgRef.current) drawSwimlane(svgRef.current, batch, topology) }, [batch, topology])
  return <svg ref={svgRef} width="100%" height="100%" style={{ display: 'block' }} />
}

export default function TimelineTab({ batches, topology, jumpBatchId, onJumpConsumed }) {
  const epochs = useMemo(() => [...new Set(batches.map(b => b.epoch))].sort((a, b) => a - b), [batches])
  const [filterEpoch, setFilterEpoch] = useState(null)
  const [selectedId, setSelectedId]   = useState(null)

  useEffect(() => {
    if (jumpBatchId != null) { setSelectedId(jumpBatchId); onJumpConsumed?.() }
  }, [jumpBatchId])

  useEffect(() => {
    if (selectedId == null && batches.length) setSelectedId(batches[batches.length - 1].batch_id)
  }, [batches])

  const filtered = useMemo(() =>
    filterEpoch != null ? batches.filter(b => b.epoch === filterEpoch) : batches,
    [batches, filterEpoch])

  const batch = useMemo(() =>
    batches.find(b => b.batch_id === selectedId) || batches[batches.length - 1],
    [batches, selectedId])

  return (
    <div style={{ flex: 1, display: 'flex', flexDirection: 'column', gap: 10, padding: 12, minHeight: 0, overflow: 'hidden' }}>
      <div style={{ display: 'flex', alignItems: 'center', gap: 10, flexShrink: 0, flexWrap: 'wrap' }}>
        <label style={{ color: C.muted, fontSize: 11 }}>Epoch</label>
        <select value={filterEpoch ?? ''} onChange={e => setFilterEpoch(e.target.value ? +e.target.value : null)} style={selectStyle}>
          <option value="">All</option>
          {epochs.map(e => <option key={e} value={e}>Epoch {e}</option>)}
        </select>
        <label style={{ color: C.muted, fontSize: 11 }}>Batch</label>
        <select value={selectedId ?? ''} onChange={e => setSelectedId(+e.target.value)} style={{ ...selectStyle, maxWidth: 300 }}>
          {filtered.map(b => (
            <option key={b.batch_id} value={b.batch_id}>
              E{b.epoch}  B{b.batch_id}  —  loss {b.loss?.toFixed(4)}  ({b.total_ms}ms)
            </option>
          ))}
        </select>
        {batch && (
          <span style={{ color: C.muted, fontSize: 11, marginLeft: 'auto' }}>
            Total: <strong style={{ color: C.green }}>{batch.total_ms}ms</strong>
            &nbsp;·&nbsp;Loss: <strong style={{ color: C.yellow }}>{batch.loss?.toFixed(4)}</strong>
          </span>
        )}
      </div>
      <div style={{ ...panel, flex: 1 }}>
        <div style={panelTitle}>Forward &amp; Backward Flow</div>
        <div style={{ flex: 1, minHeight: 0 }}>
          <Swimlane batch={batch} topology={topology} />
        </div>
      </div>
    </div>
  )
}
