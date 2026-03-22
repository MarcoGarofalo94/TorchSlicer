import { useRef, useEffect } from 'react'
import { C } from '../theme'
import { drawTopology } from '../utils/svgDraw'

export default function TopologyPanel({ topology }) {
  const svgRef  = useRef(null)
  const wrapRef = useRef(null)

  useEffect(() => {
    if (!svgRef.current) return
    const W = wrapRef.current?.clientWidth || 220
    drawTopology(svgRef.current, topology || [], W)
  }, [topology])

  return (
    <div style={{ width: 244, borderRight: `1px solid ${C.border}`, display: 'flex', flexDirection: 'column', flexShrink: 0 }}>
      <div style={{ padding: '10px 14px 8px', fontSize: 10, fontWeight: 700, letterSpacing: '0.1em', textTransform: 'uppercase', color: C.muted, borderBottom: `1px solid ${C.border}` }}>
        Topology
      </div>
      <div ref={wrapRef} style={{ flex: 1, overflowY: 'auto', overflowX: 'hidden', padding: '8px 4px 8px 6px' }}>
        <svg ref={svgRef} width="100%" style={{ overflow: 'visible', display: 'block' }} />
      </div>
    </div>
  )
}
