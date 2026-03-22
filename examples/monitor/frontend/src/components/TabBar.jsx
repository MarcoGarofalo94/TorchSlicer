import { C } from '../theme'

export default function TabBar({ active, onChange }) {
  return (
    <div style={{ display: 'flex', borderBottom: `1px solid ${C.border}`, flexShrink: 0, padding: '0 12px' }}>
      {['Training', 'Timeline', 'Batches'].map(t => (
        <button key={t} onClick={() => onChange(t)} style={{
          padding: '10px 16px', border: 'none', cursor: 'pointer', fontSize: 12,
          fontWeight: 500, background: 'transparent',
          color: active === t ? C.blue : C.muted,
          borderBottom: `2px solid ${active === t ? C.blue : 'transparent'}`,
          marginBottom: -1,
        }}>
          {t}
        </button>
      ))}
    </div>
  )
}
