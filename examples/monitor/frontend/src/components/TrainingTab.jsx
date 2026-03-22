import { useMemo } from 'react'
import {
  LineChart, Line, BarChart, Bar,
  XAxis, YAxis, CartesianGrid, Tooltip, Legend, Brush,
  ResponsiveContainer,
} from 'recharts'
import { C, panel, panelTitle } from '../theme'

const tt = { background: C.surface, border: `1px solid ${C.border}`, borderRadius: 6, fontSize: 11 }

function Empty({ text }) {
  return <div style={{ flex: 1, display: 'flex', alignItems: 'center', justifyContent: 'center', color: C.muted, fontSize: 12 }}>{text}</div>
}

function EpochLossChart({ epochs }) {
  if (!epochs.length) return <Empty text="Waiting for epoch data…" />
  return (
    <ResponsiveContainer width="100%" height="100%">
      <LineChart data={epochs} margin={{ top: 12, right: 20, bottom: 20, left: 44 }}>
        <CartesianGrid strokeDasharray="3 3" stroke="#21262d" />
        <XAxis dataKey="epoch" stroke={C.muted} tick={{ fontSize: 10, fill: C.muted }}
          label={{ value: 'Epoch', position: 'insideBottomRight', offset: -8, fill: C.muted, fontSize: 10 }} />
        <YAxis stroke={C.muted} tick={{ fontSize: 10, fill: C.muted }} tickFormatter={v => v.toFixed(3)} />
        <Tooltip contentStyle={tt} labelStyle={{ color: C.muted }}
          labelFormatter={v => `Epoch ${v}`} formatter={v => [v.toFixed(4), 'Avg Loss']} />
        <Brush dataKey="epoch" height={18} stroke={C.border} fill={C.surface} travellerWidth={6} />
        <Line type="monotone" dataKey="avg_loss" stroke={C.green} strokeWidth={2.5}
          dot={{ r: 4, fill: C.green, strokeWidth: 0 }} activeDot={{ r: 6 }} name="Avg Loss" />
      </LineChart>
    </ResponsiveContainer>
  )
}

function BatchLossChart({ batches }) {
  if (!batches.length) return <Empty text="Waiting for batch data…" />
  const data = batches.map((b, i) => ({ idx: i, loss: b.loss, epoch: b.epoch, batch_id: b.batch_id }))
  return (
    <ResponsiveContainer width="100%" height="100%">
      <LineChart data={data} margin={{ top: 12, right: 20, bottom: 20, left: 44 }}>
        <CartesianGrid strokeDasharray="3 3" stroke="#21262d" />
        <XAxis dataKey="idx" stroke={C.muted} tick={{ fontSize: 9, fill: C.muted }}
          label={{ value: 'All batches', position: 'insideBottomRight', offset: -8, fill: C.muted, fontSize: 9 }} />
        <YAxis stroke={C.muted} tick={{ fontSize: 10, fill: C.muted }} tickFormatter={v => v.toFixed(2)} />
        <Tooltip contentStyle={tt} labelStyle={{ color: C.muted }}
          labelFormatter={(_, p) => p?.[0] ? `E${p[0].payload.epoch}  B${p[0].payload.batch_id}` : ''}
          formatter={v => [v.toFixed(4), 'Loss']} />
        <Brush dataKey="idx" height={18} stroke={C.border} fill={C.surface} travellerWidth={6} />
        <Line type="monotone" dataKey="loss" stroke={C.yellow} strokeWidth={1.5} dot={false} name="Loss" />
      </LineChart>
    </ResponsiveContainer>
  )
}

function WorkerTimingChart({ batches, topology }) {
  const data = useMemo(() => {
    const hosts = topology.map(w => w.hostname)
    const short = h => h.length > 10 ? h.slice(0, 8) + '…' : h
    return hosts.map(h => {
      const vals = batches.map(b => b.workers?.[h]).filter(Boolean)
      const avg  = arr => arr.length ? Math.round(arr.reduce((a, b) => a + b, 0) / arr.length) : 0
      return {
        name:     short(h),
        Forward:  avg(vals.filter(v => v.fwd_ms).map(v => v.fwd_ms)),
        Backward: avg(vals.filter(v => v.bwd_ms).map(v => v.bwd_ms)),
      }
    })
  }, [batches, topology])

  if (!data.length) return <Empty text="No worker data yet" />
  return (
    <ResponsiveContainer width="100%" height="100%">
      <BarChart data={data} margin={{ top: 12, right: 20, bottom: 20, left: 44 }}>
        <CartesianGrid strokeDasharray="3 3" stroke="#21262d" />
        <XAxis dataKey="name" stroke={C.muted} tick={{ fontSize: 10, fill: C.muted }} />
        <YAxis stroke={C.muted} tick={{ fontSize: 10, fill: C.muted }} unit="ms" />
        <Tooltip contentStyle={tt} labelStyle={{ color: C.muted }} formatter={(v, n) => [`${v}ms`, n]} />
        <Legend wrapperStyle={{ fontSize: 10, color: C.muted }} />
        <Bar dataKey="Forward"  fill={C.blue}   radius={[3, 3, 0, 0]} />
        <Bar dataKey="Backward" fill={C.orange} radius={[3, 3, 0, 0]} />
      </BarChart>
    </ResponsiveContainer>
  )
}

export default function TrainingTab({ epochs, batches, topology }) {
  return (
    <div style={{ flex: 1, display: 'grid', gridTemplateColumns: '1fr 1fr', gridTemplateRows: '55% 45%', gap: 10, padding: 12, minHeight: 0, overflow: 'hidden' }}>
      <div style={{ ...panel, gridColumn: '1 / 3' }}>
        <div style={panelTitle}>Avg Loss / Epoch</div>
        <EpochLossChart epochs={epochs} />
      </div>
      <div style={panel}>
        <div style={panelTitle}>Per-Batch Loss (all epochs)</div>
        <BatchLossChart batches={batches} />
      </div>
      <div style={panel}>
        <div style={panelTitle}>Worker Timing (avg fwd / bwd)</div>
        <WorkerTimingChart batches={batches} topology={topology} />
      </div>
    </div>
  )
}
