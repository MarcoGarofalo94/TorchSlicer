import { C } from '../theme'

const NS = 'http://www.w3.org/2000/svg'

function el(tag, attrs, text) {
  const e = document.createElementNS(NS, tag)
  for (const [k, v] of Object.entries(attrs || {})) e.setAttribute(k, v)
  if (text != null) e.textContent = text
  return e
}

function drawBox(svg, cx, y, bw, bh, accent, title, layers, sub, mem) {
  svg.appendChild(el('rect', { x: cx-bw/2, y, width: bw, height: bh, rx: 6, fill: C.surface2, stroke: accent, 'stroke-width': 1.5 }))
  svg.appendChild(el('rect', { x: cx-bw/2+1, y: y+1, width: bw-2, height: 3, rx: 3, fill: accent, opacity: .35 }))
  svg.appendChild(el('text', { x: cx, y: y+20, 'text-anchor': 'middle', fill: C.text, 'font-size': 12, 'font-weight': 700 }, title))
  if (sub) svg.appendChild(el('text', { x: cx, y: y+34, 'text-anchor': 'middle', fill: C.muted, 'font-size': 10 }, sub))
  if (layers?.length) layers.forEach((l, i) =>
    svg.appendChild(el('text', { x: cx, y: y+50+i*14, 'text-anchor': 'middle', fill: accent, 'font-size': 10, 'font-family': C.mono }, l))
  )
  if (mem) {
    const mx = cx + bw/2 - 6, my = y + bh - 8
    svg.appendChild(el('text', { x: mx, y: my, 'text-anchor': 'end', fill: C.muted, 'font-size': 9, 'font-family': C.mono }, mem))
  }
}

export function drawTopology(svg, topology, W) {
  svg.innerHTML = ''
  if (!topology.length) {
    svg.appendChild(el('text', { x: '50%', y: 50, 'text-anchor': 'middle', fill: C.muted, 'font-size': 11 }, 'Waiting for workers…'))
    svg.setAttribute('height', 100)
    return
  }

  const BW      = Math.min(W - 88, 178)
  const BH_MIN  = 82, LH = 14, GAP = 52, CX = W / 2
  const ARROW_R = CX + BW/2 + 14
  const ARROW_L = CX - BW/2 - 14

  const defs = el('defs')
  for (const [id, fill] of [['mfwd', C.blue], ['mbwd', C.orange]]) {
    const m = el('marker', { id, markerWidth: 7, markerHeight: 5, refX: 6, refY: 2.5, orient: 'auto' })
    m.appendChild(el('polygon', { points: '0 0,7 2.5,0 5', fill }))
    defs.appendChild(m)
  }
  svg.appendChild(defs)

  const heights = [BH_MIN, ...topology.map(w => BH_MIN + Math.max(0, w.layers.length - 2) * LH)]
  const ys = []
  let y = 16
  heights.forEach(h => { ys.push(y); y += h + GAP })

  drawBox(svg, CX, ys[0], BW, heights[0], C.purple, 'Coordinator', null, 'orchestrates training')

  topology.forEach((w, i) => {
    const wi  = i + 1
    const col = w.is_last ? C.orange : C.blue
    const sub = w.is_last ? 'last slice · computes loss' : (w.next ? `→ ${w.next.split(':')[0]}` : '')
    const memParts = []
    if (w.param_mb)      memParts.push(`${w.param_mb} MB params`)
    if (w.cuda_alloc_mb) memParts.push(`${w.cuda_alloc_mb} MB GPU`)
    const mem = memParts.join('  ') || null
    drawBox(svg, CX, ys[wi], BW, heights[wi], col, w.hostname, w.layers, sub, mem)

    const prevBottom = ys[wi-1] + heights[wi-1]
    const midY       = (prevBottom + ys[wi]) / 2

    svg.appendChild(el('line', { x1: ARROW_R, y1: prevBottom+4, x2: ARROW_R, y2: ys[wi]-4, stroke: C.blue, 'stroke-width': 1.5, 'marker-end': 'url(#mfwd)' }))
    svg.appendChild(el('text', { x: ARROW_R+5, y: midY+4, fill: C.blue, 'font-size': 9, 'text-anchor': 'start' }, 'fwd'))
    svg.appendChild(el('line', { x1: ARROW_L, y1: ys[wi]-4, x2: ARROW_L, y2: prevBottom+4, stroke: C.orange, 'stroke-width': 1.5, 'stroke-dasharray': '4 2', 'marker-end': 'url(#mbwd)' }))
    svg.appendChild(el('text', { x: ARROW_L-5, y: midY+4, fill: C.orange, 'font-size': 9, 'text-anchor': 'end' }, 'bwd'))
  })

  svg.setAttribute('height', y - GAP + 20)
}

export function drawSwimlane(svg, batch, topology) {
  svg.innerHTML = ''
  if (!batch || !topology?.length) {
    svg.appendChild(el('text', { x: '50%', y: '50%', 'text-anchor': 'middle', 'dominant-baseline': 'middle', fill: C.muted, 'font-size': 12 }, 'Select a batch to inspect'))
    return
  }

  const W       = svg.clientWidth  || 700
  const H       = svg.clientHeight || 180
  const LABEL_W = 130
  const chartW  = W - LABEL_W - 12
  const workers = batch.workers || {}
  const hosts   = topology.map(w => w.hostname)

  let t = 0
  const timeline = {}
  hosts.forEach(h => { timeline[h] = [] })
  hosts.forEach(h => { const d = (workers[h]||{}).fwd_ms||0; if (d) { timeline[h].push({ s: t, d, type: 'fwd' }); } t += d })
  ;[...hosts].reverse().forEach(h => { const d = (workers[h]||{}).bwd_ms||0; if (d) { timeline[h].push({ s: t, d, type: 'bwd' }); } t += d })

  const totalMs = Math.max(t, batch.total_ms || 1)
  const scale   = chartW / totalMs
  const rows    = [{ key: 'coordinator', label: 'Coordinator' }, ...hosts.map(h => ({ key: h, label: h.length > 16 ? h.slice(0, 14) + '…' : h }))]
  const ROW_H   = Math.max(42, Math.floor((H - 26) / rows.length))
  const TOP     = 22

  const defs = el('defs')
  defs.innerHTML = `<clipPath id="sw-clip"><rect x="${LABEL_W}" y="0" width="${chartW}" height="${H}"/></clipPath>`
  svg.appendChild(defs)

  for (let i = 0; i <= 6; i++) {
    const x = LABEL_W + (i / 6) * chartW
    svg.appendChild(el('line', { x1: x, y1: TOP-5, x2: x, y2: TOP + rows.length * ROW_H, stroke: '#21262d', 'stroke-width': 1 }))
    svg.appendChild(el('text', { x, y: TOP-7, 'text-anchor': 'middle', fill: C.muted, 'font-size': 9 }, `${Math.round(i / 6 * totalMs)}ms`))
  }

  rows.forEach((row, ri) => {
    const ry = TOP + ri * ROW_H
    svg.appendChild(el('text', { x: LABEL_W-8, y: ry+ROW_H/2+4, 'text-anchor': 'end', fill: C.muted, 'font-size': 10, 'font-family': C.mono }, row.label))
    svg.appendChild(el('rect', { x: LABEL_W, y: ry+2, width: chartW, height: ROW_H-4, fill: '#161b22', rx: 3 }))

    if (row.key === 'coordinator') {
      const bw = Math.max((batch.total_ms || 0) * scale, 4)
      const bx = LABEL_W, by = ry+4, bh = ROW_H-8
      svg.appendChild(el('rect', { x: bx, y: by, width: bw, height: bh, fill: C.purple, opacity: .22, rx: 3 }))
      if (bw > 100) {
        svg.appendChild(el('text', { x: bx+bw/2, y: by+bh/2-3, 'text-anchor': 'middle', fill: C.purple, 'font-size': 9, 'font-weight': 700, 'font-family': C.mono }, `E${batch.epoch}  B${batch.batch_id}  ${batch.total_ms}ms`))
        svg.appendChild(el('text', { x: bx+bw/2, y: by+bh/2+10, 'text-anchor': 'middle', fill: C.purple, 'font-size': 8, 'font-family': C.mono }, `loss ${batch.loss?.toFixed(4)}`))
      } else if (bw > 30) {
        svg.appendChild(el('text', { x: bx+bw/2, y: by+bh/2+4, 'text-anchor': 'middle', fill: C.purple, 'font-size': 9 }, `${batch.total_ms}ms`))
      }
    } else {
      ;(timeline[row.key] || []).forEach(seg => {
        const bx   = LABEL_W + seg.s * scale
        const bw   = Math.max(seg.d * scale, 2)
        const by   = ry+4, bh = ROW_H-8
        const fill = seg.type === 'fwd' ? C.blue : C.orange
        const g    = el('g', { 'clip-path': 'url(#sw-clip)' })
        g.appendChild(el('rect', { x: bx, y: by, width: bw, height: bh, fill, opacity: .9, rx: 3 }))
        if (bw > 120) {
          g.appendChild(el('text', { x: bx+bw/2, y: by+bh/2-3, 'text-anchor': 'middle', fill: '#0d1117', 'font-size': 9, 'font-weight': 700, 'font-family': C.mono }, `${seg.type === 'fwd' ? 'Forward' : 'Backward'}  ${seg.d}ms`))
          g.appendChild(el('text', { x: bx+bw/2, y: by+bh/2+10, 'text-anchor': 'middle', fill: '#0d1117', 'font-size': 8, 'font-family': C.mono }, `E${batch.epoch}  B${batch.batch_id}`))
        } else if (bw > 40) {
          g.appendChild(el('text', { x: bx+bw/2, y: by+bh/2+4, 'text-anchor': 'middle', fill: '#0d1117', 'font-size': 9, 'font-weight': 700 }, `${seg.d}ms`))
        }
        svg.appendChild(g)
      })
    }
  })
}
