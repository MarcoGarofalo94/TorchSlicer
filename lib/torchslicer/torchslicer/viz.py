"""An interactive picture of a model's graph and where it is cut.

    sliced.view(example_input=x)                 # in a notebook: shown in the output cell
    sliced.view(example_input=x).save("split.html")
    ts.trace(model).view()                       # no cuts yet: place them by clicking

One row per graph node, top to bottom (the order slices follow). Between two rows is a gap
where a cut may go: a cut is a horizontal line, the rows between two cuts are one slice.
Edges to the next row are the short ticks between rows; every other edge (skip connections,
the model input read later) is an arc on the right, so an arc crossing a gap is exactly why
a cut there is not allowed. Bars on the left are the enclosing modules (the ``expand=``
level); a dashed purple link joins nodes that share a tensor (tied weights).

In the page: click a gap marked ``+`` to add or remove a cut; the slice table (nodes,
parameters, size, bytes sent) and the matching ``ts.slice(...)`` call update. Hover a gap
where no cut is allowed to see the arcs that cross it. The page is self-contained (no
network access, no extra package); :meth:`SliceView.svg` is the static picture.
"""
import html
import json
import re

from .core.model_graph import _human_bytes

ROW, GAP, PAD = 22, 12, 10            # row height, gap between rows (where cuts go), margin
CH = 7.2                              # width of one monospace character at 12px
ARC_STEP, BAR_W = 12, 7               # arc nesting step, module bar width


def _type_name(module):
    inner = getattr(module, "block", None)            # BlockStage / MoEBlockStage
    name = type(module).__name__
    return f"{name}({type(inner).__name__})" if inner is not None else name


def _arc_levels(edges):
    """Nesting level per arc, so that overlapping arcs never share an x."""
    levels = {}
    for u, v in sorted(edges, key=lambda e: e[1] - e[0]):
        over = [levels[e] for e in levels if not (e[1] <= u or v <= e[0])]
        levels[(u, v)] = 1 + max(over, default=0)
    return levels


class SliceView:
    """The picture of a ``SlicedModel`` (its cuts drawn) or a ``ModelGraph`` (no cuts).

    ``example_input`` adds each node's output shape and size (what a cut after it sends),
    measured on a copy of the model as in ``summary()``. Shown inline by Jupyter; ``html()``
    / ``save(path)`` give the interactive page, ``svg()`` the static picture.
    """

    def __init__(self, obj, example_input=None, aux=None, title=None, multi_tensor=None):
        sliced = obj if hasattr(obj, "partitions") else None
        graph = sliced.graph if sliced else obj
        cuts = [p.layer_indices[-1] for p in sliced.partitions[:-1]] if sliced else []
        base = getattr(graph, "base", None)
        if base is not None:            # cut with multi_tensor=True: draw the model's own graph
            graph, cuts = base, list(graph.base_cuts)
        if multi_tensor is None:
            multi_tensor = base is not None or getattr(graph, "multi_tensor", False)
        self.graph = graph
        nodes = graph.nodes
        sizes = graph._param_sizes()
        outs = graph._output_sizes(example_input, aux) if example_input is not None else None
        sends, send_bytes = self._sends(graph, outs)
        model = getattr(sliced, "_model_name", None) or type(getattr(graph, "model", graph)).__name__

        first_holder, tied = {}, set()                 # nodes sharing a tensor (tied weights)
        for n in nodes:
            for t in [*n.module.parameters(), *n.module.buffers()]:
                if id(t) in first_holder and first_holder[id(t)] != n.index:
                    tied.add((first_holder[id(t)], n.index))
                first_holder.setdefault(id(t), n.index)

        edges = [(p, n.index) for n in nodes for p in n.predecessors]
        edges += [(-1, n.index) for n in nodes if n.takes_input]       # -1: the model input
        arcs = [(u, v) for u, v in edges if v > u + 1]
        self.data = {
            "title": title or model,
            "packed": graph.is_packed(),
            "nodes": [{"name": n.name or str(n.index), "type": _type_name(n.module),
                       "params": sizes[n.index][0], "bytes": sizes[n.index][1],
                       "tied": sizes[n.index][2],
                       "shape": outs[n.index][0] if outs else None,
                       "out": outs[n.index][1] if outs else None} for n in nodes],
            "valid": graph.multi_tensor_cuts() if multi_tensor else graph.valid_cuts(),
            "single": graph.valid_cuts(),
            "multi": bool(multi_tensor),
            "sends": sends,
            "sendBytes": send_bytes,
            "cuts": cuts,
            "arcs": arcs,
            "tied": sorted(tied),
        }
        self._svg = self._layout(nodes, edges, arcs, tied, example_input)

    @staticmethod
    def _sends(graph, outs):
        """Per gap: the names of what a cut there sends (after recomputation), and its bytes."""
        from .core.multicut import _split_free, crossing, recomputable
        kind = recomputable(graph)
        names, nbytes = [], []
        for k in range(len(graph) - 1):
            sent, _ = _split_free(crossing(graph, k), graph.nodes, kind)
            names.append(["input" if v < 0 else graph.nodes[v].name for v in sent])
            nbytes.append(sum(outs[v][1] for v in sent if v >= 0) if outs else None)
        return names, nbytes

    def _cut_label(self, j, g):
        d, r = self.data, self.data["nodes"][g]
        sent = d["sends"][g]
        if len(sent) > 1 or sent != [r["name"]]:
            size = f" · {_human_bytes(d['sendBytes'][g])}" if d["sendBytes"][g] is not None else ""
            return f"cut {j} → {len(sent)} values packed{size}"
        return f"cut {j} → " + (f"{r['shape']} · {_human_bytes(r['out'])}" if r["shape"] else f"after {r['name']}")

    # ── static layout (Python), so the picture exists without JS ──────────────
    def _layout(self, nodes, edges, arcs, tied, example_input):
        d = self.data
        n = len(nodes)
        y = lambda i: PAD + (i + 1) * (ROW + GAP)          # row -1 is the model input
        mid = lambda i: y(i) + ROW / 2

        scopes = {}
        for node in nodes:
            for depth, s in enumerate(node.scope):
                lo, hi, _ = scopes.get(s, (node.index, node.index, depth))
                scopes[s] = (min(lo, node.index), max(hi, node.index), depth)
        bars = {s: v for s, v in scopes.items() if v[1] > v[0]}
        depth = 1 + max((v[2] for v in bars.values()), default=-1)

        def w(texts, minimum):
            return max(minimum, max(len(t) for t in texts) * CH) + 16
        rows = d["nodes"]
        x_idx = PAD + 10 + len(str(n - 1)) * CH                    # right edge of the row numbers
        x_bars = x_idx + 6
        x_name = x_bars + depth * (BAR_W + 3) + 2
        w_name = w([r["name"] for r in rows], 60)
        w_type = w([r["type"] for r in rows], 40)
        w_par = w([f"{r['params']:,}" for r in rows], 50)
        w_shape = w([r["shape"] or "" for r in rows], 40) if example_input is not None else 0
        w_out = w([_human_bytes(r["out"] or 0) for r in rows], 50) if example_input is not None else 0
        x_type, x_par = x_name + w_name, x_name + w_name + w_type
        x_shape = x_par + w_par
        x_out = x_shape + w_shape
        x_arc = x_out + w_out + 6
        levels = _arc_levels(arcs + list(tied))
        width = x_arc + ARC_STEP * (max(levels.values(), default=0) + 1) + 40
        height = y(n) + PAD
        e = html.escape
        out = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{width:.0f}" height="{height:.0f}" '
               f'viewBox="0 0 {width:.0f} {height:.0f}" class="sv">']
        out.append('<defs><marker id="ah" viewBox="0 0 8 8" refX="7" refY="4" markerWidth="6" '
                   'markerHeight="6" orient="auto"><path d="M0,0L8,4L0,8z" class="ahead"/></marker></defs>')
        # slice bands (recoloured by JS) and the input row
        cuts = sorted(d["cuts"])
        for i in range(n):
            k = sum(c < i for c in cuts) % 6
            out.append(f'<rect class="band" id="band{i}" x="{PAD}" y="{y(i) - GAP / 2}" '
                       f'width="{x_arc - PAD}" height="{ROW + GAP}" style="fill:var(--s{k})"/>')
            out.append(f'<rect class="stripe" id="stripe{i}" x="{PAD}" y="{y(i) - GAP / 2}" '
                       f'width="5" height="{ROW + GAP}" style="fill:var(--k{k})"/>')
        shape_in = ""
        if example_input is not None and hasattr(example_input, "shape"):
            shape_in = str(list(example_input.shape))
        out.append(f'<text class="io" x="{x_name}" y="{mid(-1) + 4}">input</text>'
                   f'<text class="num" x="{x_shape + 4}" y="{mid(-1) + 4}">{e(shape_in)}</text>')
        # module bars
        for s, (lo, hi, dep) in sorted(bars.items(), key=lambda kv: kv[1][2]):
            x = x_bars + dep * (BAR_W + 3)
            out.append(f'<rect class="bar" x="{x}" y="{y(lo) + 2}" width="{BAR_W}" '
                       f'height="{y(hi) + ROW - 2 - y(lo) - 2}" rx="3"><title>{e(s)} '
                       f'(nodes {lo}–{hi})</title></rect>')
        # rows
        for i, (node, r) in enumerate(zip(nodes, rows)):
            name = r["name"]
            parent = next((s for s in reversed(node.scope[:-1]) if name.startswith(s + ".")), "")
            label = (f'<tspan class="dim">{e(parent)}.</tspan>{e(name[len(parent) + 1:])}'
                     if parent else e(name))
            out.append(f'<g class="row" data-i="{i}"><rect class="hit" x="{PAD + 6}" y="{y(i)}" '
                       f'width="{x_arc - PAD - 6}" height="{ROW}"/>'
                       f'<text class="idx" x="{x_idx}" y="{mid(i) + 4}" text-anchor="end">{i}</text>'
                       f'<text class="name" x="{x_name + 4}" y="{mid(i) + 4}">{label}</text>'
                       f'<text class="type" x="{x_type}" y="{mid(i) + 4}">{e(r["type"])}</text>'
                       f'<text class="num" x="{x_par + w_par - 12}" y="{mid(i) + 4}" text-anchor="end">'
                       f'{r["params"]:,}{"*" if r["tied"] else ""}</text>')
            if r["shape"] is not None:
                out.append(f'<text class="num" x="{x_shape + 4}" y="{mid(i) + 4}">{e(r["shape"])}</text>'
                           f'<text class="num" x="{x_out + w_out - 12}" y="{mid(i) + 4}" '
                           f'text-anchor="end">{_human_bytes(r["out"])}</text>')
            out.append("</g>")
        # edges to the next row: ticks in the gap
        for u, v in edges:
            if v == u + 1:
                out.append(f'<line class="tick" x1="{x_name + 10}" y1="{y(u) + ROW}" '
                           f'x2="{x_name + 10}" y2="{y(v)}" marker-end="url(#ah)"/>')
        # arcs: skip connections and late reads of the input, then tied weights
        for k, (u, v) in enumerate(arcs):
            x = x_arc + ARC_STEP * levels[(u, v)]
            r = min(6, ARC_STEP * levels[(u, v)] / 2)
            out.append(f'<path class="arc" id="arc{k}" d="M{x_arc - 4},{mid(u)} H{x - r} '
                       f'Q{x},{mid(u)} {x},{mid(u) + r} V{mid(v) - r} Q{x},{mid(v)} {x - r},{mid(v)} '
                       f'H{x_arc - 2}" marker-end="url(#ah)"><title>{e(d["nodes"][u]["name"] if u >= 0 else "input")}'
                       f' → {e(d["nodes"][v]["name"])}</title></path>')
        for a, b in sorted(tied):
            x = x_arc + ARC_STEP * levels[(a, b)]
            out.append(f'<path class="tiedarc" d="M{x_arc - 4},{mid(a)} H{x} V{mid(b)} H{x_arc - 4}">'
                       f'<title>tied: {e(d["nodes"][a]["name"])} and {e(d["nodes"][b]["name"])} share a '
                       f'tensor</title></path>')
        # gaps: where cuts go (hit area, line, label filled by JS)
        for g in range(n - 1):
            gy = y(g) + ROW + GAP / 2
            cls = ("ok" if g in d["valid"] else "no") + (" on" if g in cuts else "")
            label = self._cut_label(cuts.index(g), g) if g in cuts else ""
            out.append(f'<g class="gap {cls}" data-g="{g}">'
                       f'<rect class="ghit" x="{PAD}" y="{gy - GAP / 2 - 3}" width="{x_arc - PAD}" '
                       f'height="{GAP + 6}"/>'
                       f'<line class="cutline" x1="{PAD}" y1="{gy}" x2="{x_arc}" y2="{gy}"/>'
                       f'<text class="plus" x="{PAD + 9}" y="{gy + 4}">+</text>'
                       f'<g class="pill"><rect x="{x_type - 4}" y="{gy - 8}" height="16" rx="8" '
                       f'width="{len(label) * 6.6 + 16:.0f}"/>'
                       f'<text x="{x_type + 4}" y="{gy + 4}">{html.escape(label)}</text></g></g>')
        out.append("</svg>")
        return "".join(out)

    # ── output ────────────────────────────────────────────────────────────────
    def html(self) -> str:
        return (_PAGE.replace("__TITLE__", html.escape(self.data["title"]))
                .replace("__SVG__", self._svg)
                .replace("__DATA__", json.dumps(self.data).replace("</", "<\\/")))

    def save(self, path):
        with open(path, "w") as f:
            f.write(self.html())
        return path

    def svg(self, dark: bool = False) -> str:
        """The static picture (initial cuts, no interaction), e.g. for a PNG export.
        Colours are written out (SVG renderers outside browsers don't know CSS variables)."""
        colors = dict(re.findall(r"--(\w+):(#\w+)", _DARK if dark else _COLORS))
        svg = self._svg.replace('class="sv">', 'class="sv">' + _STATIC_STYLE, 1)
        svg = re.sub(r"var\(--(\w+)\)", lambda m: colors[m.group(1)], svg)
        return svg.replace('class="sv">', f'class="sv"><rect width="100%" height="100%" fill="{colors["bg"]}"/>', 1)

    def _repr_html_(self):
        height = (len(self.data["nodes"]) + 2) * (ROW + GAP) + 260
        return (f'<iframe srcdoc="{html.escape(self.html(), quote=True)}" '
                f'style="width:100%;height:{min(height, 900)}px;border:0"></iframe>')


_COLORS = """
  --bg:#ffffff; --fg:#1f2328; --dim:#8b949e; --line:#d0d7de; --accent:#cf222e; --arc:#57606a;
  --tied:#8250df; --bar:#d8dee4; --hl:#0969da;
  --s0:#ddf4ff; --s1:#fff1e5; --s2:#dafbe1; --s3:#fbefff; --s4:#fff8c5; --s5:#ffebe9;
  --k0:#0969da; --k1:#bc4c00; --k2:#1a7f37; --k3:#8250df; --k4:#9a6700; --k5:#cf222e;"""
_DARK = """
  --bg:#0d1117; --fg:#e6edf3; --dim:#7d8590; --line:#30363d; --accent:#ff7b72; --arc:#8b949e;
  --tied:#d2a8ff; --bar:#30363d; --hl:#58a6ff;
  --s0:#0c2d4a; --s1:#3d2210; --s2:#0f3020; --s3:#2e1a45; --s4:#3a2f05; --s5:#42181a;
  --k0:#58a6ff; --k1:#f0883e; --k2:#3fb950; --k3:#d2a8ff; --k4:#d29922; --k5:#ff7b72;"""
_SVG_CSS = """
.sv text{font:12px ui-monospace,SFMono-Regular,Menlo,Consolas,monospace;fill:var(--fg)}
.sv .dim,.sv .idx,.sv .io{fill:var(--dim)} .sv .type{fill:var(--dim)} .sv .num{fill:var(--fg)}
.sv .band{fill:var(--s0)} .sv .stripe{fill:var(--k0)} .sv .bar{fill:var(--bar)}
.sv .hit{fill:transparent} .sv .tick{stroke:var(--arc);stroke-width:1.2}
.sv .ahead{fill:var(--arc)}
.sv .arc{fill:none;stroke:var(--arc);stroke-width:1.2;opacity:.75}
.sv .tiedarc{fill:none;stroke:var(--tied);stroke-width:1.4;stroke-dasharray:3 3}
.sv .cutline{stroke:transparent;stroke-width:2} .sv .plus{fill:transparent;font-weight:700}
.sv .pill{display:none} .sv .pill rect{fill:var(--accent)} .sv .pill text{fill:#fff;font-size:11px}
.sv .ghit{fill:transparent}
"""
_STATIC_STYLE = (f"<style>{_SVG_CSS}"
                 ".sv .gap.on .cutline{stroke:var(--accent)} .sv .gap.on .pill{display:inline}</style>")

_PAGE = """<!doctype html><html><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1"><title>__TITLE__ slices</title>
<style>
:root{""" + _COLORS + """}
@media (prefers-color-scheme: dark){:root:not([data-theme="light"]){""" + _DARK + """}}
:root[data-theme="dark"]{""" + _DARK + """}
body{margin:0;padding:16px;background:var(--bg);color:var(--fg);
     font:14px system-ui,-apple-system,Segoe UI,sans-serif}
h1{font-size:18px;margin:0 0 4px} .sub{color:var(--dim);margin:0 0 12px;font-size:13px}
code,pre{font:12px ui-monospace,SFMono-Regular,Menlo,Consolas,monospace}
pre{background:var(--s0);padding:8px 10px;border-radius:6px;margin:0 0 12px;white-space:pre-wrap;
    word-break:break-all}
table{border-collapse:collapse;margin-bottom:12px;font-size:13px}
td,th{padding:3px 10px;text-align:right;border-bottom:1px solid var(--line)}
th{color:var(--dim);font-weight:500} td:first-child,th:first-child{text-align:left}
.sw{display:inline-block;width:10px;height:10px;border-radius:2px;margin-right:6px;vertical-align:-1px}
.barcell{width:160px;text-align:left} .barcell div{height:8px;border-radius:2px}
.wrap{overflow-x:auto} button{font:inherit;font-size:12px;margin-right:6px;padding:2px 10px;
  background:var(--bg);color:var(--fg);border:1px solid var(--line);border-radius:6px;cursor:pointer}
#tip{position:fixed;pointer-events:none;background:var(--fg);color:var(--bg);font-size:12px;
  padding:4px 8px;border-radius:4px;display:none;max-width:420px}
""" + _SVG_CSS + """
.sv .gap.ok{cursor:pointer} .sv .gap.ok:hover .cutline{stroke:var(--accent);stroke-dasharray:4 3}
.sv .gap.ok .plus{fill:var(--dim)} .sv .gap.ok:hover .plus{fill:var(--accent)}
.sv .gap.on .cutline{stroke:var(--accent);stroke-dasharray:none} .sv .gap.on .pill{display:inline}
.sv .gap.on .plus{fill:transparent}
.sv .gap.no:hover .cutline{stroke:var(--line);stroke-dasharray:2 3}
.sv .arc.hl{stroke:var(--accent);stroke-width:2.2;opacity:1}
.sv .row:hover .hit{fill:var(--line);opacity:.35}
.legend{color:var(--dim);font-size:12px;margin:0 0 8px}
</style></head><body>
<h1>__TITLE__</h1>
<p class="sub">One row per node. Click a <b>+</b> gap to add or remove a cut. Arcs are edges that skip
rows; a gap with an arc crossing it can't be cut.</p>
<pre id="code"></pre>
<div><button id="reset">initial cuts</button><button id="clear">no cuts</button></div>
<table id="slices"></table>
<p class="legend">bars on the left: enclosing modules · <span style="color:var(--tied)">dashed purple</span>:
tied weights (* = counted at its first node) · last two columns: output shape and bytes = what a cut after
that node sends per batch</p>
<div class="wrap">__SVG__</div><div id="tip"></div>
<script>
const D = __DATA__;
const N = D.nodes.length, valid = new Set(D.valid);
let cuts = new Set(D.cuts);
const hb = n => n == null ? "" : n < 1024 ? n + " B" : n < 1048576 ? (n/1024).toFixed(1) + " KiB"
        : n < 1073741824 ? (n/1048576).toFixed(1) + " MiB" : (n/1073741824).toFixed(2) + " GiB";
const fmt = n => n.toLocaleString("en-US");
const css = v => getComputedStyle(document.documentElement).getPropertyValue(v);
function slices(){ const cs=[...cuts].sort((a,b)=>a-b), out=[]; let lo=0;
  for (const c of [...cs, N-1]) { out.push([lo,c]); lo=c+1; } return out; }
function render(){
  const sl = slices(), total = D.nodes.reduce((s,n)=>s+n.params,0) || 1;
  sl.forEach(([lo,hi],k)=>{ for(let i=lo;i<=hi;i++){
    document.getElementById("band"+i).style.fill = `var(--s${k%6})`;
    document.getElementById("stripe"+i).style.fill = `var(--k${k%6})`; }});
  document.querySelectorAll(".gap").forEach(g=>{ const i=+g.dataset.g, on=cuts.has(i);
    g.classList.toggle("on", on);
    if (on) { const k=[...cuts].sort((a,b)=>a-b).indexOf(i), n=D.nodes[i];
      const t=g.querySelector(".pill text"), r=g.querySelector(".pill rect");
      t.textContent = packed(i) ? `cut ${k} → ${D.sends[i].length} values packed${D.sendBytes[i] != null ? " · " + hb(D.sendBytes[i]) : ""}`
        : `cut ${k} → ${n.shape ? n.shape + " · " + hb(n.out) : "after " + n.name}`;
      r.setAttribute("width", t.getComputedTextLength() + 16); } });
  let rows = "<tr><th>slice</th><th>nodes</th><th>params</th><th>size</th><th class=barcell>share of params</th><th>sends</th></tr>";
  sl.forEach(([lo,hi],k)=>{ let p=0,b=0; for(let i=lo;i<=hi;i++){p+=D.nodes[i].params;b+=D.nodes[i].bytes;}
    const send = hi >= N-1 ? "output" : packed(hi) ? `${D.sends[hi].length} values${D.sendBytes[hi] != null ? ", " + hb(D.sendBytes[hi]) : ""}`
        : (D.nodes[hi].out != null ? hb(D.nodes[hi].out) : "1 tensor");
    const ties = D.tied.filter(([a,c])=>a<lo && c>=lo && c<=hi).length;
    rows += `<tr><td><span class=sw style="background:var(--k${k%6})"></span>slice ${k}${ties?" <span style='color:var(--tied)'>(tied replica)</span>":""}</td>
      <td>${lo}–${hi} (${hi-lo+1})</td><td>${fmt(p)}</td><td>${hb(b)}</td>
      <td class=barcell><div style="width:${Math.max(1,160*p/total)}px;background:var(--k${k%6})"></div></td><td>${send}</td></tr>`; });
  document.getElementById("slices").innerHTML = rows;
  const cs=[...cuts].sort((a,b)=>a-b);
  const arg = D.packed ? "[" + cs.join(", ") + "]" : "[" + cs.map(i=>JSON.stringify(D.nodes[i].name)).join(", ") + "]";
  const mt = cs.some(packed) ? ", multi_tensor=True" : "";
  document.getElementById("code").textContent = cs.length
    ? `ts.slice(model, cuts=${arg}${D.packed ? ", pack=..." : ""}${mt})` : "no cuts: one slice";
}
const tip = document.getElementById("tip");
function packed(g){ return D.sends[g].length > 1 || D.sends[g][0] !== D.nodes[g].name; }
function crossing(g){ return D.arcs.map((a,k)=>[a,k]).filter(([[u,v]])=>u<=g && v>g && u!==g); }
document.querySelectorAll(".gap").forEach(el=>{ const g=+el.dataset.g;
  el.addEventListener("click", ()=>{ if(!valid.has(g)) return; cuts.has(g)?cuts.delete(g):cuts.add(g); render(); });
  el.addEventListener("mousemove", ev=>{ let msg;
    if (valid.has(g) && packed(g)) {
      crossing(g).forEach(([,k])=>document.getElementById("arc"+k).classList.add("hl"));
      msg = (cuts.has(g) ? "remove the cut" : "cut") + ` after ${D.nodes[g].name}: sends ${D.sends[g].join(", ")}` +
        ` packed into one tensor` + (D.sendBytes[g] != null ? `, ${hb(D.sendBytes[g])} per batch` : "");
    } else if (valid.has(g)) msg = (cuts.has(g) ? "remove the cut" : "cut") + ` after ${D.nodes[g].name}` +
        (D.nodes[g].shape ? `: sends ${D.nodes[g].shape}, ${hb(D.nodes[g].out)} per batch` : "");
    else { const c = crossing(g);
      c.forEach(([,k])=>document.getElementById("arc"+k).classList.add("hl"));
      msg = c.length ? "can't cut: " + c.map(([[u,v]])=>`${u<0?"input":D.nodes[u].name} → ${D.nodes[v].name}`).join(", ")
          + " would cross too" : "can't cut here (a later node reads the model input, or nothing follows)"; }
    tip.textContent = msg; tip.style.display="block"; tip.style.left=(ev.clientX+14)+"px"; tip.style.top=(ev.clientY+10)+"px"; });
  el.addEventListener("mouseleave", ()=>{ tip.style.display="none";
    document.querySelectorAll(".arc.hl").forEach(a=>a.classList.remove("hl")); }); });
document.getElementById("reset").onclick = ()=>{ cuts = new Set(D.cuts); render(); };
document.getElementById("clear").onclick = ()=>{ cuts = new Set(); render(); };
render();
</script></body></html>"""
