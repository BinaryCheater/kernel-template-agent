// The other interiors, built from what the records and the trace say so that nothing is bound by
// hand: a pipe's classes come from machine.yaml, a rule's parameters from the engine's code
// (atlas.rules[...].reads), a unit's paths from the stages of this trace.
import { COLOR, fmt, lowerBound, pct } from './data.js';
import { BLOCK, SharedMemory, STATUS, Tensor, Unit, el } from './units.js';

const CLASS_COLOR = { mma: COLOR.tensor, fp32: COLOR.fma, fma_half_rate: COLOR.fma, half_to_float: COLOR.fma, alu: COLOR.alu, alu_wide: COLOR.alu, alu_iadd: COLOR.alu, xu: COLOR.xu, ldsm: COLOR.smem_array, stsm: COLOR.smem_array, smem_op: COLOR.smem_array, cp_async: COLOR.lsu, ldg: COLOR.lsu, stg: COLOR.store_port, red: COLOR.store_port, local: COLOR.xu, tma_load: COLOR.tma_engine, tma_store: COLOR.tma_engine };
const clsColor = (c) => CLASS_COLOR[c] ?? '#7b8089';
const UNIT_OF = { smem_array: 'smem_array', lsu: 'lsu', l2: 'l2_chip', 'l2/dram': 'l2_chip', l2_return: 'l2_return', dram: 'dram', dram_queue: 'dram', store_port: 'store_port', tma_engine: 'tma_engine', xu: 'xu', fma: 'fma', alu: 'alu', tensor: 'tensor' };
const short = (s, n) => (s.length > n ? `${s.slice(0, n - 1)}…` : s);

// where a unit stands in the machine: which level is one instance, so who shares it can be seen
function scopeGlyph(p, x, y, scope) {
  const lit = '#e8824f'; const dim = '#23282f';
  el('rect', { x, y, width: 92, height: 52, rx: 4, fill: 'none', stroke: scope === 'chip' ? lit : '#3a4049', 'stroke-dasharray': scope === 'chip' ? '' : '3 3' }, p);
  el('rect', { x: x + 4, y: y + 5, width: 84, height: 42, rx: 3, fill: 'none', stroke: scope === 'tpc' ? lit : '#3a4049' }, p);
  [0, 1].forEach((m) => {
    const X = x + 8 + m * 39; const on = m === 0 && scope === 'sm';
    el('rect', { x: X, y: y + 9, width: 37, height: 34, rx: 2, fill: on ? 'rgba(232,130,79,.25)' : 'none', stroke: on ? lit : '#3a4049' }, p);
    for (let s = 0; s < 4; s++) el('rect', { x: X + 3 + (s % 2) * 16, y: y + 12 + Math.floor(s / 2) * 14, width: 14, height: 12, rx: 1.5, fill: scope === 'smsp' && m === 0 && s === 0 ? lit : dim }, p);
  });
}
const SCOPE_TEXT = { smsp: 'one per sub-partition', sm: 'one per SM', tpc: 'shared by the two SMs of a TPC', chip: 'shared by the whole chip' };

class Base extends Unit {
  static H = 500;

  // inventory rows of this unit as nodes: dashed when not modelled, amber when partial or assumed
  mechNodes(y, res, x0 = 14, x1 = 1118, rows = 2) {
    const list = (this.app.atlas?.mechanisms ?? []).filter((m) => res.some((r) => m.resources.includes(r)));
    if (!list.length) return;
    el('text', { x: x0, y: y - 7, class: 'sec' }, this.svg, `What we found here, ${list.length} rows`);
    const cols = 4; const w = (x1 - x0 - (cols - 1) * 6) / cols; const max = cols * rows;
    list.slice(0, max).forEach((m, i) => {
      const x = x0 + (i % cols) * (w + 6); const Y = y + Math.floor(i / cols) * 30;
      const g = el('g', { class: 'go', 'data-mid': m.id }, this.svg);
      el('title', {}, g, m.name);
      el('rect', { x, y: Y, width: w, height: 25, rx: 4, class: `node ${m.code === 'X' ? 'absent' : ['A', 'P'].includes(m.code) ? 'open' : ''}` }, g);
      el('circle', { cx: x + 10, cy: Y + 12.5, r: 3, fill: { M: '#7fc77f', A: '#e6b95c', P: '#86b5ee', X: '#e88a7a' }[m.code] ?? '#7d8187' }, g);
      el('text', { x: x + 19, y: Y + 16.5, class: 'tm' }, g, short(m.name, Math.floor(w / 6.1)));
    });
  }

  // a table parameter of the machine, drawn as the curve it is
  curve(x, y, w, h, ref, title, xl, yl, logx = false) {
    const pts = this.val(ref); if (!Array.isArray(pts) || !pts.length) return;
    const g = el('g', {}, this.svg); const q = this.entry(ref);
    el('title', {}, g, `${ref}\n${q?.status}\n${q?.source ?? ''}`);
    el('rect', { x, y, width: w, height: h, rx: 6, class: 'node' }, g);
    el('circle', { cx: x + 12, cy: y + 13, r: 3, fill: STATUS[q?.status] ?? '#6e7276' }, g);
    el('text', { x: x + 20, y: y + 17, class: 'bv' }, g, title);
    const L = x + 36; const R = x + w - 10; const T = y + 26; const B = y + h - 20;
    const xs = pts.map((v) => v[0]); const ys = pts.map((v) => v[1]);
    const x0 = logx ? Math.max(1, Math.min(...xs.filter((v) => v > 0))) : Math.min(...xs); const x1 = Math.max(...xs); const y1 = Math.max(...ys);
    const X = (v) => L + (logx ? Math.log(Math.max(v, x0) / x0) / Math.log(x1 / x0 || 2) : (v - x0) / (x1 - x0 || 1)) * (R - L);
    const Y = (v) => B - (v / (y1 || 1)) * (B - T);
    el('path', { d: `M${L} ${T} V${B} H${R}`, fill: 'none', class: 'axis' }, g);
    el('polyline', { points: pts.map((v) => `${X(v[0]).toFixed(1)},${Y(v[1]).toFixed(1)}`).join(' '), fill: 'none', stroke: '#e8824f', 'stroke-width': 1.8 }, g);
    pts.forEach((v) => el('circle', { cx: X(v[0]), cy: Y(v[1]), r: 2.1, fill: '#ece9e2' }, g));
    el('text', { x: L - 5, y: T + 8, class: 'chipn', 'text-anchor': 'end' }, g, fmt(y1, y1 < 10 ? 1 : 0));
    el('text', { x: L, y: B + 13, class: 'chipn' }, g, fmt(x0, 0));
    el('text', { x: R, y: B + 13, class: 'chipn', 'text-anchor': 'end' }, g, `${fmt(x1, 0)} ${xl}`);
    el('text', { x: L + 6, y: T + 8, class: 'chipn' }, g, yl);
  }

  // a list of parameters with their status, most of them read by a rule
  paramList(x, y, refs, n, title) {
    el('text', { x, y: y - 7, class: 'sec' }, this.svg, title);
    refs.slice(0, n).forEach((r, i) => {
      const e = this.entry(r); const g = el('g', {}, this.svg); const Y = y + 6 + i * 17;
      el('title', {}, g, `${r}\n${e?.status}\n${e?.source ?? ''}`);
      el('circle', { cx: x + 3, cy: Y - 4, r: 3, fill: STATUS[e?.status] ?? '#6e7276' }, g);
      el('text', { x: x + 11, y: Y, class: 'num' }, g, short(r.replace('sim.', ''), 30));
      const v = e?.value; el('text', { x: x + 318, y: Y, class: 'val', 'text-anchor': 'end' }, g, typeof v === 'number' ? fmt(v, v % 1 ? 2 : 0) : typeof v === 'boolean' ? String(v) : 'table');
    });
    if (refs.length > n) el('text', { x, y: y + 6 + n * 17, class: 'tm' }, this.svg, `${refs.length - n} more in the inspector`);
  }

  cursorLine(parts, t, t0, t1) {
    const X = 132 + ((t - t0) / (t1 - t0)) * (1118 - 132);
    parts.push(`<line x1="${X}" x2="${X}" y1="${this.sy - 4}" y2="${this.sy + 48}" stroke="#ece9e2"/>`);
  }
}

class Pipe extends Base {
  constructor(svg, tr, smsp, app, res, title) {
    super(svg, tr, smsp, app);
    Object.assign(this, { res, title: `SMSP ${smsp}  ▸  ${title}` });
    const A = app.atlas;
    this.classes = A.classes.filter((c) => (c.hold && Object.keys(c.hold).includes(res)) || c.gates?.includes(res) || c.gates?.some((g) => g?.either?.includes(res)));
    this.busy = {};
    for (let p = 0; p < tr.nsp; p++) {
      const l = tr.pipe[`${res}:${p}`]; if (!l) continue;
      for (let i = 0; i < l.s.length; i++) if (l.own[i]) { const c = tr.ins[l.pc[i]].c; this.busy[c] = (this.busy[c] ?? 0) + l.e[i] - l.s[i]; }
    }
    this._draw();
  }

  holdRef(c) {
    if (c.hold) return c.hold[this.res] ?? c.hold.either ?? null;
    return this.app.atlas.rules[c.handler]?.reads.find((r) => /mio_|smsp_clk_per_line/.test(r)) ?? null;
  }

  _draw() {
    const { svg, tr, res } = this; const color = COLOR[res];
    el('text', { x: 14, y: 50, class: 'sec' }, svg, 'One instruction through the pipe');
    const a = this.node(svg, 14, 60, 190, 84, { title: 'Register read' });
    this.param(a, 24, 104, 'sim.reg_reads_per_clk', ' reads/clk');
    el('text', { x: 24, y: 128, class: 'tm' }, a, 'holds at least sources ÷ rate');
    this.edge(svg, 'M204 102 H234');
    const b = this.node(svg, 234, 60, 460, 84, { title: res === 'mio' ? 'Memory-instruction slot' : `${res.toUpperCase()} pipe` });
    this.bar = el('rect', { x: 246, y: 86, width: 0, height: 22, rx: 3, fill: color }, b);
    el('rect', { x: 246, y: 86, width: 436, height: 22, rx: 3, fill: 'none', stroke: '#3a4049' }, b);
    this.now = el('text', { x: 246, y: 130, class: 'num' }, b, '');
    this.edge(svg, 'M694 102 H724');
    const lat = this.classes.find((c) => c.latency)?.latency;
    const c = this.node(svg, 724, 60, 196, 84, { title: 'Result' });
    if (lat) { this.param(c, 734, 104, lat, ' clk'); el('text', { x: 734, y: 128, class: 'tm' }, c, this.entry(lat)?.status === 'assumed' ? 'assumed: no probe of this latency' : 'write counter clears'); } else el('text', { x: 734, y: 104, class: 'tm' }, c, 'by the class rule, below');
    el('text', { x: 940, y: 50, class: 'sec' }, svg, 'Each sub-partition has one');
    this.sibs = [];
    for (let s = 0; s < tr.nsp; s++) {
      const g = el('g', { class: 'go', 'data-smsp': s }, svg); const y = 60 + s * 22;
      el('rect', { x: 940, y, width: 178, height: 18, rx: 3, class: 'node' }, g);
      const f = el('rect', { x: 941, y: y + 1, width: 0, height: 16, rx: 2, fill: color, opacity: 0.5 }, g);
      el('text', { x: 948, y: y + 13, class: s === this.smsp ? 'bt' : 'tm' }, g, `SMSP ${s}`);
      this.sibs.push({ f, v: el('text', { x: 1110, y: y + 13, class: 'num', 'text-anchor': 'end' }, g, '') });
    }
    el('text', { x: 14, y: 170, class: 'sec' }, svg, 'What runs here');
    [['class', 24], ['opcodes', 150], ['holds', 448], ['result', 556], ['runs here', 664], ['time held', 764]].forEach(([s, x]) => el('text', { x, y: 170, class: 'tm' }, svg, s));
    const total = Object.values(this.busy).reduce((p, q) => p + q, 0) || 1;
    this.classes.slice(0, 6).forEach((k, i) => {
      const y = 178 + i * 30; const g = this.node(svg, 14, y, 906, 26, {});
      el('rect', { x: 14, y, width: 4, height: 26, rx: 2, fill: clsColor(k.name) }, g);
      el('text', { x: 24, y: y + 17, class: 'bt' }, g, short(k.name, 18));
      el('text', { x: 150, y: y + 17, class: 'num' }, g, short((k.base ?? [k.op_prefix]).join(' '), 42));
      const h = this.holdRef(k);
      if (h) this.param(g, 448, y + 17, h, ' clk'); else el('text', { x: 448, y: y + 17, class: 'tm' }, g, 'none');
      if (k.latency) this.param(g, 556, y + 17, k.latency, ' clk');
      const n = tr.ins.reduce((a, x) => a + (x.c === k.name ? x.n : 0), 0);
      el('text', { x: 664, y: y + 17, class: 'num' }, g, fmt(n));
      const v = this.busy[k.name] ?? 0;
      el('rect', { x: 764, y: y + 9, width: 110 * (v / total), height: 7, rx: 2, fill: clsColor(k.name) }, g);
      el('text', { x: 912, y: y + 17, class: 'num', 'text-anchor': 'end' }, g, fmt(v));
    });
    const beside = this.classes.map((k) => k.beside?.tensor).find(Boolean);
    const g = this.node(svg, 940, 178, 178, 86, { title: 'Pushes the tensor pipe', mech: 'mma issue beside FFMA', res: beside ? undefined : undefined });
    if (beside) { this.param(g, 950, 218, beside, ' clk'); el('text', { x: 950, y: 238, class: 'tm' }, g, 'per instruction issued'); el('text', { x: 950, y: 253, class: 'tm' }, g, 'while the pipe is held'); } else el('text', { x: 950, y: 218, class: 'tm' }, g, res === 'mio' ? 'copies only, see Tensor' : 'no delay recorded');
    const e = this.classes.find((k) => k.gates?.some((q) => q?.either));
    const g2 = this.node(svg, 940, 270, 178, 88, { title: 'Shared with' });
    el('text', { x: 950, y: 310, class: 'tm' }, g2, e ? `${e.name}: first free of` : 'nothing: its own pipe');
    if (e) el('text', { x: 950, y: 326, class: 'num' }, g2, e.gates.find((q) => q?.either).either.join(', '));
    this.mechNodes(378, [res], 14, 920, 1);
    this.strip(436, 0);
  }

  update(t) {
    const { tr, smsp, res } = this; const key = `${res}:${smsp}`; const c = COLOR[res];
    const h = tr.holdAt(key, t);
    if (h) {
      const { lane, k } = h; const f = (t - lane.s[k]) / (lane.e[k] - lane.s[k]);
      this.bar.setAttribute('width', 436 * f);
      this.now.textContent = `w${lane.w[k]}  ${short(tr.ins[lane.pc[k]].t, 58)}  ${fmt(lane.e[k] - lane.s[k], 1)} clk`;
    } else { this.bar.setAttribute('width', 0); this.now.textContent = 'idle'; }
    const b = Math.min(tr.nb - 1, Math.floor(t / tr.binSize));
    this.sibs.forEach((s, p) => { const u = Math.min(1, tr.util[`${res}:${p}`]?.[b] ?? 0); s.f.setAttribute('width', 176 * u); s.v.textContent = pct(u); });
    const span = 300; const t0 = t - span * 0.7; const t1 = t + span * 0.3; const parts = [];
    const put = this.stripRow(parts, 'held', this.sy, 26, 132, 1118, t0, t1); const l = tr.pipe[key];
    if (l) for (let k = Math.max(0, lowerBound(l.s, t0) - 1); k < l.s.length && l.s[k] < t1; k++) put(l.s[k], l.e[k], l.own[k] ? BLOCK[tr.warpBy.get(l.w[k]).block % 4] : c, 0.9);
    const put2 = this.stripRow(parts, 'waiting for it', this.sy + 32, 12, 132, 1118, t0, t1);
    for (const w of tr.warps) {
      if (w.smsp !== smsp) continue;
      const ts = tr.warpT.get(w.id); if (!ts) continue; const ix = tr.perWarp.get(w.id);
      for (let k = lowerBound(ts, t0); k < ts.length && tr.r[ix[k]] < t1; k++) { const y = tr.why[tr.y[ix[k]]]; if (y.kind === 'res' && y.res.includes(res)) put2(tr.r[ix[k]], tr.t[ix[k]], BLOCK[w.block % 4], 0.55); }
    }
    this.cursorLine(parts, t, t0, t1);
    this.sg.innerHTML = parts.join('');
  }
}

// traffic through a port or a memory: each class's real stages pass through one shared hub
class Path extends Base {
  constructor(svg, tr, smsp, app, cfg) {
    super(svg, tr, smsp, app);
    Object.assign(this, cfg, { title: cfg.name });
    const set = new Set(cfg.stages);
    this.flows = tr.paths().filter((p) => p.stages.some((s) => set.has(s.res))).map((p) => ({ ...p, here: p.stages.filter((s) => set.has(s.res)) }))
      .sort((a, b) => b.here.reduce((v, s) => v + s.dur, 0) - a.here.reduce((v, s) => v + s.dur, 0));
    this.set = set;
    this._draw();
  }

  _draw() {
    const { svg, tr } = this; const y0 = 44;
    const def = this.app.atlas.yaml_resources[this.res] ?? {}; const scope = this.scope ?? def.scope ?? 'sm';
    scopeGlyph(svg, 14, y0, scope);
    el('text', { x: 118, y: y0 + 16, class: 'bt' }, svg, SCOPE_TEXT[scope] ?? scope);
    if (def.kind) el('text', { x: 118, y: y0 + 32, class: 'tm' }, svg, `kind: ${def.kind}`);
    this.live = el('text', { x: 118, y: y0 + 48, class: 'val' }, svg, '');
    if (def.doc) el('text', { x: 330, y: y0 + 16, class: 'tm' }, svg, short(def.doc, 74));
    if (def.chip_share) {
      const refs = Object.values(def.chip_share);
      refs.forEach((r, i) => this.param(svg, 330 + i * 200, y0 + 40, r, '', i ? 'chip' : 'own'));
    }
    const top = 128; const rowH = 44; const hubX = 372; const hubW = 120; const list = this.flows.slice(0, 4);
    el('text', { x: 14, y: top - 8, class: 'sec' }, svg, list.length > 1 ? `${this.flows.length} kinds of traffic share this` : 'Traffic through this unit');
    if (!list.length) {
      el('text', { x: 14, y: top + 18, class: 'tm' }, svg, 'This trace has no operation on this unit. The rule and its parameters still stand, right and below.');
    } else {
      el('rect', { x: hubX, y: top, width: hubW, height: list.length * rowH - 6, rx: 6, class: 'hub' }, svg);
      el('text', { x: hubX + hubW / 2, y: top + list.length * rowH + 8, class: 'tm', 'text-anchor': 'middle' }, svg, this.name);
    }
    const chip = (x, y, w, s, n) => {
      const q = s.mech.startsWith('queue'); const to = UNIT_OF[s.res]; const part = s.n / n;
      const g = el('g', to && to !== this.res ? { class: 'go', 'data-res': to } : {}, svg);
      el('title', {}, g, `${s.res}: ${s.mech}\nmean ${fmt(s.mean, 1)} clk over the ${pct(part, 1)} of these operations that pass here${s.par ? `\n${s.par}` : ''}`);
      el('rect', { x, y, width: w, height: 30, rx: 4, fill: COLOR[s.res.split('/')[0]] ?? '#3a4049', opacity: q ? 0.28 : 0.72 }, g);
      el('text', { x: x + 5, y: y + 12, class: 'chipn', style: 'fill:#ece9e2' }, g, short(s.mech.replace(/^queue.*/, 'queue').replace(' latency', ''), Math.floor(w / 6.2)));
      el('text', { x: x + 5, y: y + 25, class: 'chipn', style: 'fill:#ece9e2' }, g, `${fmt(s.mean, s.mean < 10 ? 1 : 0)}${part < 0.95 ? ` · ${pct(part, part < 0.1 ? 1 : 0)}` : ''}`);
    };
    list.forEach((f, i) => {
      const y = top + i * rowH + 3; const first = f.stages.indexOf(f.here[0]); const last = f.stages.indexOf(f.here.at(-1));
      el('rect', { x: 14, y, width: 4, height: 30, rx: 2, fill: clsColor(f.cls) }, svg);
      el('text', { x: 24, y: y + 13, class: 'bt' }, svg, short(f.cls, 14));
      el('text', { x: 24, y: y + 27, class: 'num' }, svg, `${fmt(f.n)} ops`);
      this.edge(svg, `M112 ${y + 15} H${hubX}`); this.edge(svg, `M${hubX + hubW} ${y + 15} H784`);
      const before = f.stages.slice(0, first).filter((s) => s.res !== 'issue').slice(-3);
      before.forEach((s, k) => chip(hubX - (before.length - k) * 84, y, 80, s, f.n));
      const hw = (hubW - 8 - (f.here.length - 1) * 3) / f.here.length;
      f.here.forEach((s, k) => chip(hubX + 4 + k * (hw + 3), y, hw, s, f.n));
      f.stages.slice(last + 1).filter((s) => s.res !== 'issue').slice(0, 3).forEach((s, k) => chip(hubX + hubW + 4 + k * 84, y, 80, s, f.n));
    });
    const links = this.params ?? this.tr.machine.links[this.res] ?? [];
    const px = 796; let py = 52;
    (this.curves ?? []).slice(0, 2).forEach((c, i) => this.curve(px + i * 164, 40, 158, 118, ...c));
    if (this.curves?.length) py = 182;
    this.paramList(px, py, links, this.curves?.length ? 9 : 16, 'Parameters the rules read');
    this.mechNodes(362, this.mech ?? [this.res], 14, 784, 2);
    this.strip(436, 0);
  }

  update(t) {
    const { tr } = this; const st = tr.st; const L = tr.label;
    const act = tr.activeStages(t).filter((i) => this.set.has(L[st.r[i]]));
    const q = act.filter((i) => L[st.m[i]].startsWith('queue')).length;
    this.live.textContent = this.latency ? `${act.length - q} in flight · ${q} queued` : `${tr.portBusy(this.port, t) ? 'held' : 'free'} · ${q} queued`;
    const P = tr.port[this.port]; const span = this.span ?? 600; const t0 = t - span * 0.7; const t1 = t + span * 0.3; const parts = [];
    parts.push(`<text x="14" y="${this.sy + 16}" class="tm">in flight</text>`);
    if (P) {
      const X = (v) => 132 + ((v - t0) / (t1 - t0)) * (1118 - 132); const rows = Array(9).fill(-1e12);
      const draw = (col, op) => {
        for (let k = Math.max(0, lowerBound(col.s, t0) - 400); k < col.s.length && col.s[k] < t1; k++) {
          if (col.e[k] < t0) continue;
          const row = rows.findIndex((r) => r <= col.s[k]); if (row < 0) continue;
          rows[row] = col.e[k];
          const l = Math.max(132, X(col.s[k])); const r = Math.min(1118, X(col.e[k]));
          const c = clsColor(tr.ins[col.pc[k]]?.c);
          if (r > l) parts.push(`<rect x="${l.toFixed(1)}" y="${this.sy + row * 5.4}" width="${(r - l).toFixed(1)}" height="4.4" rx="1" fill="${c}" opacity="${op}"/>`);
        }
      };
      draw(P.que, 0.4); rows.fill(-1e12); draw(P.svc, 0.95);
    }
    this.cursorLine(parts, t, t0, t1);
    this.sg.innerHTML = parts.join('');
  }
}

// six counters per warp, live: what a warp is waiting for, on the instruction that set each
class Scheduler extends Base {
  constructor(svg, tr, smsp, app) {
    super(svg, tr, smsp, app);
    this.title = `SMSP ${smsp}  ▸  Warp scheduler`;
    this.reasons = {};
    for (let i = 0; i < tr.n; i++) {
      if (tr.warpBy.get(tr.w[i]).smsp !== smsp) continue;
      const y = tr.why[tr.y[i]]; const k = y.kind === 'res' ? y.res.split('/')[0] : y.kind; const d = tr.t[i] - tr.r[i];
      if (d > 0) this.reasons[k] = (this.reasons[k] ?? 0) + d;
    }
    this._draw();
  }

  _draw() {
    const { svg, tr } = this; const ex = tr.machine.execution;
    el('text', { x: 14, y: 50, class: 'sec' }, svg, 'Issue');
    const a = this.node(svg, 14, 60, 250, 102, { title: 'One instruction per clock', mech: 'Warp scheduler' });
    el('text', { x: 24, y: 100, class: 'tm' }, a, `${ex.issue_per_cycle_per_smsp} per sub-partition, ${ex.scheduler.replaceAll('_', ' ')}`);
    el('text', { x: 24, y: 118, class: 'tm' }, a, `warp to sub-partition: ${ex.warp_to_smsp}`);
    this.param(a, 24, 146, 'sim.older_block_first', '', 'older block first');
    const b = this.node(svg, 14, 170, 250, 110, { title: 'Instruction buffer', mech: 'Instruction cache', res: undefined });
    this.param(b, 24, 208, 'sim.fetch_l0_lines', ' lines of 128 B');
    this.param(b, 24, 228, 'sim.fetch_line_clk', ' clk per line outside');
    this.param(b, 24, 248, 'sim.branch_redirect_clk', ' clk after a branch');
    el('text', { x: 24, y: 268, class: 'tm' }, b, 'a warp\'s wait; others issue meanwhile');
    el('text', { x: 284, y: 50, class: 'sec' }, svg, `Warps on this sub-partition, counters ${ex.dependency_counters}`);
    el('text', { x: 666, y: 50, class: 'tm' }, svg, 'instruction next');
    this.rows = [];
    for (let i = 0; i < 6; i++) {
      const y = 60 + i * 38; const g = el('g', {}, svg);
      el('rect', { x: 284, y, width: 620, height: 33, rx: 5, class: 'node' }, g);
      const dot = el('circle', { cx: 303, cy: y + 16.5, r: 8 }, g);
      const id = el('text', { x: 303, y: y + 20, 'text-anchor': 'middle', style: 'font:500 9.5px var(--mono);fill:#0d0f12' }, g, '');
      const st = el('text', { x: 318, y: y + 20, class: 'tm' }, g, '');
      const cells = Array.from({ length: 6 }, (_, b2) => {
        el('rect', { x: 460 + b2 * 32, y: y + 6, width: 28, height: 21, rx: 3, class: 'track' }, g);
        const f = el('rect', { x: 460 + b2 * 32, y: y + 6, width: 28, height: 21, rx: 3, fill: '#86b5ee', opacity: 0 }, g);
        return { f, n: el('text', { x: 474 + b2 * 32, y: y + 21, 'text-anchor': 'middle', class: 'chipn' }, g, '') };
      });
      this.rows.push({ g, dot, id, st, cells, ins: el('text', { x: 660, y: y + 21, class: 'num' }, g, '') });
    }
    el('text', { x: 924, y: 50, class: 'sec' }, svg, 'Where waiting went');
    const list = Object.entries(this.reasons).sort((p, q) => q[1] - p[1]).slice(0, 8); const peak = list[0]?.[1] ?? 1;
    list.forEach(([k, v], i) => {
      const y = 62 + i * 27; const to = UNIT_OF[k]; const g = el('g', to ? { class: 'go', 'data-res': to } : {}, svg);
      el('text', { x: 924, y: y + 9, class: 'tm' }, g, { sb: 'dependency counter', bar: 'block barrier', ready: 'scheduler' }[k] ?? k);
      el('text', { x: 1118, y: y + 9, class: 'num', 'text-anchor': 'end' }, g, fmt(v));
      el('rect', { x: 924, y: y + 14, width: 194 * (v / peak), height: 5, rx: 2, fill: COLOR[k] ?? '#7b8089' }, g);
    });
    this.mechNodes(310, ['scheduler', 'instruction_buffer', 'barrier'], 14, 1118, 3);
    this.strip(436, 0);
  }

  update(t) {
    const { tr, smsp } = this;
    const alive = tr.warps.filter((w) => { const b = tr.d.blocks[w.block]; return w.smsp === smsp && b && b[0] <= t && t <= b[1]; });
    this.rows.forEach((r, i) => {
      const w = alive[i]; r.g.setAttribute('opacity', w ? 1 : 0.25);
      if (!w) { r.id.textContent = ''; r.st.textContent = ''; r.ins.textContent = ''; r.dot.setAttribute('fill', '#20252b'); r.cells.forEach((c) => { c.f.setAttribute('opacity', 0); c.n.textContent = ''; }); return; }
      const s = tr.warpState(w.id, t);
      r.dot.setAttribute('fill', BLOCK[w.block % 4]); r.id.textContent = w.id;
      r.st.textContent = s.s === 'wait' ? short(s.why.label, 22) : s.s;
      const c = tr.counters(w.id, t);
      r.cells.forEach((q, b) => { const on = c[b] > 0; const waited = s.s === 'wait' && s.why.kind === 'sb' && s.why.b === b; q.f.setAttribute('opacity', on ? (waited ? 1 : 0.45) : 0); q.f.setAttribute('fill', waited ? '#e8824f' : '#86b5ee'); q.n.textContent = on ? c[b] : ''; });
      r.ins.textContent = s.i != null ? short(tr.ins[tr.pc[s.i]].t, 38) : '';
    });
    const span = 260; const t0 = t - span * 0.7; const t1 = t + span * 0.3; const parts = [];
    const put = this.stripRow(parts, 'issued', this.sy, 26, 132, 1118, t0, t1);
    const a = lowerBound(tr.t, t0); const b = lowerBound(tr.t, t1);
    for (let i = a; i < b; i++) if (tr.warpBy.get(tr.w[i]).smsp === smsp) put(tr.t[i], tr.t[i] + 1, clsColor(tr.ins[tr.pc[i]].c), 1);
    this.cursorLine(parts, t, t0, t1);
    this.sg.innerHTML = parts.join('');
  }
}

class Chip extends Base {
  constructor(svg, tr, smsp, app) { super(svg, tr, smsp, app); this.title = 'GB10 chip'; this._draw(); }

  _draw() {
    const { svg, tr } = this; const f = tr.machine.facts; const c = tr.meta.counters ?? {};
    el('text', { x: 14, y: 50, class: 'sec' }, svg, `${f.tpcs} TPC, ${f.sms} SM. One SM is replayed, the rest enter as shares.`);
    for (let i = 0; i < f.tpcs; i++) {
      const x = 14 + (i % 8) * 58; const y = 60 + Math.floor(i / 8) * 52;
      el('rect', { x, y, width: 52, height: 46, rx: 4, fill: 'none', stroke: '#2e333b' }, svg);
      [0, 1].forEach((s) => el('rect', { x: x + 4 + s * 23, y: y + 5, width: 21, height: 36, rx: 2, fill: i === 0 && s === 0 ? '#e8824f' : '#20252b' }, svg));
    }
    el('text', { x: 500, y: 50, class: 'sec' }, svg, 'Limits on one SM');
    [['regs_per_sm', ' registers'], ['smem_per_sm_bytes', ' B shared'], ['max_threads_per_sm', ' threads'], ['max_blocks_per_sm', ' blocks']].forEach(([r, u], i) => this.param(svg, 500, 72 + i * 20, r, u));
    el('text', { x: 500, y: 172, class: 'sec' }, svg, 'Memory behind the SMs');
    [['l2_bytes', ' B L2'], ['l2_effective_bytes', ' B effective'], ['dram_read_GBs', ' GB/s DRAM read']].forEach(([r, u], i) => this.param(svg, 500, 194 + i * 20, r, u));
    el('text', { x: 700, y: 50, class: 'sec' }, svg, 'This launch');
    const rows = [['Blocks on the grid', tr.meta.grid_blocks], ['Blocks replayed here', tr.meta.blocks], ['Resident per SM', tr.meta.resident], ['SM equivalents', tr.meta.active_sms]];
    if (c['launch__registers_per_thread']) rows.push(['Registers per thread', c['launch__registers_per_thread']], ['Limit by registers', c['launch__occupancy_limit_registers']], ['Limit by shared memory', c['launch__occupancy_limit_shared_mem']]);
    rows.forEach(([k, v], i) => { el('text', { x: 700, y: 72 + i * 20, class: 'tm' }, svg, k); el('text', { x: 918, y: 72 + i * 20, class: 'val', 'text-anchor': 'end' }, svg, v == null ? '–' : fmt(v, v % 1 ? 1 : 0)); });
    this.curve(936, 40, 182, 118, 'sim.launch_block_ns_by_threads', 'block launch ns', 'threads', 'ns', true);
    // residency: which block held a slot of this SM when
    el('text', { x: 14, y: 300, class: 'sec' }, svg, 'Blocks on the replayed SM');
    const blocks = Object.entries(tr.d.blocks); const span = tr.cycles - tr.t0 || 1; const slots = [];
    const X = (v) => 132 + ((v - tr.t0) / span) * (1118 - 132);
    for (const [id, [a, b]] of blocks) {
      let s = slots.findIndex((e) => e <= a); if (s < 0) { s = slots.length; slots.push(0); }
      slots[s] = b;
      if (s > 3) continue;
      const y = 310 + s * 15;
      el('rect', { x: X(a), y, width: Math.max(1, X(b) - X(a) - 1), height: 12, rx: 2, fill: BLOCK[+id % 4], opacity: 0.6 }, svg);
      if (X(b) - X(a) > 40) el('text', { x: X(a) + 4, y: y + 10, class: 'chipn', style: 'fill:#0d0f12' }, svg, `block ${id}`);
    }
    Array.from({ length: Math.min(4, slots.length) }, (_, s) => el('text', { x: 14, y: 320 + s * 15, class: 'tm' }, svg, `slot ${s}`));
    this.cur = el('line', { y1: 304, y2: 374, stroke: '#ece9e2' }, svg); this.X = X;
    this.mechNodes(404, ['chip'], 14, 1118, 2);
  }

  update(t) { this.cur.setAttribute('x1', this.X(t)); this.cur.setAttribute('x2', this.X(t)); }
}

const pipe = (res, title) => class extends Pipe { constructor(s, t, p, a) { super(s, t, p, a, res, title); } };
const path = (cfg) => class extends Path { constructor(s, t, p, a) { super(s, t, p, a, cfg); } };

export const UNITS = {
  tensor: Tensor,
  smem_array: SharedMemory,
  fma: pipe('fma', 'FMA pipe'),
  alu: pipe('alu', 'ALU pipe'),
  xu: pipe('xu', 'XU pipe'),
  mio: pipe('mio', 'Memory-instruction slot'),
  scheduler: Scheduler,
  chip: Chip,
  lsu: path({ res: 'lsu', name: 'LSU address stage', stages: ['lsu'], port: 'lsu', mech: ['lsu'] }),
  l1: path({ res: 'l1', name: 'L1 data path', stages: ['l1', 'local_chip'], port: 'l1', mech: ['l1'] }),
  l2_return: path({ res: 'l2_return', name: 'L2 return port', stages: ['l2_return'], port: 'l2_return', span: 2400, mech: ['l2_return'] }),
  tma_engine: path({ res: 'tma_engine', name: 'TMA engine', stages: ['tma_engine'], port: 'tma_engine', mech: ['tma_engine'] }),
  store_queue: path({ res: 'store_queue', name: 'Store queue', stages: ['store_port', 'stg_chip'], port: 'store_port', mech: ['store_port'] }),
  store_port: path({ res: 'store_port', name: 'Store port', stages: ['store_port', 'stg_chip'], port: 'store_port', mech: ['store_port'], curves: [['dirty_l2_extra_us', 'dirty L2 extra µs', 'MiB', 'µs'], ['sim.red_merge_gain_by_bytes', 'atomic merge gain', 'B', '×', true]] }),
  dram_queue: path({ res: 'dram_queue', name: 'DRAM share', stages: ['dram', 'dram_queue', 'l2/dram'], port: 'dram', span: 4000, latency: true, mech: ['dram'] }),
  l2_chip: path({ res: 'l2_chip', name: 'L2', scope: 'chip', stages: ['l2', 'l2/dram', 'l2_chip'], port: 'l2', span: 2400, latency: true, mech: ['l2_chip'], curves: [['l2_warm_miss_by_MiB', 'warm miss rate', 'MiB', 'miss'], ['l2_hit_vs_reuse_distance', 'hit by reuse distance', 'MiB', 'hit']] }),
  dram: path({ res: 'dram_queue', name: 'DRAM', scope: 'chip', stages: ['dram', 'dram_queue', 'l2/dram'], port: 'dram', span: 4000, latency: true, mech: ['dram'], curves: [['dram_loaded_latency_us', 'loaded latency µs', '', 'µs'], ['dram_peak_by_segment_GBs', 'peak rate GB/s', 'seg', 'GB/s']] }),
};
