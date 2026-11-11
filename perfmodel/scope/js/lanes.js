// The time view: one canvas for the lanes (redrawn on zoom), one above it for the cursor and the
// selection (redrawn every frame). Three lenses share one time axis: resources, warps, phases.
import { COLOR, PIPES, PORTS, lowerBound, fmt } from './data.js';

const GUTTER = 172;
const RULER = 24;
const RIBBON = 16;
const WHY_COLOR = (y) => (y.kind === 'res' ? (COLOR[y.res.split('/')[0]] ?? COLOR.mio) : y.kind === 'sb' ? '#6a7079' : y.kind === 'bar' ? '#454a52' : '#32363d');
const REGION_COLOR = { mma: COLOR.tensor, fp32: COLOR.fma, alu: COLOR.alu, xu: COLOR.xu, ldsm: COLOR.smem_array, cp_async: COLOR.lsu, smem_op: COLOR.smem_array, tma_load: COLOR.lsu, stg: COLOR.store_port };

export class Lanes {
  constructor(base, over, app) {
    Object.assign(this, { base, over, app, g: base.getContext('2d', { alpha: false }), o: over.getContext('2d') });
    this.lens = 'resources'; this.sy = 0; this.rows = []; this.W = 0; this.H = 0;
    this.v0 = 0; this.v1 = 1; this.hover = null;
    new ResizeObserver(() => this.resize()).observe(base.parentElement);
    this._events();
  }

  setTrace(tr) {
    this.tr = tr; this.v0 = tr.t0; this.v1 = tr.cycles; this.sy = 0;
    this.regionColor = tr.d.regions.map((r) => REGION_COLOR[r.top[0]?.split(' ')[0]] ?? '#4a5059');
    this.resize();
  }

  setLens(l) { this.lens = l; this.sy = 0; this.layout(); this.draw(); }

  setView(a, b) {
    const tr = this.tr; const span = Math.max(8, Math.min(b - a, tr.cycles - tr.t0));
    a = Math.max(tr.t0, Math.min(a, tr.cycles - span));
    this.v0 = a; this.v1 = a + span;
    if (this.lens !== 'resources') this.layout();
    this.draw(); this.app.onView();
  }

  // blocks that ran inside the view: a trace can hold many more than fit
  _warpsInView() {
    const { tr } = this;
    return tr.warps.filter((w) => { const b = tr.d.blocks[w.block]; return b && b[1] >= this.v0 && b[0] <= this.v1; });
  }

  x(t) { return GUTTER + ((t - this.v0) / (this.v1 - this.v0)) * (this.W - GUTTER - 10); }

  tAt(x) { return this.v0 + ((x - GUTTER) / (this.W - GUTTER - 10)) * (this.v1 - this.v0); }

  resize() {
    const box = this.base.parentElement.getBoundingClientRect();
    if (!box.width) return;
    const dpr = window.devicePixelRatio || 1;
    this.W = box.width; this.H = box.height; this.dpr = dpr;
    for (const c of [this.base, this.over]) { c.width = box.width * dpr; c.height = box.height * dpr; c.style.width = `${box.width}px`; c.style.height = `${box.height}px`; }
    if (this.tr) { this.layout(); this.draw(); }
  }

  layout() {
    const tr = this.tr; if (!tr) return;
    const rows = []; let y = RULER + RIBBON + 6;
    const add = (r) => { r.y = y; rows.push(r); y += r.h; };
    if (this.lens === 'resources') {
      const free = this.H - y - 8; const ports = PORTS.filter(([p]) => tr.port[p].svc.s.length + tr.port[p].que.s.length);
      const pipes = PIPES.filter((r) => [...Array(tr.nsp).keys()].some((p) => tr.pipe[`${r}:${p}`]));
      const ph = Math.max(28, Math.min(46, (free * 0.62) / Math.max(1, pipes.length)));
      const qh = Math.max(20, Math.min(30, (free * 0.38) / Math.max(1, ports.length)));
      for (const r of pipes) add({ type: 'pipe', key: r, res: r, label: { mio: 'Memory slot', tensor: 'Tensor', fma: 'FMA', alu: 'ALU', xu: 'XU' }[r], sub: `×${tr.nsp}`, h: ph });
      for (const [p, lab] of ports) add({ type: 'port', key: p, res: p === 'l2' ? 'l2_chip' : p === 'dram' ? 'dram_queue' : p, label: lab, h: qh });
    } else if (this.lens === 'warps') {
      const warps = this._warpsInView();
      const wh = Math.max(11, Math.min(20, (this.H - y - 10) / Math.max(1, warps.length * 1.1)));
      let last = null;
      for (const w of warps) {
        if (last !== null && w.block !== last) y += wh * 0.4;
        add({ type: 'warp', key: `w${w.id}`, w: w.id, label: w.block !== last ? `Block ${w.block}` : '', sub: `w${w.id}·SP${w.smsp}`, h: wh });
        last = w.block;
      }
    } else {
      const warps = this._warpsInView();
      const wh = Math.max(14, Math.min(26, (this.H - y - 10) / Math.max(1, warps.length * 1.12)));
      let last = null;
      for (const w of warps) {
        if (last !== null && w.block !== last) y += wh * 0.5;
        add({ type: 'phase', key: `p${w.id}`, w: w.id, label: w.block !== last ? `Block ${w.block}` : '', sub: `w${w.id}${w.role ? ` ${w.role.slice(0, 4)}` : ''}`, h: wh });
        last = w.block;
      }
      if (!this.spansBy) this.spansBy = new Map();
      this.spansBy.clear();
      for (const s of tr.d.spans) (this.spansBy.get(s[0]) ?? this.spansBy.set(s[0], []).get(s[0])).push(s);
    }
    this.rows = rows; this.total = y;
    this.rowOf = new Map(rows.map((r) => [r.key, r]));
  }

  draw() {
    const { g, tr } = this; if (!tr) return;
    g.setTransform(this.dpr, 0, 0, this.dpr, 0, 0);
    g.fillStyle = '#0d0f12'; g.fillRect(0, 0, this.W, this.H);
    const top = RULER + RIBBON + 4;
    g.save(); g.beginPath(); g.rect(0, top, this.W, this.H - top); g.clip();
    for (const r of this.rows) this._rowBack(r, r.y - this.sy);
    for (const r of this.rows) {
      const y = r.y - this.sy; if (y + r.h < top || y > this.H) continue;
      if (r.type === 'pipe') this._pipe(r, y);
      else if (r.type === 'port') this._port(r, y);
      else if (r.type === 'warp') this._warp(r, y);
      else this._phase(r, y);
    }
    this._path();
    g.fillStyle = '#0d0f12'; g.fillRect(0, top, GUTTER, this.H - top);
    for (const r of this.rows) {
      const y = r.y - this.sy; if (y + r.h >= top && y <= this.H) this._rowLabel(r, y);
    }
    g.restore();
    this._ruler(); this._ribbon();
    this.drawOver();
  }

  _rowBack(r, y) {
    const { g } = this;
    if (r.type === 'pipe' || r.type === 'port') {
      g.fillStyle = '#14171b'; g.fillRect(GUTTER, y + 1, this.W - GUTTER - 10, r.h - 3);
    }
  }

  _rowLabel(r, y) {
    const { g } = this;
    g.textBaseline = 'middle'; g.fillStyle = '#c9c9c3'; g.font = '500 12px "IBM Plex Sans", system-ui';
    if (r.label) g.fillText(r.label, 14, y + r.h / 2);
    if (r.sub) {
      g.fillStyle = '#7d8187'; g.font = '11px "IBM Plex Mono", monospace';
      const tx = r.type === 'pipe' ? 96 : 84;
      g.fillText(r.sub, tx, y + r.h / 2);
    }
  }

  _pipe(r, y) {
    const { g, tr } = this; const sub = (r.h - 3) / tr.nsp; const c = COLOR[r.res];
    for (let p = 0; p < tr.nsp; p++) {
      const l = tr.pipe[`${r.res}:${p}`]; if (!l) continue;
      const yy = y + 1 + p * sub; const h = Math.max(2, sub - 1);
      let k = Math.max(0, lowerBound(l.s, this.v0) - 2); let lastPx = -1;
      for (; k < l.s.length && l.s[k] < this.v1; k++) {
        const a = this.x(l.s[k]); const b = this.x(l.e[k]);
        if (Math.floor(b) === lastPx && b - a < 1) continue;
        lastPx = Math.floor(b);
        g.fillStyle = c; g.globalAlpha = l.own[k] ? 1 : 0.4;
        g.fillRect(a, yy, Math.max(0.8, b - a - (b - a > 3 ? 0.6 : 0)), h);
      }
    }
    g.globalAlpha = 1;
  }

  // in-flight depth per pixel: faint for queued, solid for in service
  _port(r, y) {
    const { g, tr } = this; const P = tr.port[r.key]; const w = Math.floor(this.W - GUTTER - 10);
    const depth = (col) => {
      const a = new Float32Array(w + 1);
      for (let k = Math.max(0, lowerBound(col.s, this.v0) - 800); k < col.s.length && col.s[k] < this.v1; k++) {
        if (col.e[k] <= this.v0) continue;
        const x0 = Math.max(0, this.x(col.s[k]) - GUTTER); const x1 = Math.min(w, this.x(col.e[k]) - GUTTER);
        if (x1 <= 0 || x0 >= w) continue;
        for (let i = Math.floor(x0); i <= Math.min(w, Math.floor(x1)); i++) a[i] += 1;
      }
      return a;
    };
    const q = depth(P.que); const s = depth(P.svc); const peak = Math.max(1, ...q.map((v, i) => v + s[i]));
    const c = COLOR[r.key]; const H = r.h - 4;
    for (let i = 0; i <= w; i++) {
      const hs = (s[i] / peak) * H; const hq = (q[i] / peak) * H;
      if (s[i]) { g.fillStyle = c; g.fillRect(GUTTER + i, y + r.h - 2 - hs, 1, Math.max(1.2, hs)); }
      if (q[i]) { g.fillStyle = c; g.globalAlpha = 0.32; g.fillRect(GUTTER + i, y + r.h - 2 - hs - hq, 1, Math.max(1.2, hq)); g.globalAlpha = 1; }
    }
    g.fillStyle = '#7d8187'; g.font = '11px "IBM Plex Mono", monospace'; g.textAlign = 'right';
    g.fillText(`peak ${peak}`, GUTTER - 8, y + r.h / 2); g.textAlign = 'left';
  }

  _warp(r, y) {
    const { g, tr } = this; const ix = tr.perWarp.get(r.w); if (!ix) return;
    const ts = tr.warpT.get(r.w); const h = r.h - 2;
    let k = Math.max(0, lowerBound(ts, this.v0) - 1);
    g.fillStyle = '#14171b'; g.fillRect(GUTTER, y, this.W - GUTTER - 10, h);
    const CL = { mma: COLOR.tensor, fp32: COLOR.fma, fma_half_rate: COLOR.fma, half_to_float: COLOR.fma, alu: COLOR.alu, alu_wide: COLOR.alu, xu: COLOR.xu, ldsm: COLOR.smem_array, stsm: COLOR.smem_array, smem_op: COLOR.smem_array, cp_async: COLOR.lsu, ldg: COLOR.lsu, stg: COLOR.lsu, tma_load: COLOR.lsu, tma_store: COLOR.lsu };
    let lastPx = -1;
    for (; k < ix.length && ts[k] < this.v1 + 1; k++) {
      const i = ix[k]; const t = tr.t[i]; const ready = tr.r[i];
      if (t > ready) {
        const a = this.x(Math.max(ready, this.v0)); const b = this.x(t);
        if (b - a >= 0.5) { g.fillStyle = WHY_COLOR(tr.why[tr.y[i]]); g.globalAlpha = 0.34; g.fillRect(a, y, b - a, h); g.globalAlpha = 1; }
      }
      const px = Math.floor(this.x(t));
      if (px !== lastPx) { lastPx = px; g.fillStyle = CL[tr.ins[tr.pc[i]].c] ?? '#8a8f98'; g.fillRect(px, y, Math.max(1, this.x(t + 1) - this.x(t) - 0.4), h); }
    }
  }

  _phase(r, y) {
    const { g, tr } = this; const list = this.spansBy.get(r.w); if (!list) return;
    const h = r.h - 2; g.font = '11px "IBM Plex Sans", system-ui'; g.textBaseline = 'middle';
    for (const [, reg, a, b] of list) {
      if (b < this.v0 || a > this.v1) continue;
      const x0 = this.x(a); const x1 = this.x(b); const w = Math.max(1, x1 - x0 - 1.5);
      g.fillStyle = this.regionColor[reg]; g.globalAlpha = 0.55;
      g.beginPath(); g.roundRect(x0, y, w, h, Math.min(3, w / 2)); g.fill(); g.globalAlpha = 1;
      if (w > 56) {
        g.save(); g.beginPath(); g.rect(x0, y, w, h); g.clip();
        g.fillStyle = '#ece9e2'; g.fillText(`P${reg} · ${tr.d.regions[reg].top.join(', ')}`, x0 + 6, y + h / 2);
        g.restore();
      }
    }
  }

  _yOf(seg) {
    const [, , res, , , wid] = seg;
    if (this.lens === 'resources') {
      const name = res.split('/')[0]; const key = name === 'l2/dram' ? 'l2' : name === 'dram_queue' ? 'dram' : name;
      const row = this.rowOf.get(key);
      if (!row) return null;
      if (row.type === 'pipe' && wid != null) {
        const s = this.tr.warpBy.get(wid).smsp; const sub = (row.h - 3) / this.tr.nsp;
        return row.y + 1 + (s + 0.5) * sub;
      }
      return row.y + row.h / 2;
    }
    const row = wid != null ? this.rowOf.get(`${this.lens === 'warps' ? 'w' : 'p'}${wid}`) : null;
    return row ? row.y + row.h / 2 : null;
  }

  _path() {
    const { g, tr } = this; if (!this.app.showPath) return;
    const a = lowerBound(tr.critStart, this.v0) - 1; const b = lowerBound(tr.critStart, this.v1) + 1;
    if (b - a > 2400) return; // the ribbon carries it at this zoom
    g.lineJoin = 'round'; g.lineCap = 'round'; g.lineWidth = 3.5; g.strokeStyle = '#0d0f12';
    const trace = () => {
      g.beginPath(); let last = null;
      for (let i = Math.max(0, a); i < Math.min(tr.crit.length, b); i++) {
        const s = tr.crit[i]; const y0 = this._yOf(s); if (y0 == null) continue;
        const Y = y0 - this.sy; const x0 = this.x(s[0]); const x1 = this.x(s[1]);
        if (last == null) g.moveTo(x0, Y); else g.lineTo(x0, Y);
        g.lineTo(x1, Y); last = Y;
      }
      g.stroke();
    };
    trace(); g.lineWidth = 1.6; g.strokeStyle = COLOR.crit; trace();
  }

  _ruler() {
    const { g, tr } = this;
    g.fillStyle = '#101317'; g.fillRect(0, 0, this.W, RULER);
    const span = this.v1 - this.v0; const pow = 10 ** Math.floor(Math.log10(span / 8));
    const step = [1, 2, 5, 10].map((m) => m * pow).find((s) => (this.W - GUTTER) / (span / s) >= 86);
    g.font = '11px "IBM Plex Mono", monospace'; g.textBaseline = 'middle';
    for (let t = Math.ceil(this.v0 / step) * step; t < this.v1; t += step) {
      const x = this.x(t); g.fillStyle = '#272c33'; g.fillRect(x, RULER - 5, 1, 5);
      g.fillStyle = '#8b8f94'; g.fillText(`${fmt(t)}`, x + 4, RULER / 2 - 1);
    }
    g.fillStyle = '#6e7276'; g.font = '11px "IBM Plex Sans", system-ui';
    g.fillText(`${fmt(tr.us(span), 2)} µs in view`, 14, RULER / 2);
    g.fillStyle = '#272c33'; g.fillRect(0, RULER - 1, this.W, 1);
  }

  _ribbon() {
    const { g, tr } = this; const y = RULER + 3; const w = this.W - GUTTER - 10;
    g.fillStyle = '#0d0f12'; g.fillRect(0, RULER, this.W, RIBBON + 6);
    g.font = '500 12px "IBM Plex Sans", system-ui'; g.fillStyle = COLOR.crit; g.textBaseline = 'middle';
    g.fillText('Critical path', 14, y + RIBBON / 2 - 1);
    g.fillStyle = '#14171b'; g.fillRect(GUTTER, y, w, RIBBON - 2);
    const a = Math.max(0, lowerBound(tr.critStart, this.v0) - 1); const b = lowerBound(tr.critStart, this.v1);
    const best = new Array(Math.ceil(w) + 1).fill(null).map(() => ({}));
    for (let i = a; i < b; i++) {
      const s = tr.crit[i];
      const p0 = Math.max(0, Math.floor(this.x(s[0]) - GUTTER)); const p1 = Math.min(best.length - 1, Math.floor(this.x(s[1]) - GUTTER));
      for (let p = p0; p <= p1; p++) best[p][s[2]] = (best[p][s[2]] ?? 0) + s[1] - s[0];
    }
    best.forEach((m, p) => {
      let top = null; let v = 0;
      for (const k in m) if (m[k] > v) { v = m[k]; top = k; }
      if (top) { g.fillStyle = COLOR[top.split('/')[0]] ?? COLOR.lsu; g.fillRect(GUTTER + p, y, 1, RIBBON - 2); }
    });
  }

  drawOver() {
    const { o, tr, app } = this; if (!tr) return;
    o.setTransform(this.dpr, 0, 0, this.dpr, 0, 0);
    o.clearRect(0, 0, this.W, this.H);
    const top = RULER;
    if (app.sel != null) this._selection();
    if (app.selPc != null) this._pcMarks();
    const x = this.x(app.t);
    if (x >= GUTTER && x <= this.W - 10) {
      o.strokeStyle = '#ece9e2'; o.lineWidth = 1; o.beginPath(); o.moveTo(x + 0.5, top - 12); o.lineTo(x + 0.5, this.H); o.stroke();
      o.fillStyle = '#ece9e2'; o.beginPath(); o.moveTo(x - 4, 12); o.lineTo(x + 5, 12); o.lineTo(x + 0.5, 18); o.fill();
    }
  }

  _anchor(i) {
    const { tr } = this; const w = tr.w[i]; const s = tr.warpBy.get(w).smsp;
    if (this.lens === 'resources') {
      const c = tr.machine.classes.find((q) => q.name === tr.ins[tr.pc[i]].c);
      const res = c?.hold?.[0]?.[0] === 'either' ? c.either[0] : c?.hold?.[0]?.[0] ?? (c?.gates?.includes('mio') ? 'mio' : null);
      const row = res && this.rowOf.get(res);
      if (!row) return null;
      return row.y + 1 + (s + 0.5) * ((row.h - 3) / tr.nsp) - this.sy;
    }
    const row = this.rowOf.get(`${this.lens === 'warps' ? 'w' : 'p'}${w}`);
    return row ? row.y + row.h / 2 - this.sy : null;
  }

  _selection() {
    const { o, tr, app } = this; const i = app.sel;
    const steps = [{ i }, ...tr.chain(i)];
    o.lineWidth = 1.4;
    for (let k = steps.length - 1; k > 0; k--) {
      const to = steps[k - 1].i; const from = steps[k].i; if (from < 0) continue;
      const y0 = this._anchor(from); const y1 = this._anchor(to); if (y0 == null || y1 == null) continue;
      const x0 = this.x(tr.t[from]); const x1 = this.x(tr.t[to]);
      o.strokeStyle = k === 1 ? '#ffffff' : 'rgba(255,255,255,.5)'; o.fillStyle = o.strokeStyle;
      o.beginPath(); o.moveTo(x0, y0);
      const mx = (x0 + x1) / 2; o.bezierCurveTo(mx, y0, mx, y1, x1, y1); o.stroke();
      o.beginPath(); o.arc(x0, y0, 2.6, 0, 7); o.fill();
    }
    const y = this._anchor(i);
    if (y != null) {
      const x = this.x(tr.t[i]);
      o.strokeStyle = '#fff'; o.lineWidth = 2; o.beginPath(); o.arc(x, y, 5, 0, 7); o.stroke();
    }
  }

  _pcMarks() {
    const { o, tr, app } = this; const a = lowerBound(tr.t, this.v0); const b = lowerBound(tr.t, this.v1);
    o.fillStyle = '#fff'; let n = 0; let last = -9;
    for (let i = a; i < b && n < 6000; i++) {
      if (tr.pc[i] !== app.selPc) continue;
      const x = this.x(tr.t[i]); const y = this._anchor(i);
      if (y == null || Math.abs(x - last) < 0.5) continue;
      o.fillRect(x - 0.5, y - 3, Math.max(1.4, this.x(tr.t[i] + 1) - x), 6); last = x; n++;
    }
  }

  // hit test: the instruction under the pointer
  pick(x, y) {
    const { tr } = this; const t = this.tAt(x); const Y = y + this.sy;
    const r = this.rows.find((q) => Y >= q.y && Y < q.y + q.h); if (!r || x < GUTTER) return null;
    const near = 4 * ((this.v1 - this.v0) / (this.W - GUTTER));
    if (r.type === 'pipe') {
      const p = Math.min(tr.nsp - 1, Math.floor((Y - r.y - 1) / ((r.h - 3) / tr.nsp)));
      const l = tr.pipe[`${r.res}:${p}`]; if (!l) return { res: r.res };
      let k = lowerBound(l.s, t) - 1;
      if (k < 0 || t > l.e[k] + near) k = lowerBound(l.s, t - near);
      if (k >= l.s.length || Math.abs(l.s[k] - t) > Math.max(near, l.e[k] - l.s[k] + near)) return { res: r.res };
      return { i: tr.issueAt(l.w[k], l.s[k]), res: r.res };
    }
    if (r.type === 'port') return { res: r.res };
    const j = tr.warpState(r.w, t);
    return j.i != null ? { i: j.i } : null;
  }

  describe(x, y) {
    const { tr } = this; const p = this.pick(x, y); if (!p || p.i == null || p.i < 0) return null;
    const i = p.i; const ins = tr.ins[tr.pc[i]]; const wait = tr.t[i] - tr.r[i];
    return `${ins.t}\nw${tr.w[i]} · t ${fmt(tr.t[i], 1)}${wait > 0 ? ` · waited ${fmt(wait, 1)} on ${tr.why[tr.y[i]].label}` : ''}`;
  }

  _events() {
    const c = this.over; let drag = null;
    c.addEventListener('wheel', (e) => {
      e.preventDefault();
      if (e.shiftKey) { this.sy = Math.max(0, Math.min(Math.max(0, this.total - this.H + 8), this.sy + e.deltaY)); this.draw(); return; }
      const span = this.v1 - this.v0;
      if (Math.abs(e.deltaX) > Math.abs(e.deltaY)) { const d = (e.deltaX / this.W) * span; this.setView(this.v0 + d, this.v1 + d); return; }
      const box = c.getBoundingClientRect(); const anchor = this.tAt(e.clientX - box.left); const f = Math.exp(e.deltaY * 0.0022);
      this.setView(anchor - (anchor - this.v0) * f, anchor + (this.v1 - anchor) * f);
    }, { passive: false });
    c.addEventListener('pointerdown', (e) => {
      const box = c.getBoundingClientRect(); const x = e.clientX - box.left; const y = e.clientY - box.top;
      c.setPointerCapture(e.pointerId);
      drag = y < RULER ? { cursor: true } : { x, v0: this.v0, v1: this.v1, moved: false };
      if (drag.cursor) this.app.seek(this.tAt(x));
    });
    c.addEventListener('pointermove', (e) => {
      const box = c.getBoundingClientRect(); const x = e.clientX - box.left; const y = e.clientY - box.top;
      if (!drag) { this.app.tip(this.describe(x, y), e.clientX, e.clientY); return; }
      if (drag.cursor) { this.app.seek(this.tAt(x)); return; }
      if (Math.abs(x - drag.x) > 3) drag.moved = true;
      if (drag.moved) { const dt = ((drag.x - x) / (this.W - GUTTER - 10)) * (drag.v1 - drag.v0); this.setView(drag.v0 + dt, drag.v1 + dt); }
    });
    c.addEventListener('pointerleave', () => this.app.tip(null));
    c.addEventListener('pointerup', (e) => {
      const box = c.getBoundingClientRect();
      if (drag && !drag.cursor && !drag.moved) this.app.pick(this.pick(e.clientX - box.left, e.clientY - box.top));
      drag = null;
    });
    c.addEventListener('dblclick', (e) => {
      const box = c.getBoundingClientRect(); const t = this.tAt(e.clientX - box.left); const s = (this.v1 - this.v0) / 4;
      this.setView(t - s / 2, t + s / 2);
    });
  }
}
