// The machine map. Structure is hand-placed for the GB10 (another machine needs its own BOX, WIRE);
// everything live comes from the trace at the cursor: pipe occupancy, warp states, queue depth, and
// each dot on a wire, whose position is its stage's progress between that stage's start and end.
import { COLOR, PIPES, pct } from './data.js';
import { UNITS } from './units-more.js';

const NS = 'http://www.w3.org/2000/svg';
const W = 1134; const H = 340;
const SM = { x: 196, y: 12, w: 744, h: 316 };
const PIPE_W = 66; const PIPE_X = 298; const ROW_H = 70; const ROW_Y = 44;
const px = (k) => PIPE_X + k * 73;
const rowY = (i) => ROW_Y + i * ROW_H;
const BOX = {
  smem_array: { x: 700, y: 44, w: 226, h: 84, title: 'Shared memory' },
  lsu: { x: 700, y: 138, w: 70, h: 56, title: 'LSU' },
  l1: { x: 778, y: 138, w: 70, h: 56, title: 'L1' },
  tma_engine: { x: 856, y: 138, w: 70, h: 56, title: 'TMA' },
  l2_return: { x: 700, y: 204, w: 70, h: 56, title: 'L2 return' },
  store_queue: { x: 778, y: 204, w: 70, h: 56, title: 'Store q' },
  dram_queue: { x: 856, y: 204, w: 70, h: 56, title: 'DRAM q' },
  store_port: { x: 700, y: 270, w: 226, h: 46, title: 'Store port' },
  l2_chip: { x: 976, y: 30, w: 142, h: 120, title: 'L2' },
  dram: { x: 976, y: 190, w: 142, h: 110, title: 'DRAM' },
};
const BUS = 684;
// wires: name → polyline; a dot's progress runs along it
const WIRE = {
  req: [[770, 150], [770, 132], [952, 132], [952, 100], [976, 100]],
  ret: [[976, 120], [962, 120], [962, 262], [735, 262]],
  ret_smem: [[735, 204], [735, 196], [692, 196], [692, 110], [700, 110]],
  dram: [[1047, 150], [1047, 190]],
  stores: [[926, 293], [944, 293], [944, 140], [976, 140]],
  tma: [[891, 138], [891, 128]],
};
const mioWire = (s, to) => [[660, rowY(s) + 31], [BUS, rowY(s) + 31], [BUS, to[1]], to];

function el(tag, attrs = {}, parent = null, text = null) {
  const e = document.createElementNS(NS, tag);
  for (const [k, v] of Object.entries(attrs)) e.setAttribute(k, v);
  if (text != null) e.textContent = text;
  parent?.appendChild(e);
  return e;
}

function pointAt(line, f) {
  const seg = []; let total = 0;
  for (let i = 1; i < line.length; i++) { const d = Math.hypot(line[i][0] - line[i - 1][0], line[i][1] - line[i - 1][1]); seg.push(d); total += d; }
  let left = Math.max(0, Math.min(1, f)) * total;
  for (let i = 0; i < seg.length; i++) {
    if (left <= seg[i]) { const k = seg[i] ? left / seg[i] : 0; return [line[i][0] + (line[i + 1][0] - line[i][0]) * k, line[i][1] + (line[i + 1][1] - line[i][1]) * k]; }
    left -= seg[i];
  }
  return line[line.length - 1];
}

export class MachineMap {
  constructor(svg, app) {
    Object.assign(this, { svg, app, cells: {}, dots: [], warpDots: [], queues: {} });
    svg.setAttribute('viewBox', `0 0 ${W} ${H}`);
  }

  setTrace(tr) {
    this.tr = tr;
    if (this.open_) this.openUnit(this.open_.res, this.open_.smsp, false); else this._build();
    this.update(this.app.t);
  }

  // click a unit with an interior: the map zooms into it; everything live keeps following the cursor
  openUnit(res, smsp = 0, animate = true, from = null) {
    const U = UNITS[res]; if (!U) return false;
    this.open_ = { res, smsp };
    this.app.setMapOpen(true);
    const { svg } = this; svg.innerHTML = ''; svg.onclick = null;
    svg.setAttribute('viewBox', `0 0 ${W} ${U.H}`);
    this.unit = new U(svg, this.tr, smsp, this.app);
    el('text', { x: 14, y: 22, class: 'crumbs', 'data-back': '1' }, svg, `◂ SM  ▸  ${this.unit.title}`);
    svg.onclick = (e) => {
      if (e.target.closest?.('[data-back]')) { this.closeUnit(); return; }
      const sp = e.target.closest?.('[data-smsp]');
      if (sp) { this.openUnit(res, +sp.dataset.smsp, false); return; }
      const mid = e.target.closest?.('[data-mid]');
      if (mid) { this.app.openSheet(+mid.dataset.mid); return; }
      const m = e.target.closest?.('[data-mech]');
      if (m) { const hit = this.app.mechBy(m.dataset.mech); if (hit != null) this.app.openSheet(hit); return; }
      const r = e.target.closest?.('[data-res]');
      if (r) { const to = r.dataset.res; this.app.pickRes(to); if (UNITS[to]) this.openUnit(to, 0, true); else this.closeUnit(); }
    };
    if (animate) {
      const o = from ? `${(from[0] / W) * 100}% ${(from[1] / H) * 100}%` : '50% 50%';
      svg.animate([{ transform: 'scale(.3)', transformOrigin: o, opacity: 0 }, { transform: 'scale(1)', transformOrigin: o, opacity: 1 }], { duration: 260, easing: 'cubic-bezier(.2,.7,.2,1)' });
    }
    this.update(this.app.t);
    return true;
  }

  closeUnit() {
    if (!this.open_) return;
    this.open_ = null; this.unit = null;
    this.app.setMapOpen(false);
    this.svg.setAttribute('viewBox', `0 0 ${W} ${H}`);
    this._build(); this.update(this.app.t);
  }

  _box(res, x, y, w, h, title) {
    const g = el('g', { class: 'res', 'data-res': res }, this.svg);
    el('rect', { x, y, width: w, height: h, rx: 6, class: 'box' }, g);
    const t = el('text', { x: x + 9, y: y + 17, class: 'bt' }, g, title);
    const v = el('text', { x: x + 9, y: y + 34, class: 'bv' }, g, '');
    return { g, t, v, box: { x, y, w, h } };
  }

  _build() {
    const { svg, tr } = this; svg.innerHTML = '';
    const f = tr.machine.facts;
    el('text', { x: 14, y: 24, class: 'bt big' }, svg, 'Chip');
    el('text', { x: 14, y: 39, class: 'bv' }, svg, `${f.tpcs} TPC · ${f.sms} SM`);
    const chip = el('g', { class: 'res', 'data-res': 'chip' }, svg);
    for (let r = 0; r < 8; r++) {
      for (let c = 0; c < 6; c++) {
        const x = 14 + c * 20 + (c >> 1) * 4; const y = 50 + r * 20 + (r >> 1) * 5; const lit = r === 1 && c === 2;
        el('rect', { x, y, width: 16, height: 16, rx: 2, class: lit ? 'tile lit' : 'tile' }, chip);
      }
    }
    const wrap = (t, y) => el('text', { x: 14, y, class: 'bv' }, svg, t);
    wrap('One SM is replayed.', 262); wrap('The others enter as', 276); wrap('shares of L2 and DRAM.', 290);
    el('path', { d: `M72 70 L${SM.x} ${SM.y + 4}`, class: 'zoom' }, svg);
    el('path', { d: `M72 86 L${SM.x} ${SM.y + SM.h - 4}`, class: 'zoom' }, svg);
    el('rect', { x: SM.x, y: SM.y, width: SM.w, height: SM.h, rx: 9, class: 'sm' }, svg);
    el('text', { x: 210, y: 32, class: 'bt big' }, svg, 'SM');
    this.smInfo = el('text', { x: 232, y: 32, class: 'bv' }, svg, '');
    for (const line of Object.values(WIRE)) el('path', { d: this._path(line), class: 'wire' }, svg);
    el('path', { d: `M${BUS} ${rowY(0) + 31} V${rowY(3) + 31}`, class: 'wire' }, svg);
    for (const y of [86, 166, 232, 293]) el('path', { d: `M${BUS} ${y} H700`, class: 'wire' }, svg);
    this.cells = {}; this.warpDots = [];
    for (let i = 0; i < tr.nsp; i++) {
      const y = rowY(i);
      el('rect', { x: 208, y, width: 462, height: 62, rx: 6, class: 'row' }, svg);
      const sched = el('g', { class: 'res', 'data-res': 'scheduler', 'data-smsp': i }, svg);
      el('rect', { x: 212, y: y + 4, width: 78, height: 54, rx: 4, class: 'box soft' }, sched);
      el('text', { x: 218, y: y + 16, class: 'bv' }, sched, `SMSP ${i}`);
      this.warpDots[i] = [];
      for (let k = 0; k < 4; k++) this.warpDots[i][k] = el('circle', { cx: 222 + k * 15, cy: y + 34, r: 5, class: 'wd' }, sched);
      this.warpText = this.warpText ?? [];
      this.warpText[i] = el('text', { x: 218, y: y + 54, class: 'bs' }, svg, '');
      PIPES.forEach((r, k) => {
        const x = px(k); const g = el('g', { class: 'res', 'data-res': r, 'data-smsp': i }, svg);
        el('rect', { x, y: y + 8, width: PIPE_W, height: 46, rx: 4, class: 'box' }, g);
        const fill = el('rect', { x: x + 1, y: y + 53, width: PIPE_W - 2, height: 0, rx: 3, fill: COLOR[r], opacity: 0.42 }, g);
        const on = el('rect', { x: x + 6, y: y + 49, width: PIPE_W - 12, height: 2.5, rx: 1, fill: COLOR[r], opacity: 0 }, g);
        el('text', { x: x + 7, y: y + 24, class: 'bt' }, g, r === 'mio' ? 'mem slot' : r);
        if (UNITS[r]) el('text', { x: x + PIPE_W - 7, y: y + 22, class: 'open-mark', 'text-anchor': 'end' }, g, '⤢');
        const v = el('text', { x: x + 7, y: y + 41, class: 'bv mono' }, g, '');
        (this.cells[r] ??= [])[i] = { fill, on, v, y: y + 53 };
      });
      el('path', { d: `M660 ${y + 31} H${BUS}`, class: 'wire' }, svg);
    }
    this.boxes = {};
    for (const [res, b] of Object.entries(BOX)) {
      this.boxes[res] = this._box(res, b.x, b.y, b.w, b.h, b.title);
      if (UNITS[res]) el('text', { x: b.x + b.w - 8, y: b.y + 17, class: 'open-mark', 'text-anchor': 'end' }, this.boxes[res].g, '⤢');
      if (res === 'smem_array') {
        const { x, y } = b;
        this.smemBar = { back: el('rect', { x: x + 9, y: y + 44, width: 208, height: 7, rx: 3, class: 'track' }, svg), fill: el('rect', { x: x + 9, y: y + 44, width: 0, height: 7, rx: 3, fill: COLOR.smem_array }, svg) };
        this.smemQ = el('text', { x: x + 9, y: y + 71, class: 'bs' }, svg, '');
      }
    }
    this.boxes.l2_chip.note = el('text', { x: BOX.l2_chip.x + 9, y: BOX.l2_chip.y + 52, class: 'bs' }, svg, 'chip share, per wave');
    this.boxes.dram.note = el('text', { x: BOX.dram.x + 9, y: BOX.dram.y + 52, class: 'bs' }, svg, 'share of the chip rate');
    this.dotLayer = el('g', {}, svg);
    this.dots = Array.from({ length: 160 }, () => el('circle', { r: 2.8, class: 'dot', opacity: 0 }, this.dotLayer));
    const sel = this.selected; if (sel) this.highlight(sel);
    svg.onclick = (e) => {
      const g = e.target.closest?.('[data-res]'); if (!g) return;
      const box = g.getBBox();
      this.app.pickRes(g.dataset.res);
      if (UNITS[g.dataset.res]) this.openUnit(g.dataset.res, +(g.dataset.smsp ?? 0), true, [box.x + box.width / 2, box.y + box.height / 2]);
    };
  }

  _path(line) { return `M${line.map((p) => p.join(' ')).join(' L')}`; }

  highlight(res) {
    this.selected = res;
    this.svg.querySelectorAll('.res').forEach((g) => g.classList.toggle('sel', g.dataset.res === res));
  }

  // where one stage's dot sits: a wire and a progress, or a queue to count
  _place(i, t) {
    const { tr } = this; const st = tr.st; const L = tr.label;
    const res = L[st.r[i]]; const mech = L[st.m[i]];
    const f = (t - st.s[i]) / Math.max(1e-6, st.e[i] - st.s[i]);
    const s = tr.warpBy.get(tr.d.tx.w[st.x[i]])?.smsp ?? 0;
    if (mech.startsWith('queue')) return { q: res === 'l2/dram' ? 'l2_chip' : res };
    if (res === 'lsu') return { line: mioWire(s, [700, 166]), f };
    if (res === 'l2') return f < 0.5 ? { line: WIRE.req, f: f * 2 } : { line: WIRE.ret, f: (f - 0.5) * 1.1 };
    if (res === 'l2/dram') return { line: WIRE.req, f };
    if (res === 'dram') return { line: WIRE.dram, f: f < 0.5 ? f * 2 : 2 - f * 2 };
    if (res === 'l2_return') return { line: WIRE.ret, f: 0.55 + f * 0.45 };
    if (res === 'tma_engine') return { line: WIRE.tma, f };
    if (res === 'store_port') return { line: WIRE.stores, f };
    if (res === 'smem_array') {
      if (mech.includes('copy write')) return { line: WIRE.ret_smem, f };
      return { line: mioWire(s, [700, 86]).slice().reverse(), f };
    }
    return null;
  }

  update(t) {
    const { tr } = this; if (!tr) return;
    if (this.unit) { this.unit.update(t); return; }
    const b = Math.min(tr.nb - 1, Math.max(0, Math.floor(t / tr.binSize)));
    const smooth = (k) => { let v = 0; let n = 0; for (let j = Math.max(0, b - 4); j <= Math.min(tr.nb - 1, b + 4); j++) { v += tr.util[k]?.[j] ?? 0; n++; } return v / n; };
    for (const r of PIPES) {
      for (let s = 0; s < tr.nsp; s++) {
        const c = this.cells[r]?.[s]; if (!c) continue;
        const u = Math.min(1, smooth(`${r}:${s}`)); const h = 44 * u;
        c.fill.setAttribute('y', c.y - h); c.fill.setAttribute('height', h);
        const busy = tr.holdAt(`${r}:${s}`, t);
        c.on.setAttribute('opacity', busy ? 1 : 0);
        c.v.textContent = pct(u);
      }
    }
    // resident warps by state
    const alive = tr.warps.filter((w) => { const bk = tr.d.blocks[w.block]; return bk && bk[0] <= t && t <= bk[1]; });
    const stc = { run: '#ece9e2', done: '#333' };
    for (let s = 0; s < tr.nsp; s++) {
      const on = alive.filter((w) => w.smsp === s);
      this.warpDots[s].forEach((d, k) => {
        const w = on[k];
        if (!w) { d.setAttribute('class', 'wd off'); d.style.fill = ''; return; }
        const st = tr.warpState(w.id, t);
        d.setAttribute('class', 'wd');
        d.style.fill = st.s === 'wait' ? (st.why.kind === 'res' ? COLOR[st.why.res.split('/')[0]] ?? '#7b8089' : '#6a7079') : (stc[st.s] ?? '#333');
      });
      this.warpText[s].textContent = on.length ? `${on.length} warp${on.length > 1 ? 's' : ''}` : 'idle';
    }
    const live = alive.length;
    this.smInfo.textContent = `${new Set(alive.map((w) => w.block)).size} blocks resident · ${live} warps · ${tr.nsp} sub-partitions`;
    // data flow
    const q = {}; let n = 0;
    const seenTx = new Set();
    for (const i of tr.activeStages(t)) {
      const p = this._place(i, t); if (!p) continue;
      if (p.q) { q[p.q] = (q[p.q] ?? 0) + 1; continue; }
      if (n >= this.dots.length) continue;
      const key = `${tr.st.x[i]}:${tr.st.r[i]}`; if (seenTx.has(key)) continue; seenTx.add(key);
      const [x, y] = pointAt(p.line, p.f); const d = this.dots[n++];
      d.setAttribute('cx', x); d.setAttribute('cy', y); d.setAttribute('opacity', 1);
      d.style.fill = COLOR[tr.label[tr.st.r[i]]] ?? '#ece9e2';
    }
    for (let k = n; k < this.dots.length; k++) this.dots[k].setAttribute('opacity', 0);
    // ports
    const smem = smooth('smem_array');
    this.smemBar.fill.setAttribute('width', 208 * Math.min(1, smem));
    this.boxes.smem_array.v.textContent = `${pct(Math.min(1, smem))} held`;
    this.smemQ.textContent = `${q.smem_array ?? 0} queued now`;
    const num = (k, v) => { if (this.boxes[k]) this.boxes[k].v.textContent = v; };
    const depth = (p) => { const v = smooth(p); return v < 0.05 ? '0' : v < 10 ? v.toFixed(1) : v.toFixed(0); };
    num('lsu', `${pct(Math.min(1, smooth('lsu')))}${q.lsu ? ` q${q.lsu}` : ''}`);
    num('l2_return', `${q.l2_return ?? 0} queued`);
    num('tma_engine', tr.port.tma_engine.svc.s.length ? `${depth('tma_engine')} in` : 'unused');
    num('dram_queue', `${q.dram_queue ?? 0} queued`);
    num('store_queue', `${q.store_port ?? 0} queued`);
    num('l2_chip', `${depth('l2')} in flight`);
    num('dram', `${depth('dram')} in flight`);
    num('l1', 'by rule');
    num('store_port', `${pct(Math.min(1, smooth('store_port')))} busy`);
  }
}
