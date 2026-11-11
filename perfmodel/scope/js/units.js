// Inside a unit: what the model and the probes know about it, drawn where it acts. The layout is
// placed by hand; every number is a gb10.json parameter or the trace at the cursor, and each node
// shows its parameter's status. A dashed node is something the engine does not model, or a term
// whose cause is not identified: it stays on the drawing so the gap can be seen.
import { COLOR, fmt, lowerBound } from './data.js';

const NS = 'http://www.w3.org/2000/svg';
export const STATUS = { measured: '#7fc77f', fitted: '#e6b95c', structural: '#8b8f94', assumed: '#e88a7a', unidentified: '#e6b95c', override: '#86b5ee' };
export const BLOCK = ['#ece9e2', '#86b5ee', '#e6b95c', '#d98fb0'];

export function el(tag, attrs = {}, parent = null, text = null) {
  const e = document.createElementNS(NS, tag);
  for (const [k, v] of Object.entries(attrs)) e.setAttribute(k, v);
  if (text != null) e.textContent = text;
  parent?.appendChild(e);
  return e;
}

export class Unit {
  constructor(svg, tr, smsp, app) { Object.assign(this, { svg, tr, smsp, app }); }

  entry(ref) { return this.tr.machine.params[ref] ?? this.app.atlas?.params?.[ref]; }

  val(ref) { return this.entry(ref)?.value; }

  // a labelled parameter on the drawing: its status dot, value and name; hover gives the source
  param(p, x, y, ref, unit = '', label = null) {
    const q = this.entry(ref);
    const g = el('g', {}, p);
    el('title', {}, g, q ? `${ref}\n${q.status}\n${q.source ?? ''}` : `${ref}: not in this trace`);
    el('circle', { cx: x + 3, cy: y - 4, r: 3, fill: STATUS[q?.status] ?? '#6e7276' }, g);
    const v = el('text', { x: x + 11, y, class: 'val' }, g, q ? `${typeof q.value === 'number' ? fmt(q.value, q.value % 1 ? 2 : 0) : q.value}${unit}` : '–');
    if (label) el('text', { x: x + 15 + v.getComputedTextLength(), y, class: 'tm' }, g, label);
    return g;
  }

  node(p, x, y, w, h, opts = {}) {
    const g = el('g', { class: 'go', ...(opts.mech ? { 'data-mech': opts.mech } : {}), ...(opts.res ? { 'data-res': opts.res } : {}) }, p);
    el('rect', { x, y, width: w, height: h, rx: 6, class: `node ${opts.kind ?? ''}` }, g);
    if (opts.title) el('text', { x: x + 10, y: y + 18, class: 'bt' }, g, opts.title);
    return g;
  }

  edge(p, d, kind = '') { return el('path', { d, class: `edge ${kind}` }, p); }

  // the time around the cursor, one strip per lane of intervals
  strip(y, h) {
    this.sg = el('g', {}, this.svg); this.sy = y; this.sh = h;
    el('text', { x: 14, y: y - 7, class: 'sec' }, this.svg, 'Around the cursor');
  }

  stripRow(parts, label, y, h, x0, x1, t0, t1) {
    parts.push(`<text x="14" y="${y + h / 2 + 4}" class="tm">${label}</text>`);
    return (a, b, color, op = 1, text = '') => {
      const X = (v) => x0 + ((v - t0) / (t1 - t0)) * (x1 - x0);
      const l = Math.max(x0, X(a)); const r = Math.min(x1, X(b));
      if (r <= l) return;
      parts.push(`<rect x="${l.toFixed(1)}" y="${y}" width="${Math.max(1, r - l - 0.7).toFixed(1)}" height="${h}" rx="2" fill="${color}" opacity="${op}"/>`);
      if (text && r - l > 26) parts.push(`<text x="${l + 4}" y="${y + h / 2 + 4}" class="tm" style="fill:#0d0f12;font-weight:600">${text}</text>`);
    };
  }

  warpLabel(w, x, y, color) {
    return `<circle cx="${x}" cy="${y}" r="7" fill="${color}"/><text x="${x}" y="${y + 3.5}" text-anchor="middle" style="font:500 9.5px var(--mono);fill:#0d0f12">${w}</text>`;
  }
}

const SHAPE = { f16_f32_m16n8k16: 'f16 → f32', f16_f16_m16n8k16: 'f16 → f16', bf16_f32_m16n8k16: 'bf16 → f32', tf32_f32_m16n8k8: 'tf32 → f32', e4m3_f32_m16n8k32: 'e4m3 → f32', e5m2_f32_m16n8k32: 'e5m2 → f32', e2m1_f32_m16n8k32_f8f6f4: 'e2m1 → f32', s8_s32_m16n8k32: 's8 → s32' };

// which rate row an opcode is: mnemonic and shape from the instruction, the source type from its suffix
function typeOf(op) {
  const m = op.match(/^(HMMA|QMMA|OMMA|IMMA)\.(\d+)(\S*)/); if (!m) return null;
  const [, mn, sh, rest] = m;
  if (mn === 'IMMA') return 's8_s32_m16n8k32';
  if (mn === 'OMMA') return 'e2m1_f32_m16n8k32_f8f6f4';
  if (mn === 'QMMA') return /E5M2/i.test(rest) ? 'e5m2_f32_m16n8k32' : 'e4m3_f32_m16n8k32';
  if (sh === '1688') return 'tf32_f32_m16n8k8';
  if (/BF16/i.test(rest)) return 'bf16_f32_m16n8k16';
  return /F16/i.test(rest) && !/F32/i.test(rest) ? 'f16_f16_m16n8k16' : 'f16_f32_m16n8k16';
}

export class Tensor extends Unit {
  static H = 500;

  constructor(svg, tr, smsp, app) {
    super(svg, tr, smsp, app);
    this.title = `SMSP ${smsp}  ▸  Tensor pipe`;
    const key = `tensor:${smsp}`; const l = tr.pipe[key];
    // what each neighbour cost the pipe in this run, on all four sub-partitions
    this.beside = {}; this.own = 0;
    for (let p = 0; p < tr.nsp; p++) {
      const q = tr.pipe[`tensor:${p}`]; if (!q) continue;
      for (let i = 0; i < q.s.length; i++) {
        if (q.own[i]) { this.own++; continue; }
        const c = tr.ins[q.pc[i]].c;
        this.beside[c] = (this.beside[c] ?? 0) + q.e[i] - q.s[i];
      }
    }
    this.lane = l;
    this.types = {};
    for (const x of tr.ins) if (x.c === 'mma' && x.n) { const t = typeOf(x.t.split(' ')[0]); if (t) this.types[t] = (this.types[t] ?? 0) + x.n; }
    this._draw();
  }

  _draw() {
    const { svg, tr } = this; const c = COLOR.tensor;
    // flow: who waits, operands, the pipe, the result
    el('text', { x: 14, y: 50, class: 'sec' }, svg, 'One mma through the pipe');
    this.node(svg, 14, 60, 170, 106, { title: 'Waiting to issue', mech: 'Older block of an SM' });
    el('text', { x: 24, y: 96, class: 'tm' }, svg, 'between mma: fair');
    el('text', { x: 24, y: 111, class: 'tm' }, svg, 'otherwise older block first');
    this.waiters = el('g', {}, svg);
    this.edge(svg, 'M184 113 H214');

    const op = this.node(svg, 214, 60, 196, 106, { title: 'Operands', mech: 'Register-file bank' });
    [['A', 232], ['B', 264], ['C', 296]].forEach(([n, x], i) => {
      el('rect', { x: x + 74, y: 66, width: 28, height: 18, rx: 3, fill: '#20252b', stroke: '#3a4049' }, op);
      el('text', { x: x + 88, y: 79, 'text-anchor': 'middle', class: 'num' }, op, n);
      void i;
    });
    this.param(op, 224, 108, 'sim.reg_reads_per_clk', ' reads/clk');
    this.param(op, 224, 130, 'sim.mma_operand_clk_per_set', ' clk per set');
    el('text', { x: 224, y: 150, class: 'tm' }, op, 'cause of the set term not shown');
    this.edge(svg, 'M410 113 H440');

    const pipe = this.node(svg, 440, 60, 330, 106, { title: 'Tensor pipe', mech: '^Tensor pipe per' });
    this.ticks = [];
    const hold = this.val('sim.hmma_16816_pipe_clk') ?? 16;
    for (let i = 0; i < hold; i++) this.ticks.push(el('rect', { x: 452 + i * 18.6, y: 78, width: 16.6, height: 28, rx: 2, fill: '#20252b' }, pipe));
    this.param(pipe, 450, 128, 'sim.hmma_16816_pipe_clk', ' clk held', 'by every type and shape');
    this.now = el('text', { x: 450, y: 150, class: 'num' }, pipe, '');
    this.edge(svg, 'M770 113 H800');

    const res = this.node(svg, 800, 60, 150, 50, { title: 'Result', mech: 'Tensor pipe per' });
    this.param(res, 810, 96, 'sim.hmma_latency_clk', ' clk', 'stall counts cover it');
    const acc = this.node(svg, 800, 116, 150, 50, { title: 'Accumulators', kind: 'absent', mech: 'accumulators in flight' });
    el('text', { x: 810, y: 150, class: 'tm' }, acc, 'in flight: not modelled');

    el('text', { x: 970, y: 50, class: 'sec' }, svg, 'Each sub-partition has one');
    this.sibs = [];
    for (let s = 0; s < tr.nsp; s++) {
      const g = el('g', { class: 'go', 'data-smsp': s }, svg); const y = 60 + s * 27;
      el('rect', { x: 970, y, width: 148, height: 22, rx: 4, class: 'node' }, g);
      const f = el('rect', { x: 971, y: y + 1, width: 0, height: 20, rx: 3, fill: c, opacity: 0.5 }, g);
      el('text', { x: 978, y: y + 15, class: s === this.smsp ? 'bt' : 'tm' }, g, `SMSP ${s}`);
      const v = el('text', { x: 1110, y: y + 15, class: 'num', 'text-anchor': 'end' }, g, '');
      this.sibs.push({ f, v });
    }

    this._precision(); this._contention();
    this.strip(424, 0);
  }

  _precision() {
    const { svg } = this; const rates = this.val('tensor_rates_by_type');
    el('text', { x: 14, y: 196, class: 'sec' }, svg, 'Precision sets the work per hold, not the hold');
    el('text', { x: 14, y: 211, class: 'tm' }, svg, 'FLOP per clk per SM, measured. Box is the mma tile, m × n × k.');
    const list = Object.entries(rates ?? {}); const peak = Math.max(...list.map(([, v]) => v));
    const g = el('g', { class: 'go', 'data-mech': '^Tensor pipe per' }, svg);
    list.forEach(([k, v], i) => {
      const x = 14 + (i % 4) * 158; const y = 222 + Math.floor(i / 4) * 86;
      const depth = +k.match(/k(\d+)/)[1]; const on = this.types[k];
      el('rect', { x, y, width: 150, height: 78, rx: 6, class: 'node', style: on ? 'stroke:#d95926;stroke-width:1.6' : '' }, g);
      // the tile: m and n fixed, k drawn in depth
      const w = depth * 0.9; const X = x + 10; const Y = y + 20;
      el('path', { d: `M${X} ${Y} h22 v14 h-22 z`, fill: COLOR.tensor, opacity: on ? 0.9 : 0.35 }, g);
      el('path', { d: `M${X} ${Y} l${w} -${w * 0.42} h22 l-${w} ${w * 0.42} z`, fill: COLOR.tensor, opacity: on ? 0.6 : 0.22 }, g);
      el('path', { d: `M${X + 22} ${Y} l${w} -${w * 0.42} v14 l-${w} ${w * 0.42} z`, fill: COLOR.tensor, opacity: on ? 0.42 : 0.14 }, g);
      el('text', { x: x + 62, y: y + 19, class: on ? 'bt' : 'bv' }, g, SHAPE[k] ?? k);
      el('text', { x: x + 62, y: y + 35, class: 'num' }, g, `16×8×${depth}`);
      el('rect', { x: x + 10, y: y + 50, width: 130, height: 4, rx: 2, class: 'track' }, g);
      el('rect', { x: x + 10, y: y + 50, width: 130 * (v / peak), height: 4, rx: 2, fill: on ? COLOR.tensor : '#7d8187' }, g);
      el('text', { x: x + 10, y: y + 69, class: 'num' }, g, fmt(v));
      if (on) el('text', { x: x + 140, y: y + 69, class: 'num', 'text-anchor': 'end' }, g, `${fmt(on)} here`);
    });
  }

  _contention() {
    const { svg } = this;
    el('text', { x: 660, y: 196, class: 'sec' }, svg, 'Who delays the next mma on this sub-partition');
    el('text', { x: 660, y: 211, class: 'tm' }, svg, 'Added to the pipe per instruction issued while it is held.');
    const rows = [
      ['fma', 'FFMA and FMA class', 'sim.tensor_beside_fma_clk', /^(fp32|fma|half)/, COLOR.fma, 'mma issue beside FFMA'],
      ['alu', 'ALU class', 'sim.tensor_beside_alu_clk', /^alu/, COLOR.alu, 'mma issue beside FFMA'],
      ['mio', 'cp.async copy', 'sim.tensor_beside_copy_clk', /^cp_async/, COLOR.lsu, 'Issue interference'],
    ];
    const total = (re) => Object.entries(this.beside).reduce((a, [k, v]) => a + (re.test(k) ? v : 0), 0);
    const peak = Math.max(1, ...rows.map((r) => total(r[3])));
    rows.forEach(([res, name, ref, ks, color, mech], i) => {
      const y = 222 + i * 44; const g = this.node(svg, 660, y, 286, 38, { mech });
      el('rect', { x: 660, y, width: 4, height: 38, rx: 2, fill: color }, g);
      el('text', { x: 674, y: y + 16, class: 'bt' }, g, name);
      this.param(g, 674, y + 32, ref, ' clk');
      const v = total(ks);
      el('rect', { x: 782, y: y + 24, width: 150 * (v / peak), height: 5, rx: 2, fill: color }, g);
      el('text', { x: 936, y: y + 16, class: 'num', 'text-anchor': 'end' }, g, `${fmt(v)} clk here`);
      this.edge(svg, `M946 ${y + 19} H962 V190 H700 V166`, v ? 'hot' : '');
    });
    const none = this.node(svg, 660, 354, 286, 38, { mech: 'Issue interference' });
    el('text', { x: 674, y: 370, class: 'bt' }, none, 'ldmatrix, LDS');
    el('text', { x: 674, y: 385, class: 'tm' }, none, 'measured: no delay beside the pipe');
    const shared = this.node(svg, 962, 222, 156, 170, { title: 'This run', mech: '^Tensor pipe per' });
    const d = Object.values(this.beside).reduce((a, b) => a + b, 0);
    el('text', { x: 972, y: 262, class: 'big' }, shared, fmt(this.own));
    el('text', { x: 972, y: 278, class: 'tm' }, shared, 'mma holds');
    el('text', { x: 972, y: 316, class: 'big' }, shared, fmt(d));
    el('text', { x: 972, y: 332, class: 'tm' }, shared, 'clk added beside');
    el('text', { x: 972, y: 370, class: 'big' }, shared, `${fmt(this.tr.critShare.tensor * 100, 1)} %`);
    el('text', { x: 972, y: 386, class: 'tm' }, shared, 'of critical path');
  }

  update(t) {
    const { tr, smsp } = this; const key = `tensor:${smsp}`;
    const hold = tr.holdAt(key, t); const c = COLOR.tensor;
    if (hold) {
      const { lane, k } = hold; const f = (t - lane.s[k]) / (lane.e[k] - lane.s[k]);
      const n = Math.ceil(f * this.ticks.length);
      this.ticks.forEach((r, i) => { r.setAttribute('fill', i < n ? (lane.own[k] ? c : '#7b5a44') : '#20252b'); });
      const x = tr.ins[lane.pc[k]];
      this.now.textContent = lane.own[k] ? `w${lane.w[k]}  ${x.t}` : `beside: ${x.c} on w${lane.w[k]}`;
    } else { this.ticks.forEach((r) => r.setAttribute('fill', '#20252b')); this.now.textContent = 'idle'; }
    // warps in this sub-partition: on the pipe, queued for it, or elsewhere
    const alive = tr.warps.filter((w) => { const b = tr.d.blocks[w.block]; return w.smsp === smsp && b && b[0] <= t && t <= b[1]; });
    const dots = alive.map((w, i) => {
      const st = tr.warpState(w.id, t); const q = st.s === 'wait' && st.why.kind === 'res' && st.why.res.includes('tensor');
      const color = BLOCK[w.block % 4];
      return `<g opacity="${q ? 1 : 0.35}">${this.warpLabel(w.id, 32 + i * 20, 136, color)}</g>${q ? `<text x="${32 + i * 20}" y="156" text-anchor="middle" class="tm">queued</text>` : ''}`;
    });
    this.waiters.innerHTML = dots.join('');
    const b = Math.min(tr.nb - 1, Math.floor(t / tr.binSize));
    this.sibs.forEach((s, p) => { const u = Math.min(1, tr.util[`tensor:${p}`]?.[b] ?? 0); s.f.setAttribute('width', 146 * u); s.v.textContent = `${Math.round(u * 100)}%`; });
    // the strip: each hold named by its warp, delays beside it paler, waiting warps below
    const span = 260; const t0 = t - span * 0.7; const t1 = t + span * 0.3; const parts = [];
    const put = this.stripRow(parts, 'pipe', this.sy, 26, 132, 1118, t0, t1);
    const l = this.lane;
    if (l) {
      for (let k = Math.max(0, lowerBound(l.s, t0) - 1); k < l.s.length && l.s[k] < t1; k++) {
        if (l.own[k]) put(l.s[k], l.e[k], BLOCK[tr.warpBy.get(l.w[k]).block % 4], 0.9, `w${l.w[k]}`);
        else put(l.s[k], l.e[k], c, 0.9);
      }
    }
    const put2 = this.stripRow(parts, 'waiting for it', this.sy + 32, 12, 132, 1118, t0, t1);
    for (const w of tr.warps) {
      if (w.smsp !== smsp) continue;
      const ts = tr.warpT.get(w.id); if (!ts) continue; const ix = tr.perWarp.get(w.id);
      for (let k = lowerBound(ts, t0); k < ts.length && tr.r[ix[k]] < t1; k++) {
        const y = tr.why[tr.y[ix[k]]];
        if (y.kind === 'res' && y.res.includes('tensor')) put2(tr.r[ix[k]], tr.t[ix[k]], BLOCK[w.block % 4], 0.55);
      }
    }
    const X = 132 + ((t - t0) / (t1 - t0)) * (1118 - 132);
    parts.push(`<line x1="${X}" x2="${X}" y1="${this.sy - 4}" y2="${this.sy + 48}" stroke="#ece9e2"/>`);
    this.sg.innerHTML = parts.join('');
  }
}

export class SharedMemory extends Unit {
  static H = 500;

  constructor(svg, tr, smsp, app) {
    super(svg, tr, smsp, app);
    this.title = 'Shared memory array';
    // by opcode: the class smem_op holds LDS, STS and SHFL together
    const count = (re) => tr.ins.reduce((a, x) => a + (re.test(x.t.replace(/^@\S+\s+/, '')) ? x.n : 0), 0);
    const st = tr.st; const L = tr.label;
    this.queued = {};
    for (let i = 0; i < st.s.length; i++) {
      if (L[st.r[i]] === 'smem_array' && L[st.m[i]].startsWith('queue')) { const m = L[st.m[i]]; this.queued[m] = (this.queued[m] ?? 0) + st.e[i] - st.s[i]; }
    }
    this.users = [
      { name: 'ldmatrix', ops: /^LDSM/, q: 'queue (ldsm)', mio: 'sim.mio_ldsm_clk', cost: 'sim.smem_array_clk_per_512B', unit: ' clk × ways', color: COLOR.tensor, mech: 'ldmatrix latency' },
      { name: 'LDS, STS', ops: /^(LDS|STS)\b/, q: 'queue (lds)', q2: 'queue (sts)', mio: 'sim.mio_smem_op_clk', cost: 'sim.smem_op_min_clk', unit: ' clk at least', color: COLOR.alu, mech: 'Plain shared loads' },
      { name: 'SHFL', ops: /^SHFL/, q: 'queue (shfl)', mio: 'sim.mio_smem_op_clk', cost: 'sim.smem_op_min_clk', unit: ' clk at least', color: COLOR.xu, mech: 'Shuffles SHFL' },
      { name: 'STSM', ops: /^STSM/, q: 'queue (stsm)', mio: 'sim.mio_stsm_clk', cost: 'sim.smem_array_clk_per_512B', unit: ' clk/512 B', color: COLOR.fma, mech: 'Memory-instruction slot' },
      { name: 'cp.async data', ops: /^LDGSTS/, q: 'queue (copy write)', mio: 'sim.mio_ldgsts_clk', cost: 'sim.smem_array_clk_per_512B', unit: ' clk/512 B', color: COLOR.lsu, mech: 'Shared-memory array' },
      { name: 'TMA boxes', ops: /^UTMALDG/, q: 'queue (tma)', mio: null, cost: 'smem_bytes_per_clk_per_sm', unit: ' B/clk', color: COLOR.tma_engine, mech: 'Shared-memory array' },
    ];
    this.count = count;
    this._draw();
  }

  _draw() {
    const { svg, tr } = this;
    el('text', { x: 14, y: 50, class: 'sec' }, svg, 'One queue, one port: everyone below waits for the same array');
    const peak = Math.max(1, ...this.users.map((u) => (this.queued[u.q] ?? 0) + (this.queued[u.q2] ?? 0)));
    this.liveQ = {};
    el('text', { x: 148, y: 72, class: 'tm' }, svg, 'memory slot');
    el('text', { x: 262, y: 72, class: 'tm' }, svg, 'on the array');
    el('text', { x: 392, y: 72, class: 'tm' }, svg, 'queued this run');
    this.users.forEach((u, i) => {
      const y = 80 + i * 46; const n = this.count(u.ops); const q = (this.queued[u.q] ?? 0) + (this.queued[u.q2] ?? 0);
      const g = this.node(svg, 14, y, 500, 40, { mech: u.mech, kind: '' });
      el('rect', { x: 14, y, width: 4, height: 40, rx: 2, fill: u.color }, g);
      el('text', { x: 28, y: y + 17, class: 'bt' }, g, u.name);
      el('text', { x: 28, y: y + 33, class: 'num' }, g, `${fmt(n)} ops`);
      if (u.mio) this.param(g, 148, y + 25, u.mio, ' clk');
      this.param(g, 262, y + 25, u.cost, u.unit);
      el('rect', { x: 392, y: y + 23, width: 108 * (q / peak), height: 6, rx: 2, fill: u.color }, g);
      el('text', { x: 500, y: y + 16, class: 'num', 'text-anchor': 'end' }, g, q ? `${fmt(q)} clk` : '0');
      this.edge(svg, `M514 ${y + 20} C545 ${y + 20} 545 208 580 208`, q ? 'hot' : '');
    });
    // the single queue, then the array
    this.node(svg, 580, 174, 150, 68, { title: 'Queue now', mech: 'Shared-memory array' });
    this.qdots = el('g', {}, svg);
    this.edge(svg, 'M730 208 H764');
    const arr = this.node(svg, 764, 60, 354, 180, { title: 'Array', mech: 'Bank conflicts' });
    for (let b = 0; b < 32; b++) el('rect', { x: 776 + b * 10.4, y: 80, width: 8.4, height: 56, rx: 1.5, class: 'track' }, arr);
    el('text', { x: 776, y: 152, class: 'tm' }, arr, '32 banks of 4 B. Which bank is busy is not modelled.');
    this.param(arr, 776, 174, 'smem_bytes_per_clk_per_sm', ' B/clk', 'the whole port');
    this.busy = el('rect', { x: 776, y: 186, width: 0, height: 8, rx: 3, fill: COLOR.smem_array }, arr);
    el('rect', { x: 776, y: 186, width: 330, height: 8, rx: 3, fill: 'none', stroke: '#3a4049' }, arr);
    this.busyT = el('text', { x: 776, y: 212, class: 'num' }, arr, '');
    this.param(arr, 776, 230, 'smem_per_sm_bytes', ' B', 'capacity');
    // what each reader gets back
    el('text', { x: 580, y: 268, class: 'sec' }, svg, 'Back to registers');
    [['ldmatrix', 'sim.ldsm_latency_clk', ' clk', 'ldmatrix latency'], ['LDS', 'sim.lds_latency_clk', ' clk + bytes/32', 'Plain shared loads'], ['SHFL', 'sim.shfl_latency_clk', ' clk', 'Shuffles SHFL']].forEach(([n, ref, u, mech], i) => {
      const g = this.node(svg, 580 + i * 182, 278, 174, 40, { mech });
      el('text', { x: 590 + i * 182, y: 295, class: 'bt' }, g, n);
      this.param(g, 590 + i * 182, 311, ref, u);
    });
    this.edge(svg, 'M941 240 V278');
    // who holds the capacity
    const per = tr.meta.counters?.['launch__shared_mem_per_block_dynamic'];
    el('text', { x: 580, y: 344, class: 'sec' }, svg, 'Capacity held by resident blocks');
    const cap = this.val('smem_per_sm_bytes') ?? 102400; const rs = tr.meta.resident ?? 1;
    const g = this.node(svg, 580, 354, 538, 42, { mech: 'CTA residency' });
    if (per) {
      for (let i = 0; i < rs; i++) el('rect', { x: 592 + i * (per / cap) * 514, y: 362, width: (per / cap) * 514 - 2, height: 12, rx: 2, fill: BLOCK[i % 4], opacity: 0.7 }, g);
      el('text', { x: 592, y: 390, class: 'tm' }, g, `${rs} block${rs > 1 ? 's' : ''} of ${fmt(per)} B (launch counter)`);
    } else {
      el('text', { x: 592, y: 379, class: 'tm' }, g, `${rs} block${rs > 1 ? 's' : ''} resident; this row has no launch counter for its size`);
    }
    this.strip(424, 0);
  }

  update(t) {
    const { tr } = this; const st = tr.st; const L = tr.label;
    const color = Object.fromEntries(this.users.flatMap((u) => [[u.q, u.color], [u.q2, u.color]]));
    const active = tr.activeStages(t).filter((i) => L[st.r[i]] === 'smem_array');
    const queued = active.filter((i) => L[st.m[i]].startsWith('queue'));
    this.qdots.innerHTML = queued.slice(0, 34).map((i, k) => `<circle cx="${594 + (k % 12) * 11}" cy="${216 + Math.floor(k / 12) * 11}" r="4" fill="${color[L[st.m[i]]] ?? '#ece9e2'}"/>`).join('') + (queued.length ? '' : '<text x="590" y="226" class="tm">empty</text>');
    const b = Math.min(tr.nb - 1, Math.floor(t / tr.binSize));
    const u = Math.min(1, tr.util.smem_array?.[b] ?? 0);
    this.busy.setAttribute('width', 330 * u);
    this.busyT.textContent = `${tr.portBusy('smem_array', t) ? 'held' : 'free'}, ${queued.length} queued`;
    const span = 400; const t0 = t - span * 0.7; const t1 = t + span * 0.3; const parts = [];
    const P = tr.port.smem_array; const q = P.que; const a = this.stripRow(parts, 'queued', this.sy, 26, 132, 1118, t0, t1);
    const lanes = [[], [], [], [], [], [], [], []];
    for (let k = Math.max(0, lowerBound(q.s, t0) - 300); k < q.s.length && q.s[k] < t1; k++) {
      if (q.e[k] < t0) continue;
      const row = lanes.findIndex((r) => (r.at(-1) ?? -1e9) <= q.s[k]);
      if (row < 0) continue;
      lanes[row].push(q.e[k]);
      const y = this.sy + row * 6; const X = (v) => 132 + ((v - t0) / (t1 - t0)) * (1118 - 132);
      const l = Math.max(132, X(q.s[k])); const r = Math.min(1118, X(q.e[k]));
      const pc = q.pc[k];
      const col = this.users.find((us) => us.ops.test(tr.ins[pc]?.t.replace(/^@\S+\s+/, '') ?? ''))?.color ?? COLOR.smem_array;
      if (r > l) parts.push(`<rect x="${l.toFixed(1)}" y="${y}" width="${(r - l).toFixed(1)}" height="5" rx="1.5" fill="${col}" opacity=".8"/>`);
    }
    const X = 132 + ((t - t0) / (t1 - t0)) * (1118 - 132);
    parts.push(`<line x1="${X}" x2="${X}" y1="${this.sy - 4}" y2="${this.sy + 48}" stroke="#ece9e2"/>`);
    this.sg.innerHTML = parts.join(''); void a;
  }
}

