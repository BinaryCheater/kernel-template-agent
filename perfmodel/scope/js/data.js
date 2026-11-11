// A Scope trace in memory: typed columns, the lanes drawn from them, and the lookups every view
// shares. Times are engine clocks. Nothing here computes timing; it only indexes what the replay
// exported (tools/kta_perfmodel/timeline.py).

export const COLOR = {
  tensor: '#d95926', fma: '#199e70', alu: '#3987e5', xu: '#c98500', mio: '#7b8089',
  smem_array: '#9085e9', lsu: '#d55181', l2: '#d55181', l2_return: '#d55181', dram: '#b0506f',
  store_port: '#a2a6ad', tma_engine: '#c98500', issue: '#5b6068', barrier: '#5b6068',
  cta_dispatch: '#5b6068', regalloc: '#5b6068', crit: '#fff6e0',
};
export const PIPES = ['tensor', 'fma', 'alu', 'xu', 'mio'];
export const PORTS = [
  ['smem_array', 'Shared memory'], ['lsu', 'LSU'], ['l1', 'L1'], ['l2', 'L2 latency'], ['l2_return', 'L2 return'],
  ['dram', 'DRAM'], ['store_port', 'Store port'], ['tma_engine', 'TMA engine'],
];
const PORT_OF = {
  smem_array: 'smem_array', lsu: 'lsu', l1: 'l1', l2: 'l2', 'l2/dram': 'l2', l2_return: 'l2_return',
  dram: 'dram', dram_queue: 'dram', store_port: 'store_port', tma_engine: 'tma_engine',
};
const PORT_FREE = { smem_array: 'smem_array', lsu: 'lsu', l1: 'l1', l2_return: 'l2_return', dram_queue: 'dram', store_port: 'store_port', tma_engine: 'tma_engine' };

export function lowerBound(a, x, lo = 0, hi = a.length) {
  while (lo < hi) { const m = (lo + hi) >> 1; if (a[m] < x) lo = m + 1; else hi = m; }
  return lo;
}

function parseWhy(s) {
  const p = s.split(':');
  if (p[0] === 'res') return { kind: 'res', res: p[1], cls: p[2], label: `${p[1]} busy` };
  if (p[0] === 'sb') return { kind: 'sb', b: +p[1], label: `dependency counter ${p[1]}` };
  if (p[0] === 'bar') return { kind: 'bar', label: 'block barrier' };
  if (p[0] === 'ready') return { kind: 'ready', label: 'none' };
  return { kind: p[0], label: s };
}

export class Trace {
  constructor(d, file) {
    Object.assign(this, { d, file, meta: d.meta, machine: d.machine, ins: d.ins, warps: d.warps });
    const q = d.issues;
    this.n = q.t.length;
    this.t = Float64Array.from(q.t); this.w = Int32Array.from(q.w);
    this.pc = Int32Array.from(q.pc); this.r = Float64Array.from(q.r); this.y = Int32Array.from(q.y);
    this.why = d.why.map(parseWhy);
    this.cycles = d.meta.window ? d.meta.window[1] : d.meta.cycles;
    this.t0 = d.meta.window ? d.meta.window[0] : 0;
    this.nsp = this.machine.facts.smsp_per_sm;
    this.warpBy = new Map(d.warps.map((w) => [w.id, w]));
    const per = new Map();
    for (let i = 0; i < this.n; i++) (per.get(this.w[i]) ?? per.set(this.w[i], []).get(this.w[i])).push(i);
    this.perWarp = new Map([...per].map(([k, v]) => [k, Int32Array.from(v)]));
    this.warpT = new Map([...this.perWarp].map(([k, v]) => [k, Float64Array.from(v, (i) => this.t[i])]));
    this._pipes(d.holds); this._ports(d); this._rels(d.rels); this._crit(d.crit);
    this._bins(); this._totals();
  }

  _pipes(h) {
    const by = {};
    for (let i = 0; i < h.s.length; i++) {
      const k = `${h.r[i]}:${h.p[i]}`;
      (by[k] ??= []).push(i);
    }
    this.pipe = {};
    for (const [k, ix] of Object.entries(by)) {
      this.pipe[k] = {
        s: Float64Array.from(ix, (i) => h.s[i]), e: Float64Array.from(ix, (i) => h.e[i]),
        w: Int32Array.from(ix, (i) => h.w[i]), pc: Int32Array.from(ix, (i) => h.pc[i]),
        own: Uint8Array.from(ix, (i) => h.own[i]),
      };
    }
  }

  _ports(d) {
    const st = d.stages; const L = d.labels;
    this.label = L; this.st = st;
    this.txStages = Array.from({ length: d.tx.w.length }, () => []);
    const svc = {}; const que = {};
    for (let i = 0; i < st.s.length; i++) {
      this.txStages[st.x[i]].push(i);
      const port = PORT_OF[L[st.r[i]]]; const m = L[st.m[i]];
      if (!port) continue;
      if (m.startsWith('queue')) { (que[port] ??= []).push(i); continue; }
      // a port's time held comes from the recording below; a stage whose name is a latency is not
      // occupancy. What runs off an event, not an issue, only has its stage.
      if (!PORT_FREE[L[st.r[i]]] || m === 'copy write') (svc[port] ??= []).push(i);
    }
    const col = (ix) => {
      ix.sort((a, b) => st.s[a] - st.s[b]);
      return { s: Float64Array.from(ix, (i) => st.s[i]), e: Float64Array.from(ix, (i) => st.e[i]), pc: Int32Array.from(ix, (i) => d.tx.pc[st.x[i]]) };
    };
    this.port = {};
    for (const [p] of PORTS) this.port[p] = { svc: col(svc[p] ?? []), que: col(que[p] ?? []) };
    // ports as recorded at each issue, merged with the event-driven stages above
    const held = {}; const R = d.ports;
    for (let i = 0; i < R.s.length; i++) { const p = PORT_FREE[R.r[i]]; if (p) (held[p] ??= []).push(i); }
    for (const [p, ix] of Object.entries(held)) {
      const a = this.port[p].svc; const n = a.s.length + ix.length;
      const all = Array.from({ length: n }, (_, k) => (k < a.s.length ? [a.s[k], a.e[k], a.pc[k]] : [R.s[ix[k - a.s.length]], R.e[ix[k - a.s.length]], R.pc[ix[k - a.s.length]]]));
      all.sort((u, v) => u[0] - v[0]);
      this.port[p].svc = { s: Float64Array.from(all, (v) => v[0]), e: Float64Array.from(all, (v) => v[1]), pc: Int32Array.from(all, (v) => v[2]) };
    }
    this.txKey = new Map();
    for (let i = 0; i < d.tx.w.length; i++) this.txKey.set(`${d.tx.w[i]}:${d.tx.t[i]}`, i);
    // active stages by time: one list per bucket, for the data flow on the machine map
    this.bucket = Math.max(16, this.cycles / 1024);
    this.buckets = Array.from({ length: Math.ceil(this.cycles / this.bucket) + 2 }, () => []);
    for (let i = 0; i < st.s.length; i++) {
      const a = Math.floor(st.s[i] / this.bucket); const b = Math.floor(st.e[i] / this.bucket);
      for (let k = a; k <= b && k < this.buckets.length; k++) this.buckets[k]?.push(i);
    }
  }

  _rels(rels) {
    this.rel = new Map();
    for (const [w, b, tr, ti] of rels) {
      const k = `${w}:${b}`;
      (this.rel.get(k) ?? this.rel.set(k, []).get(k)).push([tr, ti]);
    }
    for (const v of this.rel.values()) v.sort((a, b) => a[0] - b[0]);
  }

  _crit(c) {
    this.crit = c;
    this.critStart = Float64Array.from(c, (s) => s[0]);
  }

  _bins() {
    // busy fraction per pipe lane and in-flight depth per port, on a fixed grid
    const nb = Math.min(1600, Math.ceil(this.cycles / 8));
    const size = this.cycles / nb;
    Object.assign(this, { nb, binSize: size });
    const add = (arr, s, e) => {
      let a = Math.max(0, Math.floor(s / size)); const b = Math.min(nb - 1, Math.floor(e / size));
      for (; a <= b; a++) arr[a] += Math.min(e, (a + 1) * size) - Math.max(s, a * size);
    };
    this.util = {};
    for (const [k, l] of Object.entries(this.pipe)) {
      const u = new Float32Array(nb);
      for (let i = 0; i < l.s.length; i++) add(u, l.s[i], l.e[i]);
      this.util[k] = u.map((v) => v / size);
    }
    for (const [p] of PORTS) {
      const u = new Float32Array(nb); const v = new Float32Array(nb);
      const { svc, que } = this.port[p];
      for (let i = 0; i < svc.s.length; i++) add(u, svc.s[i], svc.e[i]);
      for (let i = 0; i < que.s.length; i++) add(v, que.s[i], que.e[i]);
      this.util[p] = u.map((x) => x / size); this.util[`${p}|q`] = v.map((x) => x / size);
    }
  }

  _totals() {
    const m = this.meta;
    this.critShare = {}; this.waited = {}; this.pipeBusy = {};
    for (const s of this.crit) this.critShare[s[2]] = (this.critShare[s[2]] ?? 0) + (s[1] - s[0]) / this.cycles;
    for (let i = 0; i < this.n; i++) {
      const y = this.why[this.y[i]];
      if (y.kind === 'res') for (const r of y.res.split('/')) this.waited[r] = (this.waited[r] ?? 0) + (this.t[i] - this.r[i]);
    }
    for (const r of PIPES) {
      let b = 0;
      for (let p = 0; p < this.nsp; p++) {
        const l = this.pipe[`${r}:${p}`];
        if (l) for (let i = 0; i < l.s.length; i++) if (l.own[i]) b += l.e[i] - l.s[i];
      }
      this.pipeBusy[r] = b / (this.cycles * this.nsp);
    }
    this.blockN = Object.keys(this.d.blocks).length;
    this.predicted = m.predicted;
  }

  us(clk) { return clk / (this.meta.clock_GHz * 1e3); }

  /** Index of the issue of warp w at or just before time t, -1 when none. */
  issueAt(w, t) {
    const ts = this.warpT.get(w); if (!ts) return -1;
    let k = lowerBound(ts, t + 0.051) - 1;
    return k < 0 ? -1 : this.perWarp.get(w)[k];
  }

  /** What a warp is doing at t: not started, done, waiting (and on what), or running. */
  warpState(w, t) {
    const ts = this.warpT.get(w); if (!ts) return { s: 'idle' };
    const ix = this.perWarp.get(w);
    if (t < ts[0]) return { s: 'idle' };
    if (t > ts[ts.length - 1]) return { s: 'done' };
    const k = lowerBound(ts, t);
    const i = ix[Math.min(k, ix.length - 1)];
    if (this.r[i] <= t && t < this.t[i]) return { s: 'wait', why: this.why[this.y[i]], i };
    return { s: 'run', i };
  }

  /** Stages in flight at t. */
  activeStages(t) {
    const b = this.buckets[Math.floor(t / this.bucket)] ?? [];
    const st = this.st; const out = [];
    for (const i of b) if (st.s[i] <= t && t < st.e[i]) out.push(i);
    return out;
  }

  /** How each operation class travels: its stages in order with mean time, from the trace. */
  paths() {
    if (this.paths_) return this.paths_;
    const by = {}; const st = this.st; const L = this.label; const T = this.d.tx;
    this.txStages.forEach((ix, x) => {
      if (!ix.length) return;
      const c = this.ins[T.pc[x]]?.c ?? 'unknown'; const t0 = Math.min(...ix.map((i) => st.s[i]));
      const o = (by[c] ??= { cls: c, n: 0, stages: {} }); o.n++;
      for (const i of ix) {
        const k = `${L[st.r[i]]}|${L[st.m[i]]}`;
        const e = (o.stages[k] ??= { res: L[st.r[i]], mech: L[st.m[i]], par: st.p[i] >= 0 ? L[st.p[i]] : null, n: 0, dur: 0, off: 0 });
        e.n++; e.dur += st.e[i] - st.s[i]; e.off += st.s[i] - t0;
      }
    });
    this.paths_ = Object.values(by).map((o) => ({ ...o, stages: Object.values(o.stages).map((e) => ({ ...e, mean: e.dur / e.n, at: e.off / e.n })).sort((a, b) => a.at - b.at) }));
    return this.paths_;
  }

  /** Whether a port is held at t. */
  portBusy(p, t) {
    const c = this.port[p]?.svc; if (!c) return false;
    for (let k = lowerBound(c.s, t + 0.01) - 1, n = 0; k >= 0 && n < 8; k--, n++) if (c.e[k] > t) return true;
    return false;
  }

  /** Dependency counters a warp has outstanding at t. */
  counters(w, t) {
    const out = [0, 0, 0, 0, 0, 0];
    for (let b = 0; b < 6; b++) {
      const r = this.rel.get(`${w}:${b}`); if (!r) continue;
      r.times ??= Float64Array.from(r, (q) => q[0]);
      // releases after t, in release order: those issued by t are in flight; a copy group's
      // release can lie far ahead of later issues, so a bounded walk rather than an early stop
      for (let k = lowerBound(r.times, t + 0.001), n = 0; k < r.length && n < 512; k++, n++) if (r[k][1] <= t) out[b]++;
    }
    return out;
  }

  txOf(i) { return this.txKey.get(`${this.w[i]}:${this.t[i]}`); }

  /** The hold on a pipe lane covering time t. */
  holdAt(key, t) {
    const l = this.pipe[key]; if (!l) return null;
    let k = lowerBound(l.s, t + 0.01) - 1;
    for (let n = 0; k >= 0 && n < 6; k--, n++) if (l.s[k] <= t && t < l.e[k]) return { lane: l, k };
    return null;
  }

  /** Why issue i started when it did: the one step of the chain the engine recorded. */
  cause(i) {
    const w = this.w[i]; const ti = this.t[i]; const y = this.why[this.y[i]];
    const wait = ti - this.r[i];
    if (y.kind === 'sb') {
      const rels = this.rel.get(`${w}:${y.b}`);
      if (rels) {
        const k = lowerBound(rels.map((r) => r[0]), ti + 0.051) - 1;
        if (k >= 0) return { i: this.issueAt(w, rels[k][1]), text: y.label, wait, at: rels[k][0] };
      }
    } else if (y.kind === 'res') {
      const s = this.warpBy.get(w).smsp;
      for (const r of y.res.split('/')) {
        const h = this.holdAt(`${r}:${s}`, ti - 0.15);
        if (h) return { i: this.issueAt(h.lane.w[h.k], h.lane.s[h.k]), text: y.label, wait, at: h.lane.e[h.k] };
      }
    } else if (y.kind === 'bar') {
      const block = this.warpBy.get(w).block; let best = -1;
      for (const q of this.warps) {
        if (q.block !== block || q.id === w) continue;
        const j = this.issueAt(q.id, ti);
        if (j >= 0 && this.ins[this.pc[j]].c === 'barrier' && (best < 0 || this.t[j] > this.t[best])) best = j;
      }
      if (best >= 0) return { i: best, text: y.label, wait, at: this.t[best] };
    }
    return wait > 0 ? { i: -1, text: y.label, wait } : null;
  }

  chain(i, depth = 6) {
    const out = [];
    for (let c = this.cause(i); c && out.length < depth; c = c.i >= 0 ? this.cause(c.i) : null) {
      out.push(c);
      if (c.i < 0) break;
    }
    return out;
  }
}

export async function load(file) {
  const r = await fetch(`traces/${encodeURIComponent(file)}`);
  if (!r.ok) throw new Error(`${file}: ${r.status}`);
  return new Trace(await r.json(), file);
}

export const fmt = (x, d = 0) => (x == null || Number.isNaN(x) ? '–' : x.toLocaleString('en-US', { maximumFractionDigits: d, minimumFractionDigits: d }));
export const pct = (x, d = 0) => `${fmt(x * 100, d)}%`;
