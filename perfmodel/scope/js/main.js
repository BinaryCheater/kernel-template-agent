// Scope: one cursor, one selection, every view follows.
import { COLOR, fmt, load } from './data.js';
import { Lanes } from './lanes.js';
import { MachineMap } from './map.js';
import { compareView, esc, instructionView, resName, resourceView, runView } from './panel.js';
import { mechanismSheet, renderAtlas, renderCover } from './atlas.js';

const $ = (id) => document.getElementById(id);

class App {
  constructor() {
    Object.assign(this, { t: 0, sel: null, selPc: null, res: 'tensor', panel: 'inspect', showPath: true, playing: false, tr: null, atlas: null, index: [] });
    this.lanes = new Lanes($('base'), $('over'), this);
    this.map = new MachineMap($('svg'), this);
    this._bind();
  }

  async start() {
    this.index = await (await fetch('traces/index.json')).json();
    this.atlas = await (await fetch('traces/atlas.json')).json();
    const base = this.index.filter((x) => !Object.keys(x.overrides).length);
    const v = this.index.filter((x) => Object.keys(x.overrides).length);
    $('trace').innerHTML = [...base, ...v].map((x) => `<option value="${esc(x.file)}">${esc(x.row)}${Object.keys(x.overrides).length ? ' · what-if' : ''}</option>`).join('');
    const h = new URLSearchParams(location.hash.slice(1));
    const file = this.index.find((x) => x.file === h.get('trace'))?.file ?? base[0]?.file;
    if (h.get('page')) this.page(h.get('page'));
    if (file) { $('trace').value = file; await this.open(file); }
  }

  async open(file) {
    $('stats').textContent = 'Loading…';
    const tr = await load(file);
    Object.assign(this, { tr, t: tr.t0 + (tr.cycles - tr.t0) * 0.13, sel: null, selPc: null });
    const m = tr.meta; const u = m.unit === 'us' ? ' µs' : ` ${m.unit ?? ''}`;
    const err = m.measured ? (m.predicted / m.measured - 1) * 100 : null;
    $('stats').innerHTML = `<span>predicted<b>${fmt(m.predicted, 1)}${u}</b></span>${m.measured ? `<span>measured<b>${fmt(m.measured, 1)}${u}</b></span><span>error<b>${err > 0 ? '+' : ''}${fmt(err, 1)} %</b></span>` : ''}<span><b>${fmt(m.cycles)}</b> clk</span>`;
    $('scope-tag').textContent = `Replay · one SM of ${tr.machine.facts.sms}${Object.keys(m.overrides).length ? ' · what-if' : ''}`;
    $('scope-tag').title = `Engine ${m.engine}, exported ${m.created}. A replay is one SM; the other SMs, the L2 and DRAM enter as shares.`;
    this.lanes.setTrace(tr); this.map.setTrace(tr); this.map.highlight(this.res);
    this._code(); this.renderPanel(); this._mini(); this.onCursor();
    this._hash();
    if (this.page_ === 'atlas') renderAtlas($('atlas'), this);
  }

  _hash() { location.hash = new URLSearchParams({ trace: this.tr?.file ?? '', page: this.page_ ?? 'run' }).toString(); }

  page(p) {
    this.page_ = p;
    document.querySelectorAll('#pages button').forEach((b) => b.classList.toggle('on', b.dataset.page === p));
    $('run').hidden = p !== 'run'; $('atlas').hidden = p !== 'atlas'; $('cover').hidden = p !== 'cover';
    if (p === 'atlas' && this.atlas) renderAtlas($('atlas'), this);
    if (p === 'cover' && this.atlas) renderCover($('cover'), this);
    if (p === 'run') requestAnimationFrame(() => { this.lanes.resize(); this._mini(); });
    if (this.tr) this._hash();
  }

  seek(t) {
    this.t = Math.max(this.tr.t0, Math.min(this.tr.cycles, t));
    this.onCursor();
  }

  onCursor() {
    const { tr, t } = this; if (!tr) return;
    $('clock').textContent = `${fmt(t)} clk`; $('clock2').textContent = `${fmt(tr.us(t), 2)} µs`;
    this.lanes.drawOver(); this.map.update(t); this._miniOver();
  }

  onView() { this._miniOver(); }

  pick(p) {
    if (!p) return;
    if (p.i != null && p.i >= 0) { this.sel = p.i; this.seek(this.tr.t[p.i]); this.panel = 'inspect'; this.selView = 'instr'; } else if (p.res) this.pickRes(p.res);
    this.renderPanel(); this.lanes.drawOver();
  }

  pickRes(res) {
    this.res = res; this.selView = 'res'; this.panel = 'inspect'; this.map.highlight(res);
    this.renderPanel();
  }

  // a pattern on inventory row names to the row, for a node on a unit's interior
  mechBy(pattern) {
    const re = new RegExp(pattern, 'i');
    return this.atlas?.mechanisms.find((m) => re.test(m.name))?.id ?? null;
  }

  setMapOpen(on) { $('run').classList.toggle('unit-open', on); }

  gotoRes(res) { this.page('run'); this.closeSheet(); this.pickRes(res); }

  openSheet(id) {
    const m = this.atlas.mechanisms[id]; const el = $('sheet');
    el.innerHTML = `<div class="sheet-h"><div><div class="crumb">Mechanism · ${esc(m.section)}</div><h2>${esc(m.name)}</h2></div><button class="close" data-close aria-label="Close">×</button></div>${mechanismSheet(this.atlas, m, this.tr)}`;
    el.hidden = false; el.scrollTop = 0;
  }

  closeSheet() { $('sheet').hidden = true; }

  renderPanel() {
    const el = $('insp'); if (!this.tr) return;
    const seg = `<div class="seg" style="margin:0 0 10px">${[['inspect', 'Inspect'], ['run', 'Run'], ['compare', 'Compare']].map(([k, l]) => `<button data-panel="${k}" class="${this.panel === k ? 'on' : ''}">${l}</button>`).join('')}</div>`;
    let body;
    if (this.panel === 'run') body = runView(this);
    else if (this.panel === 'compare') body = compareView(this);
    else if (this.selView === 'instr' && this.sel != null) body = instructionView(this, this.sel);
    else body = resourceView(this, this.res);
    el.innerHTML = seg + body;
  }

  tip(text, x, y) {
    const t = $('tip');
    if (!text) { t.hidden = true; return; }
    t.textContent = text; t.hidden = false; t.style.left = `${Math.min(innerWidth - 530, x + 14)}px`; t.style.top = `${y + 14}px`;
  }

  _code() {
    const { tr } = this; const maxW = Math.max(1, ...tr.ins.map((i) => i.w)); const maxK = Math.max(1, ...tr.ins.map((i) => i.k));
    const starts = tr.d.regions.map((r) => r.start);
    const color = { mma: COLOR.tensor, fp32: COLOR.fma, fma_half_rate: COLOR.fma, alu: COLOR.alu, alu_wide: COLOR.alu, xu: COLOR.xu, ldsm: COLOR.smem_array, stsm: COLOR.smem_array, smem_op: COLOR.smem_array, cp_async: COLOR.lsu, ldg: COLOR.lsu, stg: COLOR.lsu };
    const rows = tr.ins.map((x, i) => {
      const r = starts.indexOf(i);
      const head = r >= 0 ? `<div class="l h"><i></i><span></span><span>P${r} · ${esc(tr.d.regions[r].top.join(', '))}</span><span></span><span></span><span></span></div>` : '';
      return `${head}<div class="l" data-pc="${i}"><i style="background:${color[x.c] ?? '#2a2e35'}"></i><span class="a">${x.a.toString(16)}</span><span class="x ${x.n ? '' : 'cold'}">${esc(x.t)}</span><span class="n">${fmt(x.n)}</span><span class="b"><s style="width:${(x.w / maxW) * 100}%;background:${COLOR.mio}"></s>${x.w ? fmt(x.w) : ''}</span><span class="b"><s style="width:${(x.k / maxK) * 100}%;background:${COLOR.tensor}"></s>${x.k ? fmt(x.k) : ''}</span></div>`;
    }).join('');
    $('code').innerHTML = `<div class="l h"><i></i><span class="a">addr</span><span>instruction</span><span class="n">runs</span><span class="b">waited clk</span><span class="b">on path clk</span></div>${rows}`;
  }

  _mini() {
    const { tr } = this; if (!tr) return;
    const c = $('mini'); const box = $('miniwrap').getBoundingClientRect(); const dpr = devicePixelRatio || 1;
    if (!box.width) return;
    c.width = box.width * dpr; c.height = box.height * dpr; c.style.width = `${box.width}px`; c.style.height = `${box.height}px`;
    this.mw = box.width; this.mh = box.height;
    const off = document.createElement('canvas'); off.width = c.width; off.height = c.height;
    const g = off.getContext('2d'); g.scale(dpr, dpr);
    const w = box.width - 16; const n = Math.floor(w / 2); const top = 7; const ph = box.height - 28;
    const col = (key) => { const a = new Float32Array(n); for (let s = 0; s < tr.nsp; s++) { const u = tr.util[`${key}:${s}`]; if (u) for (let i = 0; i < tr.nb; i++) a[Math.floor((i / tr.nb) * n)] += u[i]; } return a; };
    const cnt = new Float32Array(n); for (let i = 0; i < tr.nb; i++) cnt[Math.floor((i / tr.nb) * n)]++;
    const series = [['tensor', COLOR.tensor], ['fma', COLOR.fma], ['alu', COLOR.alu], ['xu', COLOR.xu]].map(([k, c0]) => [col(k), c0]);
    const mem = new Float32Array(n); for (let i = 0; i < tr.nb; i++) mem[Math.floor((i / tr.nb) * n)] += Math.min(1, tr.util.smem_array[i]) * tr.nsp * 0.5;
    series.push([mem, COLOR.smem_array]);
    let peak = 0; for (let i = 0; i < n; i++) { const v = series.reduce((a, [s]) => a + s[i] / (cnt[i] || 1), 0); if (v > peak) peak = v; }
    for (let i = 0; i < n; i++) {
      let y = top + ph;
      for (const [s, color] of series) { const h = ((s[i] / (cnt[i] || 1)) / peak) * ph; g.fillStyle = color; g.fillRect(8 + i * 2, y - h, 1.6, h); y -= h; }
    }
    const best = Array.from({ length: n }, () => ({}));
    for (const s of tr.crit) { const a = Math.floor(((s[0] - tr.t0) / (tr.cycles - tr.t0)) * n); const b = Math.min(n - 1, Math.floor(((s[1] - tr.t0) / (tr.cycles - tr.t0)) * n)); for (let i = Math.max(0, a); i <= b; i++) best[i][s[2]] = (best[i][s[2]] ?? 0) + s[1] - s[0]; }
    best.forEach((m, i) => { let k = null; let v = 0; for (const r in m) if (m[r] > v) { v = m[r]; k = r; } g.fillStyle = COLOR[k?.split('/')[0]] ?? '#3a4049'; g.fillRect(8 + i * 2, top + ph + 4, 2, 7); });
    this.miniBase = off; this._miniOver();
  }

  _miniOver() {
    const { tr } = this; const c = $('mini'); if (!this.miniBase || !tr) return;
    const dpr = devicePixelRatio || 1; const g = c.getContext('2d');
    g.setTransform(1, 0, 0, 1, 0, 0); g.clearRect(0, 0, c.width, c.height); g.drawImage(this.miniBase, 0, 0);
    g.setTransform(dpr, 0, 0, dpr, 0, 0);
    const X = (t) => 8 + ((t - tr.t0) / (tr.cycles - tr.t0)) * (this.mw - 16);
    const a = X(this.lanes.v0); const b = X(this.lanes.v1);
    g.fillStyle = 'rgba(13,15,18,.62)'; g.fillRect(0, 0, a, this.mh); g.fillRect(b, 0, this.mw - b, this.mh);
    g.strokeStyle = '#a9abad'; g.lineWidth = 1; g.strokeRect(a + 0.5, 2.5, Math.max(2, b - a), this.mh - 5);
    g.fillStyle = '#ece9e2'; g.fillRect(X(this.t), 0, 1.4, this.mh);
  }

  _play(now) {
    if (!this.playing) return;
    const dt = Math.min(0.05, (now - (this.last ?? now)) / 1000); this.last = now;
    const tr = this.tr; let t = this.t + dt * +$('speed').value;
    if (t >= tr.cycles) { t = tr.cycles; this.toggle(false); }
    if (t > this.lanes.v1 || t < this.lanes.v0) { const s = this.lanes.v1 - this.lanes.v0; this.lanes.setView(t - s * 0.1, t + s * 0.9); }
    this.seek(t);
    requestAnimationFrame((n) => this._play(n));
  }

  toggle(on = !this.playing) {
    this.playing = on; $('play').textContent = on ? '❚❚' : '▶'; this.last = undefined;
    if (on) requestAnimationFrame((n) => this._play(n));
  }

  _bind() {
    $('trace').addEventListener('change', (e) => this.open(e.target.value));
    $('pages').addEventListener('click', (e) => { const b = e.target.closest('button'); if (b) this.page(b.dataset.page); });
    $('play').addEventListener('click', () => this.toggle());
    $('lens').addEventListener('click', (e) => {
      const b = e.target.closest('button'); if (!b) return;
      document.querySelectorAll('#lens button').forEach((q) => q.classList.toggle('on', q === b));
      const code = b.dataset.lens === 'code'; $('code').hidden = !code;
      if (!code) this.lanes.setLens(b.dataset.lens);
    });
    $('code').addEventListener('click', (e) => {
      const l = e.target.closest('[data-pc]'); if (!l) return;
      const pc = +l.dataset.pc; this.selPc = this.selPc === pc ? null : pc;
      document.querySelectorAll('#code .on').forEach((q) => q.classList.remove('on'));
      if (this.selPc != null) l.classList.add('on');
      this.lanes.drawOver();
    });
    const panelClick = (e) => {
      const p = e.target.closest('[data-panel]'); if (p) { this.panel = p.dataset.panel; this.renderPanel(); return; }
      const m = e.target.closest('[data-mech]'); if (m) { this.openSheet(+m.dataset.mech); return; }
      const iss = e.target.closest('[data-issue]'); if (iss) { const i = +iss.dataset.issue; this.pick({ i }); const s = this.lanes.v1 - this.lanes.v0; if (this.t < this.lanes.v0 || this.t > this.lanes.v1) this.lanes.setView(this.t - s / 2, this.t + s / 2); return; }
      const sp = e.target.closest('[data-showpc]'); if (sp) { this.selPc = +sp.dataset.showpc; this.lanes.drawOver(); return; }
      const r = e.target.closest('[data-res]'); if (r && this.tr.machine.resources[r.dataset.res]) this.pickRes(r.dataset.res);
    };
    $('insp').addEventListener('click', panelClick);
    $('insp').addEventListener('change', (e) => { if (e.target.id === 'other') { this.other = e.target.value; this.renderPanel(); } });
    $('sheet').addEventListener('click', (e) => { if (e.target.closest('[data-close]')) { this.closeSheet(); return; } const r = e.target.closest('[data-res]'); if (r) this.gotoRes(r.dataset.res); });
    const mini = $('mini'); let dragging = null;
    const at = (e) => { const b = mini.getBoundingClientRect(); return this.tr.t0 + ((e.clientX - b.left - 8) / (this.mw - 16)) * (this.tr.cycles - this.tr.t0); };
    mini.addEventListener('pointerdown', (e) => { mini.setPointerCapture(e.pointerId); dragging = { from: at(e), moved: false }; });
    mini.addEventListener('pointermove', (e) => {
      if (!dragging) return;
      const t = at(e); if (Math.abs(t - dragging.from) > (this.tr.cycles / this.mw) * 3) dragging.moved = true;
      if (dragging.moved) this.lanes.setView(Math.min(t, dragging.from), Math.max(t, dragging.from));
    });
    mini.addEventListener('pointerup', (e) => {
      if (dragging && !dragging.moved) { const t = at(e); const s = this.lanes.v1 - this.lanes.v0; this.seek(t); if (t < this.lanes.v0 || t > this.lanes.v1) this.lanes.setView(t - s / 2, t + s / 2); }
      dragging = null;
    });
    mini.addEventListener('dblclick', () => this.lanes.setView(this.tr.t0, this.tr.cycles));
    new ResizeObserver(() => this._mini()).observe($('miniwrap'));
    addEventListener('keydown', (e) => {
      if (/INPUT|SELECT/.test(e.target.tagName)) return;
      if (e.code === 'Space') { e.preventDefault(); this.toggle(); }
      if (e.key === 'ArrowRight') this.seek(this.t + (e.shiftKey ? 100 : 1));
      if (e.key === 'ArrowLeft') this.seek(this.t - (e.shiftKey ? 100 : 1));
      if (e.key === 'Escape') { if ($('sheet').hidden) this.map.closeUnit(); this.closeSheet(); this.sel = null; this.selPc = null; this.selView = 'res'; this.renderPanel(); this.lanes.drawOver(); }
      if (e.key === 'f') this.lanes.setView(this.tr.t0, this.tr.cycles);
    });
  }
}

const app = new App();
window.scope = app;
app.start().catch((e) => { $('stats').textContent = `Could not load: ${e.message}`; });
