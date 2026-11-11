// The inspector: a resource, an instruction, the whole run, or a comparison of two runs.
import { COLOR, fmt, pct } from './data.js';

export const esc = (s) => String(s ?? '').replace(/[&<>"]/g, (c) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;' }[c]));

const NAME = {
  tensor: ['Tensor pipe', 'smsp'], fma: ['FMA pipe', 'smsp'], alu: ['ALU pipe', 'smsp'], xu: ['XU pipe', 'smsp'],
  mio: ['Memory-instruction slot', 'smsp'], smem_array: ['Shared memory array', 'sm'], lsu: ['LSU address stage', 'sm'],
  l1: ['L1 data path', 'sm'], l2_return: ['L2 return port', 'sm'], store_queue: ['Store queue', 'sm'],
  store_port: ['Store port', 'tpc'], dram_queue: ['DRAM share', 'sm'], tma_engine: ['TMA engine', 'sm'],
  l2_chip: ['L2', 'chip'], dram: ['DRAM', 'chip'], chip: ['Chip', 'chip'], scheduler: ['Warp scheduler', 'smsp'],
  instruction_buffer: ['Instruction buffer', 'smsp'],
};
export const resName = (r) => NAME[r]?.[0] ?? r;

const bar = (label, v, max, color, unit = '%', act = '') => `<button class="bar" ${act}><span title="${esc(label)}">${esc(label)}</span><i style="width:${Math.max(0.5, (Math.abs(v) / max) * 100)}%;background:${color}"></i><em>${fmt(v, 1)}${unit === '%' ? '' : unit}</em></button>`;

function paramRow(tr, ref) {
  const p = tr.machine.params[ref]; if (!p) return '';
  const s = tr.d.sens.find((q) => q.param === ref);
  const sens = s ? (s.minus_pct == null ? 'no number' : `${s.minus_pct > 0 ? '+' : ''}${fmt(s.minus_pct, 1)} / ${s.plus_pct > 0 ? '+' : ''}${fmt(s.plus_pct, 1)} %`) : '';
  return `<details class="param"><summary><span class="n" title="${esc(ref)}">${esc(ref.replace('sim.', ''))}</span><span class="v">${esc(typeof p.value === 'object' ? 'table' : p.value)}</span>
    <span class="m"><span class="pill ${esc(p.status)}">${esc(p.status)}</span><span class="mono" title="time at −30 % / +30 %">${sens}</span></span></summary>
    <p>${typeof p.value === 'object' ? `<code>${esc(JSON.stringify(p.value))}</code><br>` : ''}${esc(p.source ?? 'no source recorded')}</p></details>`;
}

function mechList(app, res, n = 99) {
  const list = app.atlas?.mechanisms.filter((m) => m.resources.includes(res)) ?? [];
  if (!list.length) return '<p class="empty">No inventory row names this unit.</p>';
  return `${list.slice(0, n).map((m) => `<button class="mech" data-mech="${m.id}"><span class="pill ${m.code}">${m.code}</span><span>${esc(m.name)}</span></button>`).join('')}`;
}

export function resourceView(app, res) {
  const tr = app.tr;
  const links = tr.machine.links[res === 'l2_chip' ? 'l2_chip' : res === 'dram' ? 'dram_queue' : res] ?? [];
  const def = tr.machine.resources[res];
  const busy = tr.pipeBusy[res] ?? ({ smem_array: tr.meta.busy.smem, lsu: tr.meta.busy.lsu, l2_return: tr.meta.busy.fill, store_port: tr.meta.busy.stg, dram_queue: tr.meta.busy.dram, tma_engine: tr.meta.busy.tma }[res]);
  const crit = ['l2_chip', 'dram'].includes(res) ? Object.entries(tr.critShare).filter(([k]) => k.startsWith(res === 'dram' ? 'dram' : 'l2')).reduce((a, [, v]) => a + v, 0) : tr.critShare[res] ?? 0;
  const counted = app.atlas?.mechanisms.filter((m) => m.resources.includes(res)).length ?? 0;
  const facts = res === 'chip' ? Object.entries(tr.machine.facts).map(([k, v]) => `<div class="param"><summary style="display:grid"><span class="n">${esc(k)}</span><span class="v">${fmt(v)}</span></summary></div>`).join('') : '';
  const replay = res === 'chip' ? `<p class="doc">This trace replays the busiest SM with ${tr.meta.blocks ?? '–'} of ${tr.meta.grid_blocks ?? '–'} blocks; ${fmt(tr.meta.active_sms, 1)} SM-equivalents set its shares of the L2 and DRAM.</p>` : '';
  return `<div class="crumb">${esc({ smsp: 'Chip ▸ SM ▸ SMSP', sm: 'Chip ▸ SM', tpc: 'Chip ▸ TPC', chip: 'Chip' }[NAME[res]?.[1]] ?? '')}</div>
    <h2>${esc(resName(res))}</h2>
    ${def?.doc ? `<p class="doc">${esc(def.doc)}</p>` : ''}${replay}
    <div class="tiles"><div><small>Busy over the run</small><b>${busy == null ? '–' : pct(busy)}</b></div><div><small>On critical path</small><b>${pct(crit, 1)}</b></div></div>
    <h3>What we found here <small>${counted} inventory rows</small></h3>${mechList(app, res)}
    ${facts}
    <h3>Parameters <small>value · source · time at ±30 %</small></h3>
    ${links.length ? links.map((r) => paramRow(tr, r)).join('') : '<p class="empty">None named in the description.</p>'}`;
}

export function instructionView(app, i) {
  const tr = app.tr; const ins = tr.ins[tr.pc[i]]; const w = tr.warpBy.get(tr.w[i]);
  const wait = tr.t[i] - tr.r[i]; const x = tr.txOf(i);
  const st = tr.st; const L = tr.label;
  const stages = x != null ? tr.txStages[x] : [];
  const a = stages.length ? Math.min(...stages.map((k) => st.s[k])) : 0;
  const b = stages.length ? Math.max(...stages.map((k) => st.e[k])) : 1;
  const rows = stages.map((k) => {
    const res = L[st.r[k]]; const mech = L[st.m[k]]; const q = mech.startsWith('queue');
    return `<div class="r"><span title="${esc(`${res}: ${mech}`)}">${esc(mech)}</span><div class="t"><i style="left:${((st.s[k] - a) / (b - a)) * 100}%;width:${((st.e[k] - st.s[k]) / (b - a)) * 100}%;background:${COLOR[res.split('/')[0]] ?? '#7b8089'};opacity:${q ? 0.45 : 1}"></i></div><em>${fmt(st.e[k] - st.s[k], 1)}</em></div>`;
  }).join('');
  const chain = tr.chain(i).map((c) => {
    const inner = c.i >= 0 ? `<code>${esc(tr.ins[tr.pc[c.i]].t)}</code>` : '';
    return `<button class="chain" ${c.i >= 0 ? `data-issue="${c.i}"` : ''}>${fmt(c.wait, 1)} clk on ${esc(c.text)}${c.i >= 0 ? ` · w${tr.w[c.i]}` : ''}${inner}</button>`;
  }).join('');
  return `<div class="crumb">Block ${w.block} ▸ warp ${w.id} ▸ SMSP ${w.smsp}</div><h2>${esc(ins.c)}</h2>
    <div class="sass" style="margin-top:7px">${esc(ins.t)}</div>
    <div class="tiles"><div><small>Issued at</small><b>${fmt(tr.t[i], 1)}</b></div><div><small>Waited</small><b>${fmt(wait, 1)}</b></div></div>
    <h3>Path through the machine <small>clk</small></h3>
    <div class="fall">${rows || '<p class="empty">No path stages: a pipe instruction, or nothing waited on it.</p>'}</div>
    <h3>Why it could not start earlier</h3>${chain || '<p class="empty">It issued as soon as its warp was ready.</p>'}
    <h3>This pc over the run</h3>
    <div class="tiles"><div><small>Executions</small><b>${fmt(ins.n)}</b></div><div><small>On critical path</small><b>${pct(ins.k / tr.cycles, 1)}</b></div></div>
    <button class="link" data-showpc="${tr.pc[i]}">Mark every execution in the timeline</button>`;
}

export function runView(app) {
  const tr = app.tr; const m = tr.meta; const a = m.attribution.slice(0, 12); const max = a[0]?.share ?? 1;
  const res = (s) => COLOR[s.split('/')[0]] ?? '#7b8089';
  const sens = tr.d.sens.filter((s) => s.minus_pct != null);
  const smax = Math.max(1, ...sens.map((s) => Math.max(Math.abs(s.minus_pct), Math.abs(s.plus_pct))));
  const c = m.checks;
  return `<div class="crumb">Run</div><h2>Where the time went</h2>
    <h3>Critical path by mechanism <small>% of predicted time</small></h3>
    ${a.map((x) => bar(x.mechanism, x.share * 100, max * 100, res(x.resource), '%', `data-res="${esc(x.resource)}"`)).join('')}
    <h3>Parameter ±30 % <small>time change at −30 %</small></h3>
    ${sens.length ? sens.slice(0, 10).map((s) => bar(s.param.replace('sim.', ''), s.minus_pct, smax, '#c98500')).join('') : '<p class="empty">Exported without sensitivities.</p>'}
    <h3>Assumptions used</h3>
    ${Object.keys(m.defaults).length ? Object.entries(m.defaults).map(([k, v]) => `<div class="bar"><span>${esc(k)}</span><i></i><em>${fmt(v)}</em></div>`).join('') : '<p class="empty">None counted.</p>'}
    <h3>Trace against the engine</h3>
    <p class="doc">Path ${fmt(c.path_sum)} of ${fmt(c.cycles)} clk, ${pct(c.path_located)} tied to an instruction. Pipes and ports drawn against the engine's busy counters: ${Object.entries({ ...c.pipes, ...c.ports }).filter(([, p]) => p.engine).map(([k, p]) => `${k} ${fmt((p.trace / p.engine - 1) * 100, 1)} %`).join(', ')} (a hold's part of a clock carries to the next hold, and a copy holds the LSU longer than the lines its busy counter adds; the lanes show the time held).</p>
    <p class="doc">Engine ${esc(m.engine)}${m.dirty ? ' (uncommitted)' : ''} at ${esc(m.commit)}, exported ${esc(m.created)}.</p>`;
}

export function compareView(app) {
  const tr = app.tr; const list = app.index.filter((x) => x.file !== tr.file);
  const sel = list.find((x) => x.file === app.other) ?? list.find((x) => x.row === tr.meta.row) ?? list[0];
  if (!sel) return '<div class="crumb">Compare</div><h2>Nothing to compare</h2>';
  const d = (sel.predicted / tr.meta.predicted - 1) * 100;
  const keys = [...new Set([...tr.meta.attribution.slice(0, 8), ...sel.attribution.slice(0, 8)].map((a) => a.mechanism))];
  const share = (l, k) => l.find((a) => a.mechanism === k)?.share ?? 0;
  const rows = keys.map((k) => ({ k, a: share(tr.meta.attribution, k) * 100, b: share(sel.attribution, k) * 100 })).sort((p, q) => Math.abs(q.b - q.a) - Math.abs(p.b - p.a));
  const max = Math.max(...rows.flatMap((r) => [r.a, r.b]), 1);
  return `<div class="crumb">Compare</div><h2>This run against another</h2>
    <h3>Other run</h3><select id="other" style="width:100%" class="mono">${list.map((x) => `<option value="${esc(x.file)}" ${x === sel ? 'selected' : ''}>${esc(x.row)}${Object.keys(x.overrides).length ? ` · ${esc(Object.entries(x.overrides).map(([k, v]) => `${k.replace('sim.', '')}=${v}`).join(', '))}` : ''}</option>`).join('')}</select>
    <div class="tiles"><div><small>This</small><b>${fmt(tr.meta.predicted, 1)}</b></div><div><small>Other</small><b>${fmt(sel.predicted, 1)}</b></div></div>
    <p class="doc">Predicted time ${d > 0 ? '+' : ''}${fmt(d, 1)} %. A what-if is written by <code>scope-export --set KEY=VALUE</code>.</p>
    <h3>Critical-path share <small>this · other</small></h3>
    ${rows.map((r) => `<div class="bar" style="grid-template-columns:minmax(0,1.3fr) 1fr 74px"><span title="${esc(r.k)}">${esc(r.k)}</span><div><i style="width:${(r.a / max) * 100}%;background:#c9c9c3;margin-bottom:1.5px"></i><i style="width:${(r.b / max) * 100}%;background:#d95926"></i></div><em>${fmt(r.a, 1)} · ${fmt(r.b, 1)}</em></div>`).join('')}`;
}
