// What the programme found: each mechanism with its rule, its probes against the model, the
// decisions behind it, what was rejected, and the rule as the engine runs it. All text is from the
// project's records (tools/kta_perfmodel/atlas.py); this file only arranges it.
import { fmt } from './data.js';
import { esc, resName } from './panel.js';

const CODE = { M: 'Measured', A: 'Assumed', P: 'Partial', X: 'Absent', F: 'Front end' };
const CODE_NOTE = {
  M: 'modelled, its parameter measured by a probe that isolates it',
  A: 'modelled, a parameter assumed, fitted or not checked by a probe',
  P: 'modelled for some operation classes or configurations only',
  X: 'nothing in the engine: instructions that need it get a default rule',
  F: 'a front-end input that decides how much of a mechanism is exercised',
};

const inline = (s) => esc(s).replace(/\*\*(.+?)\*\*/g, '<strong>$1</strong>').replace(/`(.+?)`/g, '<code>$1</code>');

function pyHtml(src) {
  return esc(src).split('\n').map((l) => l
    .replace(/(#.*)$/, '<span class="c">$1</span>')
    .replace(/\b(def|return|if|else|elif|for|in|and|or|not|None|True|False|while)\b/g, '<span class="k">$1</span>')
    .replace(/P\[(&quot;|")(\w+)(&quot;|")\]/g, 'P["<span class="p">$2</span>"]')).join('\n');
}

// rows that differ in one number make a curve; anything else a list of pairs
function sweep(rows) {
  if (rows.length < 4) return null;
  const toks = rows.map((r) => r.id.match(/\d+\.?\d*|\D+/g));
  if (new Set(toks.map((t) => t.length)).size !== 1) return null;
  const vary = toks[0].map((_, k) => new Set(toks.map((t) => t[k])).size > 1 ? k : -1).filter((k) => k >= 0);
  if (vary.length !== 1 || Number.isNaN(+toks[0][vary[0]])) return null;
  const k = vary[0];
  return rows.map((r, i) => ({ x: +toks[i][k], m: r.measured, p: r.predicted })).sort((a, b) => a.x - b.x);
}

function curve(pts, unit) {
  const W = 640; const H = 170; const L = 52; const B = 26; const T = 8; const R = 10;
  const xs = pts.map((p) => p.x); const ys = pts.flatMap((p) => [p.m, p.p]).filter((v) => v != null);
  const x0 = Math.min(...xs); const x1 = Math.max(...xs); const y1 = Math.max(...ys) * 1.08; const y0 = Math.min(0, Math.min(...ys));
  const lg = x1 / Math.max(x0, 1e-9) > 16 && x0 > 0;
  const X = (v) => L + (lg ? Math.log(v / x0) / Math.log(x1 / x0) : (v - x0) / (x1 - x0 || 1)) * (W - L - R);
  const Y = (v) => T + (1 - (v - y0) / (y1 - y0 || 1)) * (H - T - B);
  const grid = [0, 0.5, 1].map((f) => y0 + f * (y1 - y0));
  return `<svg viewBox="0 0 ${W} ${H}" role="img" aria-label="Measured against replayed over the sweep">
    ${grid.map((v) => `<line x1="${L}" x2="${W - R}" y1="${Y(v)}" y2="${Y(v)}" stroke="#272c33"/><text x="${L - 6}" y="${Y(v) + 4}" text-anchor="end" fill="#7d8187" font-size="11" font-family="IBM Plex Mono">${fmt(v, v < 10 ? 1 : 0)}</text>`).join('')}
    <polyline fill="none" stroke="#e8824f" stroke-width="2" stroke-linejoin="round" points="${pts.filter((p) => p.p != null).map((p) => `${X(p.x)},${Y(p.p)}`).join(' ')}"/>
    ${pts.filter((p) => p.m != null).map((p) => `<circle cx="${X(p.x)}" cy="${Y(p.m)}" r="3.6" fill="#ece9e2" stroke="#13161a" stroke-width="1.4"><title>${fmt(p.x, 2)}: measured ${fmt(p.m, 2)}, replay ${fmt(p.p, 2)} ${esc(unit)}</title></circle>`).join('')}
    ${pts.filter((_, i) => i % Math.ceil(pts.length / 7) === 0).map((p) => `<text x="${X(p.x)}" y="${H - 8}" text-anchor="middle" fill="#7d8187" font-size="11" font-family="IBM Plex Mono">${fmt(p.x, p.x % 1 ? 1 : 0)}</text>`).join('')}
  </svg>`;
}

function pairs(rows, unit) {
  const list = rows.filter((r) => r.measured != null).slice(0, 18);
  const W = 640; const row = 19; const L = 250; const R = 56; const H = list.length * row + 6;
  const vals = list.flatMap((r) => [r.measured, r.predicted]).filter((v) => v > 0);
  const lo = Math.min(...vals); const hi = Math.max(...vals); const lg = hi / lo > 8;
  const X = (v) => L + (lg ? Math.log(v / lo) / Math.log(hi / lo || 2) : v / hi) * (W - L - R);
  return `<svg viewBox="0 0 ${W} ${H}" role="img" aria-label="Measured against replayed per row">
    ${list.map((r, i) => { const y = i * row + 11; const e = r.predicted ? (r.predicted / r.measured - 1) * 100 : null;
    return `<text x="${L - 10}" y="${y + 4}" text-anchor="end" fill="#a9abad" font-size="11" font-family="IBM Plex Mono">${esc(r.id.slice(-36))}</text>
      <line x1="${X(r.measured)}" x2="${X(r.predicted ?? r.measured)}" y1="${y}" y2="${y}" stroke="#4a5059" stroke-width="2"/>
      <circle cx="${X(r.predicted ?? r.measured)}" cy="${y}" r="3.4" fill="#e8824f"/><circle cx="${X(r.measured)}" cy="${y}" r="3.4" fill="#ece9e2" stroke="#13161a"><title>measured ${fmt(r.measured, 2)}, replay ${fmt(r.predicted, 2)} ${esc(unit)}</title></circle>
      <text x="${W}" y="${y + 4}" text-anchor="end" fill="#7d8187" font-size="11" font-family="IBM Plex Mono">${e == null ? '' : `${e > 0 ? '+' : ''}${fmt(e, 0)}%`}</text>`; }).join('')}
  </svg>${rows.length > list.length ? `<p class="empty">${rows.length - list.length} more rows not drawn.</p>` : ''}`;
}

function probeBlock(atlas, name) {
  const rows = atlas.probes[name]; if (!rows?.length) return '';
  const unit = rows[0].unit ?? '';
  const pts = sweep(rows);
  return `<div class="probe"><div class="head"><b>${esc(name)}</b><span>${rows.length} rows · ${esc(unit)}${pts ? ' · white measured, orange replay' : ''}</span></div>${pts ? curve(pts, unit) : pairs(rows, unit)}</div>`;
}

export function mechanismSheet(atlas, m, trace) {
  const probes = m.probes.map((p) => probeBlock(atlas, p)).filter(Boolean);
  const noData = m.probes.filter((p) => !atlas.probes[p]);
  const decs = m.decisions.filter((d) => atlas.decisions[d]);
  const rej = m.rejected.map((i) => atlas.rejected[i]);
  const crit = trace ? m.resources.map((r) => [r, trace.critShare[r] ?? 0]).filter(([, v]) => v > 0.001) : [];
  return `<div class="chips"><span class="pill ${m.code}">${CODE[m.code] ?? m.code}</span>
      ${m.depth ? `<span class="chip" title="${m.depth === 'structural' ? 'a resource or queue, carried to another design by changing the description' : 'a formula over the probe points, valid over the measured range only'}">${m.depth}</span>` : ''}
      ${m.resources.map((r) => `<button class="chip btn" data-res="${esc(r)}" title="Matched from the row's words">${esc(resName(r))}</button>`).join('')}
      ${m.var && m.var !== '—' ? `<span class="chip" title="S: the search space changes how much of it is used. H: a hardware design variable.">var ${esc(m.var)}</span>` : ''}
      ${m.kernels ? `<span class="chip" title="Test-set kernels that use it">${esc(m.kernels)}</span>` : ''}</div>
    <p class="head-note">${CODE_NOTE[m.code] ?? ''}${crit.length ? ` · in this run: ${crit.map(([r, v]) => `${esc(resName(r))} ${fmt(v * 100, 1)} % of the critical path`).join(', ')}` : ''}</p>
    <h3>What we know</h3><p class="prose">${inline(m.status)}</p>
    ${probes.length ? `<h3>Evidence <small>measured on the GB10 against the replay</small></h3>${probes.join('')}` : ''}
    ${noData.length ? `<p class="empty">Probes with no rows in the table: ${noData.map(esc).join(', ')}.</p>` : ''}
    ${m.rules.length ? `<h3>The rule as the engine runs it <small>matched from the row's words</small></h3>${m.rules.filter((r) => atlas.rules[r]).map((r, i) => `<details class="fold" ${i ? '' : 'open'}><summary>_op_${esc(r)} <small>sm.py:${atlas.rules[r].line}</small></summary><pre class="rule">${pyHtml(atlas.rules[r].code)}</pre></details>`).join('')}` : ''}
    ${rej.length ? `<h3>Rejected on the way</h3>${rej.map((r) => `<div class="dec"><b>${esc(r.topic)}</b> ${esc(r.text)} <span class="mono">${esc(r.where)}</span></div>`).join('')}` : ''}
    ${decs.length ? `<h3>Decisions</h3>${decs.map((d) => `<details class="fold"><summary><span class="mono" style="color:var(--accent)">${d}</span> ${esc(atlas.decisions[d].title)}</summary><p class="prose">${inline(atlas.decisions[d].text)}</p></details>`).join('')}` : ''}
    ${m.sources ? `<h3>Sources</h3><p class="doc">${inline(m.sources)}</p>` : ''}`;
}

export function renderAtlas(el, app) {
  const a = app.atlas; const st = (app.atlasState ??= { q: '', code: '', id: a.mechanisms.find((m) => m.code === 'M' && m.probes.some((p) => a.probes[p]))?.id ?? 0 });
  const counts = {};
  for (const m of a.mechanisms) counts[m.code] = (counts[m.code] ?? 0) + 1;
  const hit = (m) => (!st.code || m.code === st.code) && (!st.q || `${m.name} ${m.status}`.toLowerCase().includes(st.q.toLowerCase()));
  const shown = a.mechanisms.filter(hit);
  const secs = [...new Set(a.mechanisms.map((m) => m.section))];
  const cur = a.mechanisms[st.id];
  el.innerHTML = `<div class="atlas-side"><div class="atlas-tools">
      <input id="aq" placeholder="Search rules, opcodes, D numbers" value="${esc(st.q)}" aria-label="Search mechanisms">
      <div class="filters"><button data-code="" class="${st.code ? '' : 'on'}">All ${a.mechanisms.length}</button>${Object.keys(CODE).filter((c) => counts[c]).map((c) => `<button data-code="${c}" class="${st.code === c ? 'on' : ''}">${CODE[c]} ${counts[c]}</button>`).join('')}</div>
    </div>
    <div class="atlas-list">${secs.map((s) => { const l = shown.filter((m) => m.section === s); return l.length ? `<h4>${esc(s)}</h4>${l.map((m) => `<button class="mech ${m.id === st.id ? 'on' : ''}" data-pick="${m.id}"><span class="pill ${m.code}">${m.code}</span><span>${esc(m.name)}</span></button>`).join('')}` : ''; }).join('')}${shown.length ? '' : '<p class="empty" style="margin:12px">No mechanism matches.</p>'}</div></div>
    <div class="atlas-main"><div><p class="head-note">Mechanism inventory as of round ${a.as_of?.round ?? '–'} · probe rows from the table of ${esc((a.table_date ?? '').slice(0, 10))}</p>
    <h2>${esc(cur.name)}</h2>${mechanismSheet(a, cur, app.tr)}</div></div>`;
  el.querySelector('#aq').addEventListener('input', (e) => {
    st.q = e.target.value; const pos = e.target.selectionStart; renderAtlas(el, app);
    const i = el.querySelector('#aq'); i.focus(); i.setSelectionRange(pos, pos);
  });
  el.onclick = (e) => {
    const c = e.target.closest('[data-code]'); const p = e.target.closest('[data-pick]');
    if (c) { st.code = c.dataset.code; renderAtlas(el, app); }
    if (p) { st.id = +p.dataset.pick; renderAtlas(el, app); }
    const r = e.target.closest('[data-res]');
    if (r) app.gotoRes(r.dataset.res);
  };
}

export function renderCover(el, app) {
  const a = app.atlas; const targets = [...new Set(a.matrix.map((g) => g.target))]; const ops = [...new Set(a.matrix.map((g) => g.op))];
  const st = (app.coverState ??= { key: null });
  const states = (g) => g.rows.flatMap((r) => ['w', 'b', 's', 'c'].map((k) => r[k] ?? '').filter((v) => v !== ''));
  let M = 0; let m = 0; let e = 0;
  for (const g of a.matrix) for (const s of states(g)) { if (s === 'M') M++; else if (s === 'm') m++; else e++; }
  const cell = (op, t) => {
    const g = a.matrix.find((x) => x.op === op && x.target === t);
    if (!g) return '<td><div class="none">does not occur</div></td>';
    const s = states(g); const key = `${op}|${t}`;
    return `<td><button class="cell ${st.key === key ? 'on' : ''}" data-key="${esc(key)}"><div class="g">${s.map((v) => `<span class="q ${v === 'M' || v === 'm' ? v : ''}"></span>`).join('')}</div><small>${s.filter((v) => v === 'M').length}/${s.length} ruled</small></button></td>`;
  };
  const g = st.key && a.matrix.find((x) => `${x.op}|${x.target}` === st.key);
  el.innerHTML = `<h2>What has been explored</h2>
    <p class="doc">Operation × target. Each square is one pattern × sharing cell that can occur; the record of each is in the inventory, as of round ${a.as_of?.round ?? '–'}.</p>
    <div class="key" style="margin-top:10px"><span><span class="q M"></span>measured with a rule ${M}</span><span><span class="q m"></span>measured without a rule ${m}</span><span><span class="q"></span>empty ${e}</span></div>
    <table class="cov"><tr><th></th>${targets.map((t) => `<th>${esc(t)}</th>`).join('')}</tr>${ops.map((o) => `<tr><th>${esc(o)}</th>${targets.map((t) => cell(o, t)).join('')}</tr>`).join('')}</table>
    ${g ? `<div class="covdetail"><h3>${esc(g.ops)} × ${esc(g.target)}</h3><p class="doc">${inline(g.why)}</p><table><tr><th>Pattern</th><th>one warp</th><th>one block</th><th>blocks on an SM</th><th>chip</th><th>Probes</th></tr>${g.rows.map((r) => `<tr><td>${esc(r.Pattern)}</td>${['w', 'b', 's', 'c'].map((k) => `<td>${esc(r[k] ?? '')}</td>`).join('')}<td>${inline(r.Probes ?? '')}</td></tr>`).join('')}</table></div>` : '<p class="empty">Pick a cell to see its patterns and probes.</p>'}`;
  el.onclick = (ev) => { const b = ev.target.closest('[data-key]'); if (b) { st.key = b.dataset.key; renderCover(el, app); } };
}
