// Mechanisms (WP22, D591): one page per round-4 mechanism that had backend records but no view.
// Each panel answers "has it run, and what did it say?" from GET /mechanisms; an empty panel
// names the process that writes its records. Rows whose origin is not real carry FIXTURE.
import { Link } from "react-router";
import { api } from "../data/source";
import type { MechCanary, MechPlan, MechReference, MechSelection, MechWhatIf, MechanismsView } from "../data/types";
import { useApi } from "../data/useApi";
import { Chip, Loading, Panel, RecordLink } from "../components/ui";
import { hhmm, type Tone } from "../components/format";

function Origin({ origin }: { origin: string }) {
  return origin === "real" ? null : <Chip tone="human" title="synthetic records, never evidence">FIXTURE</Chip>;
}

const us = (v: number | null | undefined) => (v === null || v === undefined ? "—" : `${v.toFixed(2)} µs`);
const VERDICT: Record<string, Tone> = { caught: "ok", missed: "fail", uninformative: "idle", no_base: "waiting" };
const STAGE: Record<string, Tone> = { planned: "waiting", "not applicable": "idle", done: "ok", failed: "fail" };

function Empty({ what }: { what: string }) {
  return <div className="empty">No records yet. {what}</div>;
}

export function Mechanisms() {
  const m = useApi(api.mechanisms, []);
  if (!m.data) return <Loading error={m.error} />;
  const v = m.data;
  return (
    <div className="stack">
      <Summary v={v} />
      <Learning rows={v.learning} stages={v.stages} />
      <Plans rows={v.plans} />
      <div className="grid" style={{ gridTemplateColumns: "minmax(0, 1fr) minmax(0, 1fr)", alignItems: "start" }}>
        <Selection rows={v.selection} />
        <Canaries v={v} />
      </div>
      <WhatIf rows={v.what_if} />
    </div>
  );
}

function Summary({ v }: { v: MechanismsView }) {
  const missed = v.canaries.rows.filter((r) => r.verdict === "missed").length;
  const beaten = v.selection.filter((s) => s.model_beaten_by_random).length;
  const items: [string, string, string, string][] = [
    ["Learning loop (WP20)", `${v.learning.length} reference(s)`, "stages per reference", "#learning"],
    ["FormulationPlan (WP19)", `${v.plans.length} plan(s)`, `${v.plans.filter((p) => p.diff.length).length} differ from their default`, "#plans"],
    ["Selection arms (WP18)", `${v.selection.length} bucket(s)`, `${beaten} where random beat the model`, "#selection"],
    ["Canaries (WP18)", `${v.canaries.rows.length} run(s)`, `${missed} missed`, "#canaries"],
    ["L2 what-if (WP16)", `${v.what_if.length} run(s)`, "wall moved → time saved", "#whatif"],
    ["X1–X12 (WP21)", `${v.experiments.count} defined`, Object.entries(v.experiments.by_status).map(([k, n]) => `${n} ${k}`).join(", "), ""],
  ];
  return (
    <Panel primary title="Round-4 mechanisms" sub="has each mechanism run, and what did it record? Click a tile to jump to its panel.">
      <div className="grid" style={{ gridTemplateColumns: "repeat(6, minmax(0, 1fr))", gap: 8 }}>
        {items.map(([name, value, foot, href]) => (
          <div key={name} className="stat">
            <div className="label">{name}</div>
            <div className="value num">{href ? <a href={href}>{value}</a> : <Link to={v.experiments.page}>{value}</Link>}</div>
            <div className="foot">{foot}</div>
          </div>
        ))}
      </div>
    </Panel>
  );
}

function Learning({ rows, stages }: { rows: MechReference[]; stages: string[] }) {
  return (
    <span id="learning">
      <Panel title="Learning-loop stages per reference" sub="intake → decompose → admit → reproduce → ablate → cover; a dry-run plan computed on read (D569): nothing is submitted yet">
        {!rows.length && <Empty what="References enter with `kta reference add`; the learn process plans them." />}
        <table className="t compact">
          <thead><tr><th>Reference</th><th>Class · tiers</th>{stages.map((s) => <th key={s}>{s}</th>)}</tr></thead>
          <tbody>
            {rows.map((r) => (
              <tr key={r.id}>
                <td><RecordLink id={r.id} label={<b>{r.name}</b>} /> <Origin origin={r.origin} /><div className="small muted">on {r.target}{r.priority !== undefined && ` · priority ${r.priority.toFixed(2)}`}</div></td>
                {r.error ? <td colSpan={stages.length + 1} className="error small">{r.error}</td> : <>
                  <td className="small">{r.source_class}<div className="muted">{r.tiers?.join(" ")}</div></td>
                  {stages.map((name) => {
                    const st = r.stages?.find((s) => s.name === name);
                    return <td key={name} className="small" title={st?.why}>{st ? <><Chip tone={STAGE[st.status] ?? "idle"}>{st.status}</Chip>{st.steps > 0 && <span className="muted"> {st.steps}</span>}</> : "—"}</td>;
                  })}
                </>}
              </tr>
            ))}
          </tbody>
        </table>
      </Panel>
    </span>
  );
}

const show = (v: unknown) => (v === null || v === undefined ? "—" : typeof v === "string" ? v : JSON.stringify(v));

function Plans({ rows }: { rows: MechPlan[] }) {
  return (
    <span id="plans">
      <Panel title="FormulationPlans and their diff against the deterministic default" sub="every non-default decision; uncited ones (no structure and evidence) are flagged">
        {!rows.length && <Empty what="The case loop writes a default plan per case; agents propose others (WP19)." />}
        <table className="t compact">
          <thead><tr><th>Plan</th><th>Family · source</th><th>Default</th><th>Differences</th></tr></thead>
          <tbody>
            {rows.map((p) => (
              <tr key={p.id}>
                <td><RecordLink id={p.id} /> <Origin origin={p.origin} /><div className="small muted">{p.author} · {hhmm(p.at)}</div></td>
                <td className="small mono">{p.family}<div className="muted">{p.chain_source}</div></td>
                <td>{p.default ? <RecordLink id={p.default} /> : <span className="muted small">{p.author === "default" ? "is a default" : "none"}</span>}</td>
                <td className="small">{!p.default ? <span className="muted">—</span> : !p.diff.length ? <Chip tone="ok">same as default</Chip> : (
                  <ul style={{ margin: 0, paddingLeft: 16 }}>
                    {p.diff.map((c) => (
                      <li key={`${c.section}:${c.key}`} className="mono">{c.section}:{c.key} {show(c.default)} → {show(c.plan)} {c.cited ? <Chip tone="ok">cited</Chip> : <Chip tone="fail">uncited</Chip>}</li>
                    ))}
                  </ul>)}
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </Panel>
    </span>
  );
}

function Selection({ rows }: { rows: MechSelection[] }) {
  return (
    <span id="selection">
      <Panel title="Selection arms per gap bucket" sub="best measured time: the model's top-e points vs a random arm of the same size vs the copied assignment (12 §3)">
        {!rows.length && <Empty what="The solve loop records them in each round's GapDecomposition (WP18)." />}
        <table className="t compact">
          <thead><tr><th>Round · bucket</th><th>Model top-e</th><th>Random</th><th>Copied</th><th /></tr></thead>
          <tbody>
            {rows.map((s) => (
              <tr key={`${s.id}:${s.bucket}`}>
                <td><RecordLink id={s.id} label={<>r{s.round} · {s.bucket}</>} /> <Origin origin={s.origin} /></td>
                <td className="num small">{us(s.model_top_e_us)}</td>
                <td className="num small">{us(s.random_best_us)} <span className="muted">n={s.random_n}</span></td>
                <td className="num small">{us(s.copied_us)}{s.copied_from && <div className="muted mono">{s.copied_from.slice(0, 10)}</div>}</td>
                <td>{s.model_beaten_by_random ? <Chip tone="fail">random beat model</Chip> : <Chip tone="ok">model holds</Chip>}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </Panel>
    </span>
  );
}

function Canaries({ v }: { v: MechanismsView }) {
  const rows: MechCanary[] = v.canaries.rows;
  return (
    <span id="canaries">
      <Panel title="Canaries" sub="planted faults the gap routing must catch; a missed one goes to the Inbox until caught again">
        {!rows.length && <Empty what="The summarize process plants one per batch (WP18)." />}
        <div className="row wrap" style={{ gap: 12, marginBottom: 8 }}>
          {Object.entries(v.canaries.by_kind).map(([kind, c]) => (
            <span key={kind} className="small"><b>{kind}</b> {Object.entries(c).map(([verdict, n]) => <Chip key={verdict} tone={VERDICT[verdict] ?? "idle"}>{n} {verdict}</Chip>)}</span>
          ))}
        </div>
        <table className="t compact">
          <thead><tr><th>Batch</th><th>Kind</th><th>Expected → got</th><th>Verdict</th></tr></thead>
          <tbody>
            {rows.map((r) => (
              <tr key={r.id}>
                <td><RecordLink id={r.id} label={r.batch} /> <Origin origin={r.origin} /></td>
                <td className="small">{r.kind}</td>
                <td className="small mono">{r.expect} → {r.got ?? "—"}</td>
                <td><Chip tone={VERDICT[r.verdict] ?? "idle"}>{r.verdict}</Chip></td>
              </tr>
            ))}
          </tbody>
        </table>
      </Panel>
    </span>
  );
}

function WhatIf({ rows }: { rows: MechWhatIf[] }) {
  const moves = (w: MechWhatIf) => [...Object.entries(w.walls).map(([k, n]) => `wall ${k}=${n}`), ...Object.entries(w.peaks).map(([k, n]) => `peak ${k}=${n}`)].join(", ");
  return (
    <span id="whatif">
      <Panel title="L2 what-if" sub="move a wall or a peak, re-simulate: the shadow price is the bottleneck idle time the move removes (WP16)">
        {!rows.length && <Empty what="kta.models.whatif.record_what_if writes them; no process or command calls it yet (WP16)." />}
        <table className="t compact">
          <thead><tr><th>Run</th><th>Move</th><th>Bottleneck</th><th>Shadow price</th><th>Total base → moved</th></tr></thead>
          <tbody>
            {rows.map((w) => (
              <tr key={w.id}>
                <td><RecordLink id={w.id} /> <Origin origin={w.origin} /><div className="small muted mono">{w.model}</div></td>
                <td className="small mono">{moves(w)}</td>
                <td className="small">{w.bottleneck ?? "—"}</td>
                <td className="num"><b>{us(w.shadow_price_us)}</b></td>
                <td className="num small">{us(w.base_total_us)} → {us(w.moved_total_us)}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </Panel>
    </span>
  );
}
