import { useMemo } from "react";
import { act, api } from "../data/source";
import type { ExperimentProtocol, ImprovementsView } from "../data/types";
import { useApi } from "../data/useApi";
import { ActionButton } from "../components/actions";
import { AXIS, EChart, TOOLTIP, type EOption } from "../components/EChart";
import { Chip, Loading, Panel, RecordLink } from "../components/ui";
import { hhmm, wordTone } from "../components/format";

// Experiments and Improvement read the record conventions of decision D345 (src/kta/api/studies.py).

export function Experiments() {
  const e = useApi(api.experiments, []);
  if (!e.data) return <Loading error={e.error} />;
  if (e.data.length === 0) return (
    <div className="empty-state">
      <b>No experiment protocol recorded yet.</b>
      <span>An experiment starts from a Protocol record with <span className="mono">type: experiment</span> (arms in <span className="mono">params.arms</span>,
        planned runs in <span className="mono">params.runs_per_arm</span>). You approve it here, which freezes it; the experiment runner (WP12) then writes
        ExperimentRun records per arm and result Reports titled <span className="mono">experiment</span>.</span>
    </div>
  );
  return <div className="stack">{e.data.map((p) => <ProtocolCard key={p.id} p={p} />)}</div>;
}

function ProtocolCard({ p }: { p: ExperimentProtocol }) {
  const approved = p.approved[0];
  const done = p.arms.reduce((s, a) => s + a.runs, 0);
  const planned = p.arms.reduce((s, a) => s + (a.planned ?? 0), 0);
  return (
    <Panel primary title={`${p.name} v${p.version}`} sub={p.description}
      right={<span className="row" style={{ gap: 8 }}>
        {approved ? <Chip tone="ok">approved {approved.at.slice(0, 10)}</Chip> : <Chip tone="human">waiting for approval</Chip>}
        <RecordLink id={p.id} />
      </span>}>
      <div className="grid" style={{ gridTemplateColumns: "minmax(0, 1fr) minmax(0, 1fr)", alignItems: "start" }}>
        <div className="stack" style={{ gap: 8 }}>
          <ExperimentState p={p} />
          <div className="caption">Arms {planned ? `· ${done}/${planned} runs` : `· ${done} runs`}</div>
          {p.arms.map((a) => (
            <div key={a.name} className="arm-bar">
              <b className="mono">{a.name}</b>
              <div className="bar" style={{ height: 12 }}><span className="used" style={{ width: `${a.planned ? Math.min(1, a.runs / a.planned) * 100 : a.runs ? 100 : 0}%` }} /></div>
              <span className="num small">{a.runs}{a.planned ? `/${a.planned}` : ""}</span>
            </div>
          ))}
          <details><summary>protocol parameters</summary><pre className="json">{JSON.stringify(p.params, null, 2)}</pre></details>
          {!approved && (
            <div className="row">
              <ActionButton op="approve-protocol" label="Approve and freeze" tone="primary" title={`Approve ${p.name} v${p.version}`}
                explain="Once approved, the protocol is frozen: the experiment may start on it and it can no longer change (a change is a new version)."
                submit={(r) => act.approve("protocol", p.id, r)} />
            </div>
          )}
          {approved && <div className="small muted">Approved by {approved.by} at {hhmm(approved.at)}: {approved.reason || "no reason given"} <RecordLink id={approved.id} /></div>}
        </div>
        <div className="stack" style={{ gap: 8 }}>
          <div className="caption">Results</div>
          {p.results.length === 0 ? <div className="empty">No result report yet.</div> : p.results.map((r) => (
            <div key={r.id}><div className="small">{r.summary} <RecordLink id={r.id} /></div><ResultTable data={r.data} /></div>
          ))}
          <div className="caption">Runs</div>
          <table className="t compact"><tbody>{p.runs.slice(-12).map((r) => (
            <tr key={r.id}><td className="mono small">{r.condition}</td><td className="small">{r.name}</td><td><Chip tone={wordTone(r.status)}>{r.status}</Chip></td><td><RecordLink id={r.id} /></td></tr>
          ))}</tbody></table>
        </div>
      </div>
    </Panel>
  );
}

function ResultTable({ data }: { data: Record<string, unknown> }) {
  const rows = Array.isArray(data.rows) ? (data.rows as Record<string, unknown>[]) : [];
  if (!rows.length) return <pre className="json">{JSON.stringify(data, null, 2)}</pre>;
  const cols = [...new Set(rows.flatMap((r) => Object.keys(r)))];
  return (
    <table className="t compact"><thead><tr>{cols.map((c) => <th key={c}>{c}</th>)}</tr></thead>
      <tbody>{rows.map((r, i) => <tr key={i}>{cols.map((c) => <td key={c} className="num small">{String(r[c] ?? "")}</td>)}</tr>)}</tbody></table>
  );
}

const ATTRIBUTION = ["space", "model", "intent_sass", "verification", "process", "granularity", "other"];
const ATTR_COLOR: Record<string, string> = {
  space: "#2a78d6", model: "#6250d6", intent_sass: "#eb6834", verification: "#008300", process: "#eda100", granularity: "#e87ba4", other: "#a3a29b",
};
const ATTR_MEANING: Record<string, string> = {
  space: "the pool cannot express the needed optimisation", model: "the performance model ranked wrongly", intent_sass: "the compiler did not do what the Template intended",
  verification: "a check was missing or wrong", process: "the agent or orchestration failed", granularity: "Templates at the wrong grain", other: "not classified",
};

export function Improvement() {
  const v = useApi(api.improvements, []);
  if (!v.data) return <Loading error={v.error} />;
  return <ImprovementBody v={v.data} />;
}

function ImprovementBody({ v }: { v: ImprovementsView }) {
  const total = Object.values(v.attribution).reduce((s, n) => s + n, 0);
  return (
    <div className="stack">
      <div className="grid" style={{ gridTemplateColumns: "minmax(0, 1.3fr) minmax(0, 1fr)", alignItems: "start" }}>
        <Panel primary title="Where the shortfalls come from" sub={`${total} Gap record(s) by attribution, per day; the mix tells which layer to improve next`}>
          {total === 0 ? <div className="empty">No Gap recorded yet.</div> : <AttributionChart v={v} />}
        </Panel>
        <Panel title="Attribution categories" sub="Gap.attribution (spec/schemas.md §4.7)">
          <table className="t compact"><tbody>{ATTRIBUTION.map((a) => (
            <tr key={a}><td><span className="sw-inline" style={{ background: ATTR_COLOR[a] }} /><b>{a}</b><div className="small muted">{ATTR_MEANING[a]}</div></td>
              <td className="r num">{v.attribution[a] ?? 0}</td></tr>
          ))}</tbody></table>
          <div className="small muted">Missing kind: {Object.entries(v.missing).map(([k, n]) => `${k} ${n}`).join(" · ") || "—"}</div>
        </Panel>
      </div>
      <Panel title="Improvement reports" sub="one per batch, written by the improvement summary (WP6 improve.py, docs/design/10 §2.3); system-code items need your approval">
        {v.reports.length === 0 ? (
          <div className="empty-state"><b>No improvement report yet.</b>
            <span>Reports are Report records titled <span className="mono">improvement</span> with <span className="mono">data.batch</span>, <span className="mono">data.attribution</span> and
              <span className="mono"> data.items</span>. Approving one hands its system-code items to a coding agent.</span></div>
        ) : v.reports.map((r) => (
          <div key={r.id} style={{ borderBottom: "1px solid #efeee9", padding: "10px 0" }}>
            <div className="row wrap" style={{ gap: 8 }}>
              <b>{r.batch ?? "batch ?"}</b><span className="small ink2">{r.summary}</span><RecordLink id={r.id} />
              <span style={{ marginLeft: "auto" }}>{r.approved.length ? <Chip tone="ok">approved by {r.approved[0].by}</Chip>
                : r.rejected.length ? <Chip tone="idle">rejected by {r.rejected[0].by}</Chip>
                : r.system_items === 0 ? <span className="small muted">no system item to decide</span> : (
                <ImprovementDecision id={r.id} label={`improvement report ${r.batch ?? ""}`} />)}</span>
            </div>
            <ul className="small" style={{ margin: "6px 0 0", paddingLeft: 18 }}>
              {r.items.map((it, i) => <li key={it.key ?? i}><span className="tag">{it.kind}</span> <b>{it.title}</b>{it.description ? ` — ${it.description}` : ""}</li>)}
            </ul>
          </div>
        ))}
      </Panel>
    </div>
  );
}

function AttributionChart({ v }: { v: ImprovementsView }) {
  const option = useMemo<EOption>(() => ({
    animation: false,
    grid: { left: 40, right: 16, top: 30, bottom: 30 },
    legend: { top: 0, textStyle: { color: "#52514e", fontSize: 11 } },
    tooltip: { ...TOOLTIP, trigger: "axis", axisPointer: { type: "shadow" } },
    xAxis: { type: "category", data: v.attribution_by_day.map((d) => d.day), ...AXIS },
    yAxis: { type: "value", minInterval: 1, ...AXIS },
    series: ATTRIBUTION.filter((a) => v.attribution[a]).map((a) => ({
      type: "bar", name: a, stack: "gaps", itemStyle: { color: ATTR_COLOR[a] }, barMaxWidth: 36,
      data: v.attribution_by_day.map((d) => Number(d[a] ?? 0)),
    })),
  }) as EOption, [v]);
  return <EChart option={option} height={260} />;
}

/** Approve or reject the system items of an improvement Report (WP11 D406; inbox item
 *  ``improvement_pending`` until one of the two HumanActions exists). */
export function ImprovementDecision({ id, label, small }: { id: string; label: string; small?: boolean }) {
  return (
    <span className="row" style={{ gap: 6, display: "inline-flex" }}>
      <ActionButton op="approve-improvement" label="Approve" tone="primary" small={small} title={`Approve ${label}`}
        explain="Hands the system-code items of this report to a coding agent." submit={(x) => act.approve("improvement", id, x)} />
      <ActionButton op="reject-improvement" label="Reject" tone="danger" small={small} title={`Reject ${label}`}
        explain="The system items of this report are not pursued; the inbox item closes." submit={(x) => act.rejectImprovement(id, x)} />
    </span>
  );
}

function ExperimentState({ p }: { p: ExperimentProtocol }) {
  const st = p.progress;
  if (!st.status && !st.started) return p.approved.length ? <div className="small muted">Approved; the experiment process has not started it yet.</div> : null;
  return (
    <div className="row wrap small" style={{ gap: 10 }}>
      <Chip tone={st.status === "done" ? "ok" : st.error ? "fail" : "running"}>{st.status ?? "approved"}</Chip>
      <span>{st.started} run(s) started · {st.finished} finished</span>
      {st.report && <span>result <RecordLink id={st.report} /></span>}
      {st.error && <span style={{ color: "#9b1f1f" }}>last error: {st.error}</span>}
    </div>
  );
}
