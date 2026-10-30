import { useState } from "react";
import { act, api } from "../data/source";
import type { ActiveRuns, BudgetNode, Machines } from "../data/types";
import { useApi } from "../data/useApi";
import { ActionButton, useCaps } from "../components/actions";
import { Chip, Loading, Panel, RecordLink, UsageBar } from "../components/ui";
import { hhmm } from "../components/format";

export function Operations() {
  const m = useApi(api.machines, []);
  const a = useApi(api.active, []);
  if (!m.data || !a.data) return <Loading error={m.error ?? a.error} />;
  return (
    <div className="stack">
      <div className="grid" style={{ gridTemplateColumns: "minmax(0, 1.25fr) minmax(0, 1fr)", alignItems: "start" }}>
        <StartTask m={m.data} />
        <div className="stack">
          <PausePanel a={a.data} />
          <BudgetPanel a={a.data} />
        </div>
      </div>
      <div className="grid" style={{ gridTemplateColumns: "minmax(0, 1.25fr) minmax(0, 1fr)", alignItems: "start" }}>
        <ActionLog />
        <CapabilityTable />
      </div>
    </div>
  );
}

const ID_SPLIT = /[\s,]+/;

function StartTask({ m }: { m: Machines }) {
  const [type, setType] = useState("T1");
  const [goal, setGoal] = useState("");
  const [inputs, setInputs] = useState("");
  const [priority, setPriority] = useState(100); // tasks you start come first (11 §4.2)
  const fams = useApi(() => api.families(), []);
  const ids = inputs.split(ID_SPLIT).filter(Boolean);
  const def = m.task_types.find((t) => t.id === type);
  const addInput = (id: string) => setInputs((v) => (v.split(ID_SPLIT).includes(id) ? v : `${v.trim()} ${id}`.trim()));
  return (
    <Panel primary title="Start a task" sub="a human-created task goes into the queue like any other; the orchestrator gives it a work unit, budget and profile">
      <div className="stack" style={{ gap: 12 }}>
        <div className="type-cards">
          {m.task_types.map((t) => (
            <button key={t.id} className={`type-card ${type === t.id ? "on" : ""}`} onClick={() => setType(t.id)}>
              <b>{t.id} · {t.name}</b>
              <span className="small ink2">{t.delivers}</span>
              <span className="small muted">by {t.by}</span>
            </button>
          ))}
        </div>
        <div className="form-grid">
          <span>Goal</span>
          <input value={goal} onChange={(e) => setGoal(e.target.value)} placeholder={type === "T1" ? "e.g. gemm_tiled for fp16 inputs (change the dtype condition)" : "what the task must deliver"} />
          <span>Input records</span>
          <textarea rows={2} value={inputs} onChange={(e) => setInputs(e.target.value)} placeholder="record IDs, separated by spaces; the first is the main subject" className="mono" />
          <span />
          <div className="row wrap small" style={{ gap: 6 }}>
            <span className="muted">add a family:</span>
            {(fams.data ?? []).map((f) => <button key={f.id} className="op small" onClick={() => addInput(f.id)}>{f.name} v{f.version}</button>)}
          </div>
          <span>Priority</span>
          <input type="number" value={priority} onChange={(e) => setPriority(Number(e.target.value))} style={{ width: 90 }} />
        </div>
        {ids.length > 0 && <div className="row wrap small" style={{ gap: 6 }}><span className="muted">inputs:</span>{ids.map((i) => <RecordLink key={i} id={i} />)}</div>}
        <div className="row">
          <ActionButton op="create-task" label={`Start ${type}`} tone="primary" title={`Start ${type}: ${def?.name ?? ""}`}
            explain={<>Goal: <b>{goal || "—"}</b><br />Inputs: {ids.length} record(s). The API checks that each input exists before handing the task to the orchestrator.</>}
            valid={goal.trim().length > 0} submit={(r) => act.createTask(type, goal.trim(), ids, priority, r)} />
          {!goal.trim() && <span className="small muted">Enter a goal first.</span>}
        </div>
      </div>
    </Panel>
  );
}

function PausePanel({ a }: { a: ActiveRuns }) {
  const scopes = ["global", "process:solve", "process:learn", "process:calibrate", ...a.loops.map((l) => `loop:${l.id}`)];
  const [scope, setScope] = useState("global");
  const name = (s: string) => (s.startsWith("loop:") ? `${s.slice(0, 5)}${a.loops.find((l) => `loop:${l.id}` === s)?.subject_name ?? s.slice(5)}` : s);
  return (
    <Panel title="Pause or resume" sub="stops new tasks from being leased in the scope; running tasks continue">
      <div className="row wrap">
        <select value={scope} onChange={(e) => setScope(e.target.value)}>{scopes.map((s) => <option key={s} value={s}>{name(s)}</option>)}</select>
        <ActionButton op="pause" label="Pause" title={`Pause ${name(scope)}`} submit={(r) => act.pause(scope, r)} />
        <ActionButton op="resume" label="Resume" title={`Resume ${name(scope)}`} submit={(r) => act.resume(scope, r)} />
      </div>
    </Panel>
  );
}

function BudgetPanel({ a }: { a: ActiveRuns }) {
  const rows = a.budgets.flatMap((tree) => {
    const resource = tree.key.split(":").slice(1).join(":");
    return [{ scope: "global", resource, node: tree, depth: 0 }, ...(tree.children ?? []).map((c) => ({ scope: c.key, resource, node: c, depth: 1 }))];
  });
  return (
    <Panel title="Budgets" sub="limits apply to scheduling from now on; the reason is required">
      {rows.length === 0 ? <div className="empty">No budget scopes are set up in this store.</div> : (
        <table className="t compact">
          <tbody>{rows.map((r) => <BudgetRow key={`${r.scope}/${r.resource}`} {...r} />)}</tbody>
        </table>
      )}
    </Panel>
  );
}

function BudgetRow({ scope, resource, node, depth }: { scope: string; resource: string; node: BudgetNode; depth: number }) {
  const [limit, setLimit] = useState(String(node.limit));
  const value = Number(limit);
  return (
    <tr>
      <td style={{ paddingLeft: 4 + depth * 14, width: 170 }}><span style={{ fontWeight: depth ? 400 : 600 }}>{depth ? node.label : resource}</span>
        <div className="muted mono" style={{ fontSize: 10.5 }}>{scope}</div></td>
      <td style={{ width: "40%" }}><UsageBar used={node.used} reserved={node.reserved} limit={node.limit} />
        <div className="small muted num">{node.used} used · {node.reserved} reserved / {node.limit} {node.unit}</div></td>
      <td className="r">
        <ActionButton op="budget" label="Change" small title={`Budget ${scope} · ${resource}`}
          fields={<label className="field"><span>New limit ({node.unit}); now {node.limit}</span>
            <input type="number" min={0} value={limit} onChange={(e) => setLimit(e.target.value)} /></label>}
          valid={Number.isFinite(value) && value >= 0}
          submit={(r) => act.budget(scope, { [resource]: value }, r)} />
      </td>
    </tr>
  );
}

function ActionLog() {
  const log = useApi(() => api.actionLog(200), []);
  return (
    <Panel title="Your actions" sub="every HumanAction record, newest first, whichever page or command wrote it">
      {!log.data ? <Loading error={log.error} /> : log.data.length === 0 ? <div className="empty">No HumanAction recorded yet.</div> : (
        <table className="t">
          <thead><tr><th>at</th><th>by</th><th>action</th><th>on</th><th>reason</th><th /></tr></thead>
          <tbody>{log.data.map((x) => (
            <tr key={x.id}>
              <td className="num small">{x.at.slice(5, 10)} {hhmm(x.at)}</td>
              <td className="small">{x.by}</td>
              <td><b>{x.verb.replaceAll("_", " ")}</b>
                {x.params && <div className="small muted mono">{Object.entries(x.params).filter(([k]) => k !== "previous").map(([k, v]) => `${k}=${JSON.stringify(v)}`).join(" ")}</div>}
                {x.args && <div className="small muted mono">{x.args}</div>}</td>
              <td className="row wrap" style={{ gap: 4 }}>{Object.entries(x.refs).map(([k, v]) => <span key={k} className="small"><span className="muted">{k} </span>{(Array.isArray(v) ? v : [v]).map((id) => <RecordLink key={id} id={id} />)}</span>)}</td>
              <td className="small">{x.reason || <span className="muted">—</span>}</td>
              <td><RecordLink id={x.id} /></td>
            </tr>
          ))}</tbody>
        </table>
      )}
    </Panel>
  );
}

function CapabilityTable() {
  const { caps, loaded } = useCaps();
  if (!loaded) return <Loading />;
  const list = Object.values(caps);
  const wired = list.filter((c) => c.wired).length;
  return (
    <Panel title="What each control does" sub={`${wired} of ${list.length} operations wired; the rest wait for WP6/WP8 and are disabled, never simulated`}>
      <table className="t compact">
        <tbody>{list.map((c) => (
          <tr key={c.name}>
            <td style={{ whiteSpace: "nowrap" }}><span className={`cap-dot ${c.wired ? "on" : "off"}`} /><b>{c.name}</b><div className="mono muted" style={{ fontSize: 10.5 }}>{c.endpoint}</div></td>
            <td className="small">{c.writes}
              {!c.wired && <div className="warn-text">needs {c.needs}</div>}
              {c.partial && <div className="warn-text">{c.partial}</div>}</td>
            <td>{c.wired ? <Chip tone="ok">wired</Chip> : <Chip tone="idle">not wired</Chip>}</td>
          </tr>
        ))}</tbody>
      </table>
    </Panel>
  );
}
