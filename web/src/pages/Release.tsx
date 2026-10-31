// Release (WP21): staged release S0–S4 of docs/design/02 §6.3. The stage in force (a controls
// knob), what each stage runs and its exit criteria, and the agent calls held by the gate. In
// S1 you read a call's card, pull log and plan, then release that one call.
import { useState } from "react";
import { api } from "../data/source";
import type { HeldCall, ReleaseStage, ReleaseView } from "../data/types";
import { useApi } from "../data/useApi";
import { Chip, Loading, Panel, RecordLink } from "../components/ui";
import { TokenField } from "./Settings";

export function Release() {
  const r = useApi(api.release, []);
  if (!r.data) return <Loading error={r.error} />;
  return (
    <div className="stack">
      <Stages view={r.data} />
      <Held view={r.data} />
    </div>
  );
}

function Stages({ view }: { view: ReleaseView }) {
  const [to, setTo] = useState(view.stage.id);
  const [reason, setReason] = useState("");
  const [msg, setMsg] = useState<string | null>(null);
  const change = async () => {
    try {
      const out = await api.setControls(view.controls, { "release.stage": to }, reason.trim());
      setMsg(`stage ${to} is controls version ${out.version}; reload`);
    } catch (e) {
      setMsg(e instanceof Error ? e.message : String(e));
    }
  };
  return (
    <Panel primary title={<>Stage in force: {view.stage.id} · {view.stage.name}</>}
      sub={<>controls version <RecordLink id={view.controls} /> · no stage opens until the previous one is understood; you judge the exit criteria</>}
      right={<TokenField />}>
      <table className="t">
        <thead><tr><th>Stage</th><th>Runs</th><th>Processes</th><th>Agent tasks</th><th>Agent calls</th><th>Exit when</th></tr></thead>
        <tbody>
          {view.stages.map((s: ReleaseStage) => (
            <tr key={s.id} className={s.id === view.stage.id ? "hl" : ""}>
              <td><b>{s.id}</b> {s.name}</td>
              <td className="small">{s.runs}</td>
              <td className="small mono">{s.processes.join(", ")}</td>
              <td className="small mono">{s.task_types.join(", ") || "none"}</td>
              <td><Chip tone={s.agent_calls === "gated" ? "human" : s.agent_calls === "none" ? "idle" : "running"}>{s.agent_calls}</Chip></td>
              <td className="small">{s.exit.length ? <ul style={{ margin: 0, paddingLeft: 16 }}>{s.exit.map((e) => <li key={e}>{e}</li>)}</ul> : "—"}</td>
            </tr>
          ))}
        </tbody>
      </table>
      <div className="row wrap" style={{ gap: 8, marginTop: 12 }}>
        <span className="small">Change stage to</span>
        <select value={to} onChange={(e) => setTo(e.target.value)}>{view.stages.map((s) => <option key={s.id}>{s.id}</option>)}</select>
        <input value={reason} onChange={(e) => setReason(e.target.value)} placeholder="reason (e.g. which exit criteria were met)" style={{ flex: 1, minWidth: 260 }} />
        <button className="op primary" disabled={to === view.stage.id || !reason.trim()} onClick={change}>Write controls version</button>
        {msg && <span className="small">{msg}</span>}
      </div>
    </Panel>
  );
}

function Held({ view }: { view: ReleaseView }) {
  return (
    <Panel title="Agent calls waiting" sub={<>queued agent tasks and why each is held · in flight: {view.in_flight.length ? view.in_flight.map((t) => <RecordLink key={t} id={t} label={t.slice(-8)} />) : "none"}</>}>
      {!view.held.length && <div className="empty">No queued agent task.</div>}
      <div className="stack" style={{ gap: 12 }}>
        {view.held.map((h) => <HeldCard key={h.task} h={h} gated={view.stage.agent_calls === "gated"} />)}
      </div>
    </Panel>
  );
}

function HeldCard({ h, gated }: { h: HeldCall; gated: boolean }) {
  const [reason, setReason] = useState("");
  const [msg, setMsg] = useState<string | null>(null);
  const c = h.card_body;
  const go = async () => {
    try {
      const out = await api.releaseCall(h.task, reason.trim());
      setMsg(`attempt ${out.attempt} released`);
    } catch (e) {
      setMsg(e instanceof Error ? e.message : String(e));
    }
  };
  return (
    <div className="panel" style={{ padding: 12 }}>
      <div className="row wrap" style={{ gap: 8 }}>
        <b>{h.type}</b> <span>{c.goal}</span>
        <span className="small muted">task {h.task.slice(-8)} · attempt {h.attempt} · card <RecordLink id={h.card} /></span>
        {h.held ? <Chip tone="waiting">{h.held}</Chip> : <Chip tone="ok">may start</Chip>}
      </div>
      <div className="grid small" style={{ gridTemplateColumns: "repeat(3, minmax(0, 1fr))", gap: 12, marginTop: 8 }}>
        <div><div className="muted">Card</div>
          done checks: <span className="mono">{c.done_checks.join(", ") || "—"}</span><br />
          budget: <span className="mono">{Object.entries(c.budget).map(([k, v]) => `${k} ${v}`).join(" · ")}</span><br />
          inputs: {c.inputs.length ? c.inputs.map((i) => <RecordLink key={i} id={i} />) : "none"}
        </div>
        <div><div className="muted">Pull log</div>{h.pull_log ? <pre className="mono">{JSON.stringify(h.pull_log, null, 1)}</pre> : <span className="muted">{h.pull_log_note}</span>}</div>
        <div><div className="muted">Plan</div>{h.plan ? <pre className="mono">{JSON.stringify(h.plan, null, 1)}</pre> : <span className="muted">{h.plan_note}</span>}</div>
      </div>
      {gated && (
        <div className="row wrap" style={{ gap: 8, marginTop: 8 }}>
          <input value={reason} onChange={(e) => setReason(e.target.value)} placeholder="what you checked before releasing (required)" style={{ flex: 1, minWidth: 260 }} />
          <button className="op primary" disabled={!reason.trim() || h.released_attempt === h.attempt} onClick={go}>
            {h.released_attempt === h.attempt ? "Released" : `Release attempt ${h.attempt}`}
          </button>
          {msg && <span className="small">{msg}</span>}
        </div>
      )}
    </div>
  );
}
