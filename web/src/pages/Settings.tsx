// Settings (WP21): the controls registry of docs/design/12. Every knob with the value in force,
// the failure it prevents, where it lives, its status and the experiment that sets it. Edits
// become one new version (operator token, a reason); nothing else is editable here.
import { useMemo, useState } from "react";
import { api, operatorToken } from "../data/source";
import type { ControlsVersionRow, ControlsView, Knob } from "../data/types";
import { useApi } from "../data/useApi";
import { Chip, Loading, Panel, RecordLink } from "../components/ui";
import { hhmm } from "../components/format";

const SECTIONS: Record<string, string> = {
  "12 §1": "Budget and rounds",
  "12 §2": "Session and context",
  "12 §3": "Against digging in, slacking and gaming",
  "12 §4": "Escalation and reliability",
  "12 §7": "Rows added from code and D325",
};

/** The text a value is edited as: numbers and strings plain, everything else JSON. */
const asText = (v: unknown): string => (typeof v === "string" || typeof v === "number" ? String(v) : JSON.stringify(v));

function parse(k: Knob, text: string): unknown {
  if (k.type === "int" || k.type === "float") {
    const n = Number(text);
    if (text.trim() === "" || Number.isNaN(n)) throw new Error(`${k.row}: not a number`);
    return n;
  }
  if (k.type === "str" || k.type === "enum") return text;
  if (k.type === "bool") return text === "true";
  return JSON.parse(text) as unknown;
}

export function Settings() {
  const c = useApi(api.controls, []);
  const v = useApi(() => api.controlsVersions(), []);
  if (!c.data) return <Loading error={c.error} />;
  return (
    <div className="stack">
      <Editor view={c.data} />
      <div className="grid" style={{ gridTemplateColumns: "minmax(0, 1fr) minmax(0, 1fr)", alignItems: "start" }}>
        <Versions rows={v.data ?? []} error={v.error} />
        <Experiments view={c.data} />
      </div>
    </div>
  );
}

export function TokenField() {
  const [token, setToken] = useState(operatorToken.get());
  return (
    <label className="row small" style={{ gap: 6 }} title="The content of [api] operator_token_file (kta controls token). Kept for this tab only.">
      <span className="muted">operator token</span>
      <input type="password" value={token} style={{ width: 180 }} onChange={(e) => { setToken(e.target.value); operatorToken.set(e.target.value); }} />
    </label>
  );
}

function Editor({ view }: { view: ControlsView }) {
  const [draft, setDraft] = useState<Record<string, string>>({});
  const [reason, setReason] = useState("");
  const [msg, setMsg] = useState<{ ok: boolean; text: string } | null>(null);
  const bySection = useMemo(() => {
    const out: Record<string, Knob[]> = {};
    for (const k of view.knobs) (out[k.section] ??= []).push(k);
    return out;
  }, [view.knobs]);
  const changed = Object.entries(draft).filter(([id, text]) => asText(view.knobs.find((k) => k.id === id)?.value) !== text);
  const save = async () => {
    try {
      const changes: Record<string, unknown> = {};
      for (const [id, text] of changed) changes[id] = parse(view.knobs.find((k) => k.id === id)!, text);
      const out = await api.setControls(view.version.id, changes, reason.trim());
      setMsg({ ok: true, text: `version ${out.version} written (${out.id.slice(0, 12)}); reload to see it in force` });
      setDraft({});
      setReason("");
    } catch (e) {
      setMsg({ ok: false, text: e instanceof Error ? e.message : String(e) });
    }
  };
  const ver = view.version;
  return (
    <Panel primary title="Controls registry"
      sub={<>version {ver.version || "—"} {ver.id ? <RecordLink id={ver.id} /> : <span className="muted">(not stored yet: version 1 is written at the first run)</span>} · every row is a knob of docs/design/12; code reads values only from here</>}
      right={<TokenField />}>
      <div className="stack" style={{ gap: 16 }}>
        {Object.entries(bySection).map(([section, knobs]) => (
          <div key={section}>
            <h3 className="small" style={{ margin: "0 0 6px" }}>{section} · {SECTIONS[section] ?? ""}</h3>
            <table className="t compact">
              <thead><tr><th>Knob</th><th>Value</th><th>Prevents</th><th>Where</th><th>Status</th><th>set_by</th></tr></thead>
              <tbody>
                {knobs.map((k) => (
                  <tr key={k.id} className={draft[k.id] !== undefined && draft[k.id] !== asText(k.value) ? "hl" : ""}>
                    <td title={`${k.id}\n${k.controls}\nread by: ${k.consumer}`}><b>{k.row}</b><div className="small muted">{k.controls}</div></td>
                    <td style={{ minWidth: 220 }}><ValueCell k={k} text={draft[k.id] ?? asText(k.value)} set={(t) => setDraft({ ...draft, [k.id]: t })} /></td>
                    <td className="small">{k.prevents}</td>
                    <td className="small">{k.where.join(", ")}</td>
                    <td className="small"><Chip tone={k.status === "set" ? "ok" : k.status === "spec" ? "idle" : "waiting"}>{k.status}</Chip> <span className="muted">{k.source}</span></td>
                    <td className="small mono">{k.set_by_cell}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        ))}
        <div className="row wrap" style={{ gap: 8 }}>
          <span className="small">{changed.length} change(s)</span>
          <input value={reason} onChange={(e) => setReason(e.target.value)} placeholder="reason for this version (required)" style={{ flex: 1, minWidth: 260 }} />
          <button className="op primary" disabled={!changed.length || !reason.trim()} onClick={save}>Write new version</button>
          {msg && <span className={msg.ok ? "small" : "error small"}>{msg.text}</span>}
        </div>
      </div>
    </Panel>
  );
}

function ValueCell({ k, text, set }: { k: Knob; text: string; set: (t: string) => void }) {
  if (!k.editable) return <span className="mono small" title="An invariant enforced in code: shown, not editable">{text} <Chip tone="idle">invariant</Chip></span>;
  if (k.type === "enum") return <select value={text} onChange={(e) => set(e.target.value)}>{k.choices.map((c) => <option key={c}>{c}</option>)}</select>;
  if (k.type === "bool") return <select value={text} onChange={(e) => set(e.target.value)}><option>true</option><option>false</option></select>;
  const wide = k.type === "mapping" || k.type === "per_task_type" || k.type === "list_str";
  return (
    <span className="row small" style={{ gap: 4 }}>
      {wide
        ? <textarea className="mono" rows={2} style={{ width: "100%" }} value={text} onChange={(e) => set(e.target.value)} />
        : <input className="mono" style={{ width: 110 }} value={text} onChange={(e) => set(e.target.value)} />}
      {k.unit && <span className="muted">{k.unit}</span>}
    </span>
  );
}

function Versions({ rows, error }: { rows: ControlsVersionRow[]; error?: string }) {
  return (
    <Panel title="Versions" sub="newest first; each names its author, reason and changes; runs record the version they used">
      {error && <div className="error">{error}</div>}
      {!rows.length && !error && <div className="empty">No stored version yet.</div>}
      <table className="t compact">
        <tbody>
          {rows.map((r) => (
            <tr key={r.id}>
              <td className="num">v{r.version}</td>
              <td><RecordLink id={r.id} /><div className="small muted">{hhmm(r.at)} · {r.author} ({r.author_type})</div></td>
              <td className="small">{r.reason}
                <ul style={{ margin: "4px 0 0", paddingLeft: 16 }}>
                  {r.changes.slice(0, 6).map((c) => <li key={c.knob} className="mono">{c.knob}: {asText(c.old)} → {asText(c.new)}</li>)}
                  {r.changes.length > 6 && <li className="muted">… {r.changes.length - 6} more</li>}
                </ul>
              </td>
            </tr>
          ))}
        </tbody>
      </table>
    </Panel>
  );
}

function Experiments({ view }: { view: ControlsView }) {
  return (
    <Panel title="System-parameter experiments X1–X12" sub="02 §11, in its order (step 1–6): defined, not run; a default stands until its experiment has run">
      <table className="t compact">
        <thead><tr><th>ID</th><th>Question and design</th><th>Arms · metrics</th><th>Sets</th></tr></thead>
        <tbody>
          {[...view.experiments].sort((a, b) => a.step - b.step).map((e) => (
            <tr key={e.id}>
              <td><b>{e.id}</b><div className="small muted">step {e.step}{e.method === "logs" ? " · from logs" : ""}</div><div><Chip tone="idle">{e.status}</Chip></div></td>
              <td className="small"><b>{e.question}</b><div className="muted">{e.design}</div></td>
              <td className="small">{e.arms.join(" | ")}<div className="muted">{e.metrics.join(", ")}</div></td>
              <td className="small mono">{[...e.sets, ...e.profile_fields.map((f) => `AgentProfile.${f}`), ...e.sets_elsewhere].join(", ") || "—"}</td>
            </tr>
          ))}
        </tbody>
      </table>
    </Panel>
  );
}
