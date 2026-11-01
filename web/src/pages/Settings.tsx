// Settings (WP21): the controls registry of docs/design/12. Every knob with the value in force,
// the failure it prevents, where it lives, its status and the experiment that sets it. Edits
// become one new version (operator token, a reason); nothing else is editable here.
import { useMemo, useState } from "react";
import { useSearchParams } from "react-router";
import { api, operatorToken } from "../data/source";
import type { ControlsVersionRow, ControlsView, Knob } from "../data/types";
import { useApi } from "../data/useApi";
import { Chip, Loading, Panel, RecordLink } from "../components/ui";
import { hhmm } from "../components/format";
import { SystemSettings } from "./SettingsSystem";

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

const TABS = [
  { id: "controls", label: "Research controls" },
  { id: "system", label: "System" },
] as const;

/** Two tabs (WP22): the versioned research controls, and the control plane's system settings. */
export function Settings() {
  const [params, setParams] = useSearchParams();
  const tab = params.get("tab") === "system" ? "system" : "controls";
  const pick = (id: string) => {
    const next = new URLSearchParams(params);
    if (id === "controls") next.delete("tab");
    else next.set("tab", id);
    setParams(next);
  };
  return (
    <div className="stack">
      <div className="tabs">{TABS.map((t) => <button key={t.id} className={tab === t.id ? "on" : ""} onClick={() => pick(t.id)}>{t.label}</button>)}</div>
      {tab === "system" ? <SystemSettings /> : <ResearchControls />}
    </div>
  );
}

function ResearchControls() {
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
  const [query, setQuery] = useState("");
  const [msg, setMsg] = useState<{ ok: boolean; text: string } | null>(null);
  const q = query.trim().toLowerCase();
  const bySection = useMemo(() => {
    const out: Record<string, Knob[]> = {};
    for (const k of view.knobs) {
      if (q && !`${k.row} ${k.id} ${k.controls} ${k.prevents}`.toLowerCase().includes(q)) continue;
      (out[k.section] ??= []).push(k);
    }
    return out;
  }, [view.knobs, q]);
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
    <Panel primary title="Research controls"
      sub={<>version {ver.version || "—"} {ver.id ? <RecordLink id={ver.id} /> : <span className="muted">(not stored yet: version 1 is written at the first run)</span>} · every knob of docs/design/12; code reads values only from here; a save writes one new version</>}
      right={<TokenField />}>
      <div className="knob-tools">
        <input className="knob-search" value={query} onChange={(e) => setQuery(e.target.value)} placeholder="Filter knobs: budget, retry, quiet…" />
        <nav className="knob-nav">{Object.keys(bySection).map((sec) => <a key={sec} href={`#knobs-${sec.replace(/\W+/g, "-")}`}>{SECTIONS[sec] ?? sec}</a>)}</nav>
      </div>
      <div className="stack" style={{ gap: 18 }}>
        {Object.entries(bySection).map(([section, knobs]) => (
          <div key={section} id={`knobs-${section.replace(/\W+/g, "-")}`}>
            <h3 className="knob-section">{SECTIONS[section] ?? section} <span className="muted small">{section} · {knobs.length}</span></h3>
            <div className="knobs">
              {knobs.map((k) => {
                const text = draft[k.id] ?? asText(k.value);
                const dirty = draft[k.id] !== undefined && draft[k.id] !== asText(k.value);
                return (
                  <div key={k.id} className={`knob ${dirty ? "dirty" : ""}`}>
                    <div className="knob-what">
                      <b>{k.row}</b>{k.unit && <span className="muted small"> ({k.unit})</span>}
                      <div className="small ink2">{k.controls}</div>
                      <div className="small muted">prevents: {k.prevents}</div>
                    </div>
                    <div className="knob-value"><ValueEditor k={k} text={text} set={(t) => setDraft({ ...draft, [k.id]: t })} />
                      {dirty && <button className="linkbtn small" onClick={() => { const d = { ...draft }; delete d[k.id]; setDraft(d); }}>undo</button>}</div>
                    <div className="knob-meta small">
                      <Chip tone={k.status === "set" ? "ok" : k.status === "spec" ? "idle" : "waiting"}>{k.status}</Chip> <span className="muted">{k.source}</span>
                      <div className="muted mono" title={`read by ${k.consumer}`}>{k.id}</div>
                      <div className="muted">{k.where.join(", ")}{k.set_by_cell ? ` · set by ${k.set_by_cell}` : ""}</div>
                    </div>
                  </div>
                );
              })}
            </div>
          </div>
        ))}
      </div>
      <div className="save-bar">
        <span className="small"><b>{changed.length}</b> change(s){changed.length ? `: ${changed.map(([id]) => view.knobs.find((k) => k.id === id)?.row).join(", ")}` : ""}</span>
        <input value={reason} onChange={(e) => setReason(e.target.value)} placeholder="reason for this version (required)" style={{ flex: 1, minWidth: 260 }} />
        <button className="op primary" disabled={!changed.length || !reason.trim()} onClick={save}>Write new version</button>
        {msg && <span className={msg.ok ? "small" : "error small"}>{msg.text}</span>}
      </div>
    </Panel>
  );
}

/** One knob's value. Mappings (and per-task-type values) are edited key by key; a nested
 *  value stays JSON in its own field; "as JSON" switches to the whole value as text. */
function ValueEditor({ k, text, set }: { k: Knob; text: string; set: (t: string) => void }) {
  const [raw, setRaw] = useState(false);
  if (!k.editable) return <span className="mono small" title="An invariant enforced in code: shown, not editable">{text} <Chip tone="idle">invariant</Chip></span>;
  if (k.type === "enum") return <select value={text} onChange={(e) => set(e.target.value)}>{k.choices.map((c) => <option key={c}>{c}</option>)}</select>;
  if (k.type === "bool") return <select value={text} onChange={(e) => set(e.target.value)}><option>true</option><option>false</option></select>;
  if (k.type === "list_str") {
    let items: string[] | null = null;
    try { items = JSON.parse(text) as string[]; } catch { items = null; }
    if (items && !raw) {
      return (
        <span className="kv">
          <textarea className="mono" rows={Math.min(8, items.length + 1)} value={items.join("\n")} onChange={(e) => set(JSON.stringify(e.target.value.split("\n").map((x) => x.trim()).filter(Boolean)))} />
          <span className="small muted">one per line · <button className="linkbtn" onClick={() => setRaw(true)}>as JSON</button></span>
        </span>
      );
    }
  }
  if (k.type === "mapping" || k.type === "per_task_type") {
    let obj: Record<string, unknown> | null = null;
    try { const v = JSON.parse(text) as unknown; obj = v && typeof v === "object" && !Array.isArray(v) ? (v as Record<string, unknown>) : null; } catch { obj = null; }
    if (obj && !raw) {
      const put = (key: string, val: string) => {
        const old = obj![key];
        let next: unknown = val;
        if (typeof old === "number") next = val.trim() === "" || Number.isNaN(Number(val)) ? val : Number(val);
        else if (typeof old === "boolean") next = val === "true";
        else if (old !== null && typeof old === "object") { try { next = JSON.parse(val); } catch { next = val; } }
        set(JSON.stringify({ ...obj, [key]: next }));
      };
      return (
        <span className="kv">
          <span className="kv-grid">
            {Object.entries(obj).map(([key, v]) => (
              <label key={key} className="kv-row">
                <span className="mono small">{key}</span>
                {typeof v === "boolean"
                  ? <select value={String(v)} onChange={(e) => put(key, e.target.value)}><option>true</option><option>false</option></select>
                  : <input className="mono" value={typeof v === "object" && v !== null ? JSON.stringify(v) : String(v)} onChange={(e) => put(key, e.target.value)} />}
              </label>
            ))}
          </span>
          <span className="small muted"><button className="linkbtn" onClick={() => setRaw(true)}>edit as JSON</button></span>
        </span>
      );
    }
  }
  const long = text.length > 24 || raw;
  return (
    <span className="kv">
      {long
        ? <textarea className="mono" rows={Math.min(10, Math.ceil(text.length / 60) + 1)} value={text} onChange={(e) => set(e.target.value)} />
        : <input className="mono" style={{ width: 140 }} value={text} onChange={(e) => set(e.target.value)} />}
      {raw && <span className="small muted"><button className="linkbtn" onClick={() => setRaw(false)}>back to fields</button></span>}
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
