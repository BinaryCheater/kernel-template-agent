// Workflow (WP22 Part D, D593–D597): how an agent task of each type runs, and what configures
// each step. The pipeline comes from GET /workflow, which derives it from machines.yaml, the
// done-check table, sessions.decide and the profile record. Clicking a node opens what
// configures it: profile templates, skills and CLI fields become a DRAFT profile version (new
// record, diff, live TASK.md preview, approval moves current_profile); knobs go through the
// controls versioning; the proxy policy through the system configuration. Done checks and
// invariants are read-only with their source.
import { useEffect, useMemo, useState } from "react";
import { useSearchParams } from "react-router";
import { api } from "../data/source";
import type { WfChanges, WfDiff, WfKnob, WfPreview, WfTaskType, WorkflowView } from "../data/types";
import { useApi } from "../data/useApi";
import { Chip, Loading, Panel, RecordLink } from "../components/ui";
import { TokenField } from "./Settings";

type NodeId = "creators" | "card" | "prompt" | "session" | "run" | "checks" | "outcome";
type Item = { kind: "template" | "skill" | "field" | "knob" | "proxy"; key: string };
const EMPTY: WfChanges = { templates: {}, skills: {}, fields: {} };
const FIELDS = ["provider", "model", "reasoning_effort", "cli_version"] as const;
const asText = (v: unknown) => (typeof v === "string" || typeof v === "number" ? String(v) : JSON.stringify(v));

function count(c: WfChanges) {
  return Object.keys(c.templates).length + Object.keys(c.skills).length + Object.keys(c.fields).length;
}

export function Workflow() {
  const [params, setParams] = useSearchParams();
  const profile = params.get("profile") ?? undefined;
  const w = useApi(() => api.workflow(profile), [profile]);
  if (!w.data) return <Loading error={w.error} />;
  return <WorkflowBody key={w.data.profile.id} w={w.data} pick={(id) => { const n = new URLSearchParams(params); n.set("profile", id); setParams(n); }} />;
}

function WorkflowBody({ w, pick }: { w: WorkflowView; pick: (id: string) => void }) {
  const [type, setType] = useState("T1");
  const [node, setNode] = useState<NodeId>("prompt");
  const [item, setItem] = useState<Item | null>({ kind: "template", key: "step:T1" });
  const [changes, setChanges] = useState<WfChanges>(EMPTY);
  const tt = w.task_types.find((t) => t.id === type)!;
  const select = (n: NodeId, i: Item | null) => { setNode(n); setItem(i); };
  return (
    <div className="stack">
      <ProfileBar w={w} pick={pick} />
      <div className="tabs">{w.task_types.map((t) => (
        <button key={t.id} className={type === t.id ? "on" : ""} onClick={() => { setType(t.id); if (item?.key.startsWith("step:")) setItem({ kind: "template", key: `step:${t.id}` }); }}>
          {t.id} <span className="muted">{t.name}</span></button>))}</div>
      <div className="grid" style={{ gridTemplateColumns: "minmax(0, 5fr) minmax(0, 6fr)", alignItems: "start", gap: 16 }}>
        <Pipeline w={w} tt={tt} node={node} item={item} select={select} changes={changes} />
        <Editor w={w} tt={tt} node={node} item={item} changes={changes} setChanges={setChanges} />
      </div>
      <DraftBar w={w} changes={changes} reset={() => setChanges(EMPTY)} pick={pick} />
    </div>
  );
}

function ProfileBar({ w, pick }: { w: WorkflowView; pick: (id: string) => void }) {
  const p = w.profile;
  const cur = w.versions.find((v) => v.current);
  const diff = useApi(() => (cur && !p.current ? api.workflowDiff(cur.id, p.id) : Promise.resolve([] as WfDiff[])), [p.id]);
  return (
    <Panel primary title={<>Profile {p.name} v{p.version}</>}
      sub={<>{p.current ? <Chip tone="ok">in force</Chip> : <Chip tone="waiting">not in force</Chip>} <RecordLink id={p.id} /> · {p.provider} · {p.model} · effort {p.reasoning_effort ?? "—"} · CLI {p.cli_version} · isolation {p.isolation}
        {p.base && <> · edited from <RecordLink id={p.base} /></>}</>}
      right={<select value={p.id} onChange={(e) => pick(e.target.value)}>{w.versions.map((v) => <option key={v.id} value={v.id}>v{v.version}{v.current ? " (in force)" : ""} · {v.author}</option>)}</select>}>
      <div className="small muted">Every task type below runs with this profile once it is in force. Edits never change a profile: they become a new version (draft), shown as a diff, approved by you; past runs keep the version they recorded.</div>
      {!p.current && cur && <DiffList title={`differences from the version in force (v${cur.version})`} rows={diff.data ?? []} />}
    </Panel>
  );
}

function Node({ id, title, active, onClick, children, edited }: { id: NodeId; title: string; active: boolean; onClick: () => void; children: React.ReactNode; edited?: boolean }) {
  return (
    <div onClick={onClick} data-node={id}
      style={{ border: `1px solid ${active ? "var(--st-human)" : "var(--hairline)"}`, borderRadius: 6, padding: "8px 10px", cursor: "pointer", background: active ? "var(--panel, #fff)" : undefined }}>
      <div className="row" style={{ justifyContent: "space-between" }}><b>{title}</b>{edited && <Chip tone="human">edited</Chip>}</div>
      <div className="small">{children}</div>
    </div>
  );
}

const Arrow = ({ label }: { label?: string }) => <div className="small muted" style={{ textAlign: "center", lineHeight: "18px" }}>↓ {label}</div>;

function Link({ onClick, children, on }: { onClick: () => void; children: React.ReactNode; on?: boolean }) {
  return <span className={on ? "rl" : "rlv"} style={{ marginRight: 8, fontWeight: on ? 700 : undefined }} onClick={(e) => { e.stopPropagation(); onClick(); }}>{children}</span>;
}

function Pipeline({ w, tt, node, item, select, changes }: {
  w: WorkflowView; tt: WfTaskType; node: NodeId; item: Item | null; select: (n: NodeId, i: Item | null) => void; changes: WfChanges;
}) {
  const t = (key: string, n: NodeId, label = key) => <Link on={item?.key === key} onClick={() => select(n, { kind: "template", key })}>{label}{changes.templates[key] !== undefined ? " *" : ""}</Link>;
  const retry = w.lifecycle.transitions.filter((e) => ["checking", "rejected", "expired", "escalated"].includes(e.from));
  const edited = (keys: string[]) => keys.some((k) => changes.templates[k] !== undefined);
  return (
    <Panel title={`${tt.id} pipeline`} sub={`delivers ${tt.delivers}; derived from machines.yaml, done_checks.py, sessions.decide and the profile`}>
      <div className="stack" style={{ gap: 2 }}>
        <Node id="creators" title="Created by" active={node === "creators"} onClick={() => select("creators", null)}>
          {tt.creators.length ? tt.creators.map((c, i) => <div key={i}>{c.kind === "loop" ? <><b>{c.machine} loop</b> {c.from} → {c.to}{c.label ? ` (${c.label})` : ""}</> : <><b>process</b> <span className="mono">{c.where}</span></>}</div>)
            : <span className="muted">no creator found (only by hand: Operations → create task)</span>}
        </Node>
        <Arrow label="TaskCard (immutable; budget fixed at creation)" />
        <Node id="card" title="Task card and inputs" active={node === "card"} onClick={() => select("card", null)}>
          goal, inputs (≤ cards_k cards), done checks, budget · knobs: {w.knobs.nodes.card.map((k) => k.row).join(", ")}
        </Node>
        <Arrow label="queued → leased" />
        <Node id="prompt" title="Prompt assembly" active={node === "prompt"} onClick={() => select("prompt", { kind: "template", key: `step:${tt.id}` })}
          edited={edited(["prefix", `step:${tt.id}`, "message:notes_seed"]) || tt.skills.some((s) => changes.skills[s] !== undefined)}>
          <div>{t("prefix", "prompt", "AGENTS.md (prefix)")}{t(`step:${tt.id}`, "prompt", `TASK.md (step-${tt.id})`)}{t("message:notes_seed", "prompt", "NOTES.md seed")}</div>
          <div>skills: {tt.skills.map((s) => <Link key={s} on={item?.key === s} onClick={() => select("prompt", { kind: "skill", key: s })}>{s}{changes.skills[s] !== undefined ? " *" : ""}</Link>)}</div>
          <div className="muted">{tt.commands.length} kta commands allowed</div>
        </Node>
        <Arrow />
        <Node id="session" title="Session: new or resume" active={node === "session"} onClick={() => select("session", null)} edited={edited(["message:new_session", "message:human_note"])}>
          <ol style={{ margin: "2px 0", paddingLeft: 18 }}>{w.session.rules.map((r, i) => <li key={i}><Chip tone={r.mode === "resume" ? "ok" : "idle"}>{r.mode}</Chip> {r.reason}</li>)}</ol>
          <div>{t("message:new_session", "session", "new-session message")}{t("message:human_note", "session", "human note")}</div>
        </Node>
        <Arrow />
        <Node id="run" title="Agent run" active={node === "run"} onClick={() => select("run", { kind: "field", key: "model" })}
          edited={Object.keys(changes.fields).length > 0 || edited(["home:config.toml", "message:budget_warning"])}>
          {changes.fields.provider ?? w.profile.provider} · {changes.fields.model ?? w.profile.model} · effort {changes.fields.reasoning_effort ?? w.profile.reasoning_effort ?? "—"} · proxy <Chip tone={w.proxy.proxy === "required" ? "ok" : "idle"}>{w.proxy.proxy}</Chip>
          <div>{t("home:config.toml", "run", "CLI config.toml")}{t("message:budget_warning", "run", "budget warning")}</div>
        </Node>
        <Arrow label="kta submit" />
        <Node id="checks" title="Done checks (fixed, read-only)" active={node === "checks"} onClick={() => select("checks", null)}>
          {tt.done_checks.map((c) => <span key={c.name} className="mono" style={{ marginRight: 8 }}>{c.name}</span>)}
        </Node>
        <Arrow />
        <Node id="outcome" title="Accept, retry or escalate" active={node === "outcome"} onClick={() => select("outcome", { kind: "template", key: "message:rejection" })} edited={edited(["message:rejection"])}>
          {retry.map((e, i) => <div key={i}>{e.from} → <b>{e.to}</b>{e.label ? ` (${e.label})` : ""}{e.when ? <span className="muted"> — {e.when}</span> : ""}</div>)}
          <div className="muted">max retries: {asText(w.knobs.nodes.retry.find((k) => k.id === "budget.retries_per_task")?.value)} · a retry resumes the session with the rejection message</div>
        </Node>
      </div>
    </Panel>
  );
}

function Editor({ w, tt, node, item, changes, setChanges }: {
  w: WorkflowView; tt: WfTaskType; node: NodeId; item: Item | null; changes: WfChanges; setChanges: (c: WfChanges) => void;
}) {
  const knobs = w.knobs.nodes[node] ?? [];
  return (
    <div className="stack">
      {item?.kind === "template" && <TemplateEditor w={w} tt={tt} k={item.key} changes={changes} setChanges={setChanges} />}
      {item?.kind === "skill" && <SkillEditor w={w} name={item.key} changes={changes} setChanges={setChanges} />}
      {node === "run" && <FieldsEditor w={w} changes={changes} setChanges={setChanges} />}
      {node === "run" && <ProxyEditor w={w} />}
      {node === "creators" && <Panel title="Who creates this task" sub="read-only: loop transitions (machines.yaml) and process call sites (source scan)">
        <table className="t compact"><tbody>{tt.creators.map((c, i) => <tr key={i}><td>{c.kind}</td><td className="small">{c.kind === "loop" ? <>{c.machine}: {c.from} → {c.to}{c.rule ? ` · rule ${c.rule}` : ""}<div className="muted">{c.when}</div></> : <span className="mono">{c.where}</span>}</td></tr>)}</tbody></table>
      </Panel>}
      {node === "checks" && <Panel title="Done checks" sub="fixed per task type in code; agents cannot change them">
        <table className="t compact"><tbody>{tt.done_checks.map((c) => <tr key={c.name}><td className="mono">{c.name}</td><td className="mono small muted">{c.source}</td></tr>)}</tbody></table>
        <h3 className="small">Invariants</h3>
        <ul className="small">{w.invariants.map((v) => <li key={v.what}>{v.what} <span className="mono muted">{v.source}</span></li>)}</ul>
      </Panel>}
      {knobs.length > 0 && <KnobEditor w={w} knobs={knobs} />}
      {node === "session" && <div className="small muted">Rules read from <span className="mono">{w.session.source}</span>, in order; the first that applies decides.</div>}
    </div>
  );
}

function TemplateEditor({ w, tt, k, changes, setChanges }: { w: WorkflowView; tt: WfTaskType; k: string; changes: WfChanges; setChanges: (c: WfChanges) => void }) {
  const tpl = w.templates[k];
  const text = changes.templates[k] ?? tpl?.text ?? "";
  const [card, setCard] = useState("");
  const [prev, setPrev] = useState<WfPreview | null>(null);
  const [err, setErr] = useState<string | null>(null);
  const key = JSON.stringify(changes);
  useEffect(() => {
    const h = setTimeout(() => {
      api.workflowPreview(w.profile.id, changes, tt.id, card.trim() || undefined).then((p) => { setPrev(p); setErr(null); }, (e: unknown) => setErr(e instanceof Error ? e.message : String(e)));
    }, 300);
    return () => clearTimeout(h);
  }, [key, tt.id, card, w.profile.id]); // eslint-disable-line react-hooks/exhaustive-deps
  if (!tpl) return <Panel title={k}><div className="empty">This profile has no {k}.</div></Panel>;
  const set = (v: string) => {
    const next = { ...changes.templates };
    if (v === tpl.text) delete next[k]; else next[k] = v;
    setChanges({ ...changes, templates: next });
  };
  return (
    <>
      <Panel title={<>Template <span className="mono">{k}</span></>} sub={tpl.placeholders ? <>placeholders: {tpl.placeholders.map((p) => <span key={p} className="mono">${p} </span>)}</> : "copied verbatim (no placeholders)"}
        right={changes.templates[k] !== undefined && <button className="op" onClick={() => set(tpl.text)}>Revert</button>}>
        <textarea className="mono" style={{ width: "100%", minHeight: 260, fontSize: 12 }} value={text} onChange={(e) => set(e.target.value)} />
      </Panel>
      <Preview prev={prev && { ...prev, warnings: prev.warnings.filter((x) => x.startsWith(`${k}:`)) }} err={err} card={card} setCard={setCard} type={tt.id} />
    </>
  );
}

function Preview({ prev, err, card, setCard, type }: { prev: WfPreview | null; err: string | null; card: string; setCard: (s: string) => void; type: string }) {
  return (
    <Panel title={`Preview: TASK.md for ${type}`} sub="rendered by the runner's own code (cards.task_md) from the draft"
      right={<input className="mono small" placeholder="real TaskCard ID (optional)" value={card} onChange={(e) => setCard(e.target.value)} style={{ width: 240 }} />}>
      {err && <div className="error small">{err}</div>}
      {prev?.errors.map((e) => <div key={e} className="error small">✗ {e}</div>)}
      {prev?.warnings.map((e) => <div key={e} className="small muted">⚠ {e}</div>)}
      {prev?.task_md ? <pre className="mono small" style={{ whiteSpace: "pre-wrap", maxHeight: 360, overflow: "auto" }}>{prev.task_md}</pre> : !err && <div className="muted small">{prev ? "not rendered: fix the errors above" : "rendering…"}</div>}
    </Panel>
  );
}

function SkillEditor({ w, name, changes, setChanges }: { w: WorkflowView; name: string; changes: WfChanges; setChanges: (c: WfChanges) => void }) {
  const s = w.skills[name];
  if (!s) return <Panel title={name}><div className="empty">Not in this profile.</div></Panel>;
  const text = changes.skills[name] ?? s.text;
  const set = (v: string) => {
    const next = { ...changes.skills };
    if (v === s.text) delete next[name]; else next[name] = v;
    setChanges({ ...changes, skills: next });
  };
  return (
    <Panel title={<>Skill <span className="mono">{name}</span></>} sub={<>SKILL.md of <span className="mono">{s.ref.slice(0, 30)}…</span> · used by {s.used_by.join(", ")}; other files of the bundle are kept</>}>
      <textarea className="mono" style={{ width: "100%", minHeight: 360, fontSize: 12 }} value={text} onChange={(e) => set(e.target.value)} />
    </Panel>
  );
}

function FieldsEditor({ w, changes, setChanges }: { w: WorkflowView; changes: WfChanges; setChanges: (c: WfChanges) => void }) {
  const set = (f: string, v: string) => {
    const cur = asText((w.profile as unknown as Record<string, unknown>)[f] ?? "");
    const next = { ...changes.fields };
    if (v === cur) delete next[f]; else next[f] = v || null;
    setChanges({ ...changes, fields: next });
  };
  return (
    <Panel title="Agent CLI" sub="profile-wide: one model and effort for every task type of this profile (AgentProfile has no per-type model)">
      <table className="t compact"><tbody>
        {FIELDS.map((f) => {
          const cur = asText((w.profile as unknown as Record<string, unknown>)[f] ?? "");
          const v = changes.fields[f] ?? cur;
          return <tr key={f} className={changes.fields[f] !== undefined ? "hl" : ""}><td>{f}</td><td>
            {f === "provider" ? <select value={v ?? ""} onChange={(e) => set(f, e.target.value)}>{["codex", "claude_code", "pi", "deepseek_harness"].map((p) => <option key={p}>{p}</option>)}</select>
              : <input className="mono" value={v ?? ""} onChange={(e) => set(f, e.target.value)} />}
            {f === "provider" && v !== "codex" && <span className="small error"> only the codex adapter is implemented</span>}
          </td></tr>;
        })}
      </tbody></table>
    </Panel>
  );
}

function ProxyEditor({ w }: { w: WorkflowView }) {
  const [policy, setPolicy] = useState(w.proxy.proxy);
  const [reason, setReason] = useState("");
  const [msg, setMsg] = useState<string | null>(null);
  const save = async () => {
    try {
      const sys = await api.system();
      const out = await api.setSystem(sys.config.revision, { providers: { [w.proxy.provider]: { proxy: policy } } }, reason.trim());
      setMsg(`system configuration revision ${out.revision}`);
    } catch (e) { setMsg(e instanceof Error ? e.message : String(e)); }
  };
  return (
    <Panel title={`Proxy for ${w.proxy.provider}`} sub="system configuration (Settings → System), not part of the profile; takes effect at the next step">
      <div className="row wrap" style={{ gap: 8 }}>
        <select value={policy} onChange={(e) => setPolicy(e.target.value as "required" | "off")}><option>required</option><option>off</option></select>
        <span className="mono small muted">{w.proxy.check_hosts.join(", ")}</span>
        <input placeholder="reason" value={reason} onChange={(e) => setReason(e.target.value)} />
        <button className="op" disabled={policy === w.proxy.proxy || !reason.trim()} onClick={save}>Save policy</button>
        {msg && <span className="small">{msg}</span>}
      </div>
    </Panel>
  );
}

function KnobEditor({ w, knobs }: { w: WorkflowView; knobs: WfKnob[] }) {
  const [draft, setDraft] = useState<Record<string, string>>({});
  const [reason, setReason] = useState("");
  const [msg, setMsg] = useState<string | null>(null);
  const changed = knobs.filter((k) => draft[k.id] !== undefined && draft[k.id] !== asText(k.value));
  const save = async () => {
    try {
      const changes: Record<string, unknown> = {};
      for (const k of changed) changes[k.id] = k.type === "int" || k.type === "float" ? Number(draft[k.id]) : k.type === "str" || k.type === "enum" ? draft[k.id] : JSON.parse(draft[k.id]);
      const out = await api.setControls(w.knobs.version, changes, reason.trim());
      setMsg(`controls version ${out.version} written`);
    } catch (e) { setMsg(e instanceof Error ? e.message : String(e)); }
  };
  return (
    <Panel title="Controls at this step" sub="research controls registry: a change is a new controls version (Settings → Research controls), recorded with every later run">
      <table className="t compact"><tbody>
        {knobs.map((k) => <tr key={k.id} title={k.prevents}><td><b>{k.row}</b><div className="mono small muted">{k.id}</div></td>
          <td>{k.editable ? <input className="mono" style={{ width: 200 }} value={draft[k.id] ?? asText(k.value)} onChange={(e) => setDraft({ ...draft, [k.id]: e.target.value })} /> : <span className="mono small">{asText(k.value)} <Chip tone="idle">invariant</Chip></span>} <span className="muted small">{k.unit}</span></td></tr>)}
      </tbody></table>
      <div className="row wrap" style={{ gap: 8, marginTop: 6 }}>
        <input placeholder="reason (required)" value={reason} onChange={(e) => setReason(e.target.value)} style={{ flex: 1 }} />
        <button className="op" disabled={!changed.length || !reason.trim()} onClick={save}>Write controls version</button>
        {msg && <span className="small">{msg}</span>}
      </div>
    </Panel>
  );
}

function DiffList({ title, rows }: { title: string; rows: WfDiff[] }) {
  if (!rows.length) return null;
  return (
    <details open style={{ marginTop: 8 }}>
      <summary className="small"><b>{title}</b> ({rows.length})</summary>
      {rows.map((d) => <div key={d.what} className="small" style={{ marginTop: 4 }}><b className="mono">{d.what}</b>
        {d.diff !== undefined ? <pre className="mono" style={{ whiteSpace: "pre-wrap", maxHeight: 220, overflow: "auto", margin: "2px 0" }}>{d.diff}</pre> : <> {asText(d.old)} → {asText(d.new)}</>}</div>)}
    </details>
  );
}

function DraftBar({ w, changes, reset, pick }: { w: WorkflowView; changes: WfChanges; reset: () => void; pick: (id: string) => void }) {
  const [reason, setReason] = useState("");
  const [saved, setSaved] = useState<{ id: string; diff: WfDiff[]; warnings: string[] } | null>(null);
  const [msg, setMsg] = useState<{ ok: boolean; text: string } | null>(null);
  const n = count(changes);
  const summary = useMemo(() => [...Object.keys(changes.templates), ...Object.keys(changes.skills).map((s) => `skill:${s}`), ...Object.keys(changes.fields)], [changes]);
  const save = async () => {
    try {
      const out = await api.saveDraft(w.profile.id, changes, reason.trim());
      setSaved(out);
      setMsg({ ok: true, text: `draft ${out.id.slice(0, 12)} written (not in force)` });
      reset();
    } catch (e) { setMsg({ ok: false, text: e instanceof Error ? e.message : String(e) }); }
  };
  const approve = async () => {
    if (!saved) return;
    try {
      const out = await api.approveDraft(saved.id, reason.trim() || "approved from the workflow page");
      setMsg({ ok: true, text: `in force; batch ${out.batch} built for its new tasks` });
      pick(saved.id);
    } catch (e) { setMsg({ ok: false, text: e instanceof Error ? e.message : String(e) }); }
  };
  return (
    <Panel title="Draft" sub={`based on ${w.profile.name} v${w.profile.version}; saving writes a new profile version (never changes this one); approving puts it in force for new tasks`} right={<TokenField />}>
      <div className="row wrap" style={{ gap: 8 }}>
        <span className="small">{n ? <>{n} change(s): <span className="mono">{summary.join(", ")}</span></> : "no change"}</span>
        <input value={reason} onChange={(e) => setReason(e.target.value)} placeholder="reason (required)" style={{ flex: 1, minWidth: 240 }} />
        <button className="op" disabled={!n} onClick={reset}>Discard</button>
        <button className="op primary" disabled={!n || !reason.trim()} onClick={save}>Save draft version</button>
        {saved && <button className="op primary" onClick={approve}>Approve draft</button>}
        {msg && <span className={msg.ok ? "small" : "error small"}>{msg.text}</span>}
      </div>
      {saved && <>
        {saved.warnings.map((x) => <div key={x} className="small muted">⚠ {x}</div>)}
        <DiffList title={`draft ${saved.id.slice(0, 12)} against v${w.profile.version}`} rows={saved.diff} />
      </>}
    </Panel>
  );
}
