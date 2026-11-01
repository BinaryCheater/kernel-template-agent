// Review (SYSDEBUG-2, D690, D691): what a run assumes and what the user said about it.
// Tab "settings": every setting the run reads, grouped by the step that reads it, with who set
// it (you / assistant default / design documents / calibration); assistant defaults that nobody
// calibrated or confirmed are marked, since those are the decisions not yet discussed. With
// ?loop=<id>, the controls versions that loop ran under. Tab "notes": every annotation, open
// first, with its subject and a way to close it.
import { useMemo, useState } from "react";
import { Link, useSearchParams } from "react-router";
import { ApiError, act, api } from "../data/source";
import type { Annotation, InForce, InForceKnob } from "../data/types";
import { useApi } from "../data/useApi";
import { useLive } from "../data/live";
import { Loading, Panel, useOpenRecord } from "../components/ui";
import { Annotate } from "../components/Annotate";
import { stateWord } from "../components/words";
import { dur, since } from "../components/format";

const ago = (iso: string) => `${dur(since(iso, Date.now()))} ago`;

const STAGE_WORDS: Record<string, string> = { agent: "Agent tasks", learning: "Learning", system: "System", other: "Not mapped yet" };
const stageWord = (s: string) => STAGE_WORDS[s] ?? stateWord(s);

export function Review() {
  const [params, setParams] = useSearchParams();
  const tab = params.get("tab") === "notes" ? "notes" : "settings";
  const loop = params.get("loop") ?? undefined;
  const set = (k: string, v: string | null) => {
    const next = new URLSearchParams(params);
    if (v) next.set(k, v);
    else next.delete(k);
    setParams(next, { replace: true });
  };
  const notes = useApi(() => api.annotations(), []);
  const open = (notes.data ?? []).filter((a) => !a.resolution).length;
  return (
    <div className="stack">
      <div className="tabs">
        <button className={tab === "settings" ? "on" : ""} onClick={() => set("tab", null)}>Settings in force</button>
        <button className={tab === "notes" ? "on" : ""} onClick={() => set("tab", "notes")}>Your notes{open ? ` (${open} open)` : ""}</button>
      </div>
      {tab === "settings" ? <SettingsInForce loop={loop} /> : <Notes items={notes.data} error={notes.error} />}
    </div>
  );
}

function fmt(v: unknown, unit?: string | null): string {
  if (v === null || v === undefined) return "—";
  if (typeof v === "object") {
    const entries = Object.entries(v as Record<string, unknown>);
    return entries.map(([k, x]) => `${k} ${typeof x === "object" ? JSON.stringify(x) : String(x)}`).join(" · ");
  }
  return `${String(v)}${unit ? ` ${unit}` : ""}`;
}

function whoTone(k: InForceKnob): string {
  if (k.set_by_you) return "you";
  if (k.calibrated) return "calibrated";
  return k.status; // default | spec
}

function SettingsInForce({ loop }: { loop?: string }) {
  const r = useApi(() => api.inForce(loop), [loop]);
  const [stage, setStage] = useState<string>("all");
  const [onlyOpen, setOnlyOpen] = useState(false);
  if (!r.data) return <Loading error={r.error} />;
  const d: InForce = r.data;
  const rank = (k: InForceKnob) => Math.min(...k.stages.map((s) => d.stages.indexOf(s)).map((i) => (i < 0 ? 99 : i)));
  const shown = [...d.knobs].sort((a, b) => rank(a) - rank(b)).filter((k) => (stage === "all" || k.stages.includes(stage)) && (!onlyOpen || (!k.set_by_you && !k.calibrated && k.status === "default")));
  const undecided = d.knobs.filter((k) => !k.set_by_you && !k.calibrated && k.status === "default").length;
  return (
    <>
      <Panel primary title="What this run assumes"
        sub={loop ? `the settings loop ${loop.slice(0, 8)} reads, and the versions it ran under` : "every setting a run reads, in the version now in force"}>
        <div className="inforce-summary">
          <div className="inforce-big"><b>{undecided}</b><span>assistant defaults nobody has calibrated or confirmed</span></div>
          <div className="inforce-big spec"><b>{d.counts.spec ?? 0}</b><span>from the design documents, not confirmed by you</span></div>
          <div className="inforce-big you"><b>{d.counts.you ?? 0}</b><span>set by you</span></div>
          <div className="inforce-big cal"><b>{d.counts.calibrated ?? 0}</b><span>set by a calibration run</span></div>
        </div>
        <div className="small ink2" style={{ marginTop: 8 }}>
          Settings version {d.version.version}
          {d.versions_seen.length > 0 && <> · this loop ran under {d.versions_seen.map((v) => `v${v.version} (${v.by}, ${ago(v.at)})`).join(", ")}</>}
        </div>
      </Panel>
      <Panel title="Settings by the step that reads them"
        right={<label className="small row" style={{ gap: 6 }}><input type="checkbox" checked={onlyOpen} onChange={(e) => setOnlyOpen(e.target.checked)} />only undecided defaults</label>}>
        <div className="facet-c" style={{ marginBottom: 12 }}>
          {["all", ...d.stages].map((s) => {
            const n = s === "all" ? d.knobs.length : d.knobs.filter((k) => k.stages.includes(s)).length;
            return <button key={s} className={`fchip ${stage === s ? "on" : ""}`} disabled={!n} onClick={() => setStage(s)}>{s === "all" ? "All steps" : stageWord(s)}<i>{n}</i></button>;
          })}
        </div>
        <table className="t inforce">
          <thead><tr><th>setting</th><th>value</th><th>who decided</th><th>read at</th><th>your verdict</th></tr></thead>
          <tbody>{shown.map((k) => (
            <tr key={k.id} className={`who-${whoTone(k)}`}>
              <td><b>{k.row}</b><div className="small ink2">{k.controls}</div><div className="small mono ink3">{k.id}</div></td>
              <td className="inforce-val">{fmt(k.value, k.unit)}{JSON.stringify(k.value) !== JSON.stringify(k.initial) && <div className="small ink3">initially {fmt(k.initial, k.unit)}</div>}</td>
              <td><span className={`who ${whoTone(k)}`}>{k.who}</span>
                {k.last_change && <div className="small ink2" title={k.last_change.reason}>v{k.last_change.version} by {k.last_change.by}: {k.last_change.reason.slice(0, 90)}{k.last_change.reason.length > 90 ? "…" : ""}</div>}
                {k.prevents && <div className="small ink3">guards against: {k.prevents}</div>}</td>
              <td className="small">{k.stages.map(stageWord).join(", ")}</td>
              <td><Annotate subject={`knob:${k.id}`} label={`setting: ${k.row}`} loop={loop} existing={k.annotations} compact /></td>
            </tr>
          ))}</tbody>
        </table>
      </Panel>
      <Hardware d={d} />
    </>
  );
}

function Hardware({ d }: { d: InForce }) {
  const open = useOpenRecord();
  const h = d.hardware;
  if (!h.model) return null;
  return (
    <Panel title="Hardware numbers the cost model (L1) uses" sub={`${h.model.target} v${h.model.version} on ${h.model.arch}`}
      right={<button className="linkbtn" onClick={() => h.model && open(h.model.id)}>record</button>}>
      <div className="small ink2" style={{ marginBottom: 8 }}>{Object.entries(h.counts).map(([s, n]) => `${n} ${s}`).join(" · ")}</div>
      <table className="t inforce">
        <thead><tr><th>name</th><th>value</th><th>status</th><th>source</th></tr></thead>
        <tbody>{h.values.map((v) => (
          <tr key={v.name} className={`hw-${v.status}`}><td className="mono small">{v.name}</td><td>{fmt(v.value)}</td><td><span className={`who ${v.status === "measured" ? "calibrated" : v.status === "default" ? "default" : "spec"}`}>{v.status}</span></td><td className="small ink2">{v.source}</td></tr>
        ))}</tbody>
      </table>
    </Panel>
  );
}

const VERDICT_WORDS: Record<string, string> = { agree: "Agree", question: "Question", change: "Change" };

function subjectLink(a: Annotation) {
  if (a.subject.startsWith("knob:")) return <Link to={`/review${a.loop ? `?loop=${a.loop}` : ""}`}>{a.label || a.subject}</Link>;
  if (a.subject.startsWith("loop:")) return <span>{a.label || a.subject}</span>;
  if (a.subject.startsWith("page:")) return <Link to={a.subject.slice(5)}>{a.label || a.subject.slice(5)}</Link>;
  return <Link to={`/records/${a.subject}`}>{a.label || `record ${a.subject.slice(0, 10)}`}</Link>;
}

function Notes({ items, error }: { items?: Annotation[]; error?: string }) {
  const sorted = useMemo(() => [...(items ?? [])].sort((a, b) => Number(!!a.resolution) - Number(!!b.resolution)), [items]);
  if (!items) return <Loading error={error} />;
  if (!items.length) return <Panel title="Your notes"><div className="empty">No notes yet. Use the ✓ ? ✎ buttons on a setting, a step or a record.</div></Panel>;
  return (
    <Panel primary title="Your notes" sub="open first; each links back to what it is about">
      <table className="t notes">
        <thead><tr><th>when</th><th>verdict</th><th>about</th><th>note</th><th>state</th></tr></thead>
        <tbody>{sorted.map((a) => <NoteRow key={a.id} a={a} />)}</tbody>
      </table>
    </Panel>
  );
}

function NoteRow({ a }: { a: Annotation }) {
  const { refresh } = useLive();
  const [err, setErr] = useState<string | null>(null);
  const close = async (outcome: "done" | "kept" | "later") => {
    try {
      await act.resolve(a.id, outcome);
      refresh();
    } catch (e) {
      setErr(e instanceof ApiError ? e.detail : String(e));
    }
  };
  return (
    <tr className={a.resolution ? "resolved" : ""}>
      <td className="small num">{ago(a.at)}</td>
      <td><span className={`verdict ${a.verdict}`}>{VERDICT_WORDS[a.verdict] ?? a.verdict}</span></td>
      <td>{subjectLink(a)}</td>
      <td>{a.note || <span className="ink3">—</span>}</td>
      <td>{a.resolution ? <span className="small ink2">{a.resolution.outcome}</span> : (
        <span className="row" style={{ gap: 4 }}>
          <button className="op plain small" onClick={() => close("done")} title="the change was made">Done</button>
          <button className="op plain small" onClick={() => close("kept")} title="discussed; keep as it is">Keep</button>
          <button className="op plain small" onClick={() => close("later")} title="not now">Later</button>
          {err && <span className="small error">{err}</span>}
        </span>
      )}</td>
    </tr>
  );
}
