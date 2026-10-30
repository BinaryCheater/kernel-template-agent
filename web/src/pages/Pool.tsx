import { useState } from "react";
import { Link } from "react-router";
import { api } from "../data/source";
import type { PoolView } from "../data/types";
import { useApi } from "../data/useApi";
import { useSnapshot } from "../components/Shell";
import { RecordActions } from "../components/actions";
import { AssuranceChip, Chip, Loading, Panel, RecordLink } from "../components/ui";
import { LEVEL_NAMES, wordTone } from "../components/format";

type Tab = "templates" | "patterns" | "structures" | "gaps" | "references";
const TABS: { id: Tab; label: string; hint: string }[] = [
  { id: "templates", label: "Templates", hint: "refinement steps: contract, level, assurance" },
  { id: "patterns", label: "Patterns", hint: "transaction patterns; new ones need your confirmation" },
  { id: "structures", label: "Structures", hint: "structural facts the solver may use; need your confirmation" },
  { id: "gaps", label: "Gaps", hint: "what the pool could not describe, with attribution" },
  { id: "references", label: "References", hint: "reference implementations and how much of each is reconstructed" },
];
const LV_COLORS: Record<number, string> = Object.fromEntries([1, 2, 3, 4, 5, 6, 7].map((l) => [l, `var(--lv${l})`]));

export function Pool() {
  const { snapshot, snapshots } = useSnapshot();
  const [all, setAll] = useState(false);
  const snap = all ? undefined : snapshot || undefined;
  const p = useApi(() => api.pool(snap), [snap]);
  const label = snapshots.find((s) => s.id === snapshot)?.label;
  return (
    <div className="stack">
      <div className="row small ink2">
        <span>Showing {all || !snapshot ? <b>every record in the store</b> : <>members of snapshot <b>{label}</b></>}.</span>
        <label className="row" style={{ gap: 4 }}><input type="checkbox" checked={all} onChange={(e) => setAll(e.target.checked)} />include records outside the snapshot</label>
        <Link to="/pool/diff" style={{ marginLeft: "auto" }}>Compare two snapshots →</Link>
        <Link to="/families">Families and their chains →</Link>
      </div>
      {!p.data ? <Loading error={p.error} /> : <PoolBody p={p.data} />}
    </div>
  );
}

function PoolBody({ p }: { p: PoolView }) {
  const [tab, setTab] = useState<Tab>("templates");
  const hint = TABS.find((t) => t.id === tab)?.hint;
  return (
    <>
      <div className="stats" style={{ gridTemplateColumns: "repeat(5, 1fr)" }}>
        {TABS.map((t) => (
          <div key={t.id} className="stat" style={{ cursor: "pointer", borderColor: tab === t.id ? "var(--st-human)" : undefined }} onClick={() => setTab(t.id)}>
            <div className="label">{t.label}</div>
            <div className="value num">{p.counts[t.id] ?? 0}</div>
            <div className="foot">{t.id === "patterns" ? `${p.other_patterns} in category other` : t.id === "gaps" ? "always all gaps" : " "}</div>
          </div>
        ))}
      </div>
      <Panel primary title={TABS.find((t) => t.id === tab)?.label} sub={hint}>
        <div className="tabs">{TABS.map((t) => <button key={t.id} className={tab === t.id ? "on" : ""} onClick={() => setTab(t.id)}>{t.label}<span className="muted">{p.counts[t.id] ?? 0}</span></button>)}</div>
        {tab === "templates" && <Templates p={p} />}
        {tab === "patterns" && <Patterns p={p} />}
        {tab === "structures" && <Structures p={p} />}
        {tab === "gaps" && <Gaps p={p} />}
        {tab === "references" && <References p={p} />}
      </Panel>
    </>
  );
}

const Status = ({ s, a }: { s: string; a?: string | null }) => <span className="row" style={{ gap: 4 }}><Chip tone={wordTone(s)}>{s}</Chip><AssuranceChip a={a} /></span>;
const Lv = ({ n }: { n: number }) => <span className="lvl" style={{ background: LV_COLORS[n] ?? "var(--k-other)" }} title={LEVEL_NAMES[n]}>{n}</span>;

function Templates({ p }: { p: PoolView }) {
  if (!p.templates.length) return <div className="empty">No Template in this view.</div>;
  return (
    <table className="t">
      <thead><tr><th>template</th><th>levels</th><th>refines → produces</th><th>contract</th><th>hardware</th><th>status</th><th /></tr></thead>
      <tbody>{p.templates.map((t) => (
        <tr key={t.id}>
          <td><b className="mono">{t.name}@{t.version}</b><div className="small ink2 ellipsis2" style={{ maxWidth: 360 }}>{t.summary}</div><RecordLink id={t.id} /></td>
          <td style={{ whiteSpace: "nowrap" }}><Lv n={t.level_from} /> → <Lv n={t.level_to} /><div className="small muted">{t.change}</div></td>
          <td className="small mono">{t.refines} → {t.produces.join(", ") || "—"}</td>
          <td className="small num">{t.variables} vars · {t.constraints} constraints · {t.obligations} obligations</td>
          <td className="small">{t.requires.join(", ") || <span className="muted">—</span>}{t.requires.length > 0 && <div className="muted">probed on {t.probed_on.join(", ") || "nothing yet"}</div>}</td>
          <td><Status s={t.status} a={t.assurance} /></td>
          <td><RecordActions id={t.id} kind="Template" name={`${t.name}@${t.version}`} small /></td>
        </tr>
      ))}</tbody>
    </table>
  );
}

function Patterns({ p }: { p: PoolView }) {
  if (!p.patterns.length) return <div className="empty">No Pattern record in this view. Seed patterns are imported by <span className="mono">kta init</span>; new ones come from T2/T3 tasks.</div>;
  return (
    <table className="t">
      <thead><tr><th>pattern</th><th>level</th><th>category</th><th>description</th><th>status</th><th>confirmed</th><th /></tr></thead>
      <tbody>{p.patterns.map((x) => (
        <tr key={x.id} className={x.category === "other" ? "hl" : ""}>
          <td><b className="mono">{x.name}</b> <RecordLink id={x.id} /></td><td><Lv n={x.level} /></td><td>{x.category}</td>
          <td className="small">{x.description}<div className="muted">{x.examples} examples</div></td>
          <td><Status s={x.status} a={x.assurance} /></td>
          <td>{x.confirmed_by ? <RecordLink id={x.confirmed_by} label="yes" /> : <span className="muted small">no</span>}</td>
          <td><RecordActions id={x.id} kind="Pattern" name={x.name} small /></td>
        </tr>
      ))}</tbody>
    </table>
  );
}

function Structures({ p }: { p: PoolView }) {
  if (!p.structures.length) return <div className="empty">No Structure record in this view.</div>;
  return (
    <table className="t">
      <thead><tr><th>structure</th><th>effect</th><th>description</th><th>evidence needed</th><th>status</th><th /></tr></thead>
      <tbody>{p.structures.map((x) => (
        <tr key={x.id}><td><b className="mono">{x.name}</b> <RecordLink id={x.id} /></td><td>{x.effect}</td><td className="small">{x.description}</td>
          <td className="small">{x.evidence_needed}</td><td><Status s={x.status} a={x.assurance} />{x.confirmed_by && <div className="small">confirmed <RecordLink id={x.confirmed_by} /></div>}</td>
          <td><RecordActions id={x.id} kind="Structure" name={x.name} small /></td></tr>
      ))}</tbody>
    </table>
  );
}

function Gaps({ p }: { p: PoolView }) {
  if (!p.gaps.length) return <div className="empty">No Gap recorded.</div>;
  return (
    <table className="t">
      <thead><tr><th>missing</th><th>description</th><th>level</th><th>pattern guess</th><th>attribution</th><th>status</th><th /></tr></thead>
      <tbody>{p.gaps.map((g) => (
        <tr key={g.id}><td><span className="tag">{g.missing}</span></td><td className="small">{g.description} <RecordLink id={g.id} /></td>
          <td>{g.level != null ? <Lv n={g.level} /> : "—"}</td><td className="mono small">{g.pattern_guess ?? "—"}</td><td>{g.attribution}</td>
          <td><Status s={g.status} a={g.assurance} /></td><td><RecordActions id={g.id} kind="Gap" small /></td></tr>
      ))}</tbody>
    </table>
  );
}

function References({ p }: { p: PoolView }) {
  if (!p.references.length) return <div className="empty">No Reference in this view.</div>;
  return (
    <table className="t">
      <thead><tr><th>reference</th><th>runnable on</th><th>reconstructed</th><th>unmatched</th><th>status</th></tr></thead>
      <tbody>{p.references.map((r) => (
        <tr key={r.id}><td><b>{r.name}</b> <RecordLink id={r.id} /></td><td className="small">{r.runnable_on.join(", ") || <span className="muted">analysis only</span>}</td>
          <td style={{ width: 220 }}>{r.total ? <><div className="bar"><span className="used" style={{ width: `${((r.matched ?? 0) / r.total) * 100}%`, background: "var(--st-ok)" }} /></div>
            <div className="small num">{r.matched}/{r.total} steps <RecordLink id={r.reconstruction} /></div></> : <span className="muted small">not decomposed yet</span>}</td>
          <td className="num">{r.unmatched ?? "—"}</td><td><Status s={r.status} a={r.assurance} /></td></tr>
      ))}</tbody>
    </table>
  );
}
