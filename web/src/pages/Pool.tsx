import { useState } from "react";
import { Link } from "react-router";
import { api } from "../data/source";
import type { PoolTemplate, PoolView } from "../data/types";
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
      <AssuranceStrip p={p} />
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
      <thead><tr><th>template</th><th>levels</th><th>refines → produces</th><th>contract</th><th>hardware</th><th>status</th><th>obligations (checker)</th><th /></tr></thead>
      <tbody>{p.templates.map((t) => (
        <tr key={t.id}>
          <td><b className="mono">{t.name}@{t.version}</b><div className="small ink2 ellipsis2" style={{ maxWidth: 360 }}>{t.summary}</div><RecordLink id={t.id} /></td>
          <td style={{ whiteSpace: "nowrap" }}><Lv n={t.level_from} /> → <Lv n={t.level_to} /><div className="small muted">{t.change}</div></td>
          <td className="small mono">{t.refines} → {t.produces.join(", ") || "—"}</td>
          <td className="small num">{t.variables} vars · {t.constraints} constraints · {t.obligations} obligations</td>
          <td className="small">{t.requires.join(", ") || <span className="muted">—</span>}{t.requires.length > 0 && <div className="muted">probed on {t.probed_on.join(", ") || "nothing yet"}</div>}</td>
          <td><Status s={t.status} a={t.assurance} /></td>
          <td><ProofCell t={t} /></td>
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

const ASSURANCE_HINT: Record<string, string> = {
  declared: "fields complete, obligations listed, nothing checked",
  tested: "a family using it passed space verification (code level)",
  bounded: "obligations checked on bounded models or bounded enumeration",
  proved: "K1–K3 proved over the full stated domain",
};

/** WP24: the assurance distribution. bounded / proved come only from live proof records. */
function AssuranceStrip({ p }: { p: PoolView }) {
  const a = p.assurance ?? { declared: 0, tested: 0, bounded: 0, proved: 0 };
  const total = Object.values(a).reduce((x, y) => x + y, 0) || 1;
  const unchecked = p.templates.filter((t) => t.proof.obligations.some((o) => o.status === "unchecked" || o.status === "never checked")).length;
  return (
    <Panel title="Assurance of the Templates in view" sub="declared → tested → bounded → proved; bounded and proved are set only by checker proof records (kta prove)">
      <div className="bar" style={{ height: 14, display: "flex" }}>
        {(["proved", "bounded", "tested", "declared"] as const).map((k) => (
          <span key={k} title={`${k}: ${a[k]}`} style={{ width: `${(a[k] / total) * 100}%`, background: k === "proved" ? "var(--st-ok)" : k === "bounded" ? "#7cc49a" : k === "tested" ? "var(--st-running)" : "var(--k-other)" }} />
        ))}
      </div>
      <div className="row small" style={{ gap: 18, marginTop: 6 }}>
        {(["declared", "tested", "bounded", "proved"] as const).map((k) => (
          <span key={k} title={ASSURANCE_HINT[k]}><b className="num">{a[k]}</b> {k}</span>
        ))}
        <span className="muted" style={{ marginLeft: "auto" }}>{unchecked} Template(s) with obligations the checker did not check (listed per row)</span>
      </div>
    </Panel>
  );
}

/** Every obligation of Σ with its checker outcome, stated domain and last run; unchecked ones stay listed. */
function ProofCell({ t }: { t: PoolTemplate }) {
  const pr = t.proof;
  const n = (s: string) => pr.obligations.filter((o) => o.status === s).length;
  const head = pr.summary
    ? `${pr.level ?? "no level"} · ${n("proved") + n("bounded")}/${pr.obligations.length} discharged` + (pr.live === false ? " · stale" : "")
    : "never checked";
  return (
    <details>
      <summary className="small"><Chip tone={wordTone(pr.level ?? (pr.summary ? "unresolved" : "never checked"))}>{head}</Chip></summary>
      <div className="small" style={{ maxWidth: 520 }}>
        {pr.last_run && <div className="muted">last run {pr.last_run.slice(0, 19)}Z · {pr.checker} · {pr.solver} · <RecordLink id={pr.summary} label="summary" /></div>}
        <table className="t">
          <tbody>
            {pr.contract.map((c) => (
              <tr key={`${c.condition}-${c.name}`}><td className="mono">{c.condition} {c.name === "*" ? "" : c.name}</td><td><Chip tone={wordTone(c.status)}>{c.status}</Chip></td><td className="ink2">{c.reason || c.domain}</td><td><RecordLink id={c.result} /></td></tr>
            ))}
            {pr.obligations.map((o) => (
              <tr key={o.name}>
                <td className="mono">K2 {o.name}<div className="muted">method {o.method}</div></td>
                <td><Chip tone={wordTone(o.status)}>{o.status}</Chip>{o.reused && <div className="muted">reused</div>}</td>
                <td className="ink2">{o.domain}{o.reason && <div className="muted">{o.reason}</div>}</td>
                <td><RecordLink id={o.result} /></td>
              </tr>
            ))}
          </tbody>
        </table>
        {pr.gaps.length > 0 && <div className="muted">why not higher: {pr.gaps.length} gap(s), e.g. {pr.gaps[0]}</div>}
      </div>
    </details>
  );
}
