// Learning a reference (LEARNDEMO-1, D722): how the expand loop turned one repository into
// pool knowledge. Read top to bottom in the loop's order: where the loop is, what each agent
// task did, then every chain step and every unexplained transaction beside the frozen source
// lines it cites, the chain coloured by where each step's Template came from, and what the
// reference added to the pool. Data: GET /references/{id}/learning and /source.
import { useEffect, useMemo, useRef, useState } from "react";
import { Link, useParams } from "react-router";
import { api } from "../data/source";
import { useApi } from "../data/useApi";
import { ChainDag } from "../components/ChainDag";
import type { ChainNode } from "../data/types";
import { Chip, Empty, Loading, Panel, RecordLink } from "../components/ui";
import type { Tone } from "../components/format";
import "../styles/learning.css";

type StepStatus = "in_pool" | "extracted" | "added_later" | "missing";
interface Stage { id: string; kind: string; work: string; does: string; status: "done" | "current" | "skipped" | "pending"; entered_at: string | null; visits: number }
interface TaskRow {
  id: string; type: "T2" | "T3"; goal: string; state: string; attempts: number; created_at: string; updated_at: string;
  submissions: string[]; outputs: Record<string, unknown>;
  checks: { id: string; passed: boolean; summary: string; reason: string; created_at: string }[];
}
interface Item {
  key: string; template: string; applies_to: string; bindings: Record<string, unknown> | null; after: string[] | null;
  status: StepStatus; template_id: string | null; summary?: string; level?: [number, number];
}
interface Unmatched { level: number; pattern_guess: string; description: string; where: string }
interface Gap { id: string; missing: string; description: string; level: number; pattern_guess: string; attribution: string }
interface Round { id: string; created_at: string; chain: string | null; workload: string | null; matched: number; total: number; notes: string | null; items: Item[]; unmatched: Unmatched[]; gaps: Gap[] }
interface Learning {
  reference: { id: string; name: string; build: string; runnable_on: string[] };
  source: { id: string; locator: string; revision: string; license: string | null; note: string | null } | null;
  problem: { id: string; name: string; reference_impl: string } | null;
  source_tree: { id: string; summary: string; files_total: number; files_included: number; tree: string; worker: string } | null;
  loop: { id: string; state: string; updated_at: string; analysis_only: string | null; round: number; note: string | null } | null;
  stages: Stage[];
  tasks: TaskRow[];
  rounds: Round[];
  extracted_templates: { id: string; name: string; summary: string | null }[];
  admission: { id: string; summary: string }[];
  coverage: { id: string; summary: string; created_at: string }[];
  notes: Record<string, string>;
}
interface RefRow { id: string; name: string; loop_state: string | null; matched: number | null; total: number | null; origin: string }
interface SourceFile { path: string; lines: string[]; truncated: boolean }

const STATUS_WORDS: Record<StepStatus, string> = {
  in_pool: "already in the pool",
  extracted: "extracted from this reference",
  added_later: "added to the pool later",
  missing: "missing from the pool",
};
const STATUS_TONE: Record<StepStatus, Tone> = { in_pool: "ok", extracted: "human", added_later: "running", missing: "fail" };
const STAGE_TONE: Record<Stage["status"], Tone> = { done: "ok", current: "running", skipped: "idle", pending: "idle" };
const TASK_WORDS: Record<string, string> = {
  T2: "decompose: read the source, list what the kernel does on the hardware, write the chain of refinement steps and match each to a Template",
  T3: "extract: write one missing Template from the source lines that implement it",
};
const TASK_TONE: Record<string, Tone> = { accepted: "ok", rejected: "fail", escalated: "fail", cancelled: "idle", suspended: "waiting" };

/** `path:12-30` citations in a free-text location (chain `applies_to`, transaction `where`). */
interface Cite { path: string; from: number; to: number }
const CITE = /([\w./-]+\.(?:cu|cuh|h|hpp|inl|cpp|cc|c|py)):(\d+)(?:-(\d+))?((?:,\s*\d+(?:-\d+)?)*)/g;
function cites(text: string): Cite[] {
  const out: Cite[] = [];
  for (const m of text.matchAll(CITE)) {
    out.push({ path: m[1], from: +m[2], to: +(m[3] ?? m[2]) });
    for (const extra of (m[4] ?? "").split(",").map((s) => s.trim()).filter(Boolean)) {
      const [a, b] = extra.split("-").map(Number);
      out.push({ path: m[1], from: a, to: b ?? a });
    }
  }
  return out;
}

export function ReferenceList() {
  const r = useApi(() => api.references<RefRow[]>(), []);
  if (!r.data) return <Loading error={r.error} />;
  if (!r.data.length) return <Empty what="every Reference (a repository at a commit) and how far the learn loop got with it" why="no Reference has been added (kta reference add)" />;
  return (
    <Panel title="References" sub="each row opens how the system learned from that repository">
      <table className="t">
        <thead><tr><th>reference</th><th>learn loop</th><th>steps matched</th><th>added by</th></tr></thead>
        <tbody>{r.data.map((x) => (
          <tr key={x.id}>
            <td><Link to={`/references/${x.id}`}>{x.name}</Link></td>
            <td>{x.loop_state ?? <span className="muted">not started</span>}</td>
            <td className="num">{x.total != null ? `${x.matched} / ${x.total}` : "—"}</td>
            <td className="ink2">{x.origin}</td>
          </tr>
        ))}</tbody>
      </table>
    </Panel>
  );
}

function now(d: Learning): string {
  const lp = d.loop;
  if (!lp) return "No learn loop has started for this reference.";
  const live = d.tasks.filter((t) => !["accepted", "cancelled", "rejected"].includes(t.state));
  const stage = d.stages.find((s) => s.id === lp.state);
  const words = stage ? stage.does : lp.state;
  const tasks = live.length ? ` Agent tasks open: ${live.map((t) => `${t.type} (${t.state})`).join(", ")}.` : "";
  return `Now: ${lp.state}${lp.round > 1 ? `, round ${lp.round}` : ""} — ${words}.${tasks}`;
}

function Flow({ stages }: { stages: Stage[] }) {
  return (
    <div className="ln-flow">
      {stages.map((s, i) => (
        <div key={s.id} className={`ln-stage st-${s.status}`} title={s.does}>
          <div className="ln-stage-head"><span className="ln-n">{i + 1}</span><b>{s.id}</b>{s.visits > 1 && <span className="ink2"> ×{s.visits}</span>}</div>
          <div className="ln-stage-who">{s.work}{s.status === "skipped" ? " · skipped (analysis only)" : ""}</div>
          <div className="ln-stage-does">{s.does}</div>
          <Chip tone={STAGE_TONE[s.status]}>{s.status}{s.entered_at ? ` ${s.entered_at.slice(11, 16)}` : ""}</Chip>
        </div>
      ))}
    </div>
  );
}

function Tasks({ tasks }: { tasks: TaskRow[] }) {
  if (!tasks.length) return <Empty what="the agent tasks of this loop: T2 decomposes the reference, T3 extracts each Template the chain needs and the pool lacks" why="the loop has not reached analyzing yet (the source is frozen first)" />;
  return (
    <table className="t ln-tasks">
      <thead><tr><th>task</th><th>what it does</th><th>state</th><th>tries</th><th>checks after submit</th><th>wrote</th></tr></thead>
      <tbody>{tasks.map((t) => (
        <tr key={t.id}>
          <td><Link to={`/tasks/${t.id}`} className="mono">{t.type}</Link><div className="ink2 small">{t.created_at.slice(11, 16)}</div></td>
          <td>{TASK_WORDS[t.type]}<div className="ink2 small">{t.goal}</div></td>
          <td><Chip tone={TASK_TONE[t.state] ?? "running"}>{t.state}</Chip></td>
          <td className="num">{t.attempts}</td>
          <td>{t.checks.length ? t.checks.map((c) => (
            <div key={c.id} className={c.passed ? "ok-t" : "fail-t"}>{c.passed ? "passed" : "refused"}: <span className="ink2">{c.passed ? c.summary : c.reason}</span></div>
          )) : <span className="muted">not checked yet</span>}</td>
          <td>{Object.entries(t.outputs).filter(([, v]) => typeof v === "string").map(([k, v]) => (
            <div key={k}><span className="ink2">{k} </span><RecordLink id={v as string} /></div>
          ))}</td>
        </tr>
      ))}</tbody>
    </table>
  );
}

type Row = { kind: "step"; item: Item } | { kind: "tx"; tx: Unmatched; n: number };

function SourcePane({ refId, cite }: { refId: string; cite: Cite | null }) {
  const f = useApi(() => (cite ? api.sourceFile<SourceFile>(refId, cite.path) : Promise.resolve(null)), [refId, cite?.path]);
  const box = useRef<HTMLDivElement>(null);
  useEffect(() => {
    const el = box.current?.querySelector(".hl");
    if (el) el.scrollIntoView({ block: "center" });
  }, [f.data, cite?.from]);
  if (!cite) return <div className="ln-src-empty">Pick a step or a transaction on the left: the source lines it cites appear here.</div>;
  if (f.error) return <div className="error">{f.error}</div>;
  if (!f.data) return <Loading />;
  const lo = Math.max(1, cite.from - 12);
  const hi = Math.min(f.data.lines.length, cite.to + 12);
  return (
    <div className="ln-src" ref={box}>
      <div className="ln-src-head mono">{cite.path}:{cite.from}{cite.to !== cite.from ? `-${cite.to}` : ""}</div>
      <pre>{f.data.lines.slice(lo - 1, hi).map((l, i) => {
        const n = lo + i;
        return <div key={n} className={n >= cite.from && n <= cite.to ? "hl" : ""}><span className="ln-no">{n}</span>{l || " "}</div>;
      })}</pre>
    </div>
  );
}

function Explain({ refId, round }: { refId: string; round: Round }) {
  const rows: Row[] = useMemo(() => [
    ...round.items.map((item) => ({ kind: "step" as const, item })),
    ...round.unmatched.map((tx, n) => ({ kind: "tx" as const, tx, n })),
  ], [round]);
  const [sel, setSel] = useState(0);
  const [citeIx, setCiteIx] = useState(0);
  const row = rows[sel];
  const rowCites = row ? cites(row.kind === "step" ? row.item.applies_to : row.tx.where) : [];
  const cite = rowCites[Math.min(citeIx, rowCites.length - 1)] ?? null;
  const pick = (i: number) => { setSel(i); setCiteIx(0); };
  return (
    <div className="ln-explain">
      <div className="ln-rows">
        <div className="ln-rows-h">Chain steps, in order <span className="ink2">— each refines one transaction of the kernel with a Template</span></div>
        {rows.map((r, i) => r.kind === "step" ? (
          <div key={`s${r.item.key}`} className={`ln-row ${i === sel ? "sel" : ""} s-${r.item.status}`} onClick={() => pick(i)}>
            <div className="ln-row-top"><span className="mono">{r.item.key}</span><b>{r.item.template}</b><Chip tone={STATUS_TONE[r.item.status]}>{STATUS_WORDS[r.item.status]}</Chip></div>
            <div className="ln-row-txt">{r.item.applies_to}</div>
          </div>
        ) : (
          <div key={`t${r.n}`} className={`ln-row tx ${i === sel ? "sel" : ""}`} onClick={() => pick(i)}>
            {r.n === 0 && <div className="ln-rows-h tx-h">Transactions no step explains <span className="ink2">— each is a Gap</span></div>}
            <div className="ln-row-top"><span className="mono">ux-{String(r.n + 1).padStart(2, "0")}</span><b>level {r.tx.level} · {r.tx.pattern_guess}</b><Chip tone="fail">unexplained</Chip></div>
            <div className="ln-row-txt">{r.tx.description}</div>
          </div>
        ))}
      </div>
      <div className="ln-side">
        {row?.kind === "step" && (
          <div className="ln-tpl">
            <div className="ln-tpl-h">
              {row.item.template_id ? <Link to={`/knowledge/templates/${row.item.template_id}`}><b>{row.item.template}</b></Link> : <b>{row.item.template}</b>}
              <Chip tone={STATUS_TONE[row.item.status]}>{STATUS_WORDS[row.item.status]}</Chip>
            </div>
            <div>{row.item.summary ?? (row.item.status === "missing" ? "The pool has no such Template yet: the loop creates a T3 to extract it from these lines." : "")}</div>
            {row.item.bindings && Object.keys(row.item.bindings).length > 0 && (
              <div className="ln-bind"><span className="k">values chosen in this kernel</span>
                {Object.entries(row.item.bindings).map(([k, v]) => <span key={k} className="pill mono">{k} = {String(v)}</span>)}</div>
            )}
          </div>
        )}
        {rowCites.length > 1 && (
          <div className="ln-cites">{rowCites.map((c, i) => (
            <button key={i} className={i === citeIx ? "on" : ""} onClick={() => setCiteIx(i)}>{c.path.split("/").pop()}:{c.from}</button>
          ))}</div>
        )}
        <SourcePane refId={refId} cite={cite} />
      </div>
    </div>
  );
}

function chainNodes(round: Round): ChainNode[] {
  return round.items.map((i) => ({
    key: i.key, template: i.template, template_record: i.template_id, applies_to: i.applies_to, after: i.after ?? [],
    level_from: i.level?.[0] ?? null, level_to: i.level?.[1] ?? null, change: null,
    assurance: i.status, summary: i.summary ?? "",
    requires: [], probed_on: [], variables: [], bindings: (i.bindings ?? {}) as ChainNode["bindings"],
  }));
}

export function LearningPage() {
  const { id = "" } = useParams();
  const r = useApi(() => api.learning<Learning>(id), [id]);
  const [roundIx, setRoundIx] = useState<number | null>(null);
  const [dagSel, setDagSel] = useState<string>();
  if (!r.data) return <Loading error={r.error} />;
  const d = r.data;
  const ri = roundIx ?? d.rounds.length - 1;
  const round = d.rounds[ri];
  const src = d.source;
  return (
    <div className="stack ln">
      <Panel primary title={<>Learning from <b>{d.reference.name}</b></>} sub={src ? <a href={`${src.locator}/tree/${src.revision}`} target="_blank" rel="noreferrer" className="mono">{src.locator.replace("https://github.com/", "")} @ {src.revision.slice(0, 10)}</a> : null}
        right={d.loop?.analysis_only ? <Chip tone="waiting" title={d.loop.analysis_only}>analysis only: no build, measurement or reproduction</Chip> : null}>
        <div className="ln-head">
          <div><span className="k">Computes</span>{d.problem?.reference_impl ?? "—"}</div>
          <div><span className="k">Frozen source</span>{d.source_tree ? <>{d.source_tree.summary} <span className="ink2">(fetched on {d.source_tree.worker})</span> <RecordLink id={d.source_tree.id} /></> : <span className="muted">not fetched yet</span>}</div>
          <div className="ln-now">{now(d)}</div>
        </div>
      </Panel>

      <Panel title="The learn loop" sub="one instance per reference; machine-made steps and agent steps alternate">
        <Flow stages={d.stages} />
      </Panel>

      <Panel title="Agent tasks" sub="what each agent was asked, and whether the deterministic checks admitted its delivery">
        <Tasks tasks={d.tasks} />
      </Panel>

      {round ? (
        <>
          <Panel title="What the code does, step by step" sub={`decomposition ${ri + 1} of ${d.rounds.length}: ${round.matched} of ${round.total} steps matched a Template that was already in the pool`}
            right={d.rounds.length > 1 ? <span className="ln-rounds">{d.rounds.map((x, i) => <button key={x.id} className={i === ri ? "on" : ""} onClick={() => setRoundIx(i)}>round {i + 1}</button>)}</span> : null}>
            {round.workload && <div className="ln-workload"><span className="k">Workload</span>{round.workload}</div>}
            <Explain key={round.id} refId={d.reference.id} round={round} />
          </Panel>

          <Panel title="The chain" sub="steps left to right in dependency order; colour = where the step's Template came from">
            <div className="ln-legend">{(Object.keys(STATUS_WORDS) as StepStatus[]).map((s) => <Chip key={s} tone={STATUS_TONE[s]}>{STATUS_WORDS[s]}</Chip>)}</div>
            <div className="ln-dag"><ChainDag nodes={chainNodes(round)} selected={dagSel} onSelect={setDagSel} /></div>
          </Panel>

          <div className="ln-two">
            <Panel title="Gaps" sub="what the pool could not express when this reference was decomposed">
              {round.gaps.length ? round.gaps.map((g) => (
                <div key={g.id} className="ln-gap">
                  <div><Chip tone="fail">missing {g.missing}</Chip> <span className="ink2">level {g.level} · {g.pattern_guess} · {g.attribution}</span> <RecordLink id={g.id} /></div>
                  <div>{g.description}</div>
                </div>
              )) : <span className="muted">none</span>}
            </Panel>
            <Panel title="What this reference added to the pool">
              {d.extracted_templates.length ? d.extracted_templates.map((t) => (
                <div key={t.id} className="ln-gap"><Link to={`/knowledge/templates/${t.id}`}><b>{t.name}</b></Link><div className="ink2">{t.summary}</div></div>
              )) : <span className="muted">no Template extracted yet</span>}
              {d.admission.map((a) => <div key={a.id} className="ln-meta"><span className="k">admission</span>{a.summary} <RecordLink id={a.id} /></div>)}
              {d.coverage.map((c) => <div key={c.id} className="ln-meta"><span className="k">coverage</span>{c.summary} <RecordLink id={c.id} /></div>)}
            </Panel>
          </div>

          {Object.keys(d.notes).filter((k) => k !== "submission").map((k) => (
            <details key={k} className="raw"><summary>The agent's {k} <RecordLink id={d.notes.submission} /></summary><pre className="ln-notes">{d.notes[k]}</pre></details>
          ))}
        </>
      ) : (
        <Panel title="What the code does, step by step">
          <Empty what="every chain step beside the source lines it refines, and every transaction no step explains" why="no decomposition has been admitted yet: the T2 task is still working or its delivery was refused" />
        </Panel>
      )}
    </div>
  );
}
