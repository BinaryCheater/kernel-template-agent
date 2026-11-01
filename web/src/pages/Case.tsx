// Cases (WP25, D666). The list separates research Cases from measurement probes (calibration
// peaks and self-tests, written by `kta measure`). A Case page reads top to bottom as the
// story of the problem: what is asked (shapes, bounds), what the current round runs with,
// then one card per round — the path it took, what it used, the candidate funnel, measured
// against predicted against the bound, and how it was judged and what that changed. Raw
// candidates, steps and records sit inside each card, collapsed.
import { useMemo, useState } from "react";
import { Link, useNavigate, useParams } from "react-router";
import { api } from "../data/source";
import type { ActiveRuns, CaseDetail, CaseIndexRow, CaseRoundProgress, Machines, Round, RoundSummary } from "../data/types";
import { useApi } from "../data/useApi";
import { useNow } from "../data/live";
import { Chip, Loading, Panel, RecordLink } from "../components/ui";
import { dur, hhmm, pct, since, stateTone, us } from "../components/format";
import { LoopControls, loopCondition } from "../components/LoopControls";
import { JournalList } from "./Activity";
import { stateWord, taskWord } from "../components/words";

export function CaseList() {
  const cases = useApi(api.cases, []);
  const m = useApi(api.machines, []);
  const [probes, setProbes] = useState(false);
  if (!cases.data || !m.data) return <Loading error={cases.error ?? m.error} />;
  const research = cases.data.filter((c) => c.purpose !== "probe");
  const probeRows = cases.data.filter((c) => c.purpose === "probe");
  return (
    <div className="stack">
      <Panel primary title="Cases">
        <CaseTable rows={research} m={m.data} />
        {research.length === 0 && <div className="empty">No case yet. Add one with <span className="mono">kta case add</span>, then start it with <span className="mono">kta solve</span>.</div>}
      </Panel>
      <Panel title={`Measurement probes (${probeRows.length})`} sub="calibration and self-tests, not problems to solve"
        right={<button className="linkbtn" onClick={() => setProbes(!probes)}>{probes ? "hide" : "show"}</button>}>
        {probes ? <CaseTable rows={probeRows} m={m.data} /> : null}
      </Panel>
    </div>
  );
}

function CaseTable({ rows, m }: { rows: CaseIndexRow[]; m: Machines }) {
  const nav = useNavigate();
  const now = useNow();
  const solve = m.machines.find((x) => x.id === "solve");
  const maxRounds = Number(solve?.limits?.max_rounds ?? 5);
  if (!rows.length) return null;
  return (
    <>
      <table className="t index">
        <thead><tr><th>case</th><th>now</th><th>rounds (of {maxRounds})</th><th>% of bound by round</th><th className="r">bound</th><th className="r">best</th><th>active task</th><th className="r">last activity</th></tr></thead>
        <tbody>{rows.map((c) => (
          <tr key={c.id} className="click" onClick={() => nav(`/cases/${c.id}`)}>
            <td><Link to={`/cases/${c.id}`}><b>{c.name}</b></Link></td>
            <td><Chip tone={c.blocked ? "fail" : c.paused ? "idle" : stateTone(solve, c.state)}>{c.blocked ? `stopped in ${c.state}` : c.paused ? `paused · ${c.state}` : c.state}</Chip></td>
            <td><RoundDots rounds={c.rounds ?? []} max={maxRounds} /></td>
            <td><Spark values={(c.rounds ?? []).map((r) => r.pct_of_bound)} /></td>
            <td className="r num">{us(c.bound_us)}</td>
            <td className="r num">{us(c.best_us)}</td>
            <td>{c.active_task ? <><span className="tag task">{c.active_task.type}</span> <span className="small ink2">{c.active_task.state}</span></> : <span className="muted small">—</span>}</td>
            <td className="r small ink2 num">{c.last_activity ? `${dur(since(c.last_activity, now))} ago` : "—"}</td>
          </tr>
        ))}</tbody>
      </table>
    </>
  );
}

function RoundDots({ rounds, max }: { rounds: CaseRoundProgress[]; max: number }) {
  return (
    <div className="round-dots">
      {Array.from({ length: max }, (_, i) => {
        const r = rounds[i];
        if (!r) return <span key={i} className="rd todo" title={`round ${i + 1}: not started`} />;
        const cls = r.open ? "open" : r.rule ? "done" : "task";
        const title = `round ${r.n}: ${r.open ? "in progress, " : "ended in "}${r.last_state}${r.rule ? ` · rule ${r.rule}` : ""}${r.tasks.length ? ` · created ${r.tasks.join(", ")}` : ""}`;
        return <span key={i} className={`rd ${cls}`} title={title}>{r.rule ? `R${r.rule}` : r.tasks[0] ?? ""}</span>;
      })}
    </div>
  );
}

/** Tiny line of % of bound per round; rounds without a measurement leave a gap. */
function Spark({ values }: { values: (number | null)[] }) {
  const W = 120;
  const H = 28;
  const n = Math.max(values.length, 2);
  const pts = values.map((v, i) => (v == null ? null : [6 + (i * (W - 12)) / (n - 1), H - 4 - Math.min(1, v) * (H - 8)] as const));
  const last = [...values].reverse().find((v) => v != null);
  if (!pts.some(Boolean)) return <span className="muted small">no measurement yet</span>;
  return (
    <span className="spark">
      <svg width={W} height={H} aria-label="% of bound by round">
        <line x1="0" x2={W} y1={4} y2={4} className="spark-ref" />
        <polyline points={pts.filter(Boolean).map((p) => p!.join(",")).join(" ")} className="spark-line" />
        {pts.map((p, i) => p && <circle key={i} cx={p[0]} cy={p[1]} r={3} className="spark-dot" />)}
      </svg>
      <span className="num small">{pct(last)}</span>
    </span>
  );
}

// ---------------------------------------------------------------- one case

export function CasePage() {
  const { id = "" } = useParams();
  const c = useApi(() => api.caseTimeline(id), [id]);
  const m = useApi(api.machines, []);
  const a = useApi(api.active, []);
  if (!c.data || !m.data) return <Loading error={c.error ?? m.error} />;
  return <CaseBody c={c.data} m={m.data} a={a.data} />;
}

function CaseBody({ c, m, a }: { c: CaseDetail; m: Machines; a?: ActiveRuns }) {
  const loop = a?.loops.find((l) => l.id === c.loop_id);
  const solve = m.machines.find((x) => x.id === "solve");
  const maxRounds = loop?.max_rounds ?? Number(solve?.limits?.max_rounds ?? 5);
  const cond = loop ? loopCondition(loop) : null;
  const rounds = [...c.rounds].reverse();
  return (
    <div className="stack">
      <div className="case-head">
        <div>
          <div className="caption">Case</div>
          <h2 className="mono" style={{ fontSize: 20 }}>{c.name}</h2>
          <div className="ink2 small">{c.shapes} · objective {c.objective.replace("_", " ")} · <RecordLink id={c.record} label="case record" /></div>
        </div>
        <div className="case-head-right">
          <div className="row">
            <Chip tone={c.blocked ? "fail" : stateTone(solve, c.state)}>{c.state}</Chip>
            {cond && <Chip tone={cond.tone}>{cond.text}</Chip>}
            <span className="small ink2">round {c.round || "—"} of {maxRounds}</span>
          </div>
          {loop ? <LoopControls loop={loop} /> : c.loop_id ? null : <span className="small muted">not started: <span className="mono">kta solve {c.id.slice(0, 12)}</span></span>}
          {c.best && <span className="small">best result {us(c.best.us)} · <Link to={`/results/${c.best.solution}`}>why it holds →</Link></span>}
        </div>
      </div>

      {c.blocked && (
        <div className="notice fail">
          <b>Stopped at {stateWord(c.blocked.state)} since {hhmm(c.blocked.at)} UTC: {c.blocked.error}</b>
          <div className="small">{c.blocked.message}</div>
        </div>
      )}

      <Question c={c} />
      <Inputs c={c} />

      <section className="stack" style={{ gap: 10 }}>
        <h2 className="section-title">Rounds</h2>
        {rounds.length === 0 && <div className="empty">No round has started.</div>}
        {rounds.map((r, i) => <RoundCard key={r.n} r={r} c={c} m={m} open={i === 0} current={i === 0 && !!c.loop_id} />)}
      </section>

      <div className="grid" style={{ gridTemplateColumns: "minmax(0, 1fr) minmax(0, 1fr)", alignItems: "start" }}>
        <Panel title="Activity of this case" right={<Link to={`/activity?case=${c.id}`}>all →</Link>}>
          <JournalList level="milestone" limit={10} case={c.id} compact />
        </Panel>
        <Panel title="Gaps and references">
          {c.gaps.length === 0 ? <div className="small muted">No gap recorded for this case.</div> : c.gaps.map((g) => <div key={g.id}><RecordLink id={g.id} label={`Gap: ${g.description}`} /></div>)}
          <table className="t compact" style={{ marginTop: 8 }}><tbody>
            {c.references.map((r) => <tr key={r.record}><td>{r.name}</td><td className="r num"><RecordLink id={r.record} label={us(r.us)} /></td></tr>)}
            {c.references.length === 0 && <tr><td className="small muted">no reference measured</td></tr>}
          </tbody></table>
        </Panel>
      </div>
    </div>
  );
}

/** What is asked: one card per shape bucket with its floor, our best and the references. */
function Question({ c }: { c: CaseDetail }) {
  const shapes = c.shapes.split("; ");
  return (
    <div className="bucket-cards">
      {c.buckets.map((b, i) => {
        const strongest = Object.entries(b.references).sort((x, y) => x[1] - y[1])[0];
        const share = b.best_us ? b.bound_us / b.best_us : null;
        return (
          <div key={b.bucket} className="bucket-card">
            <div className="caption">{b.bucket} · {shapes[i] ?? ""}</div>
            <div className="bc-row"><span className="muted">lower bound</span><b className="num"><RecordLink id={b.bound_record} label={us(b.bound_us)} /></b>{b.bottleneck && <span className="small muted">{b.bottleneck}-bound</span>}</div>
            <div className="bc-row"><span className="muted">best ours</span><b className="num">{us(b.best_us)}</b>{share != null && <span className="small">{pct(share)} of bound</span>}</div>
            <div className="bc-row"><span className="muted">strongest ref</span><span className="num">{strongest ? `${strongest[0]} ${us(strongest[1])}` : "—"}</span></div>
            <BoundBar bound={b.bound_us} best={b.best_us ?? null} reference={strongest?.[1] ?? null} />
          </div>
        );
      })}
    </div>
  );
}

/** Latency on one axis: the bound is the floor; ours and the reference sit to its right. */
function BoundBar({ bound, best, reference, predicted }: { bound: number; best: number | null; reference: number | null; predicted?: number | null }) {
  const max = Math.max(bound, best ?? 0, reference ?? 0, predicted ?? 0) * 1.08;
  const x = (v: number) => `${(v / max) * 100}%`;
  return (
    <div className="boundbar" title={`bound ${us(bound)}${predicted ? ` · predicted ${us(predicted)}` : ""}${best ? ` · measured ${us(best)}` : ""}${reference ? ` · reference ${us(reference)}` : ""}`}>
      <span className="bb-floor" style={{ width: x(bound) }} />
      {predicted != null && <span className="bb-mark pred" style={{ left: x(predicted) }} />}
      {reference != null && <span className="bb-mark ref" style={{ left: x(reference) }} />}
      {best != null && <span className="bb-mark best" style={{ left: x(best) }} />}
    </div>
  );
}

/** What the current round runs with: the family and its chain of Templates, model, controls. */
function Inputs({ c }: { c: CaseDetail }) {
  const inp = c.inputs;
  if (!inp || (!inp.families.length && !inp.model)) return null;
  return (
    <Panel title="Built from">
      {inp.families.map((f) => (
        <div key={f.id} className="inputs-family">
          <Link to={`/families/${f.id}`}><b className="mono">{f.name}@{f.version}</b></Link> <span className="small muted">{f.backend} · {f.steps} steps</span>
          <div className="chain-pills">
            {(f.step_detail ?? []).map((s, i) => (
              <span key={s.key} className="chain-pill" title={`${s.key}: level ${s.level_from}→${s.level_to}, ${s.assurance}`}>
                {i > 0 && <span className="arrow">→</span>}<span className="mono">{s.template}</span><span className={`assur-dot ${s.assurance}`} />
              </span>
            ))}
          </div>
        </div>
      ))}
      <div className="small ink2" style={{ marginTop: 6 }}>
        cost model <RecordLink id={inp.model} /> · hardware model <RecordLink id={inp.hw_model} /> · controls <RecordLink id={inp.controls} />
      </div>
    </Panel>
  );
}

// ---------------------------------------------------------------- a round

const RULE = ["", "①", "②", "③", "④", "⑤", "⑥"];

function why(r: Round, c: CaseDetail, current: boolean): string {
  const last = r.steps[r.steps.length - 1];
  if (r.outcome) {
    const t = r.tasks[r.tasks.length - 1];
    return `Judged: ${r.outcome.label}.` + (t ? ` Next: agent will ${taskWord(t.type)} (${t.state}), then round ${r.n + 1}.` : "");
  }
  if (current && c.blocked) return `Stopped at ${stateWord(c.blocked.state)}: ${c.blocked.error}.`;
  if (r.tasks.length) {
    const t = r.tasks[r.tasks.length - 1];
    return `At ${stateWord(last.state)}, handed to an agent to ${taskWord(t.type)} (${t.state}).`;
  }
  return current ? `Now at ${stateWord(last.state)}.` : `Ended at ${stateWord(last.state)}.`;
}

function RoundCard({ r, c, m, open, current }: { r: Round; c: CaseDetail; m: Machines; open: boolean; current: boolean }) {
  const [expanded, setExpanded] = useState(open);
  const now = useNow();
  const s = r.summary;
  const last = r.steps[r.steps.length - 1];
  const t0 = r.steps[0]?.start;
  const t1 = last?.end;
  const stopped = current && !!c.blocked;
  return (
    <div className={`round-card ${stopped ? "stopped" : ""} ${r.outcome ? "judged" : ""}`}>
      <header onClick={() => setExpanded(!expanded)}>
        <b>Round {r.n}</b>
        <span className="small ink2">{t0 ? hhmm(t0) : ""} → {t1 ? hhmm(t1) : "now"} · {t0 ? dur((Date.parse(t1 ?? new Date(now).toISOString()) - Date.parse(t0)) / 1000) : ""}</span>
        <span className="round-why">{why(r, c, current)}</span>
        <span className="linkbtn">{expanded ? "less" : "more"}</span>
      </header>
      <StepPath r={r} m={m} stopped={stopped} />
      {expanded && (
        <>
          <div className="round-cols">
            <div><div className="caption">used</div>
              {s?.families.length ? s.families.map((f) => <div key={f.id}><Link to={`/families/${f.id}`} className="mono">{f.name}@{f.version}</Link></div>) : <span className="muted small">no family yet</span>}
            </div>
            <div><div className="caption">searched</div>
              {s ? <><b className="num">{s.candidates}</b> candidates<div className="small muted">{Object.entries(s.buckets).map(([b, x]) => `${b}: ${x.candidates}`).join(" · ")}</div></> : "—"}
            </div>
            <div><div className="caption">compiled and checked</div>
              {s ? <Funnel s={s} /> : "—"}
            </div>
            <div><div className="caption">vs lower bound</div>
              {s && c.buckets.map((b) => {
                const x = s.buckets[b.bucket];
                return (
                  <div key={b.bucket} className="mvp">
                    <span className="small mono">{b.bucket}</span>
                    <BoundBar bound={b.bound_us} best={x?.measured_best_us ?? null} reference={null} predicted={x?.predicted_best_us ?? null} />
                    <span className="small num">{x?.measured_best_us ? us(x.measured_best_us) : <span className="muted">pred {us(x?.predicted_best_us)}</span>}</span>
                  </div>
                );
              })}
              <div className="legend tiny"><span><span className="sw floor" />bound</span><span><span className="sw pred" />predicted best</span><span><span className="sw best" />measured best</span></div>
            </div>
            <div><div className="caption">result</div>
              {r.outcome ? <div><b>{RULE[r.outcome.rule ?? 0]}</b> <RecordLink id={r.outcome.record} label={r.outcome.label} />{r.outcome.gap_class && <div className="small muted">gap class {r.outcome.gap_class} · <RecordLink id={r.outcome.decomposition} label="decomposition" /></div>}</div>
                : <span className="small muted">not judged yet</span>}
              {r.tasks.map((t) => (
                <div key={t.id} className="round-task">
                  <Link to={`/tasks/${t.id}`}><span className="tag task">agent: {taskWord(t.type)}</span></Link> <Chip tone={stateTone(m.machines.find((x) => x.id === "task"), t.state)}>{t.state}</Chip>
                  {t.rejections.length > 0 && <div className="small error">{t.rejections.length} rejection(s): {t.rejections[t.rejections.length - 1].reason}</div>}
                  {t.produced?.map((p) => <div key={p.id} className="small">produced <RecordLink id={p.id} label={`${p.kind} ${p.label}`} /></div>)}
                </div>
              ))}
            </div>
          </div>
          {current && c.candidates.length > 0 && <Candidates c={c} />}
          <details><summary>Steps of this round</summary>
            <table className="t compact"><tbody>{r.steps.map((st, i) => (
              <tr key={i}><td className="mono">{st.state}</td><td className="num small">{hhmm(st.start, true)}</td><td className="num small">{st.end ? dur((Date.parse(st.end) - Date.parse(st.start)) / 1000) : "running"}</td><td className="small muted">{st.note}</td></tr>
            ))}</tbody></table>
          </details>
        </>
      )}
    </div>
  );
}

/** The states this round went through, in order, each as wide as it took (min width). */
function StepPath({ r, m, stopped }: { r: Round; m: Machines; stopped: boolean }) {
  const now = useNow();
  const solve = m.machines.find((x) => x.id === "solve");
  const spans = r.steps.map((s) => ({ s, d: ((s.end ? Date.parse(s.end) : now) - Date.parse(s.start)) / 1000 }));
  const total = spans.reduce((p, x) => p + x.d, 0) || 1;
  return (
    <div className="step-path">
      {spans.map(({ s, d }, i) => {
        const open = !s.end;
        const tone = open && stopped ? "fail" : stateTone(solve, s.state);
        return (
          <div key={i} className={`sp ${tone} ${open ? "open" : ""}`} style={{ flexGrow: Math.max(0.6, (d / total) * 10) }} title={`${s.state}: ${dur(d)}${s.note ? ` · ${s.note}` : ""}`}>
            <span>{stateWord(s.state)}</span><span className="num">{dur(d)}</span>
          </div>
        );
      })}
    </div>
  );
}

function Funnel({ s }: { s: RoundSummary }) {
  const rows: [string, number, string][] = [
    ["proposed", s.candidates, "idle"], ["built", s.built, "running"], ["checks passed", s.check_passed, "ok"], ["measured", s.measured, "ok"],
  ];
  const max = Math.max(1, s.candidates);
  return (
    <div className="funnel">
      {rows.map(([k, v, tone]) => (
        <div key={k} className="fn-row"><span className="small">{k}</span><span className={`fn-bar ${tone}`} style={{ width: `${Math.max(2, (v / max) * 100)}%` }} /><span className="num small">{v}</span></div>
      ))}
      {(s.build_failed > 0 || s.check_failed > 0) && <div className="small error">{s.build_failed ? `${s.build_failed} builds failed ` : ""}{s.check_failed ? `${s.check_failed} checks failed` : ""}</div>}
    </div>
  );
}

function Candidates({ c }: { c: CaseDetail }) {
  const [showAll, setShowAll] = useState(false);
  const hasSelected = c.candidates.some((x) => x.selected);
  const rows = useMemo(() => (hasSelected && !showAll ? c.candidates.filter((x) => x.selected) : c.candidates), [c.candidates, hasSelected, showAll]);
  return (
    <details>
      <summary>Candidates ({hasSelected ? `${c.candidates.filter((x) => x.selected).length} of ${c.candidates.length}` : c.candidates.length})</summary>
      {hasSelected && <label className="small"><input type="checkbox" checked={showAll} onChange={(e) => setShowAll(e.target.checked)} /> show every shortlisted candidate</label>}
      <table className="t compact">
        <thead><tr><th>shape / arm</th><th>assignment / evidence</th><th className="r">predicted</th><th className="r">measured</th><th>check</th></tr></thead>
        <tbody>{rows.map((x) => (
          <tr key={x.rank}>
            <td className="small">{x.bucket ?? "—"}<div className="muted">{x.arm ?? "model"}</div></td>
            <td className="mono small"><RecordLink id={x.record} label={Object.entries(x.assignment).map(([k, v]) => `${k}=${v}`).join(" ")} /><div className="row small">
              {x.build_record && <RecordLink id={x.build_record} label={x.build_ok ? "built" : "build failed"} />}
              {x.proof_record && <RecordLink id={x.proof_record} label={x.claims ?? "candidate gate"} />}
            </div></td>
            <td className="r num">{us(x.predicted_us)}</td>
            <td className="r num">{x.measured_us != null ? <RecordLink id={x.measurement_record} label={us(x.measured_us)} /> : <span className="muted">{x.selected ? "pending" : "not selected"}</span>}</td>
            <td>{x.check_passed == null ? <span className="muted small">—</span> : <RecordLink id={x.check_record} label={x.check_passed ? "passed" : "failed"} />}</td>
          </tr>
        ))}</tbody>
      </table>
    </details>
  );
}
