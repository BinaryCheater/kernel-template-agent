import { useMemo } from "react";
import { Link, useNavigate, useParams } from "react-router";
import { api } from "../data/source";
import type { CaseDetail, CaseRoundProgress, Machines } from "../data/types";
import { useApi } from "../data/useApi";
import { useNow } from "../data/live";
import { AXIS, EChart, TOOLTIP, echarts, type EOption } from "../components/EChart";
import { Chip, Loading, Panel, RecordLink, Stat } from "../components/ui";
import { dur, fix1, hhmm, pct, since, stateTone, us } from "../components/format";

const TONE_COLOR: Record<string, string> = { running: "#2a78d6", waiting: "#fab219", ok: "#0ca30c", fail: "#d03b3b", human: "#6250d6", idle: "#898781" };

export function CaseList() {
  const cases = useApi(api.cases, []);
  const m = useApi(api.machines, []);
  const nav = useNavigate();
  const now = useNow();
  if (!cases.data || !m.data) return <Loading error={cases.error ?? m.error} />;
  const solve = m.data.machines.find((x) => x.id === "solve");
  const maxRounds = Number(solve?.limits?.max_rounds ?? 5);
  return (
    <Panel primary title="Cases" sub="one solve-loop instance per case · % of bound = lower bound ÷ best measured latency (100% = at the bound)">
      <table className="t index">
        <thead><tr><th>case</th><th>now</th><th>rounds (of {maxRounds})</th><th>% of bound by round</th><th className="r">bound</th><th className="r">accepted</th><th>active task</th><th className="r">last activity</th></tr></thead>
        <tbody>{cases.data.map((c) => (
          <tr key={c.id} className="click" onClick={() => nav(`/cases/${c.id}`)}>
            <td><Link to={`/cases/${c.id}`}><b>{c.name}</b></Link><div className="mono small muted">{c.problem}</div></td>
            <td><Chip tone={stateTone(solve, c.state)}>{c.state}</Chip></td>
            <td><RoundDots rounds={c.rounds ?? []} max={maxRounds} /></td>
            <td><Spark values={(c.rounds ?? []).map((r) => r.pct_of_bound)} max={maxRounds} /></td>
            <td className="r num">{us(c.bound_us)}</td>
            <td className="r num">{us(c.best_us)}</td>
            <td>{c.active_task ? <><span className="tag task">{c.active_task.type}</span> <span className="small ink2">{c.active_task.state}</span></> : <span className="muted small">—</span>}</td>
            <td className="r small ink2 num">{c.last_activity ? `${dur(since(c.last_activity, now))} ago` : "—"}</td>
          </tr>
        ))}</tbody>
      </table>
      <div className="legend" style={{ marginTop: 10 }}>
        <span><span className="rd done" /> round judged (R3 = judging rule 3)</span><span>sparkline: best measured candidate per round, dashed line = 100%</span><span><span className="rd task" /> round ended waiting on a task</span>
        <span><span className="rd open" /> current round</span><span><span className="rd todo" /> rounds left before escalation</span>
      </div>
    </Panel>
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

/** Tiny line of % of bound per round (0-100%); rounds without a measurement leave a gap. */
function Spark({ values, max }: { values: (number | null)[]; max: number }) {
  const W = 120;
  const H = 28;
  const pts = values.map((v, i) => (v == null ? null : [6 + (i * (W - 12)) / Math.max(1, max - 1), H - 4 - v * (H - 8)] as const));
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

export function CasePage() {
  const { id = "" } = useParams();
  const c = useApi(() => api.caseTimeline(id), [id]);
  const m = useApi(api.machines, []);
  if (!c.data || !m.data) return <Loading error={c.error ?? m.error} />;
  return <CaseBody c={c.data} m={m.data} />;
}

function CaseBody({ c, m }: { c: CaseDetail; m: Machines }) {
  const strongest = [...c.references].sort((a, b) => a.us - b.us)[0];
  const allTasks = c.rounds.flatMap((r) => r.tasks);
  return (
    <div className="stack">
      <div className="case-head">
        <div>
          <div className="caption">Case</div>
          <h2 className="mono" style={{ fontSize: 20 }}>{c.name}</h2>
          <div className="ink2">{c.shapes} · objective {c.objective.replace("_", " ")} · <RecordLink id={c.record} label="case record" /></div>
        </div>
        <div className="row" style={{ marginLeft: "auto" }}>
          <Chip tone={stateTone(m.machines.find((x) => x.id === "solve"), c.state)}>{c.state}</Chip>
          <span className="small ink2">round {c.round} of 5</span>
        </div>
      </div>
      <div className="stats">
        <Stat label="Lower bound" value={fix1(c.bound_us)} unit="µs" record={c.bound_record} foot="theoretical, from the hardware model" />
        <Stat label={strongest ? `Strongest reference · ${strongest.name}` : "Strongest reference"} value={fix1(strongest?.us)} unit={strongest ? "µs" : undefined} record={strongest?.record}
          foot={c.references.length > 1 ? `${c.references.length} references measured` : "measured under measure-v1"} />
        <Stat label="Best result" value={fix1(c.best?.us)} unit={c.best ? "µs" : undefined} record={c.best?.record}
          foot={c.best ? <Link to={`/results/${c.best.solution}`}>why this holds →</Link> : "no measured result yet"} />
        <Stat label="% of bound" value={pct(c.pct_of_bound)} foot={c.best?.us && strongest ? `${(strongest.us / c.best.us).toFixed(2)}× vs strongest reference` : "—"} />
      </div>

      <Panel primary title="Iteration timeline" sub="each row is one round of the solve loop; segments are states, ◆ tasks created, ✕ rejections; the judging outcome closes the round">
        <Timeline c={c} m={m} />
        <div className="rounds">
          {c.rounds.map((r) => (
            <div key={r.n} className="round-note">
              <b>Round {r.n}</b>
              {r.outcome ? <span> — {r.outcome.rule ? `rule ${r.outcome.rule}: ` : ""}<RecordLink id={r.outcome.record} label={r.outcome.label} /></span>
                : <span className="muted"> — {r.n < c.rounds.length ? "left the round in" : "in progress:"} {r.steps[r.steps.length - 1].state}{r.steps[r.steps.length - 1].note ? ` (${r.steps[r.steps.length - 1].note})` : ""}</span>}
            </div>
          ))}
        </div>
      </Panel>

      <div className="grid" style={{ gridTemplateColumns: "minmax(0, 1fr) minmax(0, 1fr)", alignItems: "start" }}>
        <Panel title="Per shape bucket" sub="latency, lower is better; the bound is the floor">
          <BucketChart c={c} />
        </Panel>
        <Panel title="Candidates of the last round" sub="pre-compile prediction vs measurement">
          <table className="t">
            <thead><tr><th>#</th><th>assignment</th><th className="r">predicted</th><th className="r">measured</th><th>check</th></tr></thead>
            <tbody>{c.candidates.map((x) => (
              <tr key={x.rank}>
                <td className="num">{x.rank}</td>
                <td className="mono small"><RecordLink id={x.record} label={Object.entries(x.assignment).map(([k, v]) => `${k}=${v}`).join(" ")} /></td>
                <td className="r num">{us(x.predicted_us)}</td>
                <td className="r num">{x.measured_us ? us(x.measured_us) : <span className="muted">pending</span>}</td>
                <td>{x.check_passed == null ? <span className="muted small">—</span> : <Chip tone={x.check_passed ? "ok" : "fail"}>{x.check_passed ? "passed" : "failed"}</Chip>}</td>
              </tr>
            ))}</tbody>
          </table>
        </Panel>
      </div>

      <div className="grid" style={{ gridTemplateColumns: "minmax(0, 1fr) minmax(0, 1fr)", alignItems: "start" }}>
        <Panel title="Tasks this case created" sub={`${allTasks.length} task(s)`}>
          {allTasks.length === 0 ? <div className="empty">None — every round found legal Templates and was judged without help.</div> : (
            <table className="t">
              <thead><tr><th>task</th><th>created</th><th>state</th><th>rejections</th><th>card</th></tr></thead>
              <tbody>{allTasks.map((t) => (
                <tr key={t.id}><td><span className="tag task">{t.type}</span></td><td className="num">{hhmm(t.created_at)}</td>
                  <td><Chip tone={stateTone(m.machines.find((x) => x.id === "task"), t.state)}>{t.state}</Chip></td>
                  <td className="small">{t.rejections.length ? t.rejections.map((r) => <div key={r.at}>{hhmm(r.at)} {r.reason}</div>) : "—"}</td>
                  <td><RecordLink id={t.card} /></td></tr>
              ))}</tbody>
            </table>
          )}
        </Panel>
        <Panel title="Gaps raised" sub="what the pool could not explain">
          {c.gaps.length === 0 ? <div className="empty">No gap recorded for this case.</div> : c.gaps.map((g) => <div key={g.id}><RecordLink id={g.id} label={`Gap: ${g.description}`} /></div>)}
          <details><summary>All references</summary>
            <table className="t"><tbody>{c.references.map((r) => <tr key={r.name}><td>{r.name}</td><td className="r num"><RecordLink id={r.record} label={us(r.us)} /></td></tr>)}</tbody></table>
          </details>
        </Panel>
      </div>
    </div>
  );
}

function Timeline({ c, m }: { c: CaseDetail; m: Machines }) {
  const solve = m.machines.find((x) => x.id === "solve");
  const now = useNow();
  const option = useMemo<EOption>(() => {
    const segs = c.rounds.flatMap((r, ri) => r.steps.filter((s) => s.state !== "accepted").map((s) => ({
      value: [ri, Date.parse(s.start), s.end ? Date.parse(s.end) : now], s, tone: stateTone(solve, s.state),
    })));
    const marks = c.rounds.flatMap((r, ri) => [
      ...r.tasks.map((t) => ({ value: [Date.parse(t.created_at), ri], name: `${t.type} created`, symbol: "diamond", color: "#6250d6", t })),
      ...r.tasks.flatMap((t) => t.rejections.map((x) => ({ value: [Date.parse(x.at), ri], name: `rejected: ${x.reason}`, symbol: "path://M2,2 L10,10 M10,2 L2,10", color: "#d03b3b" }))),
    ]);
    const t0 = Math.min(...segs.map((s) => s.value[1]));
    const t1 = Math.max(...segs.map((s) => s.value[2]));
    return {
      animation: false,
      grid: { left: 70, right: 20, top: 10, bottom: 30 },
      tooltip: { ...TOOLTIP },
      xAxis: { type: "time", min: t0 - 30_000, max: t1 + 30_000, ...AXIS, axisLabel: { ...AXIS.axisLabel, formatter: (v: number) => hhmm(new Date(v).toISOString()) } },
      yAxis: { type: "category", inverse: true, data: c.rounds.map((r) => `round ${r.n}`), ...AXIS, axisTick: { show: false }, splitLine: { show: false } },
      series: [
        {
          type: "custom", data: segs, encode: { x: [1, 2], y: 0 },
          tooltip: { formatter: (p: { data: (typeof segs)[number] }) => `<b>${p.data.s.state}</b><br/>${hhmm(p.data.s.start, true)} → ${p.data.s.end ? hhmm(p.data.s.end, true) : "now"} · ${dur((p.data.value[2] - p.data.value[1]) / 1000)}${p.data.s.note ? `<br/><span style="color:#52514e">${p.data.s.note}</span>` : ""}` },
          renderItem: (params: { dataIndex: number; coordSys: { x: number; y: number; width: number; height: number } }, api: { value: (i: number) => number; coord: (v: number[]) => number[]; size: (v: number[]) => number[] }) => {
            const d = segs[params.dataIndex];
            const [x0, y] = api.coord([api.value(1), api.value(0)]);
            const [x1] = api.coord([api.value(2), api.value(0)]);
            const h = Math.min(30, api.size([0, 1])[1] * 0.5);
            const rect = echarts.graphic.clipRectByRect({ x: x0, y: y - h / 2, width: Math.max(2, x1 - x0 - 1), height: h }, params.coordSys);
            if (!rect) return null;
            const wait = d.tone === "waiting";
            return { type: "group", children: [
              { type: "rect", shape: { ...rect, r: 3 }, style: { fill: TONE_COLOR[d.tone], opacity: wait ? 0.35 : d.s.end ? 0.85 : 1, stroke: d.s.end ? undefined : "#0b0b0b", lineDash: [3, 2], lineWidth: d.s.end ? 0 : 1 } },
              rect.width > 46 ? { type: "text", silent: true, style: { text: d.s.state, x: rect.x + 4, y: rect.y + rect.height / 2, verticalAlign: "middle", fontSize: 11, fill: wait ? "#0b0b0b" : "#fff", width: rect.width - 8, overflow: "truncate" } } : null,
            ].filter(Boolean) };
          },
        },
        {
          type: "scatter", data: marks.map((x) => ({ value: x.value, name: x.name, symbol: x.symbol, itemStyle: { color: x.color, borderColor: "#fff", borderWidth: 1 } })),
          symbolSize: 13, symbolOffset: [0, -22], z: 5,
          label: { show: true, position: "top", fontSize: 10.5, color: "#3d2fa0", formatter: (p: { name: string }) => (p.name.endsWith("created") ? p.name.split(" ")[0] : "") },
          tooltip: { formatter: (p: { name: string; value: number[] }) => `${hhmm(new Date(p.value[0]).toISOString(), true)} · ${p.name}` },
        },
      ],
    } as EOption;
  }, [c, solve, now]);
  return (
    <>
      <EChart option={option} height={Math.max(150, c.rounds.length * 78 + 50)} />
      <div className="legend">
        <span><span className="sw" style={{ background: TONE_COLOR.running }} />working state</span>
        <span><span className="sw" style={{ background: TONE_COLOR.waiting, opacity: 0.4 }} />waiting for a task</span>
        <span><span className="sw" style={{ background: "#fff", border: "1px dashed #0b0b0b" }} />current state</span>
        <span style={{ color: "#6250d6" }}>◆ task created</span><span style={{ color: "#d03b3b" }}>✕ rejection</span>
      </div>
    </>
  );
}

function BucketChart({ c }: { c: CaseDetail }) {
  const option = useMemo<EOption>(() => {
    const refNames = [...new Set(c.buckets.flatMap((b) => Object.keys(b.references)))];
    const series = [
      { name: "lower bound", color: "#c3c2b7", get: (b: CaseDetail["buckets"][number]) => b.bound_us },
      { name: "best (ours)", color: "#2a78d6", get: (b: CaseDetail["buckets"][number]) => b.best_us ?? null },
      ...refNames.map((n, i) => ({ name: n, color: ["#eb6834", "#1baf7a", "#e87ba4"][i % 3], get: (b: CaseDetail["buckets"][number]) => b.references[n] ?? null })),
    ];
    return {
      animation: false,
      grid: { left: 80, right: 50, top: 30, bottom: 24 },
      legend: { top: 0, textStyle: { color: "#52514e", fontSize: 11 }, itemWidth: 12, itemHeight: 10 },
      tooltip: { ...TOOLTIP, trigger: "axis", axisPointer: { type: "shadow" }, valueFormatter: (v: number) => (v == null ? "—" : `${v} µs`) },
      xAxis: { type: "value", name: "µs", ...AXIS },
      yAxis: { type: "category", data: c.buckets.map((b) => b.bucket), ...AXIS, axisTick: { show: false } },
      series: series.map((s) => ({ type: "bar", name: s.name, data: c.buckets.map(s.get), itemStyle: { color: s.color, borderRadius: [0, 4, 4, 0] }, barGap: "20%", barMaxWidth: 14,
        label: { show: true, position: "right", fontSize: 10, color: "#52514e", formatter: (p: { value: number }) => (p.value == null ? "" : p.value.toFixed(1)) } })),
    } as EOption;
  }, [c]);
  return <EChart option={option} height={Math.max(160, c.buckets.length * 110 + 60)} />;
}
