import { useMemo } from "react";
import { Link, useNavigate, useParams } from "react-router";
import { api, nowMs } from "../data/source";
import type { CaseDetail, Machines } from "../data/types";
import { useApi } from "../data/useApi";
import { AXIS, EChart, TOOLTIP, echarts, type EOption } from "../components/EChart";
import { Chip, Loading, Panel, RecordLink, Stat } from "../components/ui";
import { dur, hhmm, pct, stateTone, us } from "../components/format";

const TONE_COLOR: Record<string, string> = { running: "#2a78d6", waiting: "#fab219", ok: "#0ca30c", fail: "#d03b3b", human: "#6250d6", idle: "#898781" };

export function CaseList() {
  const cases = useApi(api.cases, []);
  const m = useApi(api.machines, []);
  const nav = useNavigate();
  if (!cases.data || !m.data) return <Loading error={cases.error ?? m.error} />;
  const solve = m.data.machines.find((x) => x.id === "solve");
  return (
    <Panel primary title="Cases" sub="one solve-loop instance per case; % of bound = lower bound ÷ best measured latency (100% = at the bound)">
      <table className="t">
        <thead><tr><th>case</th><th>problem</th><th>state</th><th>round</th><th className="r">bound</th><th className="r">best</th><th style={{ width: 260 }}>% of bound</th></tr></thead>
        <tbody>{cases.data.map((c) => (
          <tr key={c.id} className="click" onClick={() => nav(`/cases/${c.id}`)}>
            <td><Link to={`/cases/${c.id}`}><b>{c.name}</b></Link></td>
            <td className="mono small">{c.problem}</td>
            <td><Chip tone={stateTone(solve, c.state)}>{c.state}</Chip></td>
            <td className="num">{c.round}/5</td>
            <td className="r num">{us(c.bound_us)}</td>
            <td className="r num">{us(c.best_us)}</td>
            <td><div className="row"><div className="bar" style={{ flex: 1 }}><span className="used" style={{ width: `${(c.pct_of_bound ?? 0) * 100}%` }} /></div>
              <span className="num small" style={{ width: 40, textAlign: "right" }}>{pct(c.pct_of_bound)}</span></div></td>
          </tr>
        ))}</tbody>
      </table>
    </Panel>
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
        <Stat label="Lower bound" value={c.bound_us.toFixed(1)} unit="µs" record={c.bound_record} foot="theoretical, from the hardware model" />
        <Stat label={`Strongest reference · ${strongest.name}`} value={strongest.us.toFixed(1)} unit="µs" record={strongest.record}
          foot={c.references.length > 1 ? `${c.references.length} references measured` : "measured under measure-v1"} />
        <Stat label="Best result" value={c.best ? c.best.us.toFixed(1) : "—"} unit={c.best ? "µs" : undefined} record={c.best?.record}
          foot={c.best ? <Link to={`/results/${c.best.solution}`}>why this holds →</Link> : "no measured result yet"} />
        <Stat label="% of bound" value={pct(c.pct_of_bound)} foot={c.best ? `${(strongest.us / c.best.us).toFixed(2)}× vs strongest reference` : "—"} />
      </div>

      <Panel primary title="Iteration timeline" sub="each row is one round of the solve loop; segments are states, ◆ tasks created, ✕ rejections; the judging outcome closes the round">
        <Timeline c={c} m={m} />
        <div className="rounds">
          {c.rounds.map((r) => (
            <div key={r.n} className="round-note">
              <b>Round {r.n}</b>
              {r.outcome ? <span> — {r.outcome.rule ? `rule ${r.outcome.rule}: ` : ""}<RecordLink id={r.outcome.record} label={r.outcome.label} /></span>
                : <span className="muted"> — in progress ({r.steps[r.steps.length - 1].state}{r.steps[r.steps.length - 1].note ? `: ${r.steps[r.steps.length - 1].note}` : ""})</span>}
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
                <td>{x.check_passed === undefined ? <span className="muted small">—</span> : <Chip tone={x.check_passed ? "ok" : "fail"}>{x.check_passed ? "passed" : "failed"}</Chip>}</td>
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
          {c.gaps.length === 0 ? <div className="empty">No gap recorded for this case.</div> : c.gaps.map((g) => <div key={g}><RecordLink id={g} label="Gap: global-load latency exposed in the K loop (missing Template, level 3)" /></div>)}
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
  const now = nowMs();
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
