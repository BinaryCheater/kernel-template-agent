import { useMemo, useState } from "react";
import { Link } from "react-router";
import { api } from "../data/source";
import { useNow } from "../data/live";
import type { Span, TraceInfo } from "../data/types";
import { useApi } from "../data/useApi";
import { Loading, Panel } from "../components/ui";
import { dur, hhmm, pct } from "../components/format";
import { KIND_STYLE, SpanLegend, Swimlanes, kindStyle, laneRank } from "../components/Swimlanes";

const WINDOWS = [{ label: "15 min", min: 15 }, { label: "1 h", min: 60 }, { label: "3 h", min: 180 }];
const LANE_GROUPS = ["process", "agent", "worker", "gpu", "human"];

export function Pipeline() {
  const now = useNow();
  const [win, setWin] = useState(60);
  // Fetch the window rounded to the minute so the live tick does not refetch every 5 s.
  const minute = Math.floor(now / 60_000);
  const tr = useApi(() => api.trace(now - win * 60_000, now), [win, minute]);
  if (!tr.data) return <Loading error={tr.error} />;
  return <PipelineBody spans={tr.data.spans} traces={tr.data.traces} now={now} win={win} setWin={setWin} />;
}

/** Busy seconds of a lane in [from, to): union of non-waiting spans. */
function busy(spans: Span[], from: number, to: number, now: number): number {
  const iv = spans.filter((s) => !kindStyle(s.kind).wait && !s.parent_id)
    .map((s) => [Math.max(from, Date.parse(s.start_ts)), Math.min(to, s.end_ts ? Date.parse(s.end_ts) : now)])
    .filter(([a, b]) => b > a).sort((x, y) => x[0] - y[0]);
  let total = 0;
  let cur: number[] | undefined;
  for (const i of iv) {
    if (!cur || i[0] > cur[1]) { if (cur) total += cur[1] - cur[0]; cur = [...i]; } else cur[1] = Math.max(cur[1], i[1]);
  }
  if (cur) total += cur[1] - cur[0];
  return total / 1000;
}

function PipelineBody({ spans, traces, now, win, setWin }: { spans: Span[]; traces: TraceInfo[]; now: number; win: number; setWin: (m: number) => void }) {
  const [trace, setTrace] = useState("");
  const [groups, setGroups] = useState<Set<string>>(new Set(LANE_GROUPS));
  const [expanded, setExpanded] = useState<Set<string>>(new Set());
  const [picked, setPicked] = useState<Span>();
  const from = now - win * 60_000;

  const visible = useMemo(() => spans.filter((s) => groups.has(s.lane.split(":")[0]) &&
    (s.end_ts ? Date.parse(s.end_ts) : now) >= from), [spans, groups, from, now]);
  const lanes = useMemo(() => [...new Set(spans.filter((s) => !s.parent_id).map((s) => s.lane))]
    .sort((a, b) => laneRank(a) - laneRank(b) || a.localeCompare(b)), [spans]);
  const util = lanes.filter((l) => !l.startsWith("process") && l !== "human")
    .map((l) => ({ lane: l, share: busy(spans.filter((s) => s.lane === l), from, now, now) / (win * 60) }));
  const inWin = (s: Span) => Math.max(0, Math.min(now, s.end_ts ? Date.parse(s.end_ts) : now) - Math.max(from, Date.parse(s.start_ts))) / 1000;
  const model = spans.filter((s) => s.kind === "agent_model").reduce((x, s) => x + inWin(s), 0);
  const sessions = spans.filter((s) => s.kind === "agent_session").reduce((x, s) => x + inWin(s), 0);
  const gpu = util.find((u) => u.lane.startsWith("gpu:"));
  const kinds = [...new Set(spans.map((s) => s.kind))].sort((a, b) => Object.keys(KIND_STYLE).indexOf(a) - Object.keys(KIND_STYLE).indexOf(b));

  const onSpan = (s: Span) => {
    setPicked(s);
    if (s.kind === "agent_session") {
      const next = new Set(expanded);
      if (next.has(s.span_id)) next.delete(s.span_id); else next.add(s.span_id);
      setExpanded(next);
    }
  };

  return (
    <div className="stack">
      <div className="util-row">
        <div className="util-hero">
          <div className="small ink2">GPU idle in window</div>
          <div className="hero num">{gpu ? pct(1 - gpu.share) : "—"}</div>
        </div>
        <div className="util-hero">
          <div className="small ink2">Agent thinking share of session time</div>
          <div className="hero num">{sessions ? pct(model / sessions) : "—"}</div>
        </div>
        <div className="util-bars">
          <div className="caption">Lane utilisation, last {WINDOWS.find((w) => w.min === win)?.label}</div>
          {util.map((u) => (
            <div key={u.lane} className="util-line">
              <span className="mono small">{u.lane}</span>
              <div className="bar"><span className="used" style={{ width: `${Math.min(100, u.share * 100)}%` }} /></div>
              <span className="num small">{pct(u.share)}</span>
            </div>
          ))}
        </div>
      </div>

      <Panel primary title="Swimlanes" sub="time → ; one row per lane (overlapping work stacks into extra rows). Click an agent session to expand its thinking and tool calls."
        right={<span className="muted">window follows now (live)</span>}>
        <div className="filters">
          <label>Window
            <select value={win} onChange={(e) => setWin(Number(e.target.value))}>{WINDOWS.map((w) => <option key={w.min} value={w.min}>{w.label}</option>)}</select>
          </label>
          <label>Highlight
            <select value={trace} onChange={(e) => setTrace(e.target.value)}>
              <option value="">all work</option>
              {traces.map((t) => <option key={t.id} value={t.id}>{t.label}</option>)}
            </select>
          </label>
          <span className="small muted">Lanes</span>
          {LANE_GROUPS.map((g) => (
            <label key={g} className="check"><input type="checkbox" checked={groups.has(g)} onChange={() => {
              const n = new Set(groups); if (n.has(g)) n.delete(g); else n.add(g); setGroups(n);
            }} />{g}</label>
          ))}
          <span className="small muted" style={{ marginLeft: "auto" }}>drag to pan · ctrl+wheel to zoom · slider below</span>
        </div>
        <Swimlanes spans={visible} nowMs={now} from={from} to={now} expanded={expanded} highlightTrace={trace || undefined} onSpanClick={onSpan} />
        <SpanLegend kinds={kinds} />
      </Panel>

      <div className="grid" style={{ gridTemplateColumns: "minmax(0, 1fr) minmax(0, 1fr)", alignItems: "start" }}>
        <Panel title="Selected span" sub={picked ? undefined : "click any bar"}>
          {picked ? <SpanDetail s={picked} now={now} traces={traces} /> : <div className="empty">Nothing selected.</div>}
        </Panel>
        <TraceComposition spans={spans} trace={trace} traces={traces} now={now} />
      </div>
    </div>
  );
}

function SpanDetail({ s, now, traces }: { s: Span; now: number; traces: TraceInfo[] }) {
  const t = traces.find((x) => x.id === s.trace_id);
  const d = ((s.end_ts ? Date.parse(s.end_ts) : now) - Date.parse(s.start_ts)) / 1000;
  return (
    <dl className="kv">
      <dt>name</dt><dd><b>{s.name}</b></dd>
      <dt>kind</dt><dd><span className="sw-inline" style={{ background: kindStyle(s.kind).color }} />{kindStyle(s.kind).label}</dd>
      <dt>lane</dt><dd className="mono">{s.lane}</dd>
      <dt>time</dt><dd className="num">{hhmm(s.start_ts, true)} → {s.end_ts ? hhmm(s.end_ts, true) : "running"} · {dur(d)}</dd>
      <dt>status</dt><dd>{s.status}</dd>
      <dt>trace</dt><dd>{t ? t.label : s.trace_id}{t?.kind === "solve" && <> · <Link to="/cases">open case</Link></>}</dd>
      {Object.entries(s.attrs).map(([k, v]) => <div key={k} style={{ display: "contents" }}><dt>{k}</dt><dd className="mono small">{String(v)}</dd></div>)}
      <dt>span</dt><dd className="mono small">{s.span_id}{s.parent_id ? ` (child of ${s.parent_id.slice(-6)})` : ""}</dd>
    </dl>
  );
}

function TraceComposition({ spans, trace, traces, now }: { spans: Span[]; trace: string; traces: TraceInfo[]; now: number }) {
  const t = traces.find((x) => x.id === trace);
  const mine = spans.filter((s) => s.trace_id === trace && s.kind !== "loop_step" && s.kind !== "agent_session");
  const byKind = new Map<string, number>();
  for (const s of mine) byKind.set(s.kind, (byKind.get(s.kind) ?? 0) + ((s.end_ts ? Date.parse(s.end_ts) : now) - Date.parse(s.start_ts)) / 1000);
  const rows = [...byKind.entries()].sort((a, b) => b[1] - a[1]);
  const max = Math.max(1, ...rows.map((r) => r[1]));
  return (
    <Panel title="Where the time went" sub={t ? t.label : "pick a trace under Highlight"}>
      {!t ? <div className="empty">Highlight one case or reference to see its time by kind. The critical path is computed by the API (phase B).</div> : (
        <div className="stack" style={{ gap: 6 }}>
          {rows.map(([k, sec]) => (
            <div key={k} className="util-line">
              <span className="small">{kindStyle(k).label}</span>
              <div className="bar"><span style={{ width: `${(sec / max) * 100}%`, background: kindStyle(k).color, opacity: kindStyle(k).wait ? 0.4 : 1 }} /></div>
              <span className="num small">{dur(sec)}</span>
            </div>
          ))}
          <div className="small muted">Sums of span durations (parallel jobs add up); queue and lock waits are not charged to agent budgets.</div>
        </div>
      )}
    </Panel>
  );
}
