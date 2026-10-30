import { useMemo } from "react";
import type { Span } from "../data/types";
import { AXIS, EChart, TOOLTIP, echarts, type EOption } from "./EChart";
import { dur, hhmm } from "./format";

/** Span kind → colour group (web/DESIGN.md §4.3). Unknown kinds fall to "other". */
export const KIND_STYLE: Record<string, { color: string; label: string; wait?: boolean }> = {
  loop_step: { color: "#6b7280", label: "loop step" },
  task: { color: "#6250d6", label: "task" },
  agent_session: { color: "#6250d6", label: "agent session" },
  agent_model: { color: "#a79cf0", label: "agent thinking" },
  tool_call: { color: "#1baf7a", label: "kta tool call" },
  compile: { color: "#eb6834", label: "compile" },
  check: { color: "#008300", label: "check" },
  measure: { color: "#2a78d6", label: "measure" },
  profile: { color: "#2a78d6", label: "profile" },
  solve: { color: "#e87ba4", label: "solve" },
  job_run: { color: "#eda100", label: "job run" },
  lock_hold: { color: "#eda100", label: "lock held" },
  job_queue: { color: "#a3a29b", label: "queued", wait: true },
  lock_wait: { color: "#eda100", label: "lock wait", wait: true },
  human_wait: { color: "#6250d6", label: "waiting for you", wait: true },
};
export const kindStyle = (k: string) => KIND_STYLE[k] ?? { color: "#a3a29b", label: `other (${k})` };

const LANE_ORDER = ["process:", "agent:", "worker:", "gpu:", "human"];
export function laneRank(lane: string): number {
  const i = LANE_ORDER.findIndex((p) => lane.startsWith(p));
  return i < 0 ? LANE_ORDER.length : i;
}

export interface Row {
  lane: string; // display label of the row's lane
  laneKey: string;
  sub?: string; // sub-row label (expanded session)
  first: boolean;
}

/** Greedy interval packing: overlapping spans in one lane go to separate tracks. */
function pack(spans: Span[], nowMs: number): Span[][] {
  const tracks: { end: number; items: Span[] }[] = [];
  for (const s of [...spans].sort((a, b) => a.start_ts.localeCompare(b.start_ts))) {
    const a = Date.parse(s.start_ts);
    const e = s.end_ts ? Date.parse(s.end_ts) : nowMs;
    const t = tracks.find((x) => x.end <= a);
    if (t) {
      t.items.push(s);
      t.end = e;
    } else tracks.push({ end: e, items: [s] });
  }
  return tracks.map((t) => t.items);
}

export function buildRows(spans: Span[], nowMs: number, expanded: Set<string>) {
  const top = spans.filter((s) => !s.parent_id);
  const lanes = [...new Set(top.map((s) => s.lane))].sort((a, b) => laneRank(a) - laneRank(b) || a.localeCompare(b));
  const rows: Row[] = [];
  const items: { span: Span; row: number }[] = [];
  for (const lane of lanes) {
    const tracks = pack(top.filter((s) => s.lane === lane), nowMs);
    tracks.forEach((tr, i) => {
      const row = rows.length;
      rows.push({ lane, laneKey: lane, first: i === 0 });
      tr.forEach((span) => items.push({ span, row }));
      for (const span of tr) {
        if (!expanded.has(span.span_id)) continue;
        const kids = spans.filter((c) => c.parent_id === span.span_id);
        const r2 = rows.length;
        rows.push({ lane, laneKey: lane, sub: `↳ ${span.name}`, first: false });
        kids.forEach((k) => items.push({ span: k, row: r2 }));
      }
    });
  }
  return { rows, items };
}

export interface SwimlaneProps {
  spans: Span[];
  nowMs: number;
  from: number;
  to: number;
  expanded?: Set<string>;
  compact?: boolean;
  rowHeight?: number;
  highlightTrace?: string;
  onSpanClick?: (s: Span) => void;
}

export function Swimlanes(p: SwimlaneProps) {
  const expanded = p.expanded ?? new Set<string>();
  const { rows, items } = useMemo(() => buildRows(p.spans, p.nowMs, expanded), [p.spans, p.nowMs, expanded]);
  const rh = p.rowHeight ?? (p.compact ? 11 : 24);
  const height = rows.length * rh + (p.compact ? 26 : 74);
  const option = useMemo<EOption>(() => {
    const data = items.map(({ span, row }) => {
      const a = Date.parse(span.start_ts);
      const e = span.end_ts ? Date.parse(span.end_ts) : p.nowMs;
      return { value: [row, a, e], span };
    });
    return {
      animation: false,
      grid: { left: p.compact ? 120 : 190, right: 16, top: 6, bottom: p.compact ? 20 : 58 },
      tooltip: {
        ...TOOLTIP,
        formatter: (x: { data: { span: Span } }) => tip(x.data.span, p.nowMs),
      },
      xAxis: {
        type: "time", min: p.from, max: p.to, ...AXIS, position: "bottom",
        axisLabel: { ...AXIS.axisLabel, formatter: (v: number) => hhmm(new Date(v).toISOString()) },
        splitLine: { show: !p.compact, lineStyle: { color: "#efeee9" } },
      },
      yAxis: {
        type: "category", inverse: true, data: rows.map((_, i) => i), ...AXIS,
        axisTick: { show: false }, splitLine: { show: false },
        axisLabel: {
          ...AXIS.axisLabel, fontSize: p.compact ? 10 : 11, interval: 0,
          formatter: (v: string) => { const r = rows[Number(v)]; return r.sub ? r.sub.slice(0, 26) : r.first ? r.lane : ""; },
        },
      },
      dataZoom: p.compact ? [] : [
        { type: "inside", xAxisIndex: 0, filterMode: "weakFilter", zoomOnMouseWheel: "ctrl", moveOnMouseWheel: false, moveOnMouseMove: true },
        { type: "slider", xAxisIndex: 0, height: 18, bottom: 8, filterMode: "weakFilter", labelFormatter: (v: number) => hhmm(new Date(v).toISOString()) },
      ],
      series: [{
        type: "custom",
        encode: { x: [1, 2], y: 0 },
        data,
        renderItem: (params: unknown, api: { value: (i: number) => number; coord: (v: number[]) => number[]; size: (v: number[]) => number[] }) => {
          const it = data[(params as { dataIndex: number }).dataIndex];
          const s = it.span;
          const st = kindStyle(s.kind);
          const [x0, y] = api.coord([api.value(1), api.value(0)]);
          const [x1] = api.coord([api.value(2), api.value(0)]);
          const h = api.size([0, 1])[1] * (p.compact ? 0.78 : 0.72);
          const coordSys = (params as { coordSys: { x: number; y: number; width: number; height: number } }).coordSys;
          const rect = echarts.graphic.clipRectByRect({ x: x0, y: y - h / 2, width: Math.max(1.5, x1 - x0), height: h }, coordSys);
          if (!rect) return null;
          const dimmed = p.highlightTrace && s.trace_id !== p.highlightTrace;
          const children: unknown[] = [{
            type: "rect", shape: { ...rect, r: 2 },
            style: {
              fill: st.color, opacity: dimmed ? 0.12 : st.wait ? 0.28 : s.kind === "loop_step" ? 0.55 : 0.92,
              stroke: s.status === "fail" ? "#d03b3b" : s.status === "running" ? "#0b0b0b" : undefined,
              lineWidth: s.status === "fail" ? 2 : s.status === "running" ? 1 : 0, lineDash: s.status === "running" ? [3, 2] : undefined,
            },
          }];
          if (!p.compact && rect.width > 54) {
            children.push({
              type: "text",
              style: { text: s.name, x: rect.x + 4, y: rect.y + rect.height / 2, verticalAlign: "middle", fontSize: 10.5,
                fill: st.wait || s.kind === "agent_model" ? "#0b0b0b" : "#fff", width: rect.width - 8, overflow: "truncate", opacity: dimmed ? 0.3 : 1 },
              silent: true,
            });
          }
          return { type: "group", children };
        },
      }],
      markLine: undefined,
    } as EOption;
  }, [items, rows, p.from, p.to, p.nowMs, p.compact, p.highlightTrace]);
  return <EChart option={option} height={height} onClick={(x) => p.onSpanClick?.((x.data as { span: Span }).span)} />;
}

function tip(s: Span, nowMs: number): string {
  const end = s.end_ts ? Date.parse(s.end_ts) : nowMs;
  const d = (end - Date.parse(s.start_ts)) / 1000;
  const attrs = Object.entries(s.attrs).slice(0, 5).map(([k, v]) => `<div style="color:#52514e">${k}: <b>${String(v).slice(0, 48)}</b></div>`).join("");
  const st = kindStyle(s.kind);
  return `<div style="min-width:200px"><div style="font-weight:600">${s.name}</div>
    <div style="color:#52514e"><span style="display:inline-block;width:9px;height:9px;border-radius:2px;background:${st.color};margin-right:5px"></span>${st.label} · ${s.lane}</div>
    <div>${hhmm(s.start_ts, true)} → ${s.end_ts ? hhmm(s.end_ts, true) : "now"} · <b>${dur(d)}</b> · ${s.status}</div>${attrs}</div>`;
}

export function SpanLegend({ kinds }: { kinds: string[] }) {
  return (
    <div className="legend">
      {kinds.map((k) => {
        const st = kindStyle(k);
        return <span key={k}><span className="sw" style={{ background: st.color, opacity: st.wait ? 0.35 : 1 }} />{st.label}</span>;
      })}
      <span><span className="sw" style={{ background: "#fff", border: "1px dashed #0b0b0b" }} />running now</span>
      <span><span className="sw" style={{ background: "#fff", border: "2px solid #d03b3b" }} />failed</span>
    </div>
  );
}
