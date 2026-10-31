import { api } from "../data/source";
import type { StageTimeline } from "../data/types";
import { useApi } from "../data/useApi";
import { EChart } from "../components/EChart";
import { Chip, Loading, Panel } from "../components/ui";
import { hhmm, pct } from "../components/format";

/** Stage queues over time (WP23): depth, throughput and wait per stage, GPU busy and host-lock waits. */
export function PipelineStages({ now, win }: { now: number; win: number }) {
  const minute = Math.floor(now / 60_000);
  const bucket = Math.max(60, Math.round((win * 60) / 60));
  const st = useApi(() => api.stages(now - win * 60_000, now, bucket), [win, minute]);
  if (!st.data) return <Loading error={st.error} />;
  return <StagesBody d={st.data} />;
}

function StagesBody({ d }: { d: StageTimeline }) {
  const x = d.buckets.map((b) => hhmm(b));
  const names = Object.keys(d.stages).filter((k) => d.stages[k].depth.some((v) => v > 0) || d.now.stages.some((s) => s.id === k));
  const depth = {
    tooltip: { trigger: "axis" }, legend: { top: 0, type: "scroll" }, grid: { left: 40, right: 40, top: 28, bottom: 24 },
    xAxis: { type: "category", data: x },
    yAxis: [{ type: "value", name: "depth" }, { type: "value", name: "share", max: 1, min: 0 }],
    series: [
      ...names.map((k) => ({ name: k, type: "line", stack: "depth", areaStyle: {}, symbol: "none", data: d.stages[k].depth })),
      { name: "GPU busy", type: "line", yAxisIndex: 1, symbol: "none", lineStyle: { type: "dashed" }, data: d.gpu_busy },
    ],
  };
  const flow = {
    tooltip: { trigger: "axis" }, legend: { top: 0, type: "scroll" }, grid: { left: 40, right: 40, top: 28, bottom: 24 },
    xAxis: { type: "category", data: x },
    yAxis: [{ type: "value", name: "per h" }, { type: "value", name: "wait s" }],
    series: [
      ...names.map((k) => ({ name: `${k} out/h`, type: "bar", stack: "tp", data: d.stages[k].throughput_per_h })),
      { name: "host-lock wait s", type: "line", yAxisIndex: 1, symbol: "none", data: d.host_lock.wait_s },
    ],
  };
  const busy = d.gpu_busy.length ? d.gpu_busy.reduce((a, b) => a + b, 0) / d.gpu_busy.length : 0;
  return (
    <Panel title="Stage queues" sub={`WIP limits from the knob pipeline.stage_capacity; series derived from spans (bucket ${d.bucket_s} s)`}
      right={<span className="small">GPU busy in window <b className="num">{pct(busy)}</b> · host-lock waits <b className="num">{d.host_lock.waits.reduce((a, b) => a + b, 0)}</b></span>}>
      <table className="table" data-testid="stage-table">
        <thead><tr><th>stage</th><th>pool</th><th className="num">depth / capacity</th><th>does</th></tr></thead>
        <tbody>
          {d.now.stages.map((s) => (
            <tr key={s.id}>
              <td className="mono">{s.id}</td><td className="small">{s.pool}</td>
              <td className="num">{s.depth}{s.capacity != null && <> / {s.capacity} {s.depth >= s.capacity && <Chip tone="waiting">full</Chip>}</>}</td>
              <td className="small muted">{s.does}</td>
            </tr>
          ))}
        </tbody>
      </table>
      {d.now.held.length > 0 && (
        <div className="small" style={{ marginTop: 6 }}>
          <b>Held</b>: {d.now.held.map((h) => <span key={h.loop} className="mono">{h.loop.slice(-6)} ({h.state}) — {h.reason}; </span>)}
        </div>
      )}
      <div className="caption" style={{ marginTop: 8 }}>Depth per stage (stacked, time-averaged) and GPU busy share</div>
      <EChart option={depth} height={200} />
      <div className="caption">Throughput per stage and host-lock wait</div>
      <EChart option={flow} height={180} />
    </Panel>
  );
}
