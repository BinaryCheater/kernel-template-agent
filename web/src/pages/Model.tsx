import { useMemo, useState } from "react";
import { api } from "../data/source";
import type { Residual } from "../data/types";
import { useApi } from "../data/useApi";
import { useOpenRecord, Loading, Panel, RecordLink, Stat } from "../components/ui";
import { AXIS, EChart, TOOLTIP, type EOption } from "../components/EChart";
import { pct } from "../components/format";

const PALETTE = ["#2a78d6", "#eb6834", "#1baf7a", "#6250d6", "#e87ba4", "#eda100"];

export function Model() {
  const r = useApi(() => api.residuals(), []);
  if (!r.data) return <Loading error={r.error} />;
  return <ModelBody rows={r.data} />;
}

function ModelBody({ rows }: { rows: Residual[] }) {
  const models = [...new Set(rows.map((r) => r.model))];
  const [model, setModel] = useState<string>("");
  const shown = model ? rows.filter((r) => r.model === model) : rows;
  const abs = shown.map((r) => Math.abs(r.rel_error)).sort((a, b) => a - b);
  const median = abs.length ? abs[Math.floor(abs.length / 2)] : null;
  const within = abs.length ? abs.filter((x) => x <= 0.1).length / abs.length : null;
  const worst = [...shown].sort((a, b) => Math.abs(b.rel_error) - Math.abs(a.rel_error)).slice(0, 8);
  return (
    <div className="stack">
      <div className="stats">
        <Stat label="Measured candidates" value={shown.length} foot={`${models.length} model version(s)`} />
        <Stat label="Median |error|" value={pct(median, 1)} foot="relative to the measurement" />
        <Stat label="Within ±10%" value={pct(within)} foot="share of candidates" />
        <Stat label="Mean signed error" value={pct(shown.length ? shown.reduce((s, r) => s + r.rel_error, 0) / shown.length : null, 1)} foot="> 0: model is pessimistic" />
      </div>
      <div className="grid" style={{ gridTemplateColumns: "minmax(0, 1.4fr) minmax(0, 1fr)", alignItems: "start" }}>
        <Panel primary title="Predicted vs measured" sub="each point is a measured candidate; on the diagonal the model is exact" right={
          <select value={model} onChange={(e) => setModel(e.target.value)}><option value="">all models</option>{models.map((m) => <option key={m}>{m}</option>)}</select>}>
          {shown.length === 0 ? <div className="empty">No Prediction has a matching Measurement yet.</div> : <Scatter rows={shown} models={models} />}
        </Panel>
        <Panel title="Largest errors" sub="candidates the model explains worst; click to open the records">
          <table className="t">
            <thead><tr><th>model</th><th className="r">predicted</th><th className="r">measured</th><th className="r">error</th><th /></tr></thead>
            <tbody>{worst.map((r) => (
              <tr key={r.prediction}><td className="small mono">{r.model}</td><td className="r num">{r.predicted_us.toFixed(1)} µs</td><td className="r num">{r.measured_us.toFixed(1)} µs</td>
                <td className="r num" style={{ color: Math.abs(r.rel_error) > 0.2 ? "#9b1f1f" : undefined }}>{r.rel_error > 0 ? "+" : ""}{(r.rel_error * 100).toFixed(1)}%</td>
                <td className="small"><RecordLink id={r.prediction} label="pred" /> <RecordLink id={r.measurement} label="meas" /></td></tr>
            ))}</tbody>
          </table>
          <div className="small muted" style={{ marginTop: 8 }}>Residuals per hardware unit and top-k recall need per-unit Profile records and the model diagnosis task (T4, WP11).</div>
        </Panel>
      </div>
    </div>
  );
}

function Scatter({ rows, models }: { rows: Residual[]; models: string[] }) {
  const open = useOpenRecord();
  const option = useMemo<EOption>(() => {
    const max = Math.max(...rows.flatMap((r) => [r.predicted_us, r.measured_us])) * 1.08;
    return {
      animation: false,
      grid: { left: 60, right: 20, top: 30, bottom: 44 },
      legend: { top: 0, data: [...models, "exact", "±20%"], textStyle: { color: "#52514e", fontSize: 11 } },
      tooltip: { ...TOOLTIP, formatter: (x: { data: { r: Residual } }) => `${x.data.r.model}<br/>predicted <b>${x.data.r.predicted_us.toFixed(1)} µs</b><br/>measured <b>${x.data.r.measured_us.toFixed(1)} µs</b><br/>error ${(x.data.r.rel_error * 100).toFixed(1)}%` },
      xAxis: { type: "value", name: "measured µs", nameLocation: "middle", nameGap: 26, min: 0, max, ...AXIS },
      yAxis: { type: "value", name: "predicted µs", min: 0, max, ...AXIS },
      series: [
        ...models.map((m, i) => ({
          type: "scatter", name: m, symbolSize: 9, itemStyle: { color: PALETTE[i % PALETTE.length], opacity: 0.85 },
          data: rows.filter((r) => r.model === m).map((r) => ({ value: [r.measured_us, r.predicted_us], r })),
        })),
        { type: "line", name: "exact", data: [[0, 0], [max, max]], symbol: "none", lineStyle: { color: "#c3c2b7", type: "dashed" }, silent: true },
        { type: "line", name: "±20%", data: [[0, 0], [max, max * 1.2]], symbol: "none", lineStyle: { color: "#e1e0d9", type: "dotted" }, silent: true },
        { type: "line", name: "-20%", data: [[0, 0], [max, max * 0.8]], symbol: "none", lineStyle: { color: "#e1e0d9", type: "dotted" }, silent: true, legendHoverLink: false },
      ],
    } as EOption;
  }, [rows, models]);
  return <EChart option={option} height={420} onClick={(x) => { const r = (x.data as { r?: Residual }).r; if (r) open(r.prediction); }} />;
}
