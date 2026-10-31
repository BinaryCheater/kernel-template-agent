import { useEffect, useState } from "react";
import { Link } from "react-router";
import { api } from "../data/source";
import type { ModelVersionRow, Scoreboard as Board, TimelineConfig } from "../data/types";
import { useApi } from "../data/useApi";
import { Chip, Loading, Panel, RecordLink, Stat } from "../components/ui";
import { pct } from "../components/format";

const PHASE_COLOR: Record<string, string> = { load: "#2a78d6", sync: "#eda100", compute: "#1baf7a", epilogue: "#6250d6" };

/** Hardware-model regression scoreboard (02 §4 "Maintenance", WP16 D511): every ModelVersion
 *  with rank correlation and error per resource, and per kernel the L2 predicted timeline
 *  against the measured time. Fixture rows are marked FIXTURE. */
export function Scoreboard() {
  const b = useApi(() => api.scoreboard(), []);
  if (!b.data) return <Loading error={b.error} />;
  return <Body board={b.data} />;
}

function fmt(x: number | null | undefined, digits = 2) {
  return x === null || x === undefined ? "—" : x.toFixed(digits);
}

function Origin({ origin }: { origin: string }) {
  return origin === "real" ? <Chip tone="ok">real</Chip> : <Chip tone="human" title="synthetic records, never evidence">FIXTURE</Chip>;
}

function Body({ board }: { board: Board }) {
  const versions = board.versions;
  const [vid, setVid] = useState("");
  const [kernel, setKernel] = useState("");
  const v = versions.find((x) => x.id === vid) ?? versions[versions.length - 1];
  useEffect(() => {
    if (v && !v.results.some((r) => r.kernel === kernel)) setKernel(v.results[0]?.kernel ?? "");
  }, [v, kernel]);
  if (!versions.length) return <div className="empty">No ModelVersion record yet: the first regression run is pending (real runs, D320).</div>;
  return (
    <div className="stack">
      <div className="row small ink2"><Link to="/model">← Model</Link></div>
      {board.fixture && <div className="small"><Chip tone="human">FIXTURE</Chip> Some rows below are synthetic (origin fixture, written by <span className="mono">scripts/seed-dev</span>); they show the page, not the model's quality.</div>}
      <div className="stats">
        <Stat label="Model versions" value={versions.length} foot={`latest ${v.label}`} />
        <Stat label="Mean rank correlation" value={fmt(v.summary.rank_correlation_mean)} foot="Spearman ρ over cases" />
        <Stat label="Hold-out ρ" value={fmt(v.summary.holdout_rank_correlation_mean)} foot="cases held out of fitting" />
        <Stat label="Hold-out median error" value={pct(v.summary["holdout_error_median:total"] ?? null, 1)} foot={v.admitted === null ? "not decided" : v.admitted ? "admitted" : "rolled back"} />
      </div>
      <Panel primary title="Scoreboard" sub="one row per model version; error columns are median relative errors (total time, then busy time per resource)">
        <VersionTable board={board} selected={v.id} onSelect={setVid} />
      </Panel>
      <Panel title={`Predicted vs measured: ${v.label}`} sub="per configuration: the L2 timeline of the first wave (busy intervals per resource) and the two totals"
        right={<select value={kernel} onChange={(e) => setKernel(e.target.value)}>{v.results.map((r) => <option key={r.kernel}>{r.kernel}</option>)}</select>}>
        {kernel ? <Timeline version={v} kernel={kernel} /> : <div className="empty">This version has no regression result.</div>}
      </Panel>
    </div>
  );
}

function VersionTable({ board, selected, onSelect }: { board: Board; selected: string; onSelect: (id: string) => void }) {
  return (
    <table className="t">
      <thead><tr><th>version</th><th>change</th><th>hardware model</th><th>data</th><th>admitted</th><th>kernel</th><th className="r">ρ</th>
        {board.error_keys.map((k) => <th key={k} className="r">{k}</th>)}<th /></tr></thead>
      <tbody>{board.versions.flatMap((v: ModelVersionRow) => v.results.map((r, i) => (
        <tr key={`${v.id}:${r.kernel}`} className={v.id === selected ? "selected" : ""} onClick={() => onSelect(v.id)} style={{ cursor: "pointer" }}>
          <td>{i === 0 ? v.label : ""}</td><td className="small">{i === 0 ? v.change : ""}</td>
          <td className="small mono">{i === 0 ? `${v.hw_model}${v.arch ? ` on ${v.arch}` : ""}` : ""}</td>
          <td>{i === 0 ? <Origin origin={v.origin} /> : null}</td>
          <td>{i === 0 ? (v.admitted === null ? "—" : v.admitted ? <Chip tone="ok">yes</Chip> : <Chip tone="fail">no</Chip>) : null}</td>
          <td className="small">{r.kernel}{r.holdout ? " (hold-out)" : ""}</td>
          <td className="r num">{fmt(r.rank_correlation)}</td>
          {board.error_keys.map((k) => <td key={k} className="r num">{pct(r.error_by_resource[k] ?? null, 1)}</td>)}
          <td className="small">{i === 0 ? <RecordLink id={v.id} label="record" /> : null}</td>
        </tr>
      )))}</tbody>
    </table>
  );
}

function Timeline({ version, kernel }: { version: ModelVersionRow; kernel: string }) {
  const t = useApi(() => api.modelTimeline(version.id, kernel), [version.id, kernel]);
  if (!t.data) return <Loading error={t.error} />;
  const d = t.data;
  return (
    <div className="stack">
      <div className="small ink2">ρ {fmt(d.rank_correlation)} · median error {pct(d.error_by_resource.total ?? null, 1)} {d.holdout ? "· hold-out" : ""} <Origin origin={d.origin} /></div>
      {d.configurations.map((c) => <ConfigTimeline key={c.index} c={c} />)}
      <div className="small muted">Measured per-resource timelines need in-kernel timestamps (L4); until then the measured side is the total time and, where a Profile exists, busy time per resource.</div>
    </div>
  );
}

function ConfigTimeline({ c }: { c: TimelineConfig }) {
  const wave = c.intervals.full ?? c.intervals.tail ?? [];
  const span = Math.max(1e-9, ...wave.map((i) => i.end_us));
  const lanes = [...new Set(wave.map((i) => i.resource))].sort();
  const W = 640, lane = 16, top = 4, left = 90;
  const max = Math.max(c.predicted_us, c.measured_us) || 1;
  return (
    <div>
      <div className="row small">
        <span className="mono">{c.configuration ? JSON.stringify(c.configuration) : `configuration ${c.index}`}</span>
        <span style={{ marginLeft: "auto" }}>predicted <b className="num">{c.predicted_us.toFixed(1)} µs</b> · measured <b className="num">{c.measured_us.toFixed(1)} µs</b> · bottleneck {c.bottleneck ?? "—"} · {c.waves.length} waves</span>
        {c.prediction && <RecordLink id={c.prediction} label="L2" />}
        {c.measurement && <RecordLink id={c.measurement} label="meas" />}
      </div>
      <svg width="100%" viewBox={`0 0 ${W + left} ${top + lanes.length * lane + 30}`} role="img" aria-label="L2 busy intervals of the first wave">
        {lanes.map((r, k) => (
          <g key={r}>
            <text x={left - 6} y={top + k * lane + 11} textAnchor="end" fontSize="10" fill="#52514e">{r}</text>
            <rect x={left} y={top + k * lane} width={W} height={lane - 3} fill="#f2f1ec" />
          </g>
        ))}
        {wave.map((i, n) => (
          <rect key={n} x={left + (i.start_us / span) * W} y={top + lanes.indexOf(i.resource) * lane}
            width={Math.max(0.5, ((i.end_us - i.start_us) / span) * W)} height={lane - 3}
            fill={PHASE_COLOR[i.phase] ?? "#888"} opacity={0.35 + 0.65 * i.utilisation}>
            <title>{`${i.resource} ${i.phase} ${i.start_us.toFixed(2)}–${i.end_us.toFixed(2)} µs`}</title>
          </rect>
        ))}
        <g transform={`translate(0, ${top + lanes.length * lane + 4})`}>
          <text x={left - 6} y={9} textAnchor="end" fontSize="10" fill="#52514e">predicted</text>
          <rect x={left} y={0} width={(c.predicted_us / max) * W} height={10} fill="#2a78d6" />
          <text x={left - 6} y={22} textAnchor="end" fontSize="10" fill="#52514e">measured</text>
          <rect x={left} y={13} width={(c.measured_us / max) * W} height={10} fill="#eb6834" />
        </g>
      </svg>
    </div>
  );
}
