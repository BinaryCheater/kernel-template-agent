import { useMemo } from "react";
import { api } from "../data/source";
import type { CoverageReport, Machines } from "../data/types";
import { useApi } from "../data/useApi";
import { AXIS, EChart, TOOLTIP, type EOption } from "../components/EChart";
import { Chip, Loading, Panel, RecordLink } from "../components/ui";
import { pct, wordTone } from "../components/format";
import { useSnapshot } from "../components/Shell";

export function Coverage() {
  const { snapshot, snapshots } = useSnapshot();
  const cov = useApi(() => (snapshot ? api.coverage(snapshot) : Promise.reject(new Error("no snapshot yet"))), [snapshot]);
  const m = useApi(api.machines, []);
  if (cov.status === 501) {
    return (
      <Panel primary title="No coverage report for this snapshot" sub={snapshots.find((s) => s.id === snapshot)?.label}>
        <div className="empty">{cov.error}. Coverage is computed by the expand loop (WP10b/WP11) and stored as a Report; pick another snapshot above.</div>
      </Panel>
    );
  }
  if (!cov.data || !m.data) return <Loading error={snapshot ? cov.error ?? m.error : undefined} />;
  return <CoverageBody c={cov.data} m={m.data} label={snapshots.find((s) => s.id === snapshot)?.label ?? ""} />;
}

function CoverageBody({ c, m, label }: { c: CoverageReport; m: Machines; label: string }) {
  const expand = m.machines.find((x) => x.id === "expand")!;
  return (
    <div className="stack">
      <div className="scope">
        <div><span className="caption">Sampling scope</span></div>
        <div className="scope-text">Sources: {c.scope.description}, <b>{c.scope.total}</b> in total —{" "}
          {Object.entries(c.scope.by_category).map(([k, v]) => `${k} ${v}`).join(", ")}.</div>
        <div className="small ink2">All estimates below hold only for this population of sources. Snapshot <b>{label}</b> · <RecordLink id={c.record} label="coverage report record" /></div>
      </div>
      <DirectTest c={c} />

      <div className="grid" style={{ gridTemplateColumns: "minmax(0, 1.5fr) minmax(0, 1fr)", alignItems: "start" }}>
        <Panel primary title="Discovery curve" sub="sources decomposed vs distinct Templates found; flattening toward the Chao1 estimate means saturation for this kind of source">
          <Discovery c={c} />
        </Panel>
        <Panel title="Coverage-rate trend" sub="two estimates of the same thing should agree; a large gap means the estimate's assumptions fail">
          <Trend c={c} />
        </Panel>
      </div>

      <div className="grid" style={{ gridTemplateColumns: "minmax(0, 1fr) minmax(0, 1fr)", alignItems: "start" }}>
        <Panel title="Instruction families" sub="red rows: state of the art uses it, we cannot produce it">
          <table className="t">
            <thead><tr><th>instruction family</th><th>we produce</th><th>SOTA uses</th><th>via</th></tr></thead>
            <tbody>{c.instruction_families.map((r) => (
              <tr key={r.family} className={r.sota_uses && !r.we_produce ? "hl" : ""}>
                <td>{r.family}</td>
                <td>{r.we_produce ? <Chip tone="ok">yes</Chip> : <Chip tone="fail">no</Chip>}</td>
                <td>{r.sota_uses ? "yes" : <span className="muted">no</span>}</td>
                <td className="small ink2">{r.note}</td>
              </tr>
            ))}</tbody>
          </table>
        </Panel>
        <Panel title="Hardware feature × pattern" sub="Templates per cell (best assurance); empty cells are possible blind spots">
          <HwGrid c={c} />
        </Panel>
      </div>

      <div className="grid" style={{ gridTemplateColumns: "repeat(3, minmax(0, 1fr))", alignItems: "start" }}>
        <Panel title="Reference implementations" sub="expand-loop state of each">
          <table className="t">
            <tbody>{c.references.map((r) => (
              <tr key={r.name}>
                <td><RecordLink id={r.record} label={r.name} /></td>
                <td><Chip tone={r.state === "not started" ? "idle" : r.state === "covered" ? "ok" : expand.states.find((s) => s.id === r.state)?.kind === "wait" ? "waiting" : wordTone(r.state) === "idle" ? "running" : wordTone(r.state)}>{r.state}</Chip></td>
                <td className="small">{r.gaps ? `${r.gaps} gaps` : ""}</td>
              </tr>
            ))}</tbody>
          </table>
        </Panel>
        <Panel title="Gap explained per case" sub="share of the gap to the bound the model explains">
          {c.gap_explained.map((g) => (
            <div key={g.case} style={{ marginBottom: 10 }}>
              <div className="row small"><span className="mono">{g.case}</span><span className="muted" style={{ marginLeft: "auto" }}>{pct(g.unexplained)} unexplained</span></div>
              <div className="stackbar"><span style={{ width: `${g.explained * 100}%`, background: "var(--st-ok)" }} /><span style={{ width: `${g.unexplained * 100}%`, background: "var(--st-fail)" }} /></div>
            </div>
          ))}
          <div className="legend"><span><span className="sw" style={{ background: "var(--st-ok)" }} />explained</span><span><span className="sw" style={{ background: "var(--st-fail)" }} />unexplained → learn first</span></div>
        </Panel>
        <Panel title="Learning backlog" sub="what the system plans to learn next, and why">
          <ol className="backlog">{c.backlog.map((b) => <li key={b.item}><b>{b.item}</b><div className="small ink2">{b.why}</div></li>)}</ol>
        </Panel>
      </div>

      <GapsByMissing c={c} />
      <Panel title="The structure itself" sub="docs/design/21 §2.4 — where the knowledge model fails to describe what it sees">
        <div className="row wrap" style={{ gap: 32 }}>
          <div><div className="caption">Patterns filed as "other"</div><div className="hero num">{c.structure.other_patterns}</div></div>
          <div><div className="caption">Unmatched transactions by kind</div>
            <div className="row wrap">{Object.entries(c.structure.unmatched_by_kind).map(([k, v]) => <span key={k} className="tag">{k} {v}</span>)}</div></div>
          <div><div className="caption">Template assurance</div>
            <div className="row wrap">{Object.entries(c.structure.assurance).map(([k, v]) => <span key={k} className="tag">{k} {v}</span>)}</div></div>
        </div>
      </Panel>
    </div>
  );
}

function Discovery({ c }: { c: CoverageReport }) {
  const d = c.discovery;
  const option = useMemo<EOption>(() => ({
    animation: false,
    grid: { left: 50, right: 110, top: 20, bottom: 40 },
    tooltip: { ...TOOLTIP, trigger: "axis" },
    xAxis: { type: "value", name: "sources decomposed", nameLocation: "middle", nameGap: 26, min: 0, max: Math.max(c.scope.total, d.sources.length), interval: 1, ...AXIS },
    yAxis: { type: "value", name: "distinct Templates", min: 0, max: Math.ceil(d.chao1.hi + 2), ...AXIS },
    series: [
      { type: "line", name: "band low", data: d.sources.map((x, i) => [x, d.lo[i]]), stack: "band", lineStyle: { opacity: 0 }, symbol: "none", tooltip: { show: false } },
      { type: "line", name: "reshuffle range", data: d.sources.map((x, i) => [x, d.hi[i] - d.lo[i]]), stack: "band", lineStyle: { opacity: 0 }, symbol: "none", areaStyle: { color: "#2a78d6", opacity: 0.15 }, tooltip: { show: false } },
      { type: "line", name: "mean over reshuffles", data: d.sources.map((x, i) => [x, d.mean[i]]), lineStyle: { color: "#2a78d6", width: 2 }, itemStyle: { color: "#2a78d6" }, symbolSize: 7,
        markLine: { silent: true, symbol: "none", lineStyle: { color: "#6250d6", type: "dashed" }, label: { formatter: `Chao1 ≈ ${d.chao1.estimate}`, color: "#3d2fa0", fontSize: 11 }, data: [{ yAxis: d.chao1.estimate }] },
        markArea: { silent: true, itemStyle: { color: "#6250d6", opacity: 0.07 }, label: { show: true, position: "insideTopRight", formatter: `interval ${d.chao1.lo}–${d.chao1.hi}`, color: "#6250d6", fontSize: 10 }, data: [[{ yAxis: d.chao1.lo }, { yAxis: d.chao1.hi }]] } },
    ],
  }) as EOption, [c, d]);
  return (
    <>
      <EChart option={option} height={300} />
      <div className="legend"><span><span className="sw" style={{ background: "#2a78d6" }} />mean distinct Templates (band: reshuffle range)</span>
        <span><span className="sw" style={{ background: "#6250d6", opacity: 0.5 }} />Chao1 estimate of the total and its interval</span></div>
    </>
  );
}

function Trend({ c }: { c: CoverageReport }) {
  const option = useMemo<EOption>(() => ({
    animation: false,
    grid: { left: 44, right: 16, top: 30, bottom: 30 },
    legend: { top: 0, textStyle: { color: "#52514e", fontSize: 11 }, itemWidth: 14, itemHeight: 8 },
    tooltip: { ...TOOLTIP, trigger: "axis", valueFormatter: (v: number) => pct(v) },
    xAxis: { type: "category", data: c.trend.map((t) => t.snapshot), ...AXIS },
    yAxis: { type: "value", min: 0, max: 1, ...AXIS, axisLabel: { ...AXIS.axisLabel, formatter: (v: number) => pct(v) } },
    series: [
      { type: "line", name: "Good–Turing sample coverage", data: c.trend.map((t) => t.good_turing), color: "#2a78d6", symbolSize: 7, lineStyle: { width: 2 } },
      { type: "line", name: "new reference expressible without extension", data: c.trend.map((t) => t.expressible), color: "#eb6834", symbolSize: 7, lineStyle: { width: 2 } },
    ],
  }) as EOption, [c]);
  return <EChart option={option} height={300} />;
}

function HwGrid({ c }: { c: CoverageReport }) {
  const g = c.hw_patterns;
  const cell = (f: number, p: number) => g.cells.find((x) => x.f === f && x.p === p);
  return (
    <table className="hwgrid">
      <thead><tr><th />{g.patterns.map((p) => <th key={p}>{p}</th>)}</tr></thead>
      <tbody>{g.features.map((f, fi) => (
        <tr key={f}><th className="mono">{f}</th>
          {g.patterns.map((p, pi) => {
            const x = cell(fi, pi);
            return <td key={p} className={x && x.n > 0 ? `has as-${x.best}` : "none"} title={x && x.n ? `${x.n} Template(s), best assurance ${x.best}` : "no Template"}>
              {x && x.n > 0 ? <><b>{x.n}</b><span>{x.best}</span></> : ""}</td>;
          })}
        </tr>
      ))}</tbody>
    </table>
  );
}

/** The direct test (WP10b D386): references whose every step a pool Template matches, and
 *  of those, how many leave nothing unexplained; beside it, what still needs real runs. */
function DirectTest({ c }: { c: CoverageReport }) {
  const d = c.report?.direct_test;
  const pending = Object.entries(c.pending ?? {});
  if (!d && pending.length === 0) return null;
  return (
    <div className="grid" style={{ gridTemplateColumns: "minmax(0, 1fr) minmax(0, 1.4fr)", alignItems: "start" }}>
      {d && (
        <Panel title="Direct test" sub="can the pool express each decomposed reference without extension?">
          <div className="row wrap" style={{ gap: 28 }}>
            <div><div className="caption">References</div><div className="hero num">{d.references}</div></div>
            <div><div className="caption">All steps matched</div><div className="hero num">{d.all_steps_matched}</div>
              <div className="small muted">{pct(d.references ? d.all_steps_matched / d.references : null)}</div></div>
            <div><div className="caption">Nothing unexplained</div><div className="hero num">{d.nothing_unexplained}</div>
              <div className="small muted">{pct(d.references ? d.nothing_unexplained / d.references : null)}</div></div>
          </div>
        </Panel>
      )}
      {pending.length > 0 && (
        <Panel title="Not measured yet" sub="these parts of the page stay empty until real runs exist (D320)">
          <table className="t compact"><tbody>{pending.map(([k, why]) => (
            <tr key={k}><td style={{ whiteSpace: "nowrap" }}><Chip tone="waiting">{k.replaceAll("_", " ")}</Chip></td><td className="small ink2">{why}</td></tr>
          ))}</tbody></table>
        </Panel>
      )}
    </div>
  );
}

function GapsByMissing({ c }: { c: CoverageReport }) {
  const missing = Object.entries(c.report?.gaps_by_missing ?? {});
  const attribution = Object.entries(c.report?.gaps_by_attribution ?? {});
  if (missing.length === 0) return null;
  const max = Math.max(1, ...missing.map(([, n]) => n));
  return (
    <Panel title="Open Gaps by what is missing" sub="from the coverage report; a Gap names the kind of knowledge the pool lacks">
      <div className="grid" style={{ gridTemplateColumns: "minmax(0, 1fr) minmax(0, 1fr)", gap: 24 }}>
        <div className="stack" style={{ gap: 4 }}>{missing.sort((a, b) => b[1] - a[1]).map(([k, n]) => (
          <div key={k} className="arm-bar"><span className="small">{k}</span>
            <div className="bar" style={{ height: 10 }}><span className="used" style={{ width: `${(n / max) * 100}%` }} /></div>
            <span className="num small">{n}</span></div>
        ))}</div>
        {attribution.length > 0 && <div><div className="caption">by attribution</div>
          <div className="row wrap" style={{ marginTop: 6 }}>{attribution.filter(([, n]) => n).map(([k, n]) => <span key={k} className="tag">{k} {n}</span>)}</div></div>}
      </div>
    </Panel>
  );
}
