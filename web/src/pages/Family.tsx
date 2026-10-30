import { useState } from "react";
import { Link, useNavigate, useParams } from "react-router";
import { api } from "../data/source";
import type { ChainNode, FamilyDetail, FamilySummary } from "../data/types";
import { useApi } from "../data/useApi";
import { useNow } from "../data/live";
import { ChainDag } from "../components/ChainDag";
import { AssuranceChip, Loading, Panel, RecordLink } from "../components/ui";
import { LEVEL_NAMES, dur, hhmm, since } from "../components/format";
import { useSnapshot } from "../components/Shell";

export function FamilyList() {
  const { snapshots, snapshot } = useSnapshot();
  const fams = useApi(() => api.families(snapshot), [snapshot]);
  const nav = useNavigate();
  const now = useNow();
  if (!fams.data) return <Loading error={fams.error} />;
  return (
    <Panel primary title="Families" sub={`executable families in ${snapshots.find((s) => s.id === snapshot)?.label ?? "the current snapshot"}: a composition chain plus a generator`}>
      <table className="t index">
        <thead><tr><th>family</th><th>problem</th><th>composition chain — one bar per step: refinement levels it spans, colour = assurance</th>
          <th className="r">versions</th><th className="r">solutions</th><th className="r">last version</th></tr></thead>
        <tbody>{fams.data.map((f) => (
          <tr key={f.id} className="click" onClick={() => nav(`/families/${f.id}`)}>
            <td><Link to={`/families/${f.id}`} className="mono"><b>{f.name}</b>@{f.version}</Link><div className="small muted">{f.backend}</div></td>
            <td className="mono small">{f.problem}</td>
            <td><StepBars steps={f.step_detail ?? []} /></td>
            <td className="r num">{typeof f.versions === "number" ? f.versions : "—"}</td>
            <td className="r num">{f.solutions}</td>
            <td className="r small ink2 num">{f.updated_at ? `${dur(since(f.updated_at, now))} ago` : "—"}</td>
          </tr>
        ))}</tbody>
      </table>
      <div className="legend" style={{ marginTop: 10 }}>
        <span>levels 0 graph → 5 microarchitecture (bottom → top)</span>
        <span><span className="sw as-bar-declared" />declared</span><span><span className="sw as-bar-tested" />tested</span>
        <span><span className="sw as-bar-bounded" />bounded</span><span><span className="sw as-bar-proved" />proved</span>
      </div>
    </Panel>
  );
}

function StepBars({ steps }: { steps: NonNullable<FamilySummary["step_detail"]> }) {
  const H = 48;
  const unit = H / 6;
  return (
    <div className="step-bars">
      {steps.map((s) => {
        const lo = s.level_from ?? 0;
        const hi = s.level_to ?? lo;
        return (
          <div key={s.key} className="step-bar" title={`${s.key}: ${s.template} · level ${lo}→${hi} · ${s.assurance}`}>
            <div className="step-col" style={{ height: H }}>
              <span className={`as-bar-${s.assurance}`} style={{ bottom: lo * unit, height: (hi - lo + 1) * unit - 1 }} />
            </div>
            <span className="step-key mono">{s.key}</span>
          </div>
        );
      })}
    </div>
  );
}

export function FamilyPage() {
  const { id = "" } = useParams();
  const f = useApi(() => api.familyChain(id), [id]);
  if (!f.data) return <Loading error={f.error} />;
  return <FamilyBody f={f.data} />;
}

function FamilyBody({ f }: { f: FamilyDetail }) {
  const [sel, setSel] = useState<string>(f.nodes[0]?.key);
  const node = f.nodes.find((n) => n.key === sel);
  return (
    <div className="stack">
      <div className="case-head">
        <div>
          <div className="caption">Family · {f.backend}</div>
          <h2 className="mono" style={{ fontSize: 20 }}>{f.name}<span className="muted">@{f.version}</span></h2>
          <div className="ink2">{f.workload} · <RecordLink id={f.id} label="family record" /> · <RecordLink id={f.chain_record} label="chain record" /></div>
        </div>
      </div>
      <div className="grid" style={{ gridTemplateColumns: "minmax(0, 1fr) 360px", alignItems: "start" }}>
        <Panel primary title="Composition chain" sub="each box refines one transaction with one Template; arrows are 'after' dependencies — click a step">
          <ChainDag nodes={f.nodes} selected={sel} onSelect={setSel} />
          <div className="legend">
            <span>refinement levels</span>
            {LEVEL_NAMES.map((n, i) => <span key={n}><span className={`sw lvsw lv${i}`} />{i} {n}</span>)}
          </div>
          <div className="legend" style={{ marginTop: 4 }}>
            <span>assurance:</span><span className="cn-assur as-declared">declared</span><span className="cn-assur as-tested">tested</span>
            <span className="cn-assur as-bounded">bounded</span><span className="cn-assur as-proved">proved</span>
          </div>
        </Panel>
        {node ? <TemplateCard n={node} /> : <Panel title="Template"><div className="empty">Click a step.</div></Panel>}
      </div>
      <div className="grid" style={{ gridTemplateColumns: "repeat(3, minmax(0, 1fr))", alignItems: "start" }}>
        <Panel title="Family variables" sub="what the solver chooses">
          <table className="t"><tbody>{f.variables.map((v) => <tr key={v.name}><td className="mono">{v.name}</td><td className="mono small">{v.domain}</td><td className="small ink2">{v.meaning}</td></tr>)}</tbody></table>
        </Panel>
        <Panel title="Glue constraints" sub="constraints no single step owns">
          {f.glue.length === 0 ? <div className="empty">None.</div> : f.glue.map((g) => (
            <div key={g.id} style={{ marginBottom: 8 }}><div className="mono small">{g.expr}</div><div className="small ink2">{g.meaning}</div></div>
          ))}
        </Panel>
        <Panel title="Versions and use">
          <table className="t"><tbody>{f.versions.map((v) => (
            <tr key={v.version}><td className="mono"><RecordLink id={v.record} label={`@${v.version}`} /></td><td className="num small">{hhmm(v.created_at)}</td><td className="small">{v.note}</td></tr>
          ))}</tbody></table>
          <div className="caption" style={{ marginTop: 10 }}>Accepted solutions</div>
          {f.solution_ids.length === 0 ? <div className="small muted">none yet</div> : f.solution_ids.map((s) => <div key={s.id}><Link to={`/results/${s.id}`}>{s.label}</Link></div>)}
        </Panel>
      </div>
    </div>
  );
}

function TemplateCard({ n }: { n: ChainNode }) {
  return (
    <Panel title={<span className="mono">{n.template}</span>} sub="Template card" right={<RecordLink id={n.template_record} />}>
      <div className="stack" style={{ gap: 10 }}>
        <div className="row wrap"><AssuranceChip a={n.assurance} />{n.change && <span className="tag">{n.change}</span>}
          {n.change && <span className="small ink2">level {n.level_from} → {n.level_to}</span>}</div>
        <p style={{ margin: 0 }}>{n.summary}</p>
        <dl className="kv">
          <dt>refines</dt><dd>{n.applies_to}</dd>
          <dt>levels</dt><dd className="small">{n.level_from == null ? "—" : `${LEVEL_NAMES[n.level_from]} → ${LEVEL_NAMES[n.level_to ?? n.level_from]}`}</dd>
          <dt>requires hw</dt><dd className="mono small">{n.requires_hw.length ? n.requires_hw.join(", ") : "—"}</dd>
          <dt>bindings</dt><dd className="mono small">{Object.keys(n.bindings).length ? Object.entries(n.bindings).map(([k, v]) => `${k} ← ${v}`).join(", ") : "—"}</dd>
        </dl>
        <div>
          <div className="caption">variables</div>
          {n.variables.length === 0 ? <div className="small muted">none</div> : (
            <table className="t compact"><tbody>{n.variables.map((v) => <tr key={v.name}><td className="mono">{v.name}</td><td className="mono small">{v.domain}</td><td className="small muted">{v.binding}</td></tr>)}</tbody></table>
          )}
        </div>
        <div className="small muted">Constraints, obligations, cost terms and expected SASS are in the record (click the id).</div>
      </div>
    </Panel>
  );
}
