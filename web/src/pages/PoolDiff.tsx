import { useEffect, useState } from "react";
import { Link } from "react-router";
import { api } from "../data/source";
import type { PoolDiff as Diff, PoolDiffEntry } from "../data/types";
import { useApi } from "../data/useApi";
import { useSnapshot } from "../components/Shell";
import { Chip, Loading, Panel, RecordLink } from "../components/ui";
import { wordTone } from "../components/format";

/** Pool diff between two snapshots (E14, D449): what entered, what left, which named objects
 *  changed version, and how the status of shared members moved (promotions, demotions, stale). */
export function PoolDiff() {
  const { snapshots } = useSnapshot();
  const [a, setA] = useState("");
  const [b, setB] = useState("");
  useEffect(() => {
    // Default: the previous snapshot against the newest one.
    if (!a && !b && snapshots.length >= 2) {
      setA(snapshots[snapshots.length - 2].id);
      setB(snapshots[snapshots.length - 1].id);
    }
  }, [snapshots, a, b]);
  const pick = (value: string, set: (v: string) => void) => (
    <select value={value} onChange={(e) => set(e.target.value)}>
      <option value="">choose a snapshot</option>
      {snapshots.map((s) => <option key={s.id} value={s.id}>{s.label} · {s.created_at.slice(0, 16)} · {s.members} members</option>)}
    </select>
  );
  return (
    <div className="stack">
      <div className="row small ink2">
        <span>From</span>{pick(a, setA)}<span>to</span>{pick(b, setB)}
        <Link to="/pool" style={{ marginLeft: "auto" }}>← Pool</Link>
      </div>
      {a && b ? <DiffView a={a} b={b} /> : <div className="empty">{snapshots.length < 2 ? "Fewer than two pool snapshots exist yet." : "Choose two snapshots."}</div>}
    </div>
  );
}

function DiffView({ a, b }: { a: string; b: string }) {
  const d = useApi(() => api.poolDiff(a, b), [a, b]);
  if (!d.data) return <Loading error={d.error} />;
  return <DiffBody d={d.data} />;
}

function DiffBody({ d }: { d: Diff }) {
  const kinds = [...new Set([...Object.keys(d.counts_a), ...Object.keys(d.counts_b)])].sort();
  return (
    <>
      <div className="stats" style={{ gridTemplateColumns: "repeat(4, 1fr)" }}>
        <div className="stat"><div className="label">Added</div><div className="value num">{d.added.length}</div><div className="foot">records only in the second snapshot</div></div>
        <div className="stat"><div className="label">Removed</div><div className="value num">{d.removed.length}</div><div className="foot">records only in the first</div></div>
        <div className="stat"><div className="label">New versions</div><div className="value num">{d.changed.length}</div><div className="foot">same name, other record</div></div>
        <div className="stat"><div className="label">Status moves</div><div className="value num">{d.status.length}</div><div className="foot">promotions, demotions, stale</div></div>
      </div>
      <Panel primary title="Status moves" sub="shared members whose status differs between the two snapshot times">
        {d.status.length ? (
          <table className="t">
            <thead><tr><th>kind</th><th>record</th><th>before</th><th>after</th></tr></thead>
            <tbody>{d.status.map((s) => (
              <tr key={s.id}><td>{s.kind}</td><td><span className="mono">{s.label}</span> <RecordLink id={s.id} /></td>
                <td><Chip tone={wordTone(s.old)}>{s.old}</Chip></td><td><Chip tone={wordTone(s.new)}>{s.new}</Chip></td></tr>
            ))}</tbody>
          </table>
        ) : <div className="empty">No status changed between the two snapshots.</div>}
      </Panel>
      <Panel title="New versions" sub="a named object present in both snapshots as different records">
        {d.changed.length ? (
          <table className="t">
            <thead><tr><th>kind</th><th>name</th><th>before</th><th>after</th></tr></thead>
            <tbody>{d.changed.map((c) => (
              <tr key={c.new}><td>{c.kind}</td><td className="mono">{c.label}</td><td><RecordLink id={c.old} /></td><td><RecordLink id={c.new} /></td></tr>
            ))}</tbody>
          </table>
        ) : <div className="empty">No new versions.</div>}
      </Panel>
      <div className="grid" style={{ gridTemplateColumns: "minmax(0, 1fr) minmax(0, 1fr)", alignItems: "start" }}>
        <Entries title="Added" rows={d.added} />
        <Entries title="Removed" rows={d.removed} />
      </div>
      <Panel title="Members per kind" sub={`${d.a_at.slice(0, 16)} → ${d.b_at.slice(0, 16)}`}>
        <table className="t">
          <thead><tr><th>kind</th><th className="num">before</th><th className="num">after</th></tr></thead>
          <tbody>{kinds.map((k) => <tr key={k}><td>{k}</td><td className="num">{d.counts_a[k] ?? 0}</td><td className="num">{d.counts_b[k] ?? 0}</td></tr>)}</tbody>
        </table>
      </Panel>
    </>
  );
}

function Entries({ title, rows }: { title: string; rows: PoolDiffEntry[] }) {
  return (
    <Panel title={title} sub={`${rows.length} record(s)`}>
      {rows.length ? (
        <table className="t">
          <thead><tr><th>kind</th><th>record</th></tr></thead>
          <tbody>{rows.map((e) => <tr key={e.id}><td>{e.kind}</td><td><span className="mono small">{e.label}</span> <RecordLink id={e.id} /></td></tr>)}</tbody>
        </table>
      ) : <div className="empty">None.</div>}
    </Panel>
  );
}
