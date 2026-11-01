// Activity (WP25, D660): the system's history as one readable stream, newest first. Lines
// come from GET /journal, which phrases records, loop steps, jobs, task changes and stops
// with an open table (journal.yaml) and groups bursts (198 candidates = one line). Each line
// opens its evidence: the record drawer, the task, or the case.
import { useEffect, useState } from "react";
import { Link, useSearchParams } from "react-router";
import { api } from "../data/source";
import type { JournalEntry } from "../data/types";
import { useApi } from "../data/useApi";
import { useLive } from "../data/live";
import { Loading, Panel, useOpenRecord } from "../components/ui";
import { hhmm } from "../components/format";

export function Activity() {
  const [params, setParams] = useSearchParams();
  const level = params.get("level") === "all" ? "all" : "milestone";
  const kase = params.get("case") ?? undefined;
  const cases = useApi(api.cases, []);
  const set = (k: string, v: string | null) => {
    const next = new URLSearchParams(params);
    if (v) next.set(k, v);
    else next.delete(k);
    setParams(next, { replace: true });
  };
  return (
    <Panel primary title="Activity"
      right={
        <span className="row" style={{ gap: 8 }}>
          <div className="tabs small-tabs">
            <button className={level === "milestone" ? "on" : ""} onClick={() => set("level", null)}>Milestones</button>
            <button className={level === "all" ? "on" : ""} onClick={() => set("level", "all")}>Everything</button>
          </div>
          <select value={kase ?? ""} onChange={(e) => set("case", e.target.value || null)}>
            <option value="">all cases and system work</option>
            {(cases.data ?? []).filter((c) => c.purpose !== "probe").map((c) => <option key={c.id} value={c.id}>{c.name}</option>)}
          </select>
        </span>
      }>
      <JournalList level={level} limit={80} case={kase} paged />
    </Panel>
  );
}

/** The journal as a list; `paged` adds "older" loading, `compact` drops the actor column. */
export function JournalList(p: { level: "milestone" | "all"; limit: number; case?: string; compact?: boolean; paged?: boolean }) {
  const first = useApi(() => api.journal({ limit: p.limit, level: p.level, case: p.case }), [p.limit, p.level, p.case]);
  const { version } = useLive();
  const [older, setOlder] = useState<JournalEntry[]>([]);
  const [cursor, setCursor] = useState<string | null>(null);
  const [busy, setBusy] = useState(false);
  useEffect(() => { setOlder([]); setCursor(null); }, [p.level, p.case]);
  useEffect(() => { if (!older.length) setCursor(first.data?.next_before ?? null); }, [first.data, older.length, version]);
  if (!first.data) return <Loading error={first.error} />;
  const entries = [...first.data.entries, ...older];
  const more = async () => {
    if (!cursor) return;
    setBusy(true);
    const r = await api.journal({ before: cursor, limit: p.limit, level: p.level, case: p.case });
    setOlder([...older, ...r.entries]);
    setCursor(r.next_before);
    setBusy(false);
  };
  let day = "";
  const unphrased = Object.entries(first.data.unphrased);
  return (
    <div className={`journal ${p.compact ? "compact" : ""}`}>
      {entries.length === 0 && <div className="empty">Nothing recorded yet.</div>}
      {entries.map((e, i) => {
        const d = e.at.slice(0, 10);
        const head = !p.compact && d !== day ? d : null;
        day = d;
        return (
          <div key={`${e.at}-${i}`}>
            {head && <div className="journal-day">{head}</div>}
            <JournalLine e={e} compact={p.compact} />
          </div>
        );
      })}
      {p.paged && cursor && <button className="op plain small" disabled={busy} onClick={more}>{busy ? "Loading…" : "Older"}</button>}
      {p.paged && unphrased.length > 0 && (
        <div className="small muted" style={{ marginTop: 8 }}>
          Shown with the generic phrase (no entry in journal.yaml yet): {unphrased.map(([k, n]) => `${k} ×${n}`).join(", ")}
        </div>
      )}
    </div>
  );
}

function JournalLine({ e, compact }: { e: JournalEntry; compact?: boolean }) {
  const open = useOpenRecord();
  const target = e.source === "record" ? () => open(e.refs[0]) : undefined;
  return (
    <div className={`jl ${e.tone} ${e.level} ${target ? "click" : ""}`} onClick={target} title={e.count > 1 ? `${e.count} events; opens the newest` : undefined}>
      <span className="jl-time num">{hhmm(e.at)}</span>
      <span className="jl-dot" />
      <span className="jl-text">
        {e.text}
        {e.source === "task" && <> · <Link to={`/tasks/${e.refs[0]}`} onClick={(x) => x.stopPropagation()}>task</Link></>}
      </span>
      {e.case && e.case_name && !compact && <Link className="jl-case" to={`/cases/${e.case}`} onClick={(x) => x.stopPropagation()}>{e.case_name}</Link>}
      {!compact && <span className="jl-actor muted small">{e.actor}</span>}
    </div>
  );
}
