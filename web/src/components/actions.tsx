import { createContext, useContext, useEffect, useState, type ReactNode } from "react";
import { Link, useLocation } from "react-router";
import { ApiError, act, api } from "../data/source";
import { useLive } from "../data/live";
import type { Capability } from "../data/types";
import { useApi } from "../data/useApi";
import { RecordLink } from "./ui";

// Operations (docs/design/30 §3). Every control here posts to /actions/* and every call writes a
// HumanAction. Which operations are wired comes from GET /actions/capabilities: operations that
// wait for WP6/WP8 are drawn disabled with what they need, never faked (web/DESIGN.md §6).

interface Outcome {
  title: string;
  result: Record<string, unknown>;
}
interface CapCtx {
  caps: Record<string, Capability>;
  loaded: boolean;
  /** Show an operation's result at the top level: its subject may vanish from the page once
   *  the data refreshes (an inbox item that was handled, a cancelled task in a list). */
  announce: (o: Outcome) => void;
}
const CapContext = createContext<CapCtx>({ caps: {}, loaded: false, announce: () => {} });

export function CapabilitiesProvider({ children }: { children: ReactNode }) {
  const c = useApi(api.capabilities, []);
  const [outcome, setOutcome] = useState<Outcome | null>(null);
  const { pathname } = useLocation();
  useEffect(() => setOutcome(null), [pathname]); // following a link in the result closes it
  const caps = Object.fromEntries((c.data ?? []).map((x) => [x.name, x]));
  return (
    <CapContext.Provider value={{ caps, loaded: !!c.data, announce: setOutcome }}>
      {children}
      {outcome && (
        <>
          <div className="modal-scrim" onClick={() => setOutcome(null)} />
          <div className="modal" role="dialog" aria-label={outcome.title}>
            <header><h2>{outcome.title}</h2><button className="close" onClick={() => setOutcome(null)} aria-label="Close">×</button></header>
            <div className="mbody"><ActionResult result={outcome.result} /></div>
            <footer><span className="small muted">The page now shows the new state.</span>
              <button className="op primary" onClick={() => setOutcome(null)}>Done</button></footer>
          </div>
        </>
      )}
    </CapContext.Provider>
  );
}

export const useCaps = () => useContext(CapContext);

/** Operations whose reason is required by docs/design/30 §3 (the API answers 400 without one). */
const REASON_REQUIRED = new Set(["reject", "deprecate", "budget", "retry"]);

export type Tone = "danger" | "primary" | "plain";

interface ActionProps {
  op: string; // capability name
  label: string;
  title: string; // dialog title, e.g. "Reject Template async_copy@2"
  explain?: ReactNode; // what will happen, shown in the dialog
  reasonLabel?: string; // "Reason" by default; "Note for the next attempt" for retry
  tone?: Tone;
  small?: boolean;
  /** Extra form fields (budget limits, task type, ...); return false from `valid` to block submit. */
  fields?: ReactNode;
  valid?: boolean;
  submit: (reason: string) => Promise<Record<string, unknown>>;
}

/** A button that opens a confirmation dialog, posts the operation and shows what was written. */
export function ActionButton(p: ActionProps) {
  const { caps, loaded } = useCaps();
  const [open, setOpen] = useState(false);
  const cap = caps[p.op];
  const unwired = loaded && cap !== undefined && !cap.wired;
  const cls = `op ${p.tone ?? "plain"} ${p.small ? "small" : ""} ${unwired ? "unwired" : ""}`;
  if (unwired) {
    return (
      <button className={cls} disabled title={`Not wired yet: needs ${cap.needs}`}>
        {p.label}<span className="op-tag">not wired</span>
      </button>
    );
  }
  return (
    <>
      <button className={cls} onClick={(e) => { e.stopPropagation(); setOpen(true); }} title={cap?.writes ? `Writes ${cap.writes}` : undefined}>{p.label}</button>
      {open && <ActionDialog {...p} cap={cap} close={() => setOpen(false)} />}
    </>
  );
}

function ActionDialog(p: ActionProps & { cap?: Capability; close: () => void }) {
  const { refresh } = useLive();
  const { announce } = useCaps();
  const [reason, setReason] = useState("");
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<{ status: number; text: string } | null>(null);
  const required = REASON_REQUIRED.has(p.op);
  const canSubmit = !busy && (!required || reason.trim().length > 0) && p.valid !== false;
  const go = async () => {
    setBusy(true);
    setError(null);
    try {
      const result = await p.submit(reason.trim());
      announce({ title: p.title, result });
      p.close();
      refresh();
    } catch (e) {
      const status = e instanceof ApiError ? e.status : 0;
      setError({ status, text: e instanceof ApiError ? e.detail : String(e) });
      setBusy(false);
    }
  };
  return (
    <>
      <div className="modal-scrim" onClick={p.close} />
      <div className="modal" role="dialog" aria-label={p.title} onClick={(e) => e.stopPropagation()}>
        <header><h2>{p.title}</h2><button className="close" onClick={p.close} aria-label="Close">×</button></header>
        <div className="mbody">
          {p.explain && <div className="ink2">{p.explain}</div>}
          {p.cap && <div className="small muted">Records written: {p.cap.writes}{p.cap.partial ? ` — ${p.cap.partial}` : ""}</div>}
          {p.fields}
          <label className="field">
            <span>{p.reasonLabel ?? "Reason"}{required ? <b className="req"> required</b> : <span className="muted"> optional</span>}</span>
            <textarea rows={3} value={reason} onChange={(e) => setReason(e.target.value)}
              placeholder={required ? "Stored in the HumanAction record; the API refuses an empty one" : "Stored in the HumanAction record"} />
          </label>
          {error && (
            <div className={error.status === 501 ? "notice waiting" : "notice fail"}>
              {error.status === 501 ? <><b>Not available yet.</b> {error.text}. Nothing was written.</> : <><b>{error.status || "Error"}.</b> {error.text}</>}
            </div>
          )}
        </div>
        <footer>
          <span className="small muted">Acting as the single operator ([api] operator in config).</span>
          <button className="op plain" onClick={p.close}>Cancel</button>
          <button className={`op ${p.tone ?? "primary"}`} disabled={!canSubmit} onClick={go}>{busy ? "Writing…" : p.label}</button>
        </footer>
      </div>
    </>
  );
}

function ActionResult({ result }: { result: Record<string, unknown> }) {
  const action = typeof result.action === "string" ? result.action : null;
  const stale = Array.isArray(result.stale) ? (result.stale as string[]) : [];
  const followups = Array.isArray(result.followups) ? (result.followups as { what: string; state: string; needs?: string }[]) : [];
  const problems = Array.isArray(result.problems) ? (result.problems as string[]) : [];
  const task = typeof result.task === "string" ? result.task : null;
  const failed = result.state === "failed";
  return (
    <div className={`notice ${failed ? "fail" : "ok"}`}>
      <b>{failed ? "Check failed." : "Done."}</b> {action ? <>HumanAction <RecordLink id={action} /> written.</> : "The operation was carried out."}
      {typeof result.state === "string" && <div className="small">state: <b>{result.state}</b>{task && <> · <Link to={`/tasks/${task}`}>open task {task.slice(-8)}</Link></>}</div>}
      {typeof result.current === "string" && <div className="small">current_profile → <RecordLink id={result.current} /></div>}
      {problems.slice(0, 8).map((p) => <div key={p} className="small mono">{p}</div>)}
      {stale.length > 0 && <div className="small">{stale.length} dependent record(s) marked stale and needing re-verification.</div>}
      {followups.map((f) => (
        <div key={f.what} className="small">{f.what}: {f.state === "not_supported" ? <span className="warn-text">not created — needs {f.needs}</span> : f.state}</div>
      ))}
    </div>
  );
}

/** Judging controls for one record (record drawer, result page, pool rows). */
export function RecordActions({ id, kind, name, small }: { id: string; kind: string; name?: string; small?: boolean }) {
  const label = `${kind}${name ? ` ${name}` : ""}`;
  if (kind === "HumanAction" || kind === "StatusChange") return null;
  return (
    <div className="row wrap" style={{ gap: 6 }}>
      <ActionButton op="flag" label="Flag suspect" small={small} title={`Flag ${label} as suspect`}
        explain="Sets the status to suspect and asks for re-verification. Nothing is rejected." submit={(r) => act.flag(id, r)} />
      <ActionButton op="reject" label="Reject" tone="danger" small={small} title={`Reject ${label}`}
        explain="Sets the status to rejected. Every record that depends on it becomes stale and needs re-verification."
        submit={(r) => act.reject(id, r)} />
      {["Template", "Pattern", "Family", "Structure", "Protocol", "AgentProfile", "HwModel"].includes(kind) && (
        <ActionButton op="deprecate" label="Deprecate" small={small} title={`Deprecate ${label}`}
          explain="Not used for new work any more; results that already use it keep their status." submit={(r) => act.deprecate(id, r)} />
      )}
      {(kind === "Pattern" || kind === "Structure") && (
        <ActionButton op="confirm" label="Confirm" tone="primary" small={small} title={`Confirm ${label}`}
          explain="New patterns and structures need your confirmation before they count as admitted (docs/design/21)."
          submit={(r) => act.confirm(kind === "Pattern" ? "pattern" : "structure", id, r)} />
      )}
      {kind === "Protocol" && (
        <ActionButton op="approve-protocol" label="Approve protocol" tone="primary" small={small} title={`Approve ${label}`}
          explain="Approving freezes the protocol: an experiment may start on it and it cannot change afterwards."
          submit={(r) => act.approve("protocol", id, r)} />
      )}
      {kind === "AgentProfile" && (
        <ActionButton op="approve-profile" label="Make current" tone="primary" small={small} title={`Point current_profile at ${label}`}
          explain="New agent runs use this profile. Only you may move the pointer (spec/modules/agents.md)."
          submit={(r) => act.approve("profile", id, r)} />
      )}
    </div>
  );
}

/** Task controls (running page, task page, inbox). */
export function TaskActions({ task, label, small, retry = true }: { task: string; label: string; small?: boolean; retry?: boolean }) {
  return (
    <div className="row wrap" style={{ gap: 6 }}>
      {retry && (
        <ActionButton op="retry" label="Retry with note" small={small} title={`Retry ${label}`} reasonLabel="Note for the next attempt"
          submit={(r) => act.retry(task, r)} />
      )}
      <ActionButton op="cancel" label="Cancel task" tone="danger" small={small} title={`Cancel ${label}`}
        explain="Marks the task cancelled and releases its budget reservation." submit={(r) => act.cancel(task, r)} />
    </div>
  );
}
