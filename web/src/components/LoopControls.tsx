import { useState } from "react";
import { ApiError, act } from "../data/source";
import { useLive } from "../data/live";
import type { LoopInstance } from "../data/types";
import { useCaps } from "./actions";

// Advance controls for one loop instance (WP25, D664): step, round, run continuously, pause,
// retry after a block. Each click posts /actions/advance or /actions/pause and writes a
// HumanAction; there is no dialog because stepping is done many times in a row. Steps and
// rounds run in the background on the control plane; the page follows them live.

type Mode = "step" | "round" | "run" | "retry";

const HELP: Record<Mode | "pause", string> = {
  step: "Take the loop's next step now (works while the loop is paused: hand-stepping)",
  round: "Take steps until this round is judged, the loop waits for a task, ends or blocks",
  run: "Resume the loop so its process keeps advancing it",
  retry: "Clear the error mark and take the step again",
  pause: "Stop the loop after its current step; nothing is lost",
};

export function LoopControls({ loop, compact }: { loop: Pick<LoopInstance, "id" | "state" | "blocked" | "paused" | "advancing">; compact?: boolean }) {
  const { refresh } = useLive();
  const { caps, loaded } = useCaps();
  const [busy, setBusy] = useState<string | null>(null);
  const [msg, setMsg] = useState<{ ok: boolean; text: string } | null>(null);
  const terminal = ["accepted", "closed", "covered"].includes(loop.state);
  const unwired = loaded && caps.advance && !caps.advance.wired;
  if (terminal) return <span className="small muted">loop finished</span>;
  if (unwired) return <span className="small muted" title={caps.advance.needs ?? ""}>advance controls not wired on this API</span>;

  const run = async (label: string, call: () => Promise<Record<string, unknown>>, done: (r: Record<string, unknown>) => string) => {
    setBusy(label);
    setMsg(null);
    try {
      const r = await call();
      setMsg({ ok: true, text: done(r) });
      refresh();
    } catch (e) {
      setMsg({ ok: false, text: e instanceof ApiError ? e.detail : String(e) });
    } finally {
      setBusy(null);
    }
  };
  const advance = (mode: Mode) =>
    run(mode, () => act.advance(loop.id, mode), (r) =>
      mode === "run"
        ? r.process_running ? `resumed; the ${r.process} process advances it` : `resumed, but no ${r.process} process is running: start it with kta process start ${r.process}`
        : mode === "round" ? "advancing to the end of the round…" : "taking one step…");
  const pause = () => run("pause", () => act.pause(`loop:${loop.id}`, "paused from the frontend"), () => "paused");
  const working = busy !== null || !!loop.advancing;
  const loopPaused = loop.paused === `loop:${loop.id}`;
  const higherPause = loop.paused && !loopPaused;

  return (
    <span className={`loop-controls ${compact ? "compact" : ""}`} onClick={(e) => e.stopPropagation()}>
      {loop.blocked ? (
        <button className="op primary small" disabled={working} title={HELP.retry} onClick={() => advance("retry")}>Retry step</button>
      ) : (
        <>
          <button className="op plain small" disabled={working || !!higherPause} title={HELP.step} onClick={() => advance("step")}>Step</button>
          <button className="op plain small" disabled={working || !!higherPause} title={HELP.round} onClick={() => advance("round")}>Round</button>
        </>
      )}
      {loopPaused || loop.blocked
        ? <button className="op plain small" disabled={working || !!higherPause} title={HELP.run} onClick={() => advance("run")}>Run</button>
        : <button className="op plain small" disabled={working} title={HELP.pause} onClick={pause}>Pause</button>}
      {loop.advancing && <span className="small" style={{ color: "var(--st-running)" }}>{loop.advancing === "round" ? "advancing a round…" : "stepping…"}</span>}
      {higherPause && <span className="small muted">held by pause of {loop.paused}</span>}
      {msg && !compact && <span className={`small ${msg.ok ? "ink2" : "error"}`}>{msg.text}</span>}
      {msg && compact && !msg.ok && <span className="small error" title={msg.text}>failed</span>}
    </span>
  );
}

/** One line saying what holds a loop, if anything: blocked, paused, waiting, or moving. */
export function loopCondition(l: Pick<LoopInstance, "id" | "blocked" | "paused" | "waiting_on" | "advancing" | "held">): { tone: "fail" | "idle" | "waiting" | "running"; text: string } {
  if (l.blocked) return { tone: "fail", text: `stopped by ${l.blocked.error}` };
  if (l.advancing) return { tone: "running", text: l.advancing === "round" ? "advancing a round" : "stepping" };
  if (l.paused) return { tone: "idle", text: l.paused === `loop:${l.id}` ? "paused" : `paused (${l.paused})` };
  if (l.held?.reason) return { tone: "idle", text: "agent task queued, not running" };
  if (l.waiting_on) return { tone: "waiting", text: "waiting for an agent task" };
  return { tone: "running", text: "running" };
}
