// Home (WP25, D669): one sentence on the state of things with the switches that make the
// loops run; each loop as a flowchart with its cases (or references) on the step they are
// at; the list of those cases with where they are, which optimisation levels their family
// touches and their own controls; what just happened; what needs you.
import { useState } from "react";
import { Link } from "react-router";
import { ApiError, act, api } from "../data/source";
import type { ActiveRuns, LoopInstance, MachineDef, Machines } from "../data/types";
import { useApi } from "../data/useApi";
import { useLive, useNow } from "../data/live";
import { Loading, Panel } from "../components/ui";
import { dur, since } from "../components/format";
import { FlowMap, LevelStrip } from "../components/FlowMap";
import { LoopControls, loopCondition } from "../components/LoopControls";
import { stateWord, taskWord } from "../components/words";
import { JournalList } from "./Activity";

export function Overview() {
  const machines = useApi(api.machines, []);
  const active = useApi(api.active, []);
  if (!machines.data || !active.data) return <Loading error={machines.error ?? active.error} />;
  return <Home m={machines.data} a={active.data} />;
}

const OPEN = (l: LoopInstance) => !["accepted", "closed", "covered"].includes(l.state);

function Home({ m, a }: { m: Machines; a: ActiveRuns }) {
  const solve = m.machines.find((x) => x.id === "solve");
  const expand = m.machines.find((x) => x.id === "expand");
  const cases = a.loops.filter((l) => l.machine === "solve");
  const refs = a.loops.filter((l) => l.machine === "expand");
  return (
    <div className="stack home">
      <TopLine a={a} />
      {solve && <LoopSection title="Cases" def={solve} loops={cases} a={a} href={(l) => `/cases/${l.subject}`} />}
      {expand && <LoopSection title="References being learned" def={expand} loops={refs} a={a} href={(l) => `/activity?case=${l.subject}`} />}
      <div className="home-bottom">
        <Panel title="Just happened" right={<Link to="/activity">All →</Link>}>
          <JournalList level="milestone" limit={8} compact />
        </Panel>
        <NeedsYou a={a} />
      </div>
    </div>
  );
}

function TopLine({ a }: { a: ActiveRuns }) {
  const open = a.loops.filter(OPEN);
  const stopped = open.filter((l) => l.blocked).length;
  const moving = open.filter((l) => !l.blocked && !l.paused).length;
  const running = (a.processes ?? []).some((p) => p.running);
  const tone = stopped ? "fail" : running && moving ? "running" : "idle";
  const text = stopped ? `${stopped} stopped by an error`
    : !open.length ? "Nothing to run"
      : running && moving ? `${moving} running`
        : "Paused";
  return (
    <div className={`topline ${tone}`}>
      <span className="topline-state"><span className="dot" />{text}</span>
      <span className="topline-procs">{(a.processes ?? []).map((p) => <ProcessSwitch key={p.name} p={p} />)}</span>
      <span className="spacer" />
      <span className="topline-facts">agents {a.tasks.filter((t) => t.state === "running").length}/{a.agent_slots.length} · GPU {a.leases.length ? "busy" : "free"}</span>
      {a.inbox_count > 0 && <a href="#needs-you" className="needs-link">{a.inbox_count} need{a.inbox_count === 1 ? "s" : ""} you</a>}
    </div>
  );
}

const PROC_WORDS: Record<string, string> = { solve: "Solve loop", learn: "Learning loop" };

function ProcessSwitch({ p }: { p: NonNullable<ActiveRuns["processes"]>[number] }) {
  const { refresh } = useLive();
  const [err, setErr] = useState<string | null>(null);
  const [busy, setBusy] = useState(false);
  const flip = async () => {
    setBusy(true);
    setErr(null);
    try {
      await act.process(p.name, p.running ? "stop" : "start");
      refresh();
    } catch (e) {
      setErr(e instanceof ApiError ? e.detail : String(e));
    } finally {
      setBusy(false);
    }
  };
  return (
    <span className={`proc ${p.running ? "on" : "off"}`} title={err ?? (p.allowed ? "" : "not allowed in the current release stage")}>
      <span className="dot" />{PROC_WORDS[p.name] ?? p.name}
      <button className={`op small ${p.running ? "plain" : "primary"}`} disabled={busy || (!p.running && !p.allowed)} onClick={flip}>
        {p.running ? "Pause" : "Run"}
      </button>
      {err && <span className="small error">!</span>}
    </span>
  );
}

function LoopSection({ title, def, loops, a, href }: { title: string; def: MachineDef; loops: LoopInstance[]; a: ActiveRuns; href: (l: LoopInstance) => string }) {
  const open = loops.filter(OPEN);
  const [show, setShow] = useState(open.length > 0);
  return (
    <section className="panel primary loop-section">
      <header onClick={() => setShow(!show)} style={{ cursor: "pointer" }}>
        <h2>{title}</h2><span className="count-pill">{open.length}</span>
        <span className="right linkbtn">{show ? "hide" : "show"}</span>
      </header>
      {show && (
        <div className="body">
          <FlowMap def={def} loops={open} unitHref={href} />
          {open.length > 0 && <RunList loops={open} a={a} href={href} />}
        </div>
      )}
    </section>
  );
}

function RunList({ loops, a, href }: { loops: LoopInstance[]; a: ActiveRuns; href: (l: LoopInstance) => string }) {
  const now = useNow();
  return (
    <table className="t runs">
      <thead><tr><th>name</th><th>at</th><th className="r">round</th><th className="r">for</th><th>levels 1–7</th><th /></tr></thead>
      <tbody>{loops.map((l) => {
        const c = loopCondition(l);
        const task = l.waiting_on ? a.tasks.find((t) => t.id === l.waiting_on) : undefined;
        return (
          <tr key={l.id}>
            <td><span className={`run-dot ${c.tone}`} /><Link to={href(l)}><b>{l.subject_name}</b></Link></td>
            <td>{stateWord(l.state)}{task && <span className="ink2"> · {taskWord(task.type)}</span>}
              {l.blocked && <div className="run-err">{l.blocked.error}</div>}
              {!l.blocked && c.tone === "idle" && <div className="ink2 small">paused</div>}</td>
            <td className="r num">{l.round ?? "—"}{l.max_rounds ? `/${l.max_rounds}` : ""}</td>
            <td className="r num">{dur(since(l.state_since, now))}</td>
            <td><LevelStrip chain={l.chain} /></td>
            <td className="r"><LoopControls loop={l} compact /></td>
          </tr>
        );
      })}</tbody>
    </table>
  );
}

function NeedsYou({ a }: { a: ActiveRuns }) {
  const inbox = useApi(api.inbox, []);
  const items = inbox.data ?? [];
  return (
    <section className="panel" id="needs-you">
      <header><h2>Needs you</h2>{items.length > 0 && <span className="count-pill human">{items.length}</span>}</header>
      <div className="body">
        {items.length === 0 && <div className="ink2">Nothing.</div>}
        {items.map((i) => {
          const loop = i.id.startsWith("blocked:") ? a.loops.find((l) => l.id === i.id.slice(8)) : undefined;
          return (
            <div key={i.id} className="needs-item">
              <b>{loop ? `${loop.subject_name} stopped at ${stateWord(loop.state)}` : i.trigger}</b>
              <div className="small ink2">{loop?.blocked ? `${loop.blocked.error}: ${loop.blocked.message}` : i.subject}</div>
              {loop ? <LoopControls loop={loop} /> : i.task ? <Link to={`/tasks/${i.task}`}>Open →</Link> : null}
            </div>
          );
        })}
      </div>
    </section>
  );
}
