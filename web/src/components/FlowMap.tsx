// A loop drawn as a flowchart (WP25, D669; redrawn in SYSDEBUG-2, D693). The main path runs
// left to right as cards joined by solid arrows. Branches leave each card straight down onto
// one shared bus and enter their target from above (orthogonal, rounded corners), so several
// "hand to an agent" edges read as one line instead of a fan. The way back runs on its own
// dashed bus in the agent colour. Cards carry a colour band for who does the work
// (orchestrator, executor job, agent); the step a run is in glows; runs sit under it as pills.
// Layout comes from machines.yaml, so a changed machine redraws itself.
import { useNavigate } from "react-router";
import type { LoopInstance, MachineDef } from "../data/types";
import { loopCondition } from "./LoopControls";
import { stateWord } from "./words";

const W = 138;
const H = 58;
const GAP = 40;
const X0 = 14;
const Y1 = 30;
const Y2 = 236;
const BUS_DOWN = Y1 + H + 46; // branches run along this line
const BUS_UP = Y1 + H + 70; // the way back runs along this one
const CHIP = 22;
const R = 10; // corner radius of orthogonal edges

const WORK_WORDS: Record<string, string> = { orchestrator: "decided by KTA", executor: "compile / GPU job", agent: "AI agent", human: "you" };

/** From the initial state follow the first forward transition, skipping terminals and side
 *  waits (a wait left by `resume` and never entered by a normal transition). */
export function mainPath(def: MachineDef): string[] {
  const kind = Object.fromEntries(def.states.map((s) => [s.id, s.kind]));
  const side = new Set(def.states.filter((s) => s.kind === "wait"
    && def.transitions.some((t) => t.from === s.id && t.kind === "resume")
    && !def.transitions.some((t) => t.to === s.id && t.kind === "normal")).map((s) => s.id));
  const path: string[] = [];
  let cur = def.states.find((s) => s.initial)?.id ?? def.states[0].id;
  while (cur && !path.includes(cur)) {
    path.push(cur);
    const next = def.transitions.find((t) => t.from === cur && (t.kind === "normal" || t.kind === "creates_task")
      && kind[t.to] !== "terminal" && !side.has(t.to) && !path.includes(t.to));
    cur = next?.to ?? "";
  }
  return path;
}

interface Pos { x: number; y: number; row: 1 | 2 }

function layout(def: MachineDef): { pos: Record<string, Pos>; width: number; path: string[] } {
  // "new" is instantaneous: fold it into the next state's box.
  const path = mainPath(def).filter((s, _i, all) => !(s === "new" && all.length > 1));
  const pos: Record<string, Pos> = {};
  path.forEach((s, i) => (pos[s] = { x: X0 + i * (W + GAP), y: Y1, row: 1 }));
  const last = path[path.length - 1];
  const terminals = def.states.filter((s) => s.kind === "terminal").map((s) => s.id);
  const endX = X0 + path.length * (W + GAP);
  const mainEnd = def.transitions.find((t) => t.from === last && terminals.includes(t.to))?.to ?? terminals[0];
  if (mainEnd) pos[mainEnd] = { x: endX, y: Y1, row: 1 };
  terminals.filter((t) => t !== mainEnd).forEach((t, i) => (pos[t] = { x: endX - i * (W + GAP), y: Y2, row: 2 }));
  const taken: number[] = Object.values(pos).filter((p) => p.row === 2).map((p) => p.x);
  for (const s of def.states.filter((x) => !(x.id in pos) && x.id !== "new")) {
    const linked = def.transitions.filter((t) => t.from === s.id || t.to === s.id).map((t) => (t.from === s.id ? t.to : t.from))
      .map((id) => pos[id]).filter((p): p is Pos => !!p && p.row === 1);
    let x = linked.length ? linked.reduce((a, p) => a + p.x, 0) / linked.length : X0;
    while (taken.some((t) => Math.abs(t - x) < W + 12)) x += W + GAP;
    taken.push(x);
    pos[s.id] = { x, y: Y2, row: 2 };
  }
  if (def.states.some((s) => s.id === "new") && path[0]) pos.new = pos[path[0]];
  const width = Math.max(...Object.values(pos).map((p) => p.x + W)) + X0;
  return { pos, width, path };
}

type EdgeKind = "forward" | "branch" | "back" | "end" | "jump";

/** An orthogonal polyline through points, with rounded corners. */
function ortho(pts: [number, number][]): string {
  let d = `M${pts[0][0]},${pts[0][1]}`;
  for (let i = 1; i < pts.length - 1; i++) {
    const [px, py] = pts[i - 1], [x, y] = pts[i], [nx, ny] = pts[i + 1];
    const r1 = Math.min(R, Math.hypot(x - px, y - py) / 2), r2 = Math.min(R, Math.hypot(nx - x, ny - y) / 2);
    const ax = x - Math.sign(x - px) * r1, ay = y - Math.sign(y - py) * r1;
    const bx = x + Math.sign(nx - x) * r2, by = y + Math.sign(ny - y) * r2;
    d += ` L${ax},${ay} Q${x},${y} ${bx},${by}`;
  }
  const [lx, ly] = pts[pts.length - 1];
  return `${d} L${lx},${ly}`;
}

function edge(a: Pos, b: Pos, terminal: boolean, nth: number): { d: string; k: EdgeKind } {
  const ac = a.x + W / 2, bc = b.x + W / 2;
  if (a.row === 1 && b.row === 1) {
    if (b.x > a.x && b.x - a.x <= W + GAP + 1) return { d: `M${a.x + W},${a.y + H / 2} L${b.x - 4},${b.y + H / 2}`, k: terminal ? "end" : "forward" };
    const lift = Y1 - 14 - nth * 7; // a jump over other steps: over the top
    return { d: ortho([[ac, a.y], [ac, lift], [bc, lift], [bc, b.y - 4]]), k: "jump" };
  }
  if (a.row === 1 && b.row === 2 && terminal) {
    // to a terminal below: its own lane, from the card's right edge, so it never joins the agent bus
    const sx = a.x + W - 18, lane = b.y - 26;
    return { d: ortho([[sx, a.y + H], [sx, lane], [bc, lane], [bc, b.y - 4]]), k: "end" };
  }
  if (a.row === 1 && b.row === 2) {
    const off = -10; // leave left of centre, so the way back (right of centre) does not overlap
    return { d: ortho([[ac + off, a.y + H], [ac + off, BUS_DOWN], [bc + off, BUS_DOWN], [bc + off, b.y - 4]]), k: "branch" };
  }
  if (a.row === 2 && b.row === 1) {
    const off = 12;
    return { d: ortho([[ac + off, a.y], [ac + off, BUS_UP], [bc + off, BUS_UP], [bc + off, b.y + H + 4]]), k: "back" };
  }
  return { d: `M${a.x + W},${a.y + H / 2} L${b.x - 4},${b.y + H / 2}`, k: terminal ? "end" : "branch" };
}

export function FlowMap({ def, loops, unitHref }: { def: MachineDef; loops: LoopInstance[]; unitHref: (l: LoopInstance) => string }) {
  const nav = useNavigate();
  const { pos, width, path } = layout(def);
  const terminal = new Set(def.states.filter((s) => s.kind === "terminal").map((s) => s.id));
  const height = Y2 + H + 3 * CHIP + 20;
  const seen = new Set<string>();
  const into: Record<string, number> = {};
  const edges = def.transitions.filter((t) => {
    if (t.from === "*" || !pos[t.from] || !pos[t.to] || pos[t.from] === pos[t.to]) return false;
    const key = `${t.from}>${t.to}`;
    if (seen.has(key)) return false;
    seen.add(key);
    return true;
  }).map((t) => {
    const nth = (into[t.to] = (into[t.to] ?? 0) + 1) - 1;
    const { d, k } = edge(pos[t.from], pos[t.to], terminal.has(t.to), nth);
    const tip = def.transitions.filter((x) => x.from === t.from && x.to === t.to).map((x) => x.when ?? x.label ?? "").filter(Boolean).join("; ");
    return { t, d, k, tip };
  });
  const at = (s: string) => loops.filter((l) => l.state === s || (l.state === "new" && s !== "new" && pos.new === pos[s]));
  const activeIdx = Math.max(-1, ...loops.map((l) => path.indexOf(l.state === "new" ? path[0] : l.state)));
  const order = ["back", "branch", "end", "jump", "forward"] as EdgeKind[];
  return (
    <svg className="flowmap" viewBox={`0 0 ${width} ${height}`} role="img" aria-label={`${def.title} flowchart`}>
      <defs>
        {order.map((k) => (
          <marker key={k} id={`ah-${def.id}-${k}`} viewBox="0 0 10 10" refX="7.5" refY="5" markerWidth="6.5" markerHeight="6.5" orient="auto-start-reverse">
            <path d="M0,0.8 L9,5 L0,9.2 L2.4,5 z" className={`fm-head ${k}`} />
          </marker>
        ))}
        <filter id={`fm-shadow-${def.id}`} x="-10%" y="-20%" width="120%" height="150%">
          <feDropShadow dx="0" dy="1.5" stdDeviation="2" floodColor="#1a1a2e" floodOpacity="0.10" />
        </filter>
      </defs>
      {edges.sort((a, b) => order.indexOf(a.k) - order.indexOf(b.k)).map((e) => {
        const ia = path.indexOf(e.t.from), ib = path.indexOf(e.t.to);
        const done = e.k === "forward" && ib >= 0 && ib <= activeIdx && ia >= 0;
        return (
          <path key={`${e.t.from}-${e.t.to}`} d={e.d} className={`fm-edge ${e.k} ${done ? "done" : ""}`} markerEnd={`url(#ah-${def.id}-${e.k})`}>
            {e.tip && <title>{e.tip}</title>}
          </path>
        );
      })}
      {def.states.filter((s) => pos[s.id] && s.id !== "new").map((s) => {
        const p = pos[s.id];
        const here = at(s.id);
        const bad = here.some((l) => l.blocked);
        const isTerm = s.kind === "terminal";
        const work = s.work ?? "orchestrator";
        const cls = `fm-node ${s.kind} w-${work} ${here.length ? "has" : ""} ${bad ? "bad" : ""} ${s.id === "accepted" || s.id === "covered" ? "good" : ""}`;
        const shown = here.slice(0, 3);
        const step = path.indexOf(s.id);
        return (
          <g key={s.id} className="fm-g">
            {here.length > 0 && <rect className={`fm-glow ${bad ? "bad" : ""}`} x={p.x - 5} y={p.y - 5} width={W + 10} height={H + 10} rx={isTerm ? (H + 10) / 2 : 15} />}
            <rect className={cls} x={p.x} y={p.y} width={W} height={H} rx={isTerm ? H / 2 : 11} filter={`url(#fm-shadow-${def.id})`}><title>{s.does}</title></rect>
            {!isTerm && <rect className={`fm-band w-${work}`} x={p.x} y={p.y + 10} width={4} height={H - 20} rx={2} />}
            {!isTerm && step >= 0 && <text className="fm-step" x={p.x + 14} y={p.y + 19}>{step + 1}</text>}
            <text className="fm-label" x={p.x + (isTerm ? W / 2 : 14)} y={p.y + (isTerm ? H / 2 + 5 : 37)} textAnchor={isTerm ? "middle" : "start"}>{stateWord(s.id)}</text>
            {!isTerm && <text className="fm-work" x={p.x + 14} y={p.y + 50}>{WORK_WORDS[work] ?? work}</text>}
            {here.length > 0 && (
              <g>
                <circle className={`fm-count ${bad ? "bad" : ""}`} cx={p.x + W - 6} cy={p.y + 6} r={11} />
                <text className="fm-count-text" x={p.x + W - 6} y={p.y + 10} textAnchor="middle">{here.length}</text>
              </g>
            )}
            {shown.map((l, i) => {
              const c = loopCondition(l);
              const name = l.subject_name.length > 16 ? `${l.subject_name.slice(0, 15)}…` : l.subject_name;
              const y = p.y + H + 8 + i * CHIP;
              return (
                <g key={l.id} className={`fm-chip ${c.tone}`} onClick={() => nav(unitHref(l))}>
                  <title>{`${l.subject_name}: ${c.text}`}</title>
                  <rect x={p.x + 4} y={y} width={W - 8} height={CHIP - 4} rx={9} />
                  <circle className="fm-chip-dot" cx={p.x + 15} cy={y + (CHIP - 4) / 2} r={3.5} />
                  <text x={p.x + 24} y={y + 13}>{name}</text>
                </g>
              );
            })}
            {here.length > 3 && <text className="fm-more" x={p.x + W / 2} y={p.y + H + 8 + 3 * CHIP + 10} textAnchor="middle">+{here.length - 3} more</text>}
          </g>
        );
      })}
    </svg>
  );
}

/** Which of the seven optimisation levels a family's chain touches; colour = assurance. */
export function LevelStrip({ chain }: { chain?: LoopInstance["chain"] }) {
  const cells = [1, 2, 3, 4, 5, 6, 7].map((lv) => {
    const steps = (chain ?? []).filter((s) => s.level_from != null && s.level_to != null && s.level_from < lv && lv <= s.level_to);
    const best = steps.some((s) => s.assurance === "proved") ? "proved" : steps.some((s) => s.assurance === "bounded") ? "bounded"
      : steps.some((s) => s.assurance === "tested") ? "tested" : steps.length ? "declared" : "";
    return { lv, steps, best };
  });
  return (
    <span className="levelstrip">
      {cells.map((c) => (
        <span key={c.lv} className={`lv ${c.best}`} title={`level ${c.lv}${c.steps.length ? `: ${c.steps.map((s) => s.template).join(", ")}` : ": not touched"}`}>{c.lv}</span>
      ))}
    </span>
  );
}

/** The main path as a segmented bar: done, current (pulsing; red when stopped), to do. */
export function StageBar({ def, loop }: { def: MachineDef; loop: LoopInstance }) {
  const path = mainPath(def).filter((s, _i, all) => !(s === "new" && all.length > 1));
  const at = loop.state === "new" ? 0 : path.indexOf(loop.state);
  // waiting for an agent: the round got as far as judging, which handed the work over
  const cur = at < 0 && loop.state === "waiting" ? path.indexOf("judging") + 1 : at;
  const off = cur < 0; // finished, or a state off the main path
  const c = loopCondition(loop);
  return (
    <span className="stagebar" title={off ? stateWord(loop.state) : `step ${cur + 1} of ${path.length}: ${stateWord(loop.state)}`}>
      {path.map((s, i) => (
        <span key={s} className={`seg ${!off && i < cur ? "done" : ""} ${!off && i === cur ? `cur ${c.tone}` : ""}`} title={stateWord(s)} />
      ))}
      {loop.state === "waiting" && <span className={`seg agent cur ${c.tone}`} title={stateWord("waiting")} />}
    </span>
  );
}
