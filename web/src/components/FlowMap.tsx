// A loop drawn as a flowchart (WP25, D669): the main path left to right, the branches that
// hand work to an agent going down, the way back drawn dashed, terminals at the end. Layout
// comes from machines.yaml, so a changed machine redraws itself. Each live instance is a
// chip under the state it is in; conditions are tooltips, not text on the canvas.
import { useNavigate } from "react-router";
import type { LoopInstance, MachineDef } from "../data/types";
import { loopCondition } from "./LoopControls";
import { stateWord } from "./words";

const W = 116;
const H = 46;
const GAP = 34;
const X0 = 8;
const Y1 = 34;
const Y2 = 214;
const CHIP = 19;

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

function layout(def: MachineDef): { pos: Record<string, Pos>; width: number } {
  // "new" is instantaneous: fold it into the next state's box.
  const path = mainPath(def).filter((s, _i, all) => !(s === "new" && all.length > 1));
  const pos: Record<string, Pos> = {};
  path.forEach((s, i) => (pos[s] = { x: X0 + i * (W + GAP), y: Y1, row: 1 }));
  const last = path[path.length - 1];
  const terminals = def.states.filter((s) => s.kind === "terminal").map((s) => s.id);
  const endX = X0 + path.length * (W + GAP);
  // The first terminal the last step reaches sits on the main row; the others below it.
  const mainEnd = def.transitions.find((t) => t.from === last && terminals.includes(t.to))?.to ?? terminals[0];
  if (mainEnd) pos[mainEnd] = { x: endX, y: Y1, row: 1 };
  terminals.filter((t) => t !== mainEnd).forEach((t, i) => (pos[t] = { x: endX - i * (W + GAP), y: Y2, row: 2 }));
  // Side states below the main steps they connect to.
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
  return { pos, width };
}

type EdgeKind = "forward" | "branch" | "back" | "end";

function edgeKind(a: Pos, b: Pos, terminal: boolean): EdgeKind {
  if (terminal) return "end";
  if (a.row === 2 && b.row === 1) return "back";
  if (b.row === 2) return "branch";
  return "forward";
}

function path(a: Pos, b: Pos, k: EdgeKind, nth: number): string {
  if (a.row === 1 && b.row === 1) {
    if (b.x > a.x && b.x - a.x <= W + GAP + 1) return `M${a.x + W},${a.y + H / 2} L${b.x - 3},${b.y + H / 2}`;
    const lift = Y1 - 18 - nth * 6; // a jump over other steps: an arc above the row
    return `M${a.x + W / 2},${a.y} C${a.x + W / 2},${lift} ${b.x + W / 2},${lift} ${b.x + W / 2},${b.y - 3}`;
  }
  const off = k === "back" ? 14 : -14;
  if (a.row === 1) {
    const ax = a.x + W / 2 + off, bx = b.x + W / 2 + off + nth * 4;
    const m = (a.y + H + b.y) / 2;
    return `M${ax},${a.y + H} C${ax},${m} ${bx},${m} ${bx},${b.y - 3}`;
  }
  if (b.row === 1) {
    const ax = a.x + W / 2 + off + nth * 4, bx = b.x + W / 2 + off;
    const m = (a.y + b.y + H) / 2;
    return `M${ax},${a.y} C${ax},${m} ${bx},${m} ${bx},${b.y + H + 3}`;
  }
  return `M${a.x + W},${a.y + H / 2} L${b.x - 3},${b.y + H / 2}`;
}

export function FlowMap({ def, loops, unitHref }: { def: MachineDef; loops: LoopInstance[]; unitHref: (l: LoopInstance) => string }) {
  const nav = useNavigate();
  const { pos, width } = layout(def);
  const terminal = new Set(def.states.filter((s) => s.kind === "terminal").map((s) => s.id));
  const height = Y2 + H + 3 * CHIP + 16;
  const seen = new Set<string>();
  const into: Record<string, number> = {};
  const edges = def.transitions.filter((t) => {
    if (t.from === "*" || !pos[t.from] || !pos[t.to] || pos[t.from] === pos[t.to]) return false;
    const key = `${t.from}>${t.to}`;
    if (seen.has(key)) return false;
    seen.add(key);
    return true;
  }).map((t) => {
    const a = pos[t.from], b = pos[t.to];
    const k = edgeKind(a, b, terminal.has(t.to));
    const nth = (into[t.to] = (into[t.to] ?? 0) + 1) - 1;
    const tip = def.transitions.filter((x) => x.from === t.from && x.to === t.to).map((x) => x.when ?? x.label ?? "").filter(Boolean).join("; ");
    return { t, d: path(a, b, k, nth), k, tip };
  });
  const at = (s: string) => loops.filter((l) => l.state === s || (l.state === "new" && s !== "new" && pos.new === pos[s]));
  return (
    <svg className="flowmap" viewBox={`0 0 ${width} ${height}`} role="img" aria-label={`${def.title} flowchart`}>
      <defs>
        {(["forward", "branch", "back", "end"] as EdgeKind[]).map((k) => (
          <marker key={k} id={`ah-${def.id}-${k}`} viewBox="0 0 10 10" refX="8" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse">
            <path d="M0,0 L10,5 L0,10 z" className={`fm-head ${k}`} />
          </marker>
        ))}
      </defs>
      {edges.map((e) => (
        <path key={`${e.t.from}-${e.t.to}`} d={e.d} className={`fm-edge ${e.k}`} markerEnd={`url(#ah-${def.id}-${e.k})`}>
          {e.tip && <title>{e.tip}</title>}
        </path>
      ))}
      {def.states.filter((s) => pos[s.id] && s.id !== "new").map((s) => {
        const p = pos[s.id];
        const here = at(s.id);
        const bad = here.some((l) => l.blocked);
        const cls = `fm-node ${s.kind} ${here.length ? "has" : ""} ${bad ? "bad" : ""} ${s.work === "agent" ? "agent" : ""}`;
        const shown = here.slice(0, 3);
        return (
          <g key={s.id}>
            <rect className={cls} x={p.x} y={p.y} width={W} height={H} rx={s.kind === "terminal" ? H / 2 : 8}><title>{s.does}</title></rect>
            <text className="fm-label" x={p.x + W / 2} y={p.y + H / 2 + 5} textAnchor="middle">{stateWord(s.id)}</text>
            {here.length > 0 && (
              <g>
                <circle className={`fm-count ${bad ? "bad" : ""}`} cx={p.x + W - 4} cy={p.y + 4} r={10} />
                <text className="fm-count-text" x={p.x + W - 4} y={p.y + 8} textAnchor="middle">{here.length}</text>
              </g>
            )}
            {shown.map((l, i) => {
              const c = loopCondition(l);
              const name = l.subject_name.length > 15 ? `${l.subject_name.slice(0, 14)}…` : l.subject_name;
              return (
                <g key={l.id} className={`fm-chip ${c.tone}`} onClick={() => nav(unitHref(l))}>
                  <title>{`${l.subject_name}: ${c.text}`}</title>
                  <rect x={p.x} y={p.y + H + 5 + i * CHIP} width={W} height={CHIP - 3} rx={8} />
                  <text x={p.x + 8} y={p.y + H + 5 + i * CHIP + 12}>{name}</text>
                </g>
              );
            })}
            {here.length > 3 && <text className="fm-more" x={p.x + W / 2} y={p.y + H + 5 + 3 * CHIP + 10} textAnchor="middle">+{here.length - 3}</text>}
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
