import { useEffect, useMemo, useRef, useState } from "react";
import {
  BaseEdge, EdgeLabelRenderer, Handle, Position, ReactFlow, ReactFlowProvider, useReactFlow,
  type Edge, type EdgeProps, type Node, type NodeProps,
} from "@xyflow/react";
import "@xyflow/react/dist/style.css";
import type { MachineDef, StateDef, TaskTypeDef, TransitionDef } from "../data/types";
import { labelWidth, layout, roundedPath, type LNode, type Placed } from "./elk";
import { stateTone } from "./format";
import "./diagram.css";

const RULE = ["", "①", "②", "③", "④", "⑤"];
const WORK_LABEL: Record<string, string> = { orchestrator: "orchestrator", executor: "executor jobs", agent: "agent task", human: "you" };

type StateData = { def: StateDef; count: number; tone: string; selected: boolean; dim: boolean };
type EdgeData = { points: { x: number; y: number }[]; label?: string; title?: string; lx?: number; ly?: number; kind: string };

function StateNode({ data }: NodeProps<Node<StateData>>) {
  const { def, count, tone, selected, dim } = data;
  return (
    <div className={`sm-node k-${def.kind} ${count > 0 ? `active t-${tone}` : ""} ${selected ? "sel" : ""} ${dim ? "dim" : ""}`} title={`${def.id}: ${def.does}`}>
      <Handle type="target" position={Position.Left} isConnectable={false} className="h" />
      <div className="sm-head">
        <span className="sm-name">{def.id}</span>
        {count > 0 && <span className={`sm-count t-${tone}`}>{count}</span>}
      </div>
      <div className="sm-does">{def.does}</div>
      <div className="sm-work">{WORK_LABEL[def.work] ?? def.work}</div>
      <Handle type="source" position={Position.Right} isConnectable={false} className="h" />
    </div>
  );
}

function GhostNode() {
  return (
    <div className="sm-ghost">
      <Handle type="source" position={Position.Right} isConnectable={false} className="h" />
      any state
    </div>
  );
}

function ElkEdge({ id, data, markerEnd }: EdgeProps<Edge<EdgeData>>) {
  if (!data) return null;
  return (
    <>
      <BaseEdge id={id} path={roundedPath(data.points)} markerEnd={markerEnd} className={`sm-edge e-${data.kind}`} />
      {data.label && data.lx !== undefined && (
        <EdgeLabelRenderer>
          <div className={`sm-elabel e-${data.kind}`} title={data.title} style={{ transform: `translate(${data.lx}px, ${data.ly}px)` }}>
            {data.label}
          </div>
        </EdgeLabelRenderer>
      )}
    </>
  );
}

const nodeTypes = { state: StateNode, ghost: GhostNode };
const edgeTypes = { elk: ElkEdge };

/** Model order for elk: wait states follow the first state feeding them; task pills follow their source. */
function orderedStates(m: MachineDef): StateDef[] {
  const order = m.states.filter((s) => s.kind !== "wait");
  for (const w of m.states.filter((s) => s.kind === "wait")) {
    const feeder = m.transitions.find((t) => t.to === w.id && t.from !== "*")?.from;
    const at = feeder ? order.findIndex((s) => s.id === feeder) : -1;
    order.splice(at < 0 ? order.length : at + 1, 0, w);
  }
  return order;
}

function edgeTitle(t: TransitionDef, taskName: (id: string) => string): string {
  const head = `${t.from} → ${t.to}${t.rule ? ` (rule ${t.rule})` : ""}`;
  const creates = t.creates ? `\ncreates ${t.creates.map((x) => `${x} ${taskName(x)}`).join(" or ")}` : "";
  return `${head}${t.when ? `\nwhen ${t.when}` : ""}${creates}`;
}

function edgeLabel(t: TransitionDef): string | undefined {
  const glyph = t.rule ? `${RULE[t.rule]} ` : "";
  if (t.kind === "creates_task" && t.creates) return `${glyph}${t.creates.join("/")}`;
  return t.label || t.rule ? `${glyph}${t.label ?? ""}`.trim() : undefined;
}

function graphOf(m: MachineDef, taskTypes: TaskTypeDef[], size: { w: number; h: number }) {
  const nodes: LNode[] = [];
  // "From any state" transitions are listed under the diagram (AnyState), not drawn: a node
  // fed only by them would float unconnected.
  const drawn = m.transitions.filter((t) => t.from !== "*");
  const reached = new Set(drawn.flatMap((t) => [t.from, t.to]));
  for (const s of orderedStates(m)) {
    if (reached.has(s.id) || s.initial) nodes.push({ id: s.id, width: s.kind === "terminal" ? size.w - 16 : size.w, height: size.h });
  }
  const edges = m.transitions.flatMap((t, i) => (t.from === "*" ? [] : [{ id: `e${i}`, source: t.from, target: t.to, label: edgeLabel(t), t }]));
  const taskName = (id: string) => taskTypes.find((x) => x.id === id)?.name ?? id;
  return { nodes, edges, taskName };
}

/** Place a label at the edge's tail, beside its first segment (labels stay out of the elk layout). */
function tailLabel(pts: { x: number; y: number }[], label?: string, t?: TransitionDef): { label?: string; lx?: number; ly?: number } {
  if (!label || pts.length < 2) return {};
  const [a, b] = pts;
  const room = Math.abs(b.x - a.x) + Math.abs(b.y - a.y);
  // A label wider than its first segment would cover the next node: fall back to the rule glyph.
  let text = label;
  if (labelWidth(text) > room + 10 && t?.kind !== "creates_task") text = t?.rule ? RULE[t.rule] : "";
  if (!text) return {};
  const w = labelWidth(text);
  if (a.y === b.y) return { label: text, lx: b.x >= a.x ? a.x + 3 : a.x - w - 3, ly: a.y - 16 };
  return { label: text, lx: a.x + 4, ly: b.y > a.y ? a.y + 2 : a.y - 18 };
}

export interface StateMachineProps {
  machine: MachineDef;
  taskTypes: TaskTypeDef[];
  counts: Record<string, number>;
  selected?: string;
  onSelect?: (state: string | undefined) => void;
  direction?: "RIGHT" | "DOWN";
  nodeSize?: { w: number; h: number };
  /** Never shrink below this zoom; a wider graph scrolls horizontally instead. */
  minScale?: number;
  options?: Record<string, string>;
}

function Diagram(p: StateMachineProps & { placed: Placed; g: ReturnType<typeof graphOf> }) {
  const { fitView } = useReactFlow();
  const { placed, g, machine, counts, selected } = p;
  const nodes: Node[] = useMemo(() => g.nodes.map((n) => {
    const pos = placed.nodes[n.id];
    if (n.id === "__any") return { id: n.id, type: "ghost", position: { x: pos.x, y: pos.y }, data: {}, draggable: false, selectable: false };
    const def = machine.states.find((s) => s.id === n.id)!;
    return {
      id: n.id, type: "state", position: { x: pos.x, y: pos.y }, style: { width: pos.width, height: pos.height }, draggable: false,
      data: { def, count: counts[def.id] ?? 0, tone: stateTone(machine, def.id), selected: selected === def.id, dim: !!selected && selected !== def.id },
    };
  }), [g, placed, machine, counts, selected]);
  const edges: Edge[] = useMemo(() => g.edges.map((e) => {
    const pe = placed.edges[e.id];
    const kind = e.t.kind;
    return {
      id: e.id, source: e.source, target: e.target, type: "elk", selectable: false,
      markerEnd: { type: "arrowclosed" as const, width: 14, height: 14, color: kind === "creates_task" ? "#6250d6" : kind === "normal" ? "#6b6a65" : "#a3a29b" },
      data: { points: pe?.points ?? [], title: edgeTitle(e.t, g.taskName), ...tailLabel(pe?.points ?? [], e.label, e.t), kind },
    };
  }), [g, placed]);
  useEffect(() => {
    const id = requestAnimationFrame(() => fitView({ padding: 0.02, maxZoom: 1 }));
    return () => cancelAnimationFrame(id);
  }, [placed, fitView]);
  return (
    <ReactFlow nodes={nodes} edges={edges} nodeTypes={nodeTypes} edgeTypes={edgeTypes} fitView fitViewOptions={{ padding: 0.02, maxZoom: 1 }}
      nodesDraggable={false} nodesConnectable={false} elementsSelectable={false} panOnDrag={false} zoomOnScroll={false}
      zoomOnPinch={false} zoomOnDoubleClick={false} preventScrolling={false} proOptions={{ hideAttribution: true }} minZoom={0.3}
      onNodeClick={(_, n) => n.type === "state" && p.onSelect?.(p.selected === n.id ? undefined : n.id)} />
  );
}

/** One orchestrator state machine, laid out by elkjs, with live instance counts per state. */
export function StateMachine(p: StateMachineProps) {
  const size = p.nodeSize ?? { w: 150, h: 70 };
  const g = useMemo(() => graphOf(p.machine, p.taskTypes, size), [p.machine, p.taskTypes, size.w, size.h]);
  const [placed, setPlaced] = useState<Placed>();
  const box = useRef<HTMLDivElement>(null);
  const [width, setWidth] = useState(800);
  useEffect(() => {
    let live = true;
    layout(g.nodes, g.edges.map(({ label: _label, ...e }) => e), { "elk.direction": p.direction ?? "RIGHT", ...p.options }).then((r) => live && setPlaced(r));
    return () => { live = false; };
  }, [g, p.direction, p.options]);
  useEffect(() => {
    if (!box.current) return;
    const ro = new ResizeObserver(([e]) => setWidth(e.contentRect.width));
    ro.observe(box.current);
    return () => ro.disconnect();
  }, []);
  const scale = placed ? Math.max(p.minScale ?? 0, Math.min(1, width / placed.width)) : 1;
  const height = placed ? Math.ceil(placed.height * scale) + 8 : 200;
  const inner = placed ? Math.max(width, Math.ceil(placed.width * scale) + 8) : width;
  return (
    <div ref={box} className="sm-scroll" style={{ height: height + (inner > width ? 12 : 0) }}>
      <div className="sm-box" style={{ height, width: inner }}>
        {placed && (
          <ReactFlowProvider>
            <Diagram {...p} placed={placed} g={g} />
          </ReactFlowProvider>
        )}
      </div>
    </div>
  );
}

export function DiagramLegend() {
  return (
    <div className="legend sm-legend">
      <span><span className="sw lg-step" />step</span>
      <span><span className="sw lg-wait" />waits on a task</span>
      <span><span className="sw lg-term" />terminal</span>
      <span><span className="sw lg-active" />has instances now (count)</span>
      <span className="lg-line"><svg width="26" height="8"><line x1="0" y1="4" x2="26" y2="4" stroke="#6b6a65" strokeWidth="1.5" /></svg>transition</span>
      <span className="lg-line"><svg width="26" height="8"><line x1="0" y1="4" x2="26" y2="4" stroke="#6250d6" strokeWidth="1.5" /></svg>creates an agent task</span>
      <span className="lg-line"><svg width="26" height="8"><line x1="0" y1="4" x2="26" y2="4" stroke="#a3a29b" strokeWidth="1.5" strokeDasharray="4 3" /></svg>resume / retry / human</span>
      <span>①–④ judging rules in evaluation order</span>
    </div>
  );
}

/** The guarded exits of each branching state, in rule order: the conditions the diagram abbreviates. */
export function Branches({ machine, taskTypes }: { machine: MachineDef; taskTypes: TaskTypeDef[] }) {
  const guarded = machine.transitions.filter((t) => t.when);
  const froms = [...new Set(guarded.map((t) => t.from))];
  const name = (id: string) => taskTypes.find((x) => x.id === id)?.name ?? id;
  return (
    <div className="branches">
      {froms.map((f) => (
        <div key={f} className="branch">
          <div className="branch-from"><span className="mono">{f === "*" ? "any state" : f}</span></div>
          {guarded.filter((t) => t.from === f).map((t, i) => (
            <div key={i} className={`branch-row e-${t.kind}`}>
              <span className="branch-rule">{t.rule ? RULE[t.rule] : "·"}</span>
              <span className="branch-when">{t.when}</span>
              <span className="branch-to">
                {t.creates && <span className="tag task" title={t.creates.map((c) => `${c}: ${name(c)}`).join("\n")}>creates {t.creates.join(" or ")}</span>}
                → <span className="mono">{t.to}</span>
              </span>
            </div>
          ))}
        </div>
      ))}
    </div>
  );
}
