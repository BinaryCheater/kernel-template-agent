import { useEffect, useMemo, useRef, useState } from "react";
import {
  BaseEdge, Handle, Position, ReactFlow, ReactFlowProvider, useReactFlow, type Edge, type EdgeProps, type Node, type NodeProps,
} from "@xyflow/react";
import "@xyflow/react/dist/style.css";
import type { ChainNode } from "../data/types";
import { layout, roundedPath, type Placed } from "./elk";
import { LEVELS, LEVEL_NAMES } from "./format";
import "./diagram.css";

type ItemData = { n: ChainNode; selected: boolean };
const W = 210;
const H = 84;

function ItemNode({ data }: NodeProps<Node<ItemData>>) {
  const { n, selected } = data;
  const [name, ver] = n.template.split("@");
  return (
    <div className={`chain-node ${selected ? "sel" : ""} as-${n.assurance}`} title={n.summary}>
      <Handle type="target" position={Position.Left} isConnectable={false} className="h" />
      <div className="cn-levels" title={n.level_from == null ? "levels unknown (Template record missing)" : `level ${n.level_from} → ${n.level_to}: ${LEVEL_NAMES[n.level_from]} → ${LEVEL_NAMES[n.level_to ?? n.level_from]}`}>
        {LEVELS.map((l) => <span key={l} className={n.level_from != null && l >= n.level_from && l <= (n.level_to ?? n.level_from) ? `on lv${l}` : ""} />)}
      </div>
      <div className="cn-head"><span className="mono cn-key">{n.key}</span><span className={`cn-assur as-${n.assurance}`}>{n.assurance}</span></div>
      <div className="cn-tpl"><b>{name}</b><span className="muted">@{ver}</span></div>
      <div className="cn-applies">{n.applies_to}</div>
      <Handle type="source" position={Position.Right} isConnectable={false} className="h" />
    </div>
  );
}

function ElkEdge({ id, data, markerEnd }: EdgeProps<Edge<{ points: { x: number; y: number }[] }>>) {
  return data ? <BaseEdge id={id} path={roundedPath(data.points)} markerEnd={markerEnd} className="sm-edge" /> : null;
}

const nodeTypes = { item: ItemNode };
const edgeTypes = { elk: ElkEdge };

function Inner({ nodes: items, placed, selected, onSelect }: { nodes: ChainNode[]; placed: Placed; selected?: string; onSelect: (k: string) => void }) {
  const { fitView } = useReactFlow();
  const nodes: Node[] = useMemo(() => items.map((n) => ({
    id: n.key, type: "item", position: { x: placed.nodes[n.key].x, y: placed.nodes[n.key].y }, style: { width: W, height: H },
    data: { n, selected: n.key === selected }, draggable: false,
  })), [items, placed, selected]);
  const edges: Edge[] = useMemo(() => items.flatMap((n) => n.after.map((a) => ({
    id: `${a}->${n.key}`, source: a, target: n.key, type: "elk", markerEnd: { type: "arrowclosed" as const, color: "#6b6a65", width: 14, height: 14 },
    data: { points: placed.edges[`${a}->${n.key}`]?.points ?? [] },
  }))), [items, placed]);
  useEffect(() => {
    const id = requestAnimationFrame(() => fitView({ padding: 0.04, maxZoom: 1 }));
    return () => cancelAnimationFrame(id);
  }, [placed, fitView]);
  return (
    <ReactFlow nodes={nodes} edges={edges} nodeTypes={nodeTypes} edgeTypes={edgeTypes} fitView fitViewOptions={{ padding: 0.04, maxZoom: 1 }}
      nodesDraggable={false} nodesConnectable={false} elementsSelectable={false} zoomOnScroll={false} preventScrolling={false}
      proOptions={{ hideAttribution: true }} onNodeClick={(_, n) => onSelect(n.id)} />
  );
}

/** Composition chain as a DAG: nodes are chain items, edges are `after` dependencies (elk layered, left → right). */
export function ChainDag({ nodes, selected, onSelect }: { nodes: ChainNode[]; selected?: string; onSelect: (k: string) => void }) {
  const [placed, setPlaced] = useState<Placed>();
  const box = useRef<HTMLDivElement>(null);
  const [width, setWidth] = useState(800);
  useEffect(() => {
    let live = true;
    layout(nodes.map((n) => ({ id: n.key, width: W, height: H })),
      nodes.flatMap((n) => n.after.map((a) => ({ id: `${a}->${n.key}`, source: a, target: n.key }))),
      { "elk.direction": "RIGHT", "elk.layered.spacing.nodeNodeBetweenLayers": "56", "elk.spacing.nodeNode": "28" })
      .then((p) => live && setPlaced(p));
    return () => { live = false; };
  }, [nodes]);
  useEffect(() => {
    if (!box.current) return;
    const ro = new ResizeObserver(([e]) => setWidth(e.contentRect.width));
    ro.observe(box.current);
    return () => ro.disconnect();
  }, []);
  const height = placed ? Math.ceil(placed.height * Math.min(1, width / placed.width)) + 20 : 240;
  return (
    <div ref={box} className="sm-box" style={{ height }}>
      {placed && <ReactFlowProvider><Inner nodes={nodes} placed={placed} selected={selected} onSelect={onSelect} /></ReactFlowProvider>}
    </div>
  );
}
