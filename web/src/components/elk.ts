// Automatic layered layout (elkjs) shared by the state-machine and chain diagrams.

import ELK, { type ElkExtendedEdge, type ElkNode } from "elkjs/lib/elk.bundled.js";

const elk = new ELK();

export interface LNode {
  id: string;
  width: number;
  height: number;
}
export interface LEdge {
  id: string;
  source: string;
  target: string;
  label?: string;
}
export interface Placed {
  nodes: Record<string, { x: number; y: number; width: number; height: number }>;
  edges: Record<string, { points: { x: number; y: number }[]; label?: { x: number; y: number; width: number; height: number } }>;
  width: number;
  height: number;
}

export const labelWidth = (text: string) => Math.ceil(text.length * 6.1) + 12;

export async function layout(nodes: LNode[], edges: LEdge[], options: Record<string, string>): Promise<Placed> {
  const graph: ElkNode = {
    id: "root",
    layoutOptions: {
      "elk.algorithm": "layered",
      "elk.edgeRouting": "ORTHOGONAL",
      "elk.layered.spacing.nodeNodeBetweenLayers": "40",
      "elk.spacing.nodeNode": "22",
      "elk.spacing.edgeNode": "14",
      "elk.spacing.edgeEdge": "8",
      "elk.spacing.edgeLabel": "4",
      "elk.layered.spacing.edgeNodeBetweenLayers": "14",
      "elk.layered.considerModelOrder.strategy": "NODES_AND_EDGES",
      "elk.layered.cycleBreaking.strategy": "MODEL_ORDER",
      "elk.layered.nodePlacement.strategy": "BRANDES_KOEPF",
      "elk.edgeLabels.inline": "false",
      "elk.padding": "[top=12,left=12,bottom=12,right=12]",
      ...options,
    },
    children: nodes.map((n) => ({ id: n.id, width: n.width, height: n.height })),
    edges: edges.map(
      (e): ElkExtendedEdge => ({
        id: e.id,
        sources: [e.source],
        targets: [e.target],
        labels: e.label ? [{ text: e.label, width: labelWidth(e.label), height: 16, layoutOptions: { "elk.edgeLabels.placement": "TAIL" } }] : [],
      }),
    ),
  };
  const out = await elk.layout(graph);
  const placed: Placed = { nodes: {}, edges: {}, width: out.width ?? 0, height: out.height ?? 0 };
  for (const c of out.children ?? []) placed.nodes[c.id] = { x: c.x ?? 0, y: c.y ?? 0, width: c.width ?? 0, height: c.height ?? 0 };
  for (const e of (out.edges ?? []) as ElkExtendedEdge[]) {
    const s = e.sections?.[0];
    const points = s ? [s.startPoint, ...(s.bendPoints ?? []), s.endPoint] : [];
    const l = e.labels?.[0];
    placed.edges[e.id] = { points, label: l ? { x: l.x ?? 0, y: l.y ?? 0, width: l.width ?? 0, height: l.height ?? 0 } : undefined };
  }
  return placed;
}

/** SVG path through orthogonal points with small rounded corners. */
export function roundedPath(pts: { x: number; y: number }[], r = 6): string {
  if (pts.length < 2) return "";
  let d = `M ${pts[0].x} ${pts[0].y}`;
  for (let i = 1; i < pts.length - 1; i++) {
    const [a, b, c] = [pts[i - 1], pts[i], pts[i + 1]];
    const l1 = Math.hypot(b.x - a.x, b.y - a.y);
    const l2 = Math.hypot(c.x - b.x, c.y - b.y);
    const k = Math.min(r, l1 / 2, l2 / 2);
    const p1 = { x: b.x - ((b.x - a.x) / (l1 || 1)) * k, y: b.y - ((b.y - a.y) / (l1 || 1)) * k };
    const p2 = { x: b.x + ((c.x - b.x) / (l2 || 1)) * k, y: b.y + ((c.y - b.y) / (l2 || 1)) * k };
    d += ` L ${p1.x} ${p1.y} Q ${b.x} ${b.y} ${p2.x} ${p2.y}`;
  }
  const z = pts[pts.length - 1];
  return `${d} L ${z.x} ${z.y}`;
}
