// A record body rendered for reading (WP25): objects as label/value rows, lists of like
// objects as tables, long text wrapped, expressions in code type. The raw JSON stays one
// click away for whoever needs it.
import type { ReactNode } from "react";
import { api } from "../data/source";
import type { RecordEnvelope } from "../data/types";
import { useApi } from "../data/useApi";
import { useOpenRecord } from "./ui";

const HASH = /^[0-9a-f]{64}$/;

/** What a person calls a record: its name or title, else its kind. */
export const recordName = (r: RecordEnvelope) => {
  const b = r.body as Record<string, unknown>;
  const n = [b.name, b.title, b.action].find((x) => typeof x === "string" && !HASH.test(x));
  return typeof n === "string" ? `${n}${typeof b.version === "number" ? ` v${b.version}` : ""}` : r.kind;
};

/** A linked record by kind and name, not by hash. */
export function RefName({ id }: { id: string }) {
  const { data, error } = useApi(() => api.record(id), [id]);
  const open = useOpenRecord();
  if (error) return <span className="muted" title={id}>fingerprint</span>;
  const name = data ? recordName(data) : "…";
  return <span className="rlv" onClick={() => open(id)}>{data && name !== data.kind ? `${data.kind}: ${name}` : name}</span>;
}

const label = (k: string) => k.replace(/_/g, " ");
const isObj = (v: unknown): v is Record<string, unknown> => !!v && typeof v === "object" && !Array.isArray(v);
const empty = (v: unknown) => v == null || v === "" || (Array.isArray(v) && !v.length) || (isObj(v) && !Object.keys(v).length);

function Leaf({ v }: { v: unknown }): ReactNode {
  if (v == null || v === "") return <span className="muted">—</span>;
  if (typeof v === "boolean") return v ? "yes" : "no";
  if (typeof v === "number") return <span className="num">{v}</span>;
  const s = String(v);
  if (HASH.test(s)) return <RefName id={s} />;
  return s.length > 120 || s.includes("\n") ? <div className="bv-text">{s}</div> : s;
}

export function BodyView({ v, depth = 0 }: { v: unknown; depth?: number }): ReactNode {
  if (Array.isArray(v)) {
    if (!v.length) return <span className="muted">none</span>;
    if (v.every((x) => !isObj(x) && !Array.isArray(x))) return <span>{v.map(String).join(", ")}</span>;
    if (v.every(isObj) && depth < 3) {
      const keys = [...new Set(v.flatMap((x) => Object.keys(x)))].filter((k) => v.some((x) => !empty(x[k])));
      return (
        <table className="t bv-table">
          <thead><tr>{keys.map((k) => <th key={k}>{label(k)}</th>)}</tr></thead>
          <tbody>{v.map((x, i) => <tr key={i}>{keys.map((k) => <td key={k}><BodyView v={x[k]} depth={depth + 1} /></td>)}</tr>)}</tbody>
        </table>
      );
    }
    return <ol className="bv-list">{v.map((x, i) => <li key={i}><BodyView v={x} depth={depth + 1} /></li>)}</ol>;
  }
  if (isObj(v)) {
    const keys = Object.keys(v).filter((k) => !empty(v[k]));
    if (!keys.length) return <span className="muted">—</span>;
    return (
      <dl className={`bv ${depth ? "nested" : ""}`}>
        {keys.map((k) => <div key={k} className="bv-row"><dt>{label(k)}</dt><dd><BodyView v={v[k]} depth={depth + 1} /></dd></div>)}
      </dl>
    );
  }
  return <Leaf v={v} />;
}
