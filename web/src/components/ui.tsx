import type { ReactNode } from "react";
import { useSearchParams } from "react-router";
import { shortId, type Tone } from "./format";

export function Panel(props: { title?: ReactNode; sub?: ReactNode; right?: ReactNode; primary?: boolean; className?: string; children: ReactNode; bodyClass?: string }) {
  return (
    <section className={`panel ${props.primary ? "primary" : ""} ${props.className ?? ""}`}>
      {props.title && (
        <header>
          <h2>{props.title}</h2>
          {props.sub && <span className="sub">{props.sub}</span>}
          {props.right && <span className="right">{props.right}</span>}
        </header>
      )}
      <div className={`body ${props.bodyClass ?? ""}`}>{props.children}</div>
    </section>
  );
}

export function Chip({ tone, children, title }: { tone: Tone; children: ReactNode; title?: string }) {
  return (
    <span className={`chip ${tone}`} title={title}>
      <span className="dot" />
      {children}
    </span>
  );
}

export const AssuranceChip = ({ a }: { a?: string }) => (a ? <span className="chip assur" title="Assurance level">{a}</span> : null);

/** Hook to open the record drawer; the open record lives in `?rec=` so it is linkable. */
export function useOpenRecord() {
  const [params, setParams] = useSearchParams();
  return (id: string) => {
    const next = new URLSearchParams(params);
    next.set("rec", id);
    setParams(next);
  };
}

/** A record id rendered as a link that opens the record drawer. */
export function RecordLink({ id, label, title }: { id: string; label?: ReactNode; title?: string }) {
  const open = useOpenRecord();
  if (!id) return <span className="muted">—</span>;
  return (
    <span className={label === undefined ? "rl" : "rlv"} title={title ?? `Open record ${id}`} onClick={(e) => { e.stopPropagation(); open(id); }}>
      {label ?? shortId(id)}
    </span>
  );
}

export function Stat(props: { label: string; value: ReactNode; unit?: string; foot?: ReactNode; record?: string }) {
  const v = (
    <>
      {props.value}
      {props.unit && <small>{props.unit}</small>}
    </>
  );
  return (
    <div className="stat">
      <div className="label">{props.label}</div>
      <div className="value num">{props.record ? <RecordLink id={props.record} label={v} /> : v}</div>
      {props.foot && <div className="foot">{props.foot}</div>}
    </div>
  );
}

/** Horizontal used / reserved / limit bar. */
export function UsageBar({ used, reserved = 0, limit, width }: { used: number; reserved?: number; limit: number; width?: number }) {
  const u = limit > 0 ? Math.min(1, used / limit) : 0;
  const r = limit > 0 ? Math.min(1 - u, reserved / limit) : 0;
  return (
    <div className="bar" style={{ width }} title={`${Math.round(used)} used · ${Math.round(reserved)} reserved · ${limit} limit`}>
      <span className="reserved" style={{ left: `${u * 100}%`, width: `${r * 100}%` }} />
      <span className={used > limit ? "over" : "used"} style={{ width: `${u * 100}%` }} />
    </div>
  );
}

export function Loading({ error }: { error?: string }) {
  return error ? <div className="error">Could not load: {error}</div> : <div className="empty">Loading…</div>;
}
