// Annotate (SYSDEBUG-2, D691): the user's verdict on whatever they are looking at — a step,
// a record, a setting. Three buttons; "question" and "change" ask for a sentence. Every note
// lands in the review list (/review?tab=notes) with a link back to its subject.
import { useState } from "react";
import { ApiError, act } from "../data/source";
import type { Annotation } from "../data/types";
import { useLive } from "../data/live";

const VERDICTS = [
  { v: "agree", label: "Agree", icon: "✓", hint: "I have looked at this and it is fine" },
  { v: "question", label: "Question", icon: "?", hint: "I do not understand this, or doubt it" },
  { v: "change", label: "Change", icon: "✎", hint: "This should be different" },
] as const;

export function Annotate(p: { subject: string; label: string; loop?: string; existing?: Annotation[]; compact?: boolean }) {
  const { refresh } = useLive();
  const [open, setOpen] = useState<null | "agree" | "question" | "change">(null);
  const [note, setNote] = useState("");
  const [busy, setBusy] = useState(false);
  const [err, setErr] = useState<string | null>(null);
  const [done, setDone] = useState<string | null>(null);
  const mine = (p.existing ?? []).filter((a) => !a.resolution);
  const send = async (verdict: "agree" | "question" | "change", text: string) => {
    setBusy(true);
    setErr(null);
    try {
      await act.annotate(p.subject, verdict, text, p.label, p.loop);
      setDone(verdict);
      setOpen(null);
      setNote("");
      refresh();
    } catch (e) {
      setErr(e instanceof ApiError ? e.detail : String(e));
    } finally {
      setBusy(false);
    }
  };
  return (
    <span className={`annot ${p.compact ? "compact" : ""}`} onClick={(e) => e.stopPropagation()}>
      {VERDICTS.map((x) => (
        <button key={x.v} type="button" className={`annot-btn ${x.v} ${open === x.v || done === x.v ? "on" : ""}`} title={x.hint} disabled={busy}
          onClick={() => (x.v === "agree" ? send("agree", "") : setOpen(open === x.v ? null : x.v))}>
          <span className="annot-icon">{x.icon}</span>{!p.compact && <span>{x.label}</span>}
        </button>
      ))}
      {mine.length > 0 && <span className="annot-count" title={mine.map((a) => `${a.verdict}: ${a.note || "(no note)"}`).join("\n")}>{mine.length} open</span>}
      {open && (
        <span className="annot-pop">
          <textarea autoFocus rows={2} placeholder={open === "question" ? "What is unclear?" : "What should it be, and why?"} value={note} onChange={(e) => setNote(e.target.value)} />
          <span className="row" style={{ gap: 6 }}>
            <button type="button" className="op primary small" disabled={busy || !note.trim()} onClick={() => send(open, note.trim())}>Save note</button>
            <button type="button" className="op plain small" onClick={() => setOpen(null)}>Cancel</button>
          </span>
        </span>
      )}
      {err && <span className="small error">{err}</span>}
    </span>
  );
}
