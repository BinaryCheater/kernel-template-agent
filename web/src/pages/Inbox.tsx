import { Link } from "react-router";
import { act, api } from "../data/source";
import type { InboxItem } from "../data/types";
import { useApi } from "../data/useApi";
import { useNow } from "../data/live";
import { ActionButton, TaskActions, useCaps } from "../components/actions";
import { Loading, Panel, RecordLink } from "../components/ui";
import { dur, since } from "../components/format";

// docs/design/30 §2: what reaches the inbox, why it needs you, and the default suggestion.
const TRIGGERS: { trigger: string; why: string }[] = [
  { trigger: "Task over retry limit", why: "the agent cannot go on" },
  { trigger: "Loop over its round limit", why: "may need a human judgement" },
  { trigger: "Result far above the strongest reference or > 95% of bound", why: "measurement or check may be wrong" },
  { trigger: "Same check failing in ≥ 3 tasks", why: "checker bug or systematic defect" },
  { trigger: "Model error large several times in a row", why: "model may miss a factor" },
  { trigger: "Checker crash, GPU or worker down", why: "infrastructure" },
  { trigger: "Budget at 80%", why: "extend or stop" },
  { trigger: "Experiment protocol waiting for approval", why: "frozen once approved" },
  { trigger: "System-code improvement item", why: "a coding agent must change code" },
  { trigger: "New Template or pattern affecting several families", why: "wide impact" },
];

export function Inbox() {
  const items = useApi(api.inbox, []);
  if (!items.data) return <Loading error={items.error} />;
  return <InboxBody items={items.data} />;
}

function InboxBody({ items }: { items: InboxItem[] }) {
  const now = useNow();
  const sorted = [...items].sort((a, b) => urgency(b) - urgency(a) || (a.at ?? "").localeCompare(b.at ?? ""));
  return (
    <div className="grid" style={{ gridTemplateColumns: "minmax(0, 1fr) 340px", alignItems: "start" }}>
      <Panel primary title="Needs you" sub={`${items.length} item${items.length === 1 ? "" : "s"}, most urgent first; each comes with the system's suggestion`}>
        {sorted.length === 0 ? <div className="empty">Nothing needs you right now.</div> : (
          <div className="inbox-list">
            {sorted.map((it) => (
              <div key={it.id} className={`inbox-item u${urgency(it)}`}>
                <div className="urg" />
                <div className="what">
                  <div className="row small muted"><span className="caption" style={{ color: "#3d2fa0" }}>{it.trigger}</span>{it.at && <span>waiting {dur(since(it.at, now))}</span>}</div>
                  <h3>{it.subject}</h3>
                  {it.detail && <div className="small ink2">Last rejection: {it.detail}</div>}
                  <div className="suggest">Suggestion: {it.suggestion}</div>
                  <div className="small muted row" style={{ gap: 8 }}>
                    record <RecordLink id={it.record} />
                    {it.task && <Link to={`/tasks/${it.task}`}>open task →</Link>}
                  </div>
                </div>
                <div className="acts">
                  {it.task ? <TaskActions task={it.task} label={it.subject} small retry={(it.actions ?? []).includes("retry")} /> : it.record ? (
                    <ActionButton op="flag" label="Flag subject" small title={`Flag ${it.subject}`}
                      explain="Marks the subject suspect and asks for re-verification." submit={(r) => act.flag(it.record ?? "", r)} />
                  ) : null}
                </div>
              </div>
            ))}
          </div>
        )}
      </Panel>
      <div className="stack">
        <WiringNote />
        <Panel title="What reaches the inbox" sub="docs/design/30 §2; items are computed by the orchestrator from stored state (D270)">
          <table className="t compact">
            <tbody>{TRIGGERS.map((t) => (
              <tr key={t.trigger}><td>{t.trigger}<div className="small muted">{t.why}</div></td>
                </tr>
            ))}</tbody>
          </table>
        </Panel>
      </div>
    </div>
  );
}

// Items the system cannot continue without (a task over its limit, a crashed checker) first.
const urgency = (it: InboxItem) => (it.task ? 2 : 1);

function WiringNote() {
  const { caps } = useCaps();
  const partial = ["cancel", "retry"].filter((n) => caps[n]?.partial);
  if (partial.length === 0) return null;
  return (
    <div className="notice waiting small">
      <b>What the task controls do not do yet</b>
      {partial.map((n) => <span key={n}>{n}: {caps[n].partial}</span>)}
    </div>
  );
}
