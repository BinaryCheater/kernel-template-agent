// Knowledge (WP25, D670, D671): what the research has accumulated. Templates first, as one
// filterable list (level, source, checks) whose rows open a page per Template; then the
// kernel families built from them, the cost model's track record, the pool's other records
// and coverage.
import { useSearchParams } from "react-router";
import { FamilyList } from "./Family";
import { Pool } from "./Pool";
import { Coverage } from "./Coverage";
import { Model } from "./Model";
import { TemplateList } from "./Templates";
import { ReferenceList } from "./Learning";

const TABS = [
  { id: "templates", label: "Templates" },
  { id: "references", label: "Learning from references" },
  { id: "families", label: "Kernel families" },
  { id: "model", label: "Cost model" },
  { id: "pool", label: "Patterns, gaps, references" },
  { id: "coverage", label: "Coverage" },
] as const;

export function Knowledge() {
  const [params, setParams] = useSearchParams();
  const tab = TABS.find((t) => t.id === params.get("tab"))?.id ?? "templates";
  const pick = (id: string) => {
    const next = new URLSearchParams();
    if (id !== "templates") next.set("tab", id);
    setParams(next);
  };
  return (
    <div className="stack">
      <div className="tabs">{TABS.map((t) => <button key={t.id} className={tab === t.id ? "on" : ""} onClick={() => pick(t.id)}>{t.label}</button>)}</div>
      {tab === "templates" && <TemplateList />}
      {tab === "references" && <ReferenceList />}
      {tab === "families" && <FamilyList />}
      {tab === "model" && <Model />}
      {tab === "pool" && <Pool without="templates" />}
      {tab === "coverage" && <Coverage />}
    </div>
  );
}
