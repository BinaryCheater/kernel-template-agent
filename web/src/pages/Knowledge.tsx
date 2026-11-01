// Knowledge (WP25, D670): what the research has accumulated, in one place. The first tab
// lays the learned Templates out on the seven optimisation levels, coloured by how far each
// is checked; the other tabs are the existing detail pages.
import { useSearchParams } from "react-router";
import { api } from "../data/source";
import type { PoolTemplate } from "../data/types";
import { useApi } from "../data/useApi";
import { Loading, useOpenRecord } from "../components/ui";
import { LEVEL_NAMES } from "../components/format";
import { FamilyList } from "./Family";
import { Pool } from "./Pool";
import { Coverage } from "./Coverage";
import { Model } from "./Model";

const TABS = [
  { id: "levels", label: "By level" },
  { id: "families", label: "Kernel families" },
  { id: "templates", label: "Templates" },
  { id: "coverage", label: "Coverage" },
  { id: "model", label: "Cost model" },
] as const;

export function Knowledge() {
  const [params, setParams] = useSearchParams();
  const tab = TABS.find((t) => t.id === params.get("tab"))?.id ?? "levels";
  const pick = (id: string) => {
    const next = new URLSearchParams();
    if (id !== "levels") next.set("tab", id);
    setParams(next);
  };
  return (
    <div className="stack">
      <div className="tabs">{TABS.map((t) => <button key={t.id} className={tab === t.id ? "on" : ""} onClick={() => pick(t.id)}>{t.label}</button>)}</div>
      {tab === "levels" && <ByLevel />}
      {tab === "families" && <FamilyList />}
      {tab === "templates" && <Pool />}
      {tab === "coverage" && <Coverage />}
      {tab === "model" && <Model />}
    </div>
  );
}

const ASSURANCE = ["declared", "tested", "bounded", "proved"] as const;

function ByLevel() {
  const p = useApi(() => api.pool(), []);
  const open = useOpenRecord();
  if (!p.data) return <Loading error={p.error} />;
  const live = p.data.templates.filter((t) => !["deprecated", "rejected"].includes(t.status));
  const at = (lv: number) => live.filter((t) => t.level_to === lv);
  const a = p.data.assurance;
  return (
    <section className="panel primary">
      <header>
        <h2>Templates by optimisation level</h2>
        <span className="right legend">{ASSURANCE.map((x) => <span key={x}><span className={`sw assur-${x}`} />{x} {a[x] ?? 0}</span>)}</span>
      </header>
      <div className="body levels-board">
        {[1, 2, 3, 4, 5, 6, 7].map((lv) => (
          <div key={lv} className="level-col">
            <div className="level-head"><b>{lv}</b> {LEVEL_NAMES[lv]}<span className="count-pill">{at(lv).length}</span></div>
            {at(lv).map((t: PoolTemplate) => (
              <button key={t.id} className={`tpl assur-${t.assurance ?? "declared"}`} onClick={() => open(t.id)} title={t.summary}>
                {t.name}<span className="v">@{t.version}</span>
              </button>
            ))}
          </div>
        ))}
      </div>
    </section>
  );
}
