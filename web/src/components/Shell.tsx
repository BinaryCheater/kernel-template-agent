import { createContext, useContext, useEffect, useState, type ReactNode } from "react";
import { NavLink, matchPath, useLocation } from "react-router";
import { USING_SAMPLE, api, nowMs } from "../data/source";
import type { Snapshot } from "../data/types";
import { useApi } from "../data/useApi";
import { hhmm } from "./format";
import { RecordDrawer } from "./RecordDrawer";

export interface PageMeta {
  path: string;
  title: string;
  question: string;
}

interface SnapCtx {
  snapshots: Snapshot[];
  snapshot: string;
  setSnapshot: (id: string) => void;
}
const SnapshotContext = createContext<SnapCtx>({ snapshots: [], snapshot: "", setSnapshot: () => {} });
export const useSnapshot = () => useContext(SnapshotContext);

const NAV: { group: string; items: { to: string; label: string; later?: boolean; badge?: "inbox" }[] }[] = [
  { group: "Watch", items: [{ to: "/", label: "Overview" }, { to: "/running", label: "Running" }, { to: "/pipeline", label: "Pipeline" }] },
  { group: "Understand", items: [{ to: "/cases", label: "Cases" }, { to: "/families", label: "Families" }, { to: "/coverage", label: "Coverage" }] },
  { group: "Intervene", items: [{ to: "/inbox", label: "Inbox", later: true, badge: "inbox" }, { to: "/pool", label: "Pool", later: true },
    { to: "/model", label: "Model", later: true }, { to: "/experiments", label: "Experiments", later: true }] },
];

export function Shell({ pages, children }: { pages: PageMeta[]; children: ReactNode }) {
  const loc = useLocation();
  const meta = pages.find((p) => matchPath({ path: p.path, end: true }, loc.pathname)) ?? pages[0];
  const snaps = useApi(api.snapshots, []);
  const active = useApi(api.active, []);
  const [snapshot, setSnapshot] = useState("");
  useEffect(() => {
    if (!snapshot && snaps.data?.length) setSnapshot(snaps.data[snaps.data.length - 1].id);
  }, [snaps.data, snapshot]);
  useEffect(() => {
    document.title = `${meta.title} · KTA`;
  }, [meta.title]);

  return (
    <SnapshotContext.Provider value={{ snapshots: snaps.data ?? [], snapshot, setSnapshot }}>
      <div className="shell">
        <nav className="sidebar">
          <div className="brand"><b>KTA</b><span>Kernel Template Agent</span></div>
          {NAV.map((g) => (
            <div className="navgroup" key={g.group}>
              <div className="caption">{g.group}</div>
              {g.items.map((it) => (
                <NavLink key={it.to} to={it.to} end={it.to === "/"} className={({ isActive }) => (it.later ? "disabled" : isActive ? "active" : "")}
                  title={it.later ? "Planned for WP9b" : undefined}>
                  <span>{it.label}</span>
                  {it.badge === "inbox" && active.data ? <span className="badge human">{active.data.inbox_count}</span> : it.later ? <span className="later">WP9b</span> : null}
                </NavLink>
              ))}
            </div>
          ))}
          <div className="foot">Read-only view (WP9a). Actions arrive in WP9b.</div>
        </nav>
        <div className="main">
          {USING_SAMPLE && (
            <div className="sample-banner" role="note">
              <b>SAMPLE DATA</b>
              <span>Everything on this page comes from a hand-written sample in <span className="mono">web/src/sample/</span> (origin = fixture). No real runs are shown until the store and API are connected (D95).</span>
            </div>
          )}
          <header className="topbar">
            <div className="title">
              <h1>{meta.title}</h1>
              <p>{meta.question}</p>
            </div>
            <div className="spacer" />
            <label>
              Pool snapshot
              <select value={snapshot} onChange={(e) => setSnapshot(e.target.value)}>
                {(snaps.data ?? []).map((s) => <option key={s.id} value={s.id}>{s.label}</option>)}
              </select>
            </label>
            <span className="asof" title="Time of the data shown">
              <span className="chip idle"><span className="dot" />{USING_SAMPLE ? "static sample" : "live"}</span>
              as of {hhmm(new Date(nowMs()).toISOString(), true)} UTC
            </span>
          </header>
          <div className="content">{children}</div>
        </div>
      </div>
      <RecordDrawer />
    </SnapshotContext.Provider>
  );
}
