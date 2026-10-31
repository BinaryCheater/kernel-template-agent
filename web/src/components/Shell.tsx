import { createContext, useContext, useEffect, useState, type ReactNode } from "react";
import { NavLink, matchPath, useLocation } from "react-router";
import { api } from "../data/source";
import { useLive } from "../data/live";
import type { Snapshot } from "../data/types";
import { useApi } from "../data/useApi";
import { hhmm } from "./format";
import { RecordDrawer } from "./RecordDrawer";
import { CapabilitiesProvider } from "./actions";

const STATIC = new URLSearchParams(window.location.search).has("static");

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
  { group: "Understand", items: [{ to: "/cases", label: "Cases" }, { to: "/families", label: "Families" }, { to: "/pool", label: "Pool" },
    { to: "/coverage", label: "Coverage" }, { to: "/model", label: "Model" }, { to: "/model/scoreboard", label: "Scoreboard" }] },
  { group: "Intervene", items: [{ to: "/inbox", label: "Inbox", badge: "inbox" }, { to: "/operations", label: "Operations" },
    { to: "/experiments", label: "Experiments" }, { to: "/improvement", label: "Improvement" }, { to: "/agents", label: "Agents" },
    { to: "/release", label: "Release" }, { to: "/settings", label: "Settings" }] },
];

export function Shell({ pages, children }: { pages: PageMeta[]; children: ReactNode }) {
  const loc = useLocation();
  const meta = pages.find((p) => matchPath({ path: p.path, end: true }, loc.pathname)) ?? pages[0];
  const snaps = useApi(api.snapshots, []);
  const active = useApi(api.active, []);
  const metaApi = useApi(api.meta, []);
  const live = useLive();
  const origins = metaApi.data?.origins ?? {};
  const synthetic = (origins.fixture ?? 0) + (origins.simulated ?? 0);
  const [snapshot, setSnapshot] = useState("");
  useEffect(() => {
    if (!snapshot && snaps.data?.length) setSnapshot(snaps.data[snaps.data.length - 1].id);
  }, [snaps.data, snapshot]);
  useEffect(() => {
    document.title = `${meta.title} · KTA`;
  }, [meta.title]);

  return (
    <SnapshotContext.Provider value={{ snapshots: snaps.data ?? [], snapshot, setSnapshot }}>
     <CapabilitiesProvider>
      <div className="shell">
        <nav className="sidebar">
          <div className="brand"><b>KTA</b><span>Kernel Template Agent</span></div>
          {NAV.map((g) => (
            <div className="navgroup" key={g.group}>
              <div className="caption">{g.group}</div>
              {g.items.map((it) => (
                <NavLink key={it.to} to={it.to} end={it.to === "/" || it.to === "/model"} className={({ isActive }) => (it.later ? "disabled" : isActive ? "active" : "")}
                  title={it.later ? "Planned for WP9b" : undefined}>
                  <span>{it.label}</span>
                  {it.badge === "inbox" && active.data ? <span className="badge human">{active.data.inbox_count}</span> : it.later ? <span className="later">WP9b</span> : null}
                </NavLink>
              ))}
            </div>
          ))}
          <div className="foot">Every action writes a HumanAction record. Controls that wait for WP6/WP8 are shown dashed.</div>
        </nav>
        <div className="main">
          {metaApi.error && (
            <div className="sample-banner down" role="alert"><b>API UNREACHABLE</b><span>{metaApi.error}. Start it with <span className="mono">KTA_ENV=dev KTA_WP=WP9a uv run python -m kta.api</span>.</span></div>
          )}
          {synthetic > 0 && (
            <div className="sample-banner" role="note">
              <b>FIXTURE DATA</b>
              <span>This store ({metaApi.data?.env}) holds {synthetic} synthetic records (origin fixture/simulated, written by <span className="mono">scripts/seed-dev</span>) and {origins.real ?? 0} real ones. Synthetic records are never evidence.</span>
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
            <span className="asof" title={live.lastEvent ? `last event ${hhmm(new Date(live.lastEvent).toISOString(), true)} UTC` : "no event yet"}>
              {STATIC ? <span className="chip idle"><span className="dot" />static view</span>
                : <span className={`chip ${live.connected ? "ok" : "fail"}`}><span className="dot" />{live.connected ? "live" : "disconnected"}</span>}
              {hhmm(new Date(live.now).toISOString())} UTC
            </span>
          </header>
          <div className="content">{children}</div>
        </div>
      </div>
      <RecordDrawer />
     </CapabilitiesProvider>
    </SnapshotContext.Provider>
  );
}
