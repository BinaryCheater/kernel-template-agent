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

type NavItem = { to: string; label: string; badge?: "inbox"; also?: string[] };

/** Navigation (WP25, D665): the four views used while watching are always shown; knowledge,
 *  control and studies are groups that open when one of their pages is active or on click. */
const PRIMARY: NavItem[] = [
  { to: "/", label: "Home", also: ["/running"] },
  { to: "/activity", label: "Activity" },
  { to: "/pipeline", label: "Pipeline" },
  { to: "/cases", label: "Cases", also: ["/results", "/tasks"] },
];
const GROUPS: { group: string; hint: string; items: NavItem[] }[] = [
  { group: "Knowledge", hint: "what the system has learned", items: [{ to: "/families", label: "Families" }, { to: "/pool", label: "Pool" },
    { to: "/coverage", label: "Coverage" }, { to: "/model", label: "Cost model" }, { to: "/model/scoreboard", label: "Model scoreboard" }] },
  { group: "Control", hint: "decide, configure, intervene", items: [{ to: "/inbox", label: "Needs you", badge: "inbox" }, { to: "/settings", label: "Settings" },
    { to: "/workflow", label: "Agent workflow and prompts" }, { to: "/agents", label: "Agent skills and profiles" }, { to: "/release", label: "Release stage" },
    { to: "/operations", label: "Actions log" }] },
  { group: "Studies", hint: "experiments about the system itself", items: [{ to: "/experiments", label: "Experiments" }, { to: "/improvement", label: "Improvement" },
    { to: "/mechanisms", label: "Mechanisms" }] },
];

const OPEN_KEY = "kta-nav-open";
function loadOpen(): Record<string, boolean> {
  try {
    return JSON.parse(localStorage.getItem(OPEN_KEY) ?? "{}") as Record<string, boolean>;
  } catch {
    return {};
  }
}

function NavEntry({ it, inbox }: { it: NavItem; inbox?: number }) {
  const loc = useLocation();
  const extra = it.also?.some((p) => loc.pathname.startsWith(p));
  return (
    <NavLink to={it.to} end={it.to === "/" || it.to === "/model"} className={({ isActive }) => (isActive || extra ? "active" : "")}>
      <span>{it.label}</span>
      {it.badge === "inbox" && inbox ? <span className="badge human">{inbox}</span> : null}
    </NavLink>
  );
}

function Nav({ inbox }: { inbox?: number }) {
  const loc = useLocation();
  const [open, setOpen] = useState<Record<string, boolean>>(loadOpen);
  const toggle = (g: string, now: boolean) => {
    const next = { ...open, [g]: !now };
    setOpen(next);
    try {
      localStorage.setItem(OPEN_KEY, JSON.stringify(next));
    } catch {
      // storage unavailable: the choice lasts for this page only
    }
  };
  return (
    <>
      <div className="navgroup primary-nav">{PRIMARY.map((it) => <NavEntry key={it.to} it={it} inbox={inbox} />)}</div>
      {GROUPS.map((g) => {
        const here = g.items.some((it) => matchPath({ path: it.to, end: false }, loc.pathname));
        const isOpen = open[g.group] ?? here;
        const count = g.items.some((it) => it.badge === "inbox") ? inbox : undefined;
        return (
          <div className="navgroup" key={g.group}>
            <button className="nav-toggle" onClick={() => toggle(g.group, isOpen)} title={g.hint}>
              <span className="caption">{isOpen ? "▾" : "▸"} {g.group}</span>
              {!isOpen && count ? <span className="badge human">{count}</span> : null}
            </button>
            {isOpen && g.items.map((it) => <NavEntry key={it.to} it={it} inbox={inbox} />)}
          </div>
        );
      })}
    </>
  );
}

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
          <Nav inbox={active.data?.inbox_count} />
          <div className="foot">Every button that acts writes a HumanAction record (who, when, why). Controls the API cannot do yet are shown disabled with what they need.</div>
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
