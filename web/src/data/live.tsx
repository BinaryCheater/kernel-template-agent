import { createContext, useContext, useEffect, useRef, useState, type ReactNode } from "react";

// Live updates over SSE (GET /events): every ops change, span start/end or new record bumps
// `version`, at most once per THROTTLE_MS, and every useApi call re-fetches. `now` ticks so
// elapsed times keep moving between events.

const THROTTLE_MS = 2000;
const TICK_MS = 5000;

interface Live {
  version: number;
  now: number;
  connected: boolean;
  lastEvent: number | null;
}

const LiveContext = createContext<Live>({ version: 0, now: Date.now(), connected: false, lastEvent: null });

export function LiveProvider({ children }: { children: ReactNode }) {
  const [version, setVersion] = useState(0);
  const [now, setNow] = useState(Date.now());
  const [connected, setConnected] = useState(false);
  const [lastEvent, setLastEvent] = useState<number | null>(null);
  const pending = useRef<number | null>(null);

  useEffect(() => {
    // ?static=1 skips the event stream (printing, headless screenshots): data loads once.
    if (new URLSearchParams(window.location.search).has("static")) return;
    const es = new EventSource("/api/events");
    const bump = () => {
      setLastEvent(Date.now());
      if (pending.current !== null) return;
      pending.current = window.setTimeout(() => {
        pending.current = null;
        setVersion((v) => v + 1);
      }, THROTTLE_MS);
    };
    es.addEventListener("hello", () => setConnected(true));
    es.addEventListener("ops", bump);
    es.addEventListener("record", bump);
    es.onerror = () => setConnected(false);
    es.onopen = () => setConnected(true);
    const tick = window.setInterval(() => setNow(Date.now()), TICK_MS);
    return () => {
      es.close();
      window.clearInterval(tick);
      if (pending.current !== null) window.clearTimeout(pending.current);
    };
  }, []);

  return <LiveContext.Provider value={{ version, now, connected, lastEvent }}>{children}</LiveContext.Provider>;
}

export const useLive = () => useContext(LiveContext);
export const useNow = () => useContext(LiveContext).now;
