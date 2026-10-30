import { useEffect, useState } from "react";
import { useLive } from "./live";

export interface Loaded<T> {
  data?: T;
  error?: string;
  status?: number;
}

/** Run an API call when its key changes or live data changes; errors are shown, never swallowed.
 *  The previous data stays on screen while a refresh is in flight. */
export function useApi<T>(call: () => Promise<T>, key: unknown[]): Loaded<T> {
  const { version } = useLive();
  const [state, setState] = useState<Loaded<T>>({});
  useEffect(() => {
    let live = true;
    call().then(
      (data) => live && setState({ data }),
      (e: unknown) => live && setState({ error: e instanceof Error ? e.message : String(e), status: (e as { status?: number }).status }),
    );
    return () => {
      live = false;
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [...key, version]);
  return state;
}
