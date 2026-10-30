import { useSearchParams } from "react-router";
import { api } from "../data/source";
import { useApi } from "../data/useApi";
import { wordTone } from "./format";
import { AssuranceChip, Chip, Loading, RecordLink } from "./ui";
import { RecordActions } from "./actions";

/** Side drawer showing one record (envelope, body, refs, blobs); driven by `?rec=<id>`. */
export function RecordDrawer() {
  const [params, setParams] = useSearchParams();
  const id = params.get("rec");
  const close = () => {
    const next = new URLSearchParams(params);
    next.delete("rec");
    setParams(next);
  };
  if (!id) return null;
  return (
    <>
      <div className="drawer-scrim" onClick={close} />
      <aside className="drawer" role="dialog" aria-label="Record">
        <DrawerBody id={id} close={close} />
      </aside>
    </>
  );
}

function DrawerBody({ id, close }: { id: string; close: () => void }) {
  const { data: r, error } = useApi(() => api.record(id), [id]);
  return (
    <>
      <header>
        <div>
          <div className="caption">Record{r ? ` · ${r.kind}` : ""}</div>
          <h2 className="mono" style={{ fontSize: 12, wordBreak: "break-all" }}>{id}</h2>
        </div>
        <button className="close" onClick={close} aria-label="Close">×</button>
      </header>
      <div className="dbody">
        {!r ? (
          <Loading error={error} />
        ) : (
          <>
            <div className="row wrap">
              {r.status && <Chip tone={wordTone(r.status)}>{r.status}</Chip>}
              <AssuranceChip a={r.assurance} />
              <span className="chip origin" title="origin: only real records can support conclusions">{r.origin}</span>
            </div>
            <RecordActions id={r.id} kind={r.kind} small />
            <dl className="kv">
              <dt>kind</dt><dd>{r.kind} <span className="muted">schema {r.schema}</span></dd>
              <dt>created</dt><dd className="num">{r.created_at}</dd>
              <dt>actor</dt><dd>{r.actor.type} · {r.actor.name}{r.actor.task && <> · task <span className="mono">{r.actor.task}</span></>}</dd>
            </dl>
            {Object.keys(r.refs).length > 0 && (
              <div>
                <div className="caption">refs</div>
                <dl className="kv" style={{ marginTop: 6 }}>
                  {Object.entries(r.refs).map(([k, v]) => (
                    <div key={k} style={{ display: "contents" }}>
                      <dt>{k}</dt>
                      <dd className="row wrap" style={{ gap: 6 }}>{(Array.isArray(v) ? v : [v]).map((x) => <RecordLink key={x} id={x} />)}</dd>
                    </div>
                  ))}
                </dl>
              </div>
            )}
            <div>
              <div className="caption">body</div>
              <pre className="json" style={{ marginTop: 6 }}>{JSON.stringify(r.body, null, 2)}</pre>
            </div>
            {Object.keys(r.blobs).length > 0 && (
              <div>
                <div className="caption">blobs</div>
                <dl className="kv">{Object.entries(r.blobs).map(([k, v]) => <div key={k} style={{ display: "contents" }}><dt>{k}</dt><dd className="mono">{v}</dd></div>)}</dl>
              </div>
            )}
          </>
        )}
      </div>
    </>
  );
}
