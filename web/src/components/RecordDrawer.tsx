import { Link, useParams, useSearchParams } from "react-router";
import { api } from "../data/source";
import { useApi } from "../data/useApi";
import { wordTone } from "./format";
import { AssuranceChip, Chip, Loading } from "./ui";
import { RecordActions } from "./actions";
import { BodyView, RefName, recordName } from "./BodyView";
import { Annotate } from "./Annotate";

/** The same view as a page of its own (`/records/:id`), for keeping open or linking. */
export function RecordPage() {
  const { id = "" } = useParams();
  return <section className="panel primary record-page"><DrawerBody id={id} /></section>;
}

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

function DrawerBody({ id, close }: { id: string; close?: () => void }) {
  const { data: r, error } = useApi(() => api.record(id), [id]);
  return (
    <>
      <header>
        <div>
          <div className="caption">{r?.kind ?? "Record"}</div>
          <h2>{r ? recordName(r) : "…"}</h2>
        </div>
        <span className="row" style={{ gap: 10 }}>
          {r && <Annotate subject={id} label={`${r.kind}: ${recordName(r)}`} compact />}
          {r?.kind === "Template" ? <Link to={`/knowledge/templates/${id}`}>Open page</Link>
            : close && <Link to={`/records/${id}`}>Open as page</Link>}
          {close && <button className="close" onClick={close} aria-label="Close">×</button>}
        </span>
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
              <dt>created</dt><dd className="num">{r.created_at.slice(0, 19).replace("T", " ")}</dd>
              <dt>actor</dt><dd>{r.actor.type} · {r.actor.name}{r.actor.task && <> · task <span className="mono">{r.actor.task}</span></>}</dd>
            </dl>
            {Object.keys(r.refs).length > 0 && (
              <div>
                <div className="caption">Linked records</div>
                <dl className="kv" style={{ marginTop: 6 }}>
                  {Object.entries(r.refs).map(([k, v]) => (
                    <div key={k} style={{ display: "contents" }}>
                      <dt>{k}</dt>
                      <dd className="row wrap" style={{ gap: 6 }}>{(Array.isArray(v) ? v : [v]).map((x) => <RefName key={x} id={x} />)}</dd>
                    </div>
                  ))}
                </dl>
              </div>
            )}
            <BodyView v={r.body} />
            <details className="raw">
              <summary>Raw record</summary>
              <div className="ink2 mono small">{r.id} · schema {r.schema}</div>
              <pre className="json">{JSON.stringify(r.body, null, 2)}</pre>
            </details>
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

