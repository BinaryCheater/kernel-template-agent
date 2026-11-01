// Templates (WP25, D671): every learned refinement step in one list, filtered by level, by
// where it was learned and by how far it is checked; a row opens the Template's own page,
// which renders the body as sections (what it does, knobs, rules, cost, checks) instead of JSON.
import { useState, type ReactNode } from "react";
import { Link, useNavigate, useParams, useSearchParams } from "react-router";
import { api } from "../data/source";
import type { PoolTemplate, TemplateBody, TemplatePage as Page, TemplateProof, TplVar } from "../data/types";
import { useApi } from "../data/useApi";
import { Chip, Loading } from "../components/ui";
import { LEVEL_NAMES, wordTone } from "../components/format";
import { LEVEL_SHORT, sourceWord } from "../components/words";
import { RecordActions } from "../components/actions";

const touches = (lv: number, from: number, to: number) => (from === to ? lv === to : from < lv && lv <= to);

/** The seven levels as cells, the ones this step moves through filled. */
export function LevelSpan({ from, to }: { from: number; to: number }) {
  return (
    <span className="lvspan" title={`${from} ${LEVEL_NAMES[from]} → ${to} ${LEVEL_NAMES[to]}`}>
      {[1, 2, 3, 4, 5, 6, 7].map((lv) => <span key={lv} className={touches(lv, from, to) ? "on" : ""}>{lv}</span>)}
    </span>
  );
}

const hasCounterexample = (p: TemplateProof) =>
  p.obligations.some((o) => o.status === "counterexample") || p.contract.some((c) => c.status === "counterexample");

const CHECKS = [
  { id: "strong", label: "proved or bounded", test: (t: PoolTemplate) => t.assurance === "proved" || t.assurance === "bounded" },
  { id: "tested", label: "tested", test: (t: PoolTemplate) => t.assurance === "tested" },
  { id: "declared", label: "only declared", test: (t: PoolTemplate) => !t.assurance || t.assurance === "declared" },
  { id: "cex", label: "counterexample found", test: (t: PoolTemplate) => hasCounterexample(t.proof) },
] as const;

export function TemplateList() {
  const p = useApi(() => api.pool(), []);
  const [params, setParams] = useSearchParams();
  const nav = useNavigate();
  if (!p.data) return <Loading error={p.error} />;
  const all = p.data.templates.filter((t) => !["deprecated", "rejected"].includes(t.status));
  const lv = Number(params.get("lv")) || 0;
  const src = params.get("src") ?? "";
  const chk = params.get("chk") ?? "";
  const q = (params.get("q") ?? "").toLowerCase();
  const set = (k: string, v: string) => {
    const next = new URLSearchParams(params);
    if (v && params.get(k) !== v) next.set(k, v);
    else next.delete(k);
    setParams(next, { replace: true });
  };
  const byLv = (t: PoolTemplate) => !lv || touches(lv, t.level_from, t.level_to);
  const bySrc = (t: PoolTemplate) => !src || t.sources.includes(src);
  const byChk = (t: PoolTemplate) => !chk || CHECKS.find((c) => c.id === chk)!.test(t);
  const byQ = (t: PoolTemplate) => !q || `${t.name} ${t.summary} ${t.refines}`.toLowerCase().includes(q);
  const shown = all.filter((t) => byLv(t) && bySrc(t) && byChk(t) && byQ(t))
    .sort((a, b) => a.level_to - b.level_to || a.level_from - b.level_from || a.name.localeCompare(b.name));
  const sources = [...new Set(all.flatMap((t) => t.sources))].sort((a, b) => sourceWord(a).localeCompare(sourceWord(b)));
  // Each facet counts what the other facets leave, so a chip says what clicking it would show.
  const n = (f: (t: PoolTemplate) => boolean, skip: string) =>
    all.filter((t) => f(t) && (skip === "lv" || byLv(t)) && (skip === "src" || bySrc(t)) && (skip === "chk" || byChk(t)) && byQ(t)).length;
  return (
    <section className="panel primary">
      <header>
        <h2>Templates</h2><span className="count-pill">{shown.length}{shown.length !== all.length ? ` of ${all.length}` : ""}</span>
        <span className="right"><input className="tsearch" placeholder="Search" value={params.get("q") ?? ""} onChange={(e) => set("q", e.target.value)} /></span>
      </header>
      <div className="body">
        <div className="facets">
          <Facet label="Level">
            {[1, 2, 3, 4, 5, 6, 7].map((l) => {
              const c = n((t) => touches(l, t.level_from, t.level_to), "lv");
              return <button key={l} className={`fchip ${lv === l ? "on" : ""}`} disabled={!c && lv !== l} onClick={() => set("lv", String(l))}><b>{l}</b> {LEVEL_SHORT[l]}<i>{c}</i></button>;
            })}
          </Facet>
          <Facet label="Learned from">
            {sources.map((s) => <button key={s} className={`fchip ${src === s ? "on" : ""}`} onClick={() => set("src", s)}>{sourceWord(s)}<i>{n((t) => t.sources.includes(s), "src")}</i></button>)}
          </Facet>
          <Facet label="Checked">
            {CHECKS.map((c) => <button key={c.id} className={`fchip ${c.id === "cex" ? "bad" : ""} ${chk === c.id ? "on" : ""}`} onClick={() => set("chk", c.id)}>{c.label}<i>{n(c.test, "chk")}</i></button>)}
          </Facet>
        </div>
        <table className="t tlist">
          <thead><tr><th>template</th><th>levels</th><th>learned from</th><th>applies when</th><th>appears in</th><th>checked</th></tr></thead>
          <tbody>{shown.map((t) => {
            const refs = t.used_by.filter((u) => u.kind === "reference");
            const fams = t.used_by.filter((u) => u.kind === "family");
            return (
              <tr key={t.id} className="click" onClick={() => nav(`/knowledge/templates/${t.id}`)}>
                <td><b>{t.name}</b>{t.version > 1 && <span className="ver">v{t.version}</span>}<div className="tsum">{t.summary}</div></td>
                <td><LevelSpan from={t.level_from} to={t.level_to} /></td>
                <td>{t.sources.map(sourceWord).join(", ")}</td>
                <td>{t.scope === "True" ? <span className="unstated">not stated</span> : <code className="expr clip" title={t.scope}>{t.scope}</code>}</td>
                <td title={t.used_by.map((u) => sourceWord(u.name)).join(", ")}>
                  {refs.length > 0 && <div>{refs.length} reference{refs.length > 1 ? "s" : ""}</div>}
                  {fams.length > 0 && <div className="usedk">{fams.length} kernel famil{fams.length > 1 ? "ies" : "y"}</div>}
                  {!t.used_by.length && <span className="unstated">nowhere yet</span>}
                </td>
                <td><Checked a={t.assurance} p={t.proof} /></td>
              </tr>
            );
          })}</tbody>
        </table>
        {!shown.length && <div className="empty">No Template matches.</div>}
      </div>
    </section>
  );
}

const Facet = ({ label, children }: { label: string; children: ReactNode }) => (
  <div className="facet"><span className="facet-l">{label}</span><span className="facet-c">{children}</span></div>
);

function Checked({ a, p }: { a?: string | null; p: TemplateProof }) {
  return (
    <span className="row" style={{ gap: 4 }}>
      <span className={`achip ${a ?? "declared"}`}>{a ?? "declared"}</span>
      {hasCounterexample(p) && <Chip tone="fail">counterexample</Chip>}
    </span>
  );
}

// ---- one Template ----

const CLS_WORDS: Record<string, string> = {
  semantic: "correctness", resource: "resource", implementation: "implementation", performance_assumption: "performance assumption",
};
const COND_WORDS: Record<string, string> = {
  K1: "Rules imply the preconditions",
  K3: "Some setting satisfies the rules",
};

const domain = (d: TplVar["domain"]) => Array.isArray(d)
  ? d.map(String).join(", ")
  : `${d.lo} … ${d.hi}${d.pow2 ? ", powers of 2" : d.step && d.step > 1 ? `, step ${d.step}` : ""}`;

export function TemplatePage() {
  const { id = "" } = useParams();
  const p = useApi(() => api.template(id), [id]);
  if (!p.data) return <Loading error={p.error} />;
  return <TemplateView p={p.data} />;
}

function TemplateView({ p }: { p: Page }) {
  const b = p.body;
  const refs = p.used_by.filter((u) => u.kind === "reference");
  const fams = p.used_by.filter((u) => u.kind === "family");
  return (
    <div className="stack tpage">
      <div className="tp-head">
        <Link to="/knowledge" className="back">← Templates</Link>
        <div className="tp-title">
          <h1>{b.name}</h1>
          {p.versions.length > 1 && <span className="ver">v{b.version}</span>}
          <span className={`achip ${p.assurance ?? "declared"}`}>{p.assurance ?? "declared"}</span>
          {p.status !== "proposed" && <Chip tone={wordTone(p.status)}>{p.status}</Chip>}
          <span className="spacer" />
          <RecordActions id={p.id} kind="Template" name={`${b.name}@${b.version}`} small />
        </div>
        <p className="tp-sum">{b.summary}</p>
        <div className="tp-facts">
          <Fact k="Moves">
            <LevelSpan from={b.level_from} to={b.level_to} />
            <span>{LEVEL_SHORT[b.level_from]} → {LEVEL_SHORT[b.level_to]}</span>
          </Fact>
          <Fact k="Refines"><span className="pat">{b.refines}</span> → {b.produces.map((x) => <span key={x} className="pat">{x}</span>)}</Fact>
          <Fact k="Learned from">{p.sources.map((s) => s.locator?.startsWith("http")
            ? <a key={s.id} href={s.locator} target="_blank" rel="noreferrer">{sourceWord(s.name)}</a>
            : <span key={s.id}>{sourceWord(s.name)}</span>)}</Fact>
          <Fact k="Appears in">
            {!p.used_by.length && <span className="unstated">nowhere yet</span>}
            {fams.map((f) => <Link key={f.id} className="pill fam" to={`/families/${f.id}`}>{f.name}</Link>)}
            {refs.map((r) => <span key={r.id} className="pill">{sourceWord(r.name)}</span>)}
          </Fact>
        </div>
      </div>

      <div className="tp-grid">
        <div className="stack">
          <Sec title="What it does">
            <div className="tp-obj"><span className="pat">{b.object.type}</span> {b.object.description}</div>
            <pre className="prog">{b.program.description}</pre>
            <div className="tp-line"><span className="k">Back to the original</span>{b.backward_map}</div>
            {b.signature && (
              <dl className="kv2">
                <dt>Idea</dt><dd>{b.signature.algorithm}</dd>
                {b.signature.reuse && <><dt>Reuse</dt><dd>{b.signature.reuse}</dd></>}
                {b.signature.storage && <><dt>Storage</dt><dd>{b.signature.storage}</dd></>}
                {b.signature.boundary && <><dt>Edges</dt><dd>{b.signature.boundary}</dd></>}
              </dl>
            )}
          </Sec>

          <Sec title="Knobs" n={b.variables.length}>
            <table className="t">
              <thead><tr><th>name</th><th>values</th><th>meaning</th><th>fixed at</th></tr></thead>
              <tbody>{b.variables.map((v) => (
                <tr key={v.name}>
                  <td><code>{v.name}</code></td>
                  <td className="vals">{domain(v.domain)}{v.unit && <span className="ink2"> {v.unit}</span>}</td>
                  <td>{v.meaning}{v.when && <div className="ink2">only when <code>{v.when}</code></div>}</td>
                  <td>{v.binding === "compile" ? "compile time" : "run time"}{v.role !== "software" && <div className="ink2">{v.role}</div>}</td>
                </tr>
              ))}</tbody>
            </table>
          </Sec>

          <Sec title="Rules" n={b.constraints.length + b.preconditions.length}>
            <table className="t">
              <thead><tr><th>rule</th><th>meaning</th><th>kind</th></tr></thead>
              <tbody>
                {b.constraints.map((c) => (
                  <tr key={c.id}><td><code className="expr">{c.expr}</code></td><td>{c.meaning}</td><td>{CLS_WORDS[c.cls] ?? c.cls}</td></tr>
                ))}
                {b.preconditions.map((c) => (
                  <tr key={c.id} className="pre"><td><code className="expr">{c.expr}</code></td><td>{c.meaning}</td>
                    <td>precondition{c.same_as && <div className="ink2">same as a rule above</div>}</td></tr>
                ))}
              </tbody>
            </table>
          </Sec>

          {b.effects && <Effects e={b.effects} />}
        </div>

        <div className="stack">
          <Sec title="Applies when">
            {b.scope.predicate === "True"
              ? <div className="unstated big">No condition stated: it is offered for every problem of pattern <b>{b.refines}</b>.</div>
              : <code className="expr block">{b.scope.predicate}</code>}
            {b.scope.text && <p className="tp-text">{b.scope.text}</p>}
            <div className="tp-line"><span className="k">Needs hardware</span>
              {b.requires.length ? b.requires.map((r) => <span key={r} className="pill">{r.replace(/_/g, " ")}</span>) : <span className="ink2">nothing special</span>}</div>
            <div className="tp-line"><span className="k">Ran on</span>
              {b.probed_on.length ? b.probed_on.join(", ") : <span className="unstated">no GPU yet</span>}</div>
          </Sec>

          <Checks b={b} p={p.proof} />

          {p.relations.length > 0 && (
            <Sec title="Related templates">
              {p.relations.map((r, i) => (
                <div key={i} className="rel">
                  <span className="rel-t">{r.type.replace(/_/g, " ")}</span>
                  {r.id ? <Link to={`/knowledge/templates/${r.id}`}>{r.target}</Link> : <span>{r.target}</span>}
                  {r.note && <div className="ink2">{r.note}</div>}
                </div>
              ))}
            </Sec>
          )}

          {p.versions.length > 1 && (
            <Sec title="Versions">
              {p.versions.map((v) => (
                <div key={v.id} className="rel">
                  {v.id === p.id ? <b>v{v.version}</b> : <Link to={`/knowledge/templates/${v.id}`}>v{v.version}</Link>}
                  <span className="ink2"> {v.created_at.slice(0, 10)}</span>
                </div>
              ))}
            </Sec>
          )}

          <details className="raw">
            <summary>Raw record</summary>
            <div className="ink2 mono small">{p.id}</div>
            <pre className="json">{JSON.stringify(b, null, 2)}</pre>
          </details>
        </div>
      </div>
    </div>
  );
}

const Fact = ({ k, children }: { k: string; children: ReactNode }) => (
  <div className="fact"><span className="k">{k}</span><span className="v">{children}</span></div>
);

function Sec({ title, n, children }: { title: string; n?: number; children: ReactNode }) {
  return (
    <section className="panel tsec">
      <header><h2>{title}</h2>{n != null && <span className="count-pill">{n}</span>}</header>
      <div className="body">{children}</div>
    </section>
  );
}

function Effects({ e }: { e: NonNullable<TemplateBody["effects"]> }) {
  return (
    <Sec title="Cost it declares" n={e.transactions.length}>
      <table className="t">
        <thead><tr><th>work</th><th>how many</th><th>each</th></tr></thead>
        <tbody>{e.transactions.map((t) => (
          <tr key={t.id}>
            <td><b>{t.kind.replace(/_/g, " ")}</b><div className="ink2">{t.pattern}{t.src || t.dst ? ` · ${t.src ?? "?"} → ${t.dst ?? "?"}` : ""}</div>
              {t.when && <div className="ink2">only when <code>{t.when}</code></div>}</td>
            <td><code className="expr">{t.count}</code></td>
            <td>{Object.entries(t.amounts).map(([k, v]) => <div key={k}>{k.replace(/_/g, " ")} <code>{v}</code></div>)}</td>
          </tr>
        ))}</tbody>
      </table>
      {e.storage.map((s) => (
        <div key={s.id} className="tp-line"><span className="k">Holds in {s.level}</span><code className="expr">{s.bytes}</code> bytes per {s.scope ?? "cta"}{s.when && <> when <code>{s.when}</code></>}</div>
      ))}
      {e.overlap.map((o, i) => (
        <div key={i} className="tp-line"><span className="k">Overlaps</span>
          {o.between.map((x) => x.replace(/^pattern:|^kind:/, "")).join(" with ")}{o.between.length === 1 ? " with everything else" : ""}
          {o.when === "False" ? <b> never (serial)</b> : o.when ? <> while <code>{o.when}</code></> : " always"}</div>
      ))}
    </Sec>
  );
}

function Checks({ b, p }: { b: TemplateBody; p: TemplateProof }) {
  const [open, setOpen] = useState<string | null>(null);
  const statement = Object.fromEntries(b.obligations.map((o) => [o.name, o.statement]));
  const rows = [
    ...p.contract.map((c) => ({ key: `${c.condition}-${c.name}`, what: COND_WORDS[c.condition] ?? c.condition, detail: c.name === "*" ? "" : c.name, status: c.status, reason: c.reason, domain: c.domain })),
    ...p.obligations.map((o) => ({ key: o.name, what: o.name.replace(/_/g, " "), detail: statement[o.name] ?? "", status: o.status, reason: o.reason, domain: o.domain, method: o.method })),
  ];
  return (
    <Sec title="Checks">
      {!p.summary && <div className="unstated big">The checker has not run on this Template.</div>}
      <div className="checks">{rows.map((r) => (
        <div key={r.key} className={`chk ${r.status.replace(/ /g, "-")}`} onClick={() => setOpen(open === r.key ? null : r.key)}>
          <span className="chk-s"><Chip tone={wordTone(r.status)}>{r.status}</Chip></span>
          <span className="chk-w"><b>{r.what}</b>{r.detail && <div>{r.detail}</div>}
            {r.reason && <div className="chk-r">{r.reason}</div>}
            {open === r.key && r.domain && <div className="ink2">checked over: {r.domain}</div>}
          </span>
        </div>
      ))}</div>
      {b.numeric_risk && <div className="tp-line"><span className="k">Numeric risk</span>{b.numeric_risk}</div>}
    </Sec>
  );
}
