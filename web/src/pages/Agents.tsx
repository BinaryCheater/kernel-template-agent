import { api } from "../data/source";
import type { AgentsView, BatchEnv, ProfileRow, RuntimeSkill } from "../data/types";
import { useApi } from "../data/useApi";
import { ActionButton, RecordActions } from "../components/actions";
import { Chip, Loading, Panel, RecordLink } from "../components/ui";
import { hhmm, wordTone } from "../components/format";

const TYPES = ["T1", "T2", "T3", "T4", "T5", "T6"];

export function Agents() {
  const a = useApi(api.agents, []);
  if (!a.data) return <Loading error={a.error} />;
  return <AgentsBody a={a.data} />;
}

function AgentsBody({ a }: { a: AgentsView }) {
  const current = a.profiles.find((p) => p.current) ?? null;
  return (
    <div className="stack">
      <div className="intro">
        <p>
          A worker agent is controlled by three things only: the context it is given, its loop policy and its tools. All three are in a versioned
          <b> AgentProfile</b> record; <span className="mono">current_profile</span> points at the one new runs use, and only you can move it.
          Each batch of runs gets an isolated environment generated from one profile and one code commit, locked by <span className="mono">env.lock.json</span>.
        </p>
        <div className="intro-now">
          <span><b>{a.profiles.length}</b> profile versions</span>
          <span>current: {current ? <><b>{current.name} v{current.version}</b> <RecordLink id={current.id} /></> : <b className="warn-text">none</b>}</span>
          <span><b>{a.skills.skills.length}</b> runtime skills{a.skills.planned.length ? `, ${a.skills.planned.length} planned` : ""}</span>
          <span><b>{a.batches.length}</b> batch environments</span>
        </div>
      </div>

      <Panel primary title="Runtime skills × task types" sub="which skills each task type's worker receives in its workspace (.agents/skills/), under the current profile">
        <SkillMatrix skills={[...a.skills.skills, ...a.skills.absent.map((x) => ({ ...x, absent: true }))]} current={current} />
        <div className="skill-grid" style={{ marginTop: 12 }}>
          {a.skills.skills.map((s) => <SkillCard key={s.name} s={s} />)}
          {a.skills.planned.map((r) => (
            <div key={r.skill} className="skill-card planned">
              <h3>{(r.skill ?? "").replaceAll("`", "")}</h3>
              <div className="small ink2">{Object.entries(r).filter(([k]) => k !== "skill").map(([k, v]) => `${k}: ${v}`).join(" · ")}</div>
              <div className="small muted">planned in the catalogue; no SKILL.md yet</div>
            </div>
          ))}
        </div>
        {!a.skills.has_catalogue && <div className="small muted" style={{ marginTop: 8 }}>No catalogue table in <span className="mono">src/kta/agents/skills/README.md</span> yet (D321, WP8): the list above is the skill directories found.</div>}
      </Panel>

      <div className="grid" style={{ gridTemplateColumns: "minmax(0, 1.2fr) minmax(0, 1fr)", alignItems: "start" }}>
        <Panel title="Profile versions" sub="newest last; the filled node is current_profile">
          {a.profiles.length === 0 ? (
            <div className="empty-state"><b>No AgentProfile record in this store.</b>
              WP8 writes profile versions (<span className="mono">kta.agents.profile.write_profile</span>). Until one exists, tasks carry the placeholder profile
              and no worker can start.</div>
          ) : a.profiles.map((p) => <ProfileEntry key={p.id} p={p} />)}
        </Panel>
        <div className="stack">
          <Panel title="Pointer history" sub="every move of current_profile (ops log) and your approvals">
            {a.history.length === 0 && a.approvals.length === 0 ? <div className="empty">The pointer has never been set.</div> : (
              <table className="t compact"><tbody>
                {a.history.map((h, i) => <tr key={i}><td className="num small">{h.at.slice(5, 10)} {hhmm(h.at)}</td><td className="small">{h.from ? <RecordLink id={h.from} /> : "none"} → {h.to ? <RecordLink id={h.to} /> : "none"}</td></tr>)}
                {a.approvals.map((x) => <tr key={x.id}><td className="num small">{x.at.slice(5, 10)} {hhmm(x.at)}</td><td className="small">approved by {x.by}: {x.reason || "—"} <RecordLink id={x.id} /></td></tr>)}
              </tbody></table>
            )}
          </Panel>
          <Panel title="Batch environments" sub={a.batches_root ? <span className="mono">{a.batches_root}</span> : "no data directory"}>
            {a.batches.length === 0 ? (
              <div className="empty-state"><b>No batch environment built yet.</b>
                <span><span className="mono">kta agents env build &lt;batch&gt; --profile &lt;id&gt;</span> (WP8, D322) creates one; every run then starts with
                <span className="mono"> kta agents env verify</span>.</span></div>
            ) : a.batches.map((b) => <BatchCard key={b.batch} b={b} />)}
          </Panel>
        </div>
      </div>
    </div>
  );
}

function SkillMatrix({ skills, current }: { skills: { name: string; used_by: string[]; absent?: boolean }[]; current: ProfileRow | null }) {
  if (!current) return <div className="small muted">No current profile, so no skill is assigned to a task type yet. Skills found in the source tree are listed below.</div>;
  return (
    <table className="matrix">
      <thead><tr><th />{TYPES.map((t) => <th key={t}>{t}</th>)}</tr></thead>
      <tbody>{skills.map((s) => (
        <tr key={s.name}><th>{s.name}{s.absent && <span className="warn-text" title="named by the current profile; its bundle is stored as a blob, but there is no source directory in this tree"> · no source</span>}</th>{TYPES.map((t) => <td key={t} className={s.used_by.includes(t) ? "on" : ""}>{s.used_by.includes(t) ? "●" : ""}</td>)}</tr>
      ))}</tbody>
    </table>
  );
}

function SkillCard({ s }: { s: RuntimeSkill }) {
  return (
    <div className="skill-card">
      <h3>{s.name}</h3>
      <div className="small ink2 ellipsis2" title={s.description}>{s.description || <span className="muted">no description</span>}</div>
      <div className="ttype-row">{TYPES.map((t) => <span key={t} className={`ttype ${s.used_by.includes(t) ? "on" : ""}`}>{t}</span>)}</div>
      <div className="small muted">{s.lines} lines · {s.files.length} file(s) · sha {s.sha256.slice(0, 10)}</div>
      {s.catalogue && <div className="small muted">{Object.entries(s.catalogue).filter(([k]) => k !== "skill").map(([k, v]) => `${k}: ${v}`).join(" · ")}</div>}
    </div>
  );
}

function ProfileEntry({ p }: { p: ProfileRow }) {
  return (
    <div className="profile-line" style={{ marginBottom: 12 }}>
      <span className={`profile-node ${p.current ? "current" : ""}`} />
      <div className="stack" style={{ gap: 4 }}>
        <div className="row wrap" style={{ gap: 8 }}>
          <b>{p.name} v{p.version}</b>
          {p.current && <Chip tone="human">current</Chip>}
          <Chip tone={wordTone(p.status)}>{p.status}</Chip>
          {p.origin !== "real" && <span className="chip origin">{p.origin}</span>}
          <RecordLink id={p.id} />
          <span className="small muted">{p.created_at.slice(0, 10)}</span>
        </div>
        <div className="small ink2">{p.provider} {p.cli_version} · {p.model}{p.reasoning_effort ? ` (${p.reasoning_effort})` : ""} · {p.auth} · isolation {p.isolation}</div>
        <div className="small mono muted">{p.cli_flags.join(" ")}</div>
        {p.changed && <div className="small">changed from the previous version: {p.changed.length ? p.changed.map((c) => <span key={c} className="tag" style={{ marginRight: 4 }}>{c}</span>) : <span className="muted">nothing</span>}</div>}
        <details><summary>policies, templates and skill bundles</summary>
          <dl className="kv small">
            <dt>loop policy</dt><dd className="mono">{JSON.stringify(p.loop_policy)}</dd>
            <dt>context policy</dt><dd className="mono">{JSON.stringify(p.context_policy)}</dd>
            <dt>commands</dt><dd>{Object.entries(p.allowed_commands).map(([t, n]) => `${t}: ${n}`).join(" · ")}</dd>
            {Object.entries(p.templates).map(([k, v]) => <div key={k} style={{ display: "contents" }}><dt>{k}</dt><dd className="mono">{v?.slice(0, 16)}</dd></div>)}
            {Object.entries(p.skill_bundles).map(([k, v]) => <div key={k} style={{ display: "contents" }}><dt>{k.split("@")[0]}</dt><dd className="mono">{k.split("@")[1]?.slice(0, 12)} → blob {v.slice(0, 12)}</dd></div>)}
          </dl>
        </details>
        {!p.current && <RecordActions id={p.id} kind="AgentProfile" name={`${p.name} v${p.version}`} small />}
      </div>
    </div>
  );
}

function BatchCard({ b }: { b: BatchEnv }) {
  const lock = b.lock ?? {};
  return (
    <div style={{ borderBottom: "1px solid #efeee9", padding: "8px 0" }}>
      <div className="row wrap" style={{ gap: 8 }}>
        <b className="mono">{b.batch}</b>
        {b.verify.state === "not_supported" ? <Chip tone="idle">verify not wired</Chip> : <Chip tone={wordTone(b.verify.state)}>{b.verify.state}</Chip>}
        <span style={{ marginLeft: "auto" }}>
          <ActionButton op="env-verify" label="Verify now" small title={`Verify batch ${b.batch}`}
            explain="Regenerates the template parts from the profile and compares them with env.lock.json. Any difference refuses new runs."
            submit={() => api.verifyBatch(b.batch)} />
        </span>
      </div>
      {b.lock ? (
        <dl className="kv small" style={{ marginTop: 4 }}>
          {Object.entries(lock).map(([k, v]) => <div key={k} style={{ display: "contents" }}><dt>{k}</dt><dd className="mono">{typeof v === "string" && v.length > 30 ? <RecordLink id={v} label={v.slice(0, 16)} /> : JSON.stringify(v)}</dd></div>)}
          <dt>locked files</dt><dd>{b.files ?? "—"}</dd>
          <dt>lock sha256</dt><dd className="mono">{b.lock_sha256?.slice(0, 16)}</dd>
          <dt>work units</dt><dd>{b.work_units.length ? b.work_units.join(", ") : <span className="muted">none yet</span>}</dd>
        </dl>
      ) : <div className="small warn-text">No env.lock.json: this directory is not a complete batch environment.</div>}
      {b.verify.needs && <div className="small muted">verify needs {b.verify.needs}</div>}
    </div>
  );
}
