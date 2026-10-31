// Settings → System (WP22): infrastructure of the control plane, separate from the research
// controls. The agent proxy (detected value and source, override, per-provider policy, Test),
// provider CLIs (version, logged in — never credentials), workers (commit vs control plane,
// GPU, Test job), the data directory and the release stage. Writes need the operator token
// and a reason; each becomes a HumanAction set_system.
import { useState } from "react";
import { Link } from "react-router";
import { api } from "../data/source";
import type { PreflightResult, ProviderRow, SystemView, WorkerRow, WorkerTestResult } from "../data/types";
import { useApi } from "../data/useApi";
import { Chip, Loading, Panel, RecordLink } from "../components/ui";
import { hhmm } from "../components/format";
import { TokenField } from "./Settings";

type Draft = {
  override?: string;
  providers: Record<string, { proxy?: string; check_hosts?: string }>;
  workers: Record<string, string>;
};
const EMPTY: Draft = { providers: {}, workers: {} };
const SOURCE_NOTE: Record<string, string> = {
  scutil: "macOS system proxy (scutil --proxy)",
  env: "HTTPS_PROXY / HTTP_PROXY / ALL_PROXY of the control-plane environment",
  default: "nothing detected: Clash's default HTTP port",
  override: "override in the system configuration",
};

/** The changes object POST /system takes, from the draft (only what differs). */
function changesOf(view: SystemView, d: Draft): Record<string, unknown> {
  const out: Record<string, Record<string, unknown>> = {};
  if (d.override !== undefined && (d.override.trim() || null) !== view.config.proxy.override) {
    out.proxy = { override: d.override.trim() || null };
  }
  for (const [name, p] of Object.entries(d.providers)) {
    const cur = view.config.providers[name];
    const patch: Record<string, unknown> = {};
    if (p.proxy && p.proxy !== cur?.proxy) patch.proxy = p.proxy;
    if (p.check_hosts !== undefined) {
      const hosts = p.check_hosts.split(",").map((h) => h.trim()).filter(Boolean);
      if (hosts.join(",") !== (cur?.check_hosts ?? []).join(",")) patch.check_hosts = hosts;
    }
    if (Object.keys(patch).length) (out.providers ??= {})[name] = patch;
  }
  for (const [name, url] of Object.entries(d.workers)) {
    if ((url.trim() || null) !== (view.config.workers[name]?.intake_proxy ?? null)) (out.workers ??= {})[name] = { intake_proxy: url.trim() || null };
  }
  return out;
}

export function SystemSettings() {
  const v = useApi(api.system, []);
  const [draft, setDraft] = useState<Draft>(EMPTY);
  const [reason, setReason] = useState("");
  const [msg, setMsg] = useState<{ ok: boolean; text: string } | null>(null);
  if (!v.data) return <Loading error={v.error} />;
  const view = v.data;
  const changes = changesOf(view, draft);
  const n = Object.keys(changes).length;
  const save = async () => {
    try {
      const out = await api.setSystem(view.config.revision, changes, reason.trim());
      setMsg({ ok: true, text: `revision ${out.revision} written; reload to see it in force` });
      setDraft(EMPTY);
      setReason("");
    } catch (e) {
      setMsg({ ok: false, text: e instanceof Error ? e.message : String(e) });
    }
  };
  return (
    <div className="stack">
      <ProxyPanel view={view} draft={draft} setDraft={setDraft} />
      <div className="row wrap" style={{ gap: 8 }}>
        <span className="small">{n ? `${n} section(s) changed` : "no change"} · revision {view.config.revision}
          {view.config.updated_by && <span className="muted"> (by {view.config.updated_by} {hhmm(view.config.updated_at ?? "")})</span>}</span>
        <input value={reason} onChange={(e) => setReason(e.target.value)} placeholder="reason for this change (required)" style={{ flex: 1, minWidth: 260 }} />
        <TokenField />
        <button className="op primary" disabled={!n || !reason.trim()} onClick={save}>Write system configuration</button>
        {msg && <span className={msg.ok ? "small" : "error small"}>{msg.text}</span>}
      </div>
      <div className="grid" style={{ gridTemplateColumns: "minmax(0, 1fr) minmax(0, 1fr)", alignItems: "start" }}>
        <Providers rows={view.providers} />
        <ControlPlane view={view} />
      </div>
      <Workers rows={view.workers} draft={draft} setDraft={setDraft} />
      <Audit view={view} />
    </div>
  );
}

function ProxyPanel({ view, draft, setDraft }: { view: SystemView; draft: Draft; setDraft: (d: Draft) => void }) {
  const [test, setTest] = useState<Record<string, PreflightResult> | null>(null);
  const [busy, setBusy] = useState(false);
  const [err, setErr] = useState<string | null>(null);
  const p = view.proxy;
  const run = async (provider?: string) => {
    setBusy(true);
    setErr(null);
    try {
      const out = await api.proxyTest(provider);
      setTest({ ...(provider ? test ?? {} : {}), ...out.results });
    } catch (e) {
      setErr(e instanceof Error ? e.message : String(e));
    } finally {
      setBusy(false);
    }
  };
  const inForce = p.in_force;
  return (
    <Panel primary title="Agent proxy"
      sub="Codex and Claude Code ignore the macOS system proxy: the runner injects it. A required provider whose preflight fails is not started (no direct fallback)."
      right={<button className="op" disabled={busy} onClick={() => run()}>{busy ? "Testing…" : "Test all required"}</button>}>
      <div className="grid" style={{ gridTemplateColumns: "minmax(0, 1fr) minmax(0, 1fr)", gap: 16, alignItems: "start" }}>
        <div className="stack" style={{ gap: 8 }}>
          <div><span className="muted small">In force</span>
            <div className="mono" style={{ fontSize: 18 }}>{inForce ? inForce.endpoint.url : "—"}</div>
            {inForce && <div className="small"><Chip tone={inForce.source === "default" ? "waiting" : "ok"}>{inForce.source}</Chip> <span className="muted">{SOURCE_NOTE[inForce.source]}</span></div>}
            {p.error && <div className="error small">{p.error}</div>}
          </div>
          <div className="small"><span className="muted">Detected:</span> <span className="mono">{p.detected.endpoint.url}</span> via <b>{p.detected.source}</b> <span className="muted">({p.detected.detail})</span>
            {p.detected.credentials_dropped && <div className="muted">credentials in the source were dropped; they belong in secrets.env (KTA_PROXY_USERINFO)</div>}
          </div>
          <label className="small stack" style={{ gap: 4 }}>
            <span className="muted">Override (empty = auto-detect). Clash: http://127.0.0.1:7897 or socks5://127.0.0.1:7897. No user:password here.</span>
            <input className="mono" value={draft.override ?? p.override ?? ""} placeholder="auto-detect"
              onChange={(e) => setDraft({ ...draft, override: e.target.value })} />
          </label>
          <div className="small muted">NO_PROXY always holds 127.0.0.1, localhost, ::1 and the KTA_API host{p.no_proxy.length ? ` + ${p.no_proxy.join(", ")}` : ""}.
            Credentials: {p.credentials_from_secrets ? "KTA_PROXY_USERINFO from the agent host's secrets.env" : "none"}.</div>
          {err && <div className="error small">{err}</div>}
        </div>
        <table className="t compact">
          <thead><tr><th>Provider</th><th>Policy</th><th>Preflight hosts</th><th>Last preflight</th><th /></tr></thead>
          <tbody>
            {view.providers.map((r) => (
              <PolicyRow key={r.name} r={r} draft={draft} setDraft={setDraft} test={test?.[r.name]} onTest={() => run(r.name)} busy={busy} />
            ))}
          </tbody>
        </table>
      </div>
    </Panel>
  );
}

function PolicyRow({ r, draft, setDraft, test, onTest, busy }: {
  r: ProviderRow; draft: Draft; setDraft: (d: Draft) => void; test?: PreflightResult; onTest: () => void; busy: boolean;
}) {
  const d = draft.providers[r.name] ?? {};
  const set = (patch: { proxy?: string; check_hosts?: string }) => setDraft({ ...draft, providers: { ...draft.providers, [r.name]: { ...d, ...patch } } });
  const policy = d.proxy ?? r.proxy;
  return (
    <>
      <tr className={d.proxy || d.check_hosts !== undefined ? "hl" : ""}>
        <td><b>{r.name}</b></td>
        <td><select value={policy} onChange={(e) => set({ proxy: e.target.value })}><option>required</option><option>off</option></select></td>
        <td><input className="mono small" style={{ width: "100%" }} disabled={policy === "off"} value={d.check_hosts ?? r.check_hosts.join(", ")} onChange={(e) => set({ check_hosts: e.target.value })} /></td>
        <td className="small">{r.status ? <><Chip tone={r.status.ok ? "ok" : "fail"}>{r.status.ok ? "passed" : "failed"}</Chip> {hhmm(r.status.at)} {r.status.report && <RecordLink id={r.status.report} />}
          {!r.status.ok && <div className="error">{r.status.error} · tasks of this provider are held</div>}</> : <span className="muted">{r.proxy === "off" ? "direct, never proxied" : "not run yet"}</span>}</td>
        <td>{r.proxy === "required" && <button className="op" disabled={busy} onClick={onTest}>Test</button>}</td>
      </tr>
      {test && (
        <tr><td colSpan={5} className="small">
          <Chip tone={test.ok ? "ok" : "fail"}>{test.ok ? "preflight passed" : "preflight failed"}</Chip>{" "}
          port {test.port?.ok ? `ok ${test.port.ms} ms` : <span className="error">{test.port?.error}</span>}
          {test.hosts.map((h) => <span key={h.host} style={{ marginLeft: 10 }}><span className="mono">{h.host}:{h.port}</span> {h.ok ? `${h.tls} ${h.ms} ms` : <span className="error">{h.error}</span>}</span>)}
          {test.error && !test.hosts.length && <span className="error"> {test.error}</span>}
          <div className="muted">run from the control host; a launch re-runs it on the agent host {test.report && <RecordLink id={test.report} />}</div>
        </td></tr>
      )}
    </>
  );
}

function Providers({ rows }: { rows: ProviderRow[] }) {
  return (
    <Panel title="Providers" sub="CLI on the control host and whether credentials are present (status only)">
      <table className="t compact">
        <thead><tr><th>Provider</th><th>CLI version</th><th>Auth</th><th>Logged in</th></tr></thead>
        <tbody>
          {rows.map((r) => (
            <tr key={r.name}>
              <td><b>{r.name}</b>{r.cli.command && <div className="mono small muted">{r.cli.command}</div>}</td>
              <td className="mono small">{r.cli.version ?? <span className="muted">{r.cli.command ? "not installed here" : "not checked"}</span>}</td>
              <td className="small">{r.auth.kind}</td>
              <td className="small">{r.auth.logged_in === null ? <span className="muted">{r.auth.note}</span>
                : <><Chip tone={r.auth.logged_in ? "ok" : "waiting"}>{r.auth.logged_in ? "yes" : "no"}</Chip> <span className="mono muted">{r.auth.where}</span></>}</td>
            </tr>
          ))}
        </tbody>
      </table>
    </Panel>
  );
}

function ControlPlane({ view }: { view: SystemView }) {
  const c = view.control;
  return (
    <Panel title="Control plane" sub="where the store, API and agent runner live">
      <table className="t compact">
        <tbody>
          <tr><td className="muted">Host · env</td><td>{c.host} · <b>{c.env}</b></td></tr>
          <tr><td className="muted">Data directory</td><td className="mono small">{c.data_dir}</td></tr>
          <tr><td className="muted">System configuration</td><td className="mono small">{c.config_file}</td></tr>
          <tr><td className="muted">Commit</td><td className="mono small">{c.commit.slice(0, 12)}</td></tr>
          <tr><td className="muted">Agent CLIs run on</td><td>{view.agents.error ? <span className="error small">{view.agents.error}</span> : <><b>{view.agents.worker}</b> <span className="muted small">(KTA_API {view.agents.api_url || "not set"})</span></>}</td></tr>
          <tr><td className="muted">Release stage</td><td><Link to={view.release.page}><b>{view.release.id}</b> {view.release.name}</Link></td></tr>
        </tbody>
      </table>
    </Panel>
  );
}

function Workers({ rows, draft, setDraft }: { rows: WorkerRow[]; draft: Draft; setDraft: (d: Draft) => void }) {
  const [tests, setTests] = useState<Record<string, WorkerTestResult | "running">>({});
  const run = async (name: string) => {
    setTests((t) => ({ ...t, [name]: "running" }));
    try {
      const out = await api.workerTest(name);
      setTests((t) => ({ ...t, [name]: out }));
    } catch (e) {
      setTests((t) => ({ ...t, [name]: { ok: false, worker: name, ran: false, error: e instanceof Error ? e.message : String(e) } }));
    }
  };
  return (
    <Panel title="Workers" sub="registered commit against the control plane's; Test runs one trivial job through the Executor. The intake proxy (default off) is for git clone and package downloads only">
      <table className="t compact">
        <thead><tr><th>Worker</th><th>Address</th><th>Commit</th><th>GPU</th><th>Health</th><th>Intake proxy</th><th>Test</th></tr></thead>
        <tbody>
          {rows.map((w) => {
            const t = tests[w.name];
            return (
              <tr key={w.name}>
                <td><b>{w.name}</b><div className="small muted">{Object.entries(w.pools).map(([k, n]) => `${k}:${n}`).join(" ")}</div></td>
                <td className="small">{w.transport} · <span className="mono">{w.address}</span>{w.root && <div className="mono muted">{w.root}</div>}</td>
                <td className="small">{w.registered ? <><RecordLink id={w.registered} label={<span className="mono">{w.commit?.slice(0, 10)}</span>} />{" "}
                  <Chip tone={w.commit_match ? "ok" : "fail"}>{w.commit_match ? "matches" : `control ${w.control_commit.slice(0, 10)}`}</Chip></> : <span className="muted">not registered</span>}</td>
                <td className="small">{w.gpu ?? <span className="muted">none</span>}</td>
                <td className="small">{w.healthy === null ? <span className="muted">—</span> : <Chip tone={w.healthy ? "ok" : "fail"}>{w.healthy ? "healthy" : "unhealthy"}</Chip>}</td>
                <td><input className="mono small" style={{ width: 200 }} placeholder="off" value={draft.workers[w.name] ?? w.intake_proxy ?? ""}
                  onChange={(e) => setDraft({ ...draft, workers: { ...draft.workers, [w.name]: e.target.value } })} /></td>
                <td className="small" style={{ minWidth: 200 }}>
                  <button className="op" disabled={t === "running"} onClick={() => run(w.name)}>{t === "running" ? "Running…" : "Test"}</button>
                  {t && t !== "running" && (t.ran
                    ? <div><Chip tone={t.ok ? "ok" : "fail"}>{t.ok ? "job ok" : t.error_kind}</Chip> {t.wall_s?.toFixed(1)} s<div className="mono muted">{t.stdout?.trim()}</div></div>
                    : <div className="error">{t.error}</div>)}
                </td>
              </tr>
            );
          })}
        </tbody>
      </table>
    </Panel>
  );
}

function Audit({ view }: { view: SystemView }) {
  return (
    <Panel title="Changes" sub="every write of the system configuration, newest first (HumanAction set_system)">
      {!view.audit.length && <div className="empty">No change yet: the defaults are in force.</div>}
      <table className="t compact">
        <tbody>
          {view.audit.map((a, i) => (
            <tr key={i}>
              <td className="small">{hhmm(a.at)}</td>
              <td className="small">{a.by}</td>
              <td className="small">{a.reason}</td>
              <td className="mono small">{JSON.stringify((a.params as { changes?: unknown } | undefined)?.changes ?? a.params ?? {})}</td>
            </tr>
          ))}
        </tbody>
      </table>
    </Panel>
  );
}
