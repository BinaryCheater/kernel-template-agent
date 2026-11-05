# Controlled source-attention replay

Scientific settings and execution order come from the controlled-replay campaign
manifest. A recorded native session is allowed only after P1–P7 content-bound
receipts and immutable inputs have been sealed. Earlier exploratory runs are separate.

`seal_campaign(design_path, root, receipts, inventory)` in `seal.py` creates
`root/seal.json`. Receipts map P1 through P7 to JSON files with `stage`, `artifacts`
(`path`, `sha256`) and passing `checks` (`name`, `status`, artifact `evidence`).
The inventory keys are `source`, `evaluator`, `packages`, `common_prompt`, `subset18`,
`library`, `cache_settings`, `settings`, `schedule` and `question_chain`. Source may
be the retained exact source/build inventory. The question chain binds the declared
question, exact original source, both outputs, comparator, arms, schedule and actual
P1–P6 receipt identities. P7 is the independent review of those linked receipts.

Sealing checks scientific settings against the repository design, all artifact
bytes, the 18/320 rule treatments, public common files/prompts, P/S shared bytes and
nine initial workspaces against the arm prototypes. It rejects symlinks including
symlink directories and ancestors. `verify_seal(root)` rechecks the chain; fresh
launches additionally verify that slot's pristine workspace. A readiness boolean
never substitutes for a receipt. Changing frozen code or inputs requires a new
protocol revision and new seal. The controlled runner never launches from the old
exploratory launch command.

`session.run(root, slot, resume=False)` executes through the existing Codex adapter,
with one implementation process at a time and the sealed slot order. Fresh native
homes and declared empty local compiler-cache roots are checked before launch.
Only provider connection and native credential links are inherited. The cache
manifest binds the verified remote roots and installed TileLang cache effect;
parent preparation supplies the actual remote empty-root inventory. Public tools
receive `KTA_SOURCE_REPLAY_ROOT`, `KTA_SOURCE_REPLAY_SLOT` and `SOURCE_REPLAY_CLI`.
Full-access source/read isolation remains instruction-only and must be audited.

The campaign Executor job replaces the exploratory fixed wall timeout with a live
active-time watchdog. It records `remaining-active.json` and `native-session.json`
every 250 ms. Active time is elapsed wall minus verified GPU queue, plus retained
active time from an interrupted prior turn; evaluator work is never added again.
The queue ledger admits one synchronous request at a time, deduplicates retries by
submission identity, and uses locally observed order while retaining raw remote
queue timestamps. Open waits exclude only confirmed queued/waiting heartbeats;
polling lag is conservatively charged active. Closed waits use the conservative
actual acquisition duration, including the heartbeat/acquisition race. Idle uses
the same active cap, so it adds no earlier scientific stopping threshold.

At a budget stop the watchdog sends native SIGINT, preserves the transcript/home,
and records closure. Manual interruption preserves a resume point without closing
the slot. Resuming uses the same native session and remaining active budget, and
refuses an unresolved prior evaluator request. `model-closed.json` permits the fixed
coordinator-only hidden validation and confirmation stage. Queue handoff to the
worker starts its timeout after lock acquisition with the remaining active budget.

Raw native transcripts, per-turn archives, prompts, stdout/stderr, manifests,
AgentRun records and spans remain retained. Native total usage is kept once per
session; per-turn differences prevent resumed-token double counting. Cost remains
null when the native provider does not report billing. Tool-duration reconstruction
is explicitly unavailable in AgentRun; original tool events and timestamps remain
in the native transcript for later analysis.

P6 uses a real non-target native start/resume smoke and installed multi-output graph
and cache-effect evidence. Watchdog tests additionally launch a native-boundary test
process that waits longer than its wall/idle cap while verified GPU queue is open,
and prove an active-budget SIGINT preserves its native state. These boundary tests
call no model or GPU. Paid target sessions remain disabled until the complete seal.

The installed native CLI enables child agents by default. Campaign argv explicitly
sets both `features.multi_agent=false` and `features.multi_agent_v2=false`; a read-only
feature-resolution receipt verifies this. Common task instructions also prohibit
starting another model/CLI/API or delegating work outside the recorded session.
Native shell access remains full-access, so the transcript audit checks that rule.
The non-target smoke used no tools and created no child sessions.
