# Runtime skill catalogue

Skills that **KTA's worker agents** read while doing tasks T1–T6. The runner copies the
skills a task type needs into the worker workspace's `.agents/skills/` from the batch
root (spec/modules/agents.md "Skills, backup and regenerable batch environments", D321).

These are not the skills used to *build* KTA. Construction skills live in the repository
root's `.agents/skills/` (catalogue: `.agents/skills/README.md`), are read in place by
coding agents, and are never copied into a worker workspace. A runtime skill never refers
to construction skills, the repository layout or `spec/`, and contains method only: no
conclusion about a particular GPU or kernel (docs/design/10 §6).

| Skill | Task types | Author WP | Status |
| --- | --- | --- | --- |
| `use-kta` | all | WP8 | written |
| `find-knowledge` | all | WP8 | written |
| `write-family` | T1, T6 | WP8 | written |
| `decompose-kernel` | T2 | WP10a draft, WP8 finished | written |
| `write-template` | T2, T3 | WP8 | written |
| `diagnose-model` | T4 | WP11 | written |
| `write-microbench` | T5 | WP11 | written |

Which skills a task type receives is fixed by the AgentProfile (`skills`, D309); planned
skills join a new profile version when written. `diagnose-model` and `write-microbench`
are in `codex-v2` (`kta.agents.profile.CODEX_V2`, D407); `codex-v1` is unchanged.

## Versions and backup

- A skill version is the deterministic tar.gz of its directory (sorted entries, zero
  mtime, uid/gid, mode 644), named `<name>@<sha256 of the archive>`.
- Recording a profile version (`kta.agents.profile.write_profile`) stores every bundle as
  a blob and lists it in `AgentProfile.skill_bundles`; `kta agents export-profile <id>
  <dir>` writes them back out.
- A batch root holds the unpacked bundles under `skills/<name>@<hash>/`; workers read
  them from there, never from this directory (D322).

Editing a file here changes the hash: the change reaches workers only through a new
profile version and a new batch.
