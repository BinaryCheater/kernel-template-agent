# Test doubles

Replacements for external boundaries (GPU, remote hosts, LLMs, clock) used only by tests.

- Doubles live here and nowhere else; `src/` never imports this directory (gate: "test doubles").
- Every class defined here must be exercised by a contract test in `tests/contract/` that
  also runs against the real implementation (gate: "fake coverage"), so a double cannot be
  looser than the real thing.
- Register doubles with `Registry.register(..., fixture=True)`. The registry refuses
  `fixture=False` from outside `src/kta`, and refuses to select a fixture in `dev` or `main`.
- Do not replace the object under test; only its external boundary.
