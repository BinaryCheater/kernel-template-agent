# Scope

One page for the GB10 performance model: how a kernel runs on the machine, and what the programme has found about that machine. Design: D968 to D970.

| View | Shows | Comes from |
| --- | --- | --- |
| Execution | Machine map, lanes by resource, warp or phase, code, critical path, transport | One replay, `timeline.py` |
| Unit interiors | Every unit opens on the map (⤢): structure, who shares it and who delays it, each kind of traffic through it, measured curves, parameters with status, the cursor's activity | `js/units.js`, `js/units-more.js`, from `machine.yaml`, the engine's code and the trace |
| Inspector | A unit's parameters with source and ±30 % effect; an instruction's path and why it waited; the run's attribution; another run beside it | The trace |
| Mechanisms | Every inventory row: state, rule, probes measured against the replay, the engine's rule code, rejected hypotheses, decisions | `atlas.py`, from the records |
| Coverage | Operation × target, each cell's patterns and probes | The inventory's matrix |

```bash
scripts/scope-export --demo
```

```bash
scripts/scope-export --engine .claude/worktrees/perfmodel-rounds-634f00 "fa_fwd_s1024_h16_d64 kernel"
```

```bash
scripts/scope-export "fa_fwd_s1024_h16_d64 kernel" --sens 0 --set sim.hmma_16816_pipe_clk=8
```

```bash
python3 perfmodel/scope/serve.py
```

Then open http://127.0.0.1:8792. Traces are written to `perfmodel/scope/traces/`, outside Git. A row above 1.5 M issues needs `--window A:B`.

Reachable from elsewhere, user `scope`, password from a file (a reachable server refuses to start without one); gzip each file beside itself first, as `.gz`:

```bash
python3 serve.py --host 0.0.0.0 --port 28792 --password-file /root/.scope-password
```

Keys: space plays, ←/→ step a clock (shift: 100), `f` fits, esc clears. Wheel zooms, drag pans, shift-wheel scrolls lanes, double-click zooms in; drag on the bottom strip picks a range.

## What is real

- Lanes, warp states, critical path, instruction paths and the position of each dot on a wire are the engine's own times. `Run` in the inspector states how the trace compares with the engine's totals.
- A replay is one SM. Other SMs, the L2 and DRAM enter as shares, so the chip view lights one SM, and L2 and DRAM show requests in flight from that SM, not chip traffic.
- Shared memory is not modelled per bank, so no bank activity is drawn.
- Phases are stretches of code between synchronisation points; there is no source map, so they carry no algorithm names.
- The map's structure is placed by hand for the GB10 (`BOX`, `WIRE` in `js/map.js`). Another machine needs its own.
- A unit's interior is a drawing of what the model and the probes know, not of the silicon. Solid nodes are modelled; dashed ones are not modelled, or their cause is not identified. The 32 banks are structure only.
- Which resources and rule code a mechanism shows is matched from its words, and marked so.
