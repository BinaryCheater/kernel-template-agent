# 27 The performance model: an architecture-neutral specification

Status: specification set in discussion with the user on 2026-11-03. It describes what the model *is*, independent of any one chip. The programme that builds its first instance on the GB10 is [28](28-gb10-programme.md). Nothing here is implemented yet.

Related: 25 (inventory of the two upstream simulators and the KTA seed), 26 (GB10 calibration method), 01 (thinking model).

### 1. What the model is

> **Performance is the timeline produced when a program — a set of concurrent operation streams with their dependencies and synchronisation — executes on a resource network — a set of hardware resources with capacities, service rules and paths — under fixed issue and dependency semantics.**

The model is *transaction-level*: it tracks when each operation may start, which resources it holds and for how long, and what it waits for. It computes **no data values**. Where a value matters for performance (a loop count, an address, a data-dependent amount of work) it enters as an *input* to the program side, never by executing arithmetic.

This single choice separates it from two neighbouring kinds of tool:

| Kind | Example | Executes values | Speed | New architecture |
| --- | --- | --- | --- | --- |
| Functional cycle simulator | GPGPU-Sim, Accel-Sim, manufacturers' in-house cycle models | yes | hours per kernel | months of work; many internal details are not published |
| Analytical / learned predictor | roofline, PipeWeave | no | microseconds | no mechanism; cannot explain or extrapolate to a new design |
| **This model** | — | **no** | **seconds** | **a new resource-network description** |

And it is the natural level for hardware co-design: it sits *above* RTL (RTL is cycle-exact and slow; this model is its abstraction and is calibrated from it) and *below* the analytical and MIP forms used for search (which are derived from it, §6).

The GB10, a hardware design of our own, or a different accelerator family (for example an Ascend AI Core) are **instances** of the same specification: the same engine, the same program representation, the same analysis outputs, a different resource-network description and a different ISA front end.

### 2. The three parts

```
      PROGRAM                                MACHINE
  operation streams + dependencies +     resource network: units, ports,
  synchronisation + task structure       queues, memories, paths, rules
  (from an ISA front end or a template)  (one description per hardware instance)
                 \                              /
                  \                            /
                   ENGINE  (discrete-event, transaction-level, architecture-neutral)
                            issue → wait → occupy → release → complete
                                      |
                                      v
         ANALYSIS: time · per-resource occupancy · critical path by mechanism ·
                   what-if sensitivity · assumption ledger · analytical projections
```

The engine contains only generic rules. Nothing in it knows about a particular kernel or a particular chip. Overlap, contention and bottlenecks are not modelled; they *emerge* from concurrency, dependencies and shared resources. Consequently the measure of coverage is **completeness of primitives** (§2.3), not the number of kernels tried.

#### 2.1 Program representation

| Element | Content |
| --- | --- |
| Task structure | How many tasks (CTAs, blocks, tiles of a persistent worker); which execution contexts (warps, pipes) each task contains; task-level resource claims (registers, scratch memory) that govern residency |
| Operation stream | Per execution context, an ordered list of operations. Each operation has a *class* (which selects a path through the machine), an *amount* (bytes, elements, MMA shape, sectors, wavefronts), and *dependency tokens* it sets and waits on (register scoreboards, completion groups, barrier phases, memory flags) |
| Control structure | Loops with trip counts (closed-form in the program's parameters when known), branches with taken/fall-through decisions or probabilities, early exits |
| Dynamic facts | Active lanes per operation; address ranges and access patterns per memory operation (enough to derive sectors, bank wavefronts and cache footprint); data-dependent work per task (from actual data or a declared distribution) |
| Provenance | For every dynamic fact: computed from the ISA, supplied by a template parameter, or defaulted — and how many times each default was used |

A front end produces this from an ISA (SASS today; a Ventus ISA later; other families' instruction streams) or **directly from a template** without compiling, which is how a whole search space is covered at low cost. When a value cannot be computed, the front end supplies an explicit approximation and counts it; it never refuses.

#### 2.2 Machine description (the architecture specification)

| Category | What is described | GB10 instance | Ascend-style instance | Own design |
| --- | --- | --- | --- | --- |
| Hierarchy | levels and multiplicities | 48 SMs × 4 sub-partitions | n AI Cores | from the design |
| Execution contexts | what issues operations; issue width; scheduling policy | warps, one scheduler per sub-partition | scalar unit dispatching to pipe queues | from the ISA |
| Compute resources | type, scope (per sub-partition / core / chip), occupancy function, latency, issue interval, supported operation classes and precisions | tensor, FMA, ALU, XU | Cube, Vector, Scalar | from the design |
| Movement engines | throughput, latency, completion mechanism | LSU, cp.async, TMA | MTE1/2/3 | LSU / DMA |
| Memory spaces | capacity, banks, granule; hardware-managed (cache) or program-managed (buffer); replacement rule | registers, shared, L1, L2, DRAM | L0A/B/C, L1, UB, HBM | from the design |
| Paths | the ordered resources an operation class passes through | load: address stage → L1 → L2 slice → DRAM queue → return port | MTE2: HBM → L1 | from the design |
| Synchronisation semantics | how dependency and completion are expressed | scoreboards, barrier, mbarrier | set_flag / wait_flag, barrier | from the ISA |
| Residency rules | which capacities bound concurrency | registers, shared, warp slots | buffer capacity | from the design |
| Parameter provenance | per value: measured / fitted / structural / assumed / design value, with its probe or source | measured | measured or documented | RTL or design value |

A description that fills this table is executable by the engine. Writing three dissimilar architectures into the same table is the test that the abstraction is architecture-neutral; the Ascend column is a structural check in the first phase, not a measured validation.

#### 2.3 Primitive set the engine must support

issue scheduling · register dependencies · compute pipes with occupancy and latency · memory paths (hit/miss, queues, return) · program-managed buffers · asynchronous engines with completion objects · synchronisation objects (barrier, phase barrier, memory flag with polling) · residency and task scheduling onto cores · multi-kernel sequences.

A kernel the engine cannot express is reported as **unsupported with the missing primitive named**; it is never silently approximated and never dropped from a validation set.

### 3. Outputs

1. **Time** — per task, per kernel, per operator (with launch overhead).
2. **Work counts** — per operation class, per resource, in the same units as the hardware counters used to check the front end (instructions per pipe, sectors, wavefronts).
3. **Occupancy** — busy cycles and utilisation per resource instance and per core, by *our* definition (demand ÷ capacity × predicted time). This need not coincide with any profiler's definition.
4. **Critical path by mechanism** — walking back from the end along the constraint that actually bound each step, aggregated per (resource, mechanism): "40 % tensor-pipe queueing, 30 % TMA latency chain, 20 % shared-memory port contention between LDSM and TMA writes". What is not on the path is what overlap hid.
5. **What-if sensitivity** — rerun with a program parameter or a *machine* parameter changed; the latter is the hardware co-design signal.
6. **Assumption ledger** — every default used and how often; every residual term (28 §2, R3) and its share of the critical path.
7. **Analytical projections** — per-resource demand/capacity bounds and the dependency-cycle bound (§6).
8. Optional timeline trace for debugging.

### 4. Granularity and fidelity targets

The model is expected to reach about 95 % on mechanisms that matter and to *stop* there. Hardware is deterministic, so every residual error has a findable cause; but finding it is only worth doing when the error budget says so (28 §2). A speed budget accompanies the accuracy budget: *proposed* ≤ 60 s per kernel on a laptop, with CTA sampling and steady-state fast-forwarding where needed.

### 5. The calibration method is part of the specification

Probes are defined in terms of the resource network, not of an ISA: *saturate resource R*; *dependency latency along path P*; *contention between R1 and R2*; *hand-off delay of synchronisation object S*; *pipeline of depth d between engine E and pipe R*. Each architecture instantiates them as its own kernels. The same probe set runs on real hardware, on the model, and (for our own design) on RTL. The ladder in 28 §3 is how an instance is built from them.

### 6. Three fidelity levels from one description

| Level | Content | Use |
| --- | --- | --- |
| L0 | per-resource demand ÷ capacity (multi-dimensional roofline) | fast screening |
| L1 | L0 + dependency-cycle bound (latency chain ÷ iteration distance from stages / warp groups) + residency and waves | MIP over the search space, with program *and* machine variables |
| L2 | full transaction-level execution | mechanism analysis, overlap, validation, what-if |

L0 and L1 are projections of the same description; the engine spot-checks MIP picks and reveals which constraints actually bind. Below L2, RTL calibrates the description's parameters.

### 7. Position among related work

| Work | What it has | What it lacks for us |
| --- | --- | --- |
| Accel-Sim / GPGPU-Sim | detail | value execution, speed, portability |
| Microarchitecture characterisation studies (e.g. the SM120 report) | mechanisms | no executable, reusable model |
| Roofline / PipeWeave | speed, generality across shapes and GPUs | no mechanism, no explanation, no new designs |
| Timeloop / Accelergy | architecture description + mapping → performance | loop-nest analytics; no instruction streams, no dynamic synchronisation |
| SystemC TLM / SST | transaction-level frameworks | no accelerator modelling specification, no calibration method |

Claimed position: **an architecture-description-driven, transaction-level, value-free performance model with a probe-ladder calibration method; it explains, supports what-if, calibrates downward from RTL and projects upward to MIP.** A short related-work survey precedes the written specification to confirm this claim (Timeloop/Accelergy, SST, Accel-Sim, public Ascend modelling material).
