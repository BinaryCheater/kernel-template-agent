"Replay SM120 striped cursors, register capture and two-phase order protocol."

from __future__ import annotations

import ast
import copy
import hashlib
import itertools
import json
import sys
from collections import deque
from pathlib import Path
from types import SimpleNamespace

import yaml
from frontend_contract import DTYPES, Symbols
from moe_dense import ScalarSource

from kta.core.schemas.knowledge import Chain, Template

STUDY = Path(__file__).resolve().parents[1]
PREFIX = "examples/python/CuTeDSL/cute/blackwell_geforce/kernel/blockscaled_gemm/"


def main() -> None:
    manifest = json.loads((STUDY / "inputs/sources.json").read_text())
    root = Path(manifest["tree"])
    names = [
        "dense_blockscaled_gemm_persistent_cooperative.py",
        "dense_blockscaled_gemm_persistent_pingpong.py",
    ]
    trees = {}
    for path in [*(PREFIX + name for name in names), "python/CuTeDSL/cutlass/pipeline/sm90.py"]:
        data = (root / path).read_bytes()
        entry = next(x for x in manifest["files"] if x["path"] == path)
        assert hashlib.sha256(data).hexdigest() == entry["sha256"]
        trees[Path(path).name] = ast.parse(data.decode())
    classes = {
        name: next(x for x in trees[name].body if isinstance(x, ast.ClassDef)) for name in names
    }
    ping = classes[names[1]]
    jump = copy.deepcopy(
        next(x for x in ping.body if isinstance(x, ast.FunctionDef) and x.name == "advance")
    )
    ns = {}
    exec(
        compile(
            ast.fix_missing_locations(ScalarSource().visit(ast.Module([jump], []))),
            "retained-stage-jump",
            "exec",
        ),
        ns,
    )
    advance = ns["advance"]
    jumps = 0
    for stages in range(1, 17):
        for index, phase, jump_by in itertools.product(
            range(stages),
            (0, 1),
            sorted({0, 1, stages - 1, stages, stages + 1, 3 * stages + 2, 511}),
        ):
            state = SimpleNamespace(stages=stages, _index=index, _phase=phase, _count=7)
            advance(None, state, jump_by)
            assert state._index == (index + jump_by) % stages
            assert state._phase == phase ^ (((index + jump_by) // stages) % 2)
            assert state._count == 7 + jump_by
            jumps += 1
    stripes = 0
    for stages, steps, items in itertools.product(range(1, 9), range(1, 10), range(18)):
        consumed = set()
        for group in (0, 1):
            state = SimpleNamespace(stages=stages, _index=0, _phase=0, _count=0)
            advance(None, state, group * steps)
            for item in range(group, items, 2):
                state._count = 0  # source resets only the item-local count
                for k in range(steps):
                    q = item * steps + k
                    assert (state._index, state._phase) == (q % stages, (q // stages) % 2)
                    assert (item, k) not in consumed
                    consumed.add((item, k))
                    advance(None, state, 1)
                advance(None, state, steps)  # no wait/release for the other owner
        assert consumed == set(itertools.product(range(items), range(steps)))
        stripes += 1
    assert (3 // 2) % 2 == 1  # phase reset would misidentify group1's first K3/S2 generation

    # A corrected four-slot abstract order protocol; no native barriers are replaced.
    order_states = overlap_branches = 0
    for items in range(13):
        owned = [len(range(g, items, 2)) for g in (0, 1)]
        queue = deque([(0, 0, (0, 0, 0, 0))])
        seen = set()
        while queue:
            a, b, barriers = queue.popleft()
            key = (a, b, barriers)
            if key in seen:
                continue
            seen.add(key)
            counts = [a, b]
            enabled = []
            for group in (0, 1):
                if counts[group] == 2 * owned[group]:
                    continue
                depth_index = counts[group] % 2
                phase = int(group == 0) ^ ((counts[group] // 2) % 2)
                if barriers[depth_index * 2 + group] != phase:
                    enabled.append(group)
            if not enabled:
                assert counts == [2 * n for n in owned]
            if len(enabled) == 2:
                overlap_branches += 1
            for group in enabled:
                item = group + 2 * (counts[group] // 2)
                stage = counts[group] % 2
                completed = (
                    sum((x + 1) // 2 for x in counts) if stage == 0 else sum(x // 2 for x in counts)
                )
                assert item == completed  # MMA and epilogue each remain globally ordered
                updated = list(barriers)
                updated[stage * 2 + (group + 1) % 2] ^= 1
                new_counts = counts[:]
                new_counts[group] += 1
                queue.append((*new_counts, tuple(updated)))
        order_states += len(seen)
    assert overlap_branches > 0
    order = next(
        x
        for x in trees["sm90.py"].body
        if isinstance(x, ast.ClassDef) and x.name == "PipelineOrder"
    )
    create = next(x for x in order.body if isinstance(x, ast.FunctionDef) and x.name == "create")
    assert "num_stages = depth * length" in ast.unparse(create)
    get_index = next(
        x
        for x in order.body
        if isinstance(x, ast.FunctionDef) and x.name == "get_barrier_for_current_stage_idx"
    )
    assert "state.index * self.length + group_id" in ast.unparse(get_index)
    allocation = next(
        x
        for x in ast.walk(ping)
        if isinstance(x, ast.AnnAssign)
        and ast.unparse(x.target) == "math_wg_order_barrier_array_ptr"
    )
    declared = ast.literal_eval(allocation.annotation.slice.elts[1])
    assert (
        declared == 2
        and max(stage * 2 + g for stage, g in itertools.product(range(2), repeat=2)) == 3
    )
    assert declared < 2 * 2

    # Logical register lifetime replay of the source's preload/steady/hoisted-final structure.
    capture_domains = 0
    for tiles, blocks in itertools.product(range(1, 18), (1, 2, 4, 8)):
        registers = {0: (0, 0, False)}
        loaded = {(0, 0)}
        mma = []
        released = []
        for tile in range(tiles):
            final = tile == tiles - 1
            for block in range(blocks):
                nxt = (block + 1) % blocks
                if block == blocks - 1:
                    assert all((tile, b) in loaded for b in range(blocks))
                    released.append(tile)
                if final and nxt > 0:
                    registers[nxt] = (tile, nxt, False)
                    loaded.add((tile, nxt))
                held_tile, held_block, prepared = registers[block]
                assert (held_tile, held_block) == (tile, block) and not prepared
                registers[block] = (held_tile, held_block, True)
                mma.append((held_tile, held_block))
                if not final:
                    next_tile = tile + 1 if nxt == 0 else tile
                    registers[nxt] = (next_tile, nxt, False)
                    loaded.add((next_tile, nxt))
        assert mma == list(itertools.product(range(tiles), range(blocks)))
        assert released == list(range(tiles))
        capture_domains += 1
    zero_k_sites = []
    output_tails = {}
    for name, cls in classes.items():
        kernel = next(x for x in cls.body if isinstance(x, ast.FunctionDef) and x.name == "kernel")
        zero_k_sites.append(
            sum(
                isinstance(x, ast.Call) and ast.unparse(x.func) == "cute.gemm"
                for x in ast.walk(kernel)
            )
        )
        text = ast.unparse(kernel)
        assert "accumulators.fill(0.0)" in text
        assert "range(0, k_tile_cnt - 1, 1, unroll=1)" in text
        assert "peek_ab_full_status = cutlass.Boolean(1)" in text
        # Hoisted final MMA exists even if the outer K loop has no iterations.
        math_loops = [
            x
            for x in ast.walk(kernel)
            if isinstance(x, ast.While) and ast.unparse(x.test) == "work_tile.is_valid_tile"
        ]
        math_body = next(
            loop for loop in math_loops if "accumulators.fill(0.0)" in ast.unparse(loop)
        )
        hoisted = [
            x
            for x in math_body.body
            if isinstance(x, ast.For) and ast.unparse(x.target) == "k_block_idx"
        ]
        assert len(hoisted) == 1 and any(
            isinstance(x, ast.Call) and ast.unparse(x.func) == "cute.gemm"
            for x in ast.walk(hoisted[0])
        )
        next_blocks = [
            x.value
            for x in ast.walk(math_body)
            if isinstance(x, ast.Assign) and ast.unparse(x.targets[0]) == "k_block_next"
        ]
        assert len(next_blocks) == 2
        for expression in next_blocks:
            for blocks in (1, 2, 4, 8):
                for block in range(blocks):
                    got = eval(
                        compile(ast.Expression(expression), "retained-next-block", "eval"),
                        {"k_block_idx": block, "num_k_blocks": blocks},
                    )
                    assert got == (block + 1) % blocks
        advances = [
            x
            for x in math_body.body
            if isinstance(x, ast.Expr)
            and isinstance(x.value, ast.Call)
            and ast.unparse(x.value.func) == "tile_sched.advance_to_next_work"
        ]
        assert len(advances) == (2 if name == names[1] else 1)
        output_tails[name] = text.count("tma_store_pipeline.producer_tail()")
        assert "mainloop_pipeline.producer_tail(mainloop_producer_state)" in text
        alignment = copy.deepcopy(
            next(
                x
                for x in cls.body
                if isinstance(x, ast.FunctionDef) and x.name == "is_valid_tensor_alignment"
            )
        )
        mod = ast.Module([alignment], [])
        symbols = {"widths": DTYPES}
        exec(
            compile(ast.fix_missing_locations(Symbols().visit(mod)), "retained-alignment", "exec"),
            symbols,
        )
        predicate = symbols["is_valid_tensor_alignment"]
        assert predicate(128, 128, 0, 1, "Float8E4M3FN", "Float16", "k", "k", "n")
        assert predicate(128, 128, 16, 1, "Float8E4M3FN", "Float16", "k", "k", "n")
        assert not predicate(128, 128, 16, 1, "Float4E2M1FN", "Float16", "k", "k", "n")
    assert output_tails[names[0]] == 0 and output_tails[names[1]] == 1
    consumer_wait = next(
        f
        for c in trees["sm90.py"].body
        if isinstance(c, ast.ClassDef) and c.name == "PipelineAsync"
        for f in c.body
        if isinstance(f, ast.FunctionDef) and f.name == "consumer_wait"
    )
    assert "try_wait_token is None or try_wait_token == 0" in ast.unparse(consumer_wait)
    # token1 bypasses the wait, but cannot establish valid data for the hoisted final MMA.
    token = 1
    assert not (token is None or token == 0)
    chain = Chain.model_validate(
        yaml.safe_load((STUDY / "delta/geforce-schedule-chain.yaml").read_text())
    )
    previous = set()
    for item in chain.items:
        assert item.key not in previous and set(item.after) <= previous
        rule = Template.model_validate(
            yaml.safe_load((STUDY / "delta" / item.template / "template.yaml").read_text())
        )
        assert set(item.bindings) == {v.name for v in rule.variables}
        for variable in rule.variables:
            value = item.bindings[variable.name]
            if isinstance(variable.domain, list):
                assert value in variable.domain
            else:
                assert variable.domain.lo <= value <= variable.domain.hi
        previous.add(item.key)
    result = {
        "outcome": "PASS",
        "partial_chain_steps": len(chain.items),
        "source_cursor_jump_cases": jumps,
        "striped_stream_domains": stripes,
        "corrected_order_model_states": order_states,
        "overlap_branch_states": overlap_branches,
        "register_capture_domains": capture_domains,
        "source_order_slots": {
            "declared": declared,
            "required": 4,
            "maximum_index": 3,
            "declaration_alone_sufficient": False,
            "physical_padding_backing": "open",
        },
        "source_output_tail_calls": output_tails,
        "zero_k_alignment_accepted": True,
        "zero_k_hoisted_mma_remains": True,
        "reverse_mixed_alignment_witness": {
            "A": "FP8",
            "B": "FP4",
            "K": 16,
            "a_type_check_passes": True,
            "b_type_check_passes": False,
        },
        "boundary": (
            "Source scalar cursors and abstract register/order models only; so"
            "urce storage/tail/K0 obligations remain and no native progress is"
            " claimed."
        ),
    }
    (STUDY / "checks/geforce-schedules-result.json").write_text(json.dumps(result, indent=2) + "\n")
    sys.stdout.write("PASS: striped cursors, register capture and source order-storage contract.\n")


if __name__ == "__main__":
    main()
