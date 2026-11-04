"Replay FMHA mask classes, per-element identities and offset retirement channels."

from __future__ import annotations

import ast
import copy
import enum
import hashlib
import itertools
import json
import math
import sys
from collections import deque
from pathlib import Path
from types import SimpleNamespace

from moe_dense import ScalarSource

STUDY = Path(__file__).resolve().parents[1]


class MaskSource(ScalarSource):
    def visit_Call(self, node):
        if ast.unparse(node.func) == "cutlass.range":
            node.func = ast.Name("range", ast.Load())
            node.keywords = []
        return super().visit_Call(node)


def channel_states(steps, prologue=True, tail=True):
    # R: capture scores and signal peer; W: wait for peer read then store P.
    g0 = [event for i in range(steps) for event in (("R0", i), ("W0", i))]
    if tail:
        g0.append(("TAIL", steps))
    g1 = ([("PROLOGUE", 0)] if prologue else []) + [
        event for i in range(steps) for event in (("R1", i), ("W1", i))
    ]
    initial = (0, 0, 0, 0, 0, 0)
    queue, seen = deque([initial]), {initial}
    completed = unsafe = stranded = 0
    while queue:
        state = queue.popleft()
        p0, p1, ap, ac, bp, bc = state
        enabled = []
        for group, pc, events in ((0, p0, g0), (1, p1, g1)):
            if pc == len(events):
                continue
            op, index = events[pc]
            nxt = list(state)
            if op in {"R0", "TAIL"}:
                nxt[2] += 1
            elif op == "R1":
                nxt[4] += 1
            elif op == "W0":
                if bc == bp:
                    continue
                if bc != index:
                    unsafe += 1
                nxt[5] += 1
            else:  # prologue or P1 store waits on the A channel
                if ac == ap:
                    continue
                expected = 0 if op == "PROLOGUE" else index + 1
                if ac != expected:
                    unsafe += 1
                nxt[3] += 1
            nxt[group] += 1
            if nxt[2] - nxt[3] > 1 or nxt[4] - nxt[5] > 1:
                unsafe += 1
            enabled.append(tuple(nxt))
        if not enabled:
            if p0 == len(g0) and p1 == len(g1):
                completed += 1
            else:
                stranded += 1
        for nxt in enabled:
            if nxt not in seen:
                seen.add(nxt)
                queue.append(nxt)
    return len(seen), completed, unsafe, stranded


def main() -> None:
    manifest = json.loads((STUDY / "inputs/sources.json").read_text())
    root = Path(manifest["tree"])
    paths = [
        "examples/python/CuTeDSL/helpers/fmha_helpers.py",
        "examples/python/CuTeDSL/cute/blackwell/kernel/attention/fmha/fmha.py",
        "python/CuTeDSL/cutlass/pipeline/sm90.py",
    ]
    trees = {}
    for path in paths:
        blob = (root / path).read_bytes()
        record = next(x for x in manifest["files"] if x["path"] == path)
        assert hashlib.sha256(blob).hexdigest() == record["sha256"]
        trees[Path(path).name] = ast.parse(blob)
    helpers = trees["fmha_helpers.py"]
    cls = next(x for x in helpers.body if isinstance(x, ast.ClassDef) and x.name == "FusedMask")
    masks = next(x for x in helpers.body if isinstance(x, ast.ClassDef) and x.name == "MaskEnum")
    ns = {
        "enum": enum,
        "Float32": SimpleNamespace(inf=math.inf),
        "tuple_size": len,
        "cute": SimpleNamespace(
            ceil_div=lambda a, b: -(-a // b), arch=SimpleNamespace(thread_idx=lambda: (0, 0, 0))
        ),
    }
    exec(
        compile(
            ast.fix_missing_locations(
                MaskSource().visit(ast.Module([copy.deepcopy(masks), copy.deepcopy(cls)], []))
            ),
            "retained-fused-mask",
            "exec",
        ),
        ns,
    )
    mask = ns["FusedMask"]
    kinds = ns["MaskEnum"]
    domains = elements = 0
    failures = []
    helper_only_tails = []
    empty_rows = 0
    small_domains = itertools.product(
        (1, 3, 5, 9),
        (1, 3, 5, 9),
        (2, 4),
        (2, 4),
        (None, 0, 2),
        (None, 0, 2),
        (kinds.RESIDUAL_MASK, kinds.WINDOW_MASK, kinds.WINDOW_MASK_INFERENCE),
    )
    canonical_domains = [
        (257, 129, 256, 128, None, None, kinds.RESIDUAL_MASK),
        (257, 129, 256, 128, None, 0, kinds.WINDOW_MASK),
        (257, 129, 256, 128, 2, 0, kinds.WINDOW_MASK_INFERENCE),
    ]
    for sq, sk, bm, bn, left, right, kind in itertools.chain(small_domains, canonical_domains):
        if kind == kinds.RESIDUAL_MASK and (left is not None or right is not None):
            continue
        offset = sk - sq if kind == kinds.WINDOW_MASK_INFERENCE else 0
        for qb in range(math.ceil(sq / bm)):
            info = mask.get_masked_info(kind, (qb, 0, (0, 0)), (bm, bn, 64), sq, sk, left, right)
            start, end, leading, unmasked, trailing = info
            assert end == start + leading + unmasked + trailing
            classification = {i: True for i in range(start, start + leading)}
            classification.update(
                {i: False for i in range(start + leading, start + leading + unmasked)}
            )
            classification.update({i: True for i in range(start + leading + unmasked, end)})
            for kb in range(max(math.ceil(sk / bn), end)):
                coords = list(
                    itertools.product(range(qb * bm, (qb + 1) * bm), range(kb * bn, (kb + 1) * bn))
                )
                scores = list(range(1, len(coords) + 1))
                mask.apply_mask(kind, scores, coords, sq, sk, left, right)
                valid = []
                for index, (q, k) in enumerate(coords):
                    keep = q < sq and k < sk
                    if kind != kinds.RESIDUAL_MASK:
                        keep &= left is None or k >= q + offset - left
                        keep &= right is None or k <= q + offset + right
                    unmasked_helper = kind != kinds.RESIDUAL_MASK and left is None and right is None
                    helper_keep = True if unmasked_helper else keep
                    assert scores[index] == (index + 1 if helper_keep else -math.inf)
                    if unmasked_helper and q < sq and k >= sk and len(helper_only_tails) < 3:
                        helper_only_tails.append(
                            {"sq_sk": [sq, sk], "q_k": [q, k], "tile": [bm, bn], "mask": kind.name}
                        )
                    valid.append(keep)
                    elements += 1
                bad = any(valid) and kb not in classification
                # An unmasked loop may include padded query rows; these produce
                # no visible output. Only live query rows need every K valid.
                bad |= (
                    kb in classification
                    and not classification[kb]
                    and any(q < sq and not keep for (q, _), keep in zip(coords, valid, strict=True))
                )
                selected_scope = not (
                    kind != kinds.RESIDUAL_MASK and left is None and right is None and sk % bn != 0
                )
                if bad and selected_scope and len(failures) < 12:
                    failures.append(
                        {
                            "sq_sk": [sq, sk],
                            "tile": [bm, bn],
                            "query_block": qb,
                            "left_right": [left, right],
                            "mask": kind.name,
                            "classified_info": info,
                            "key_block": kb,
                        }
                    )
            for q in range(qb * bm, min((qb + 1) * bm, sq)):
                if not any(
                    k < sk
                    and (left is None or k >= q + offset - left)
                    and (right is None or k <= q + offset + right)
                    for k in range(sk)
                ):
                    empty_rows += 1
            domains += 1
    assert not failures and helper_only_tails
    forward = next(x for x in trees["fmha.py"].body if isinstance(x, ast.ClassDef))
    init = next(x for x in forward.body if isinstance(x, ast.FunctionDef) and x.name == "__init__")
    offsets = {
        ast.unparse(x.targets[0]): ast.literal_eval(x.value)
        for x in init.body
        if isinstance(x, ast.Assign)
        and ast.unparse(x.targets[0])
        in {
            "self.tmem_s0_offset",
            "self.tmem_s1_offset",
            "self.tmem_p0_offset",
            "self.tmem_p1_offset",
            "self.tmem_skip_softmax0_offset",
            "self.tmem_skip_softmax1_offset",
        }
    }
    assert offsets == {
        "self.tmem_s0_offset": 0,
        "self.tmem_s1_offset": 128,
        "self.tmem_p0_offset": 160,
        "self.tmem_p1_offset": 32,
        "self.tmem_skip_softmax0_offset": 136,
        "self.tmem_skip_softmax1_offset": 8,
    }
    # Nominal128-row, FP16-P source witness; native TMEM fragment composition
    # must independently establish this contiguous64-column footprint.
    for producer, peer_score in ((0, 1), (1, 0)):
        base = offsets[f"self.tmem_s{peer_score}_offset"]
        pbase = offsets[f"self.tmem_p{producer}_offset"]
        flag = offsets[f"self.tmem_skip_softmax{producer}_offset"]
        assert base <= pbase and pbase + 64 <= base + 128
        assert base <= flag < pbase
    guard = next(
        x
        for x in ast.walk(trees["fmha.py"])
        if isinstance(x, ast.FunctionDef) and x.name == "check_seqlen_valid"
    )
    guard_ns = {}
    exec(
        compile(
            ast.fix_missing_locations(ScalarSource().visit(ast.Module([copy.deepcopy(guard)], []))),
            "source-window-host-guard",
            "exec",
        ),
        guard_ns,
    )
    guard_fn = guard_ns["check_seqlen_valid"]
    assert guard_fn(3, 1, None, 0, True)
    assert not guard_fn(3, 3, 0, 0, False)
    empty_example = [q for q in range(3) if not any(k <= q + 1 - 3 for k in range(1))]
    assert empty_example == [0, 1]
    states = completed = 0
    for steps in range(13):
        count, terminal, unsafe, stranded = channel_states(steps)
        assert terminal == 1 and not unsafe and not stranded
        states += count
        completed += terminal
    missing_tail = channel_states(3, tail=False)
    missing_prologue = channel_states(3, prologue=False)
    assert missing_tail[3] > 0 and missing_prologue[2] > 0
    # The native participant helper returns an immutable handle; waiting alone
    # does not release the ordinary empty credit. Here producer never acquires
    # that credit, so the cross-channel protocol must supply reuse ordering.
    consumer = next(
        x
        for x in trees["sm90.py"].body
        if isinstance(x, ast.ClassDef) and x.name == "PipelineConsumer"
    )
    wait = next(
        x for x in consumer.body if isinstance(x, ast.FunctionDef) and x.name == "wait_and_advance"
    )
    assert not any(
        isinstance(x, ast.Call) and isinstance(x.func, ast.Attribute) and x.func.attr == "release"
        for x in ast.walk(wait)
    )
    result = {
        "outcome": "PASS",
        "mask_domains": domains,
        "distinct_score_elements": elements,
        "classified_live_domain_counterexamples": failures,
        "unselected_unmasked_helper_tails": helper_only_tails,
        "empty_visible_rows": empty_rows,
        "source_tmem_offsets": offsets,
        "host_accepts_empty_rows": {
            "sq_sk": [3, 1],
            "bottom_right_causal": True,
            "empty_query_rows": empty_example,
        },
        "host_rejects_nonempty_diagonal_window": {"sq_sk": [3, 3], "left_right": [0, 0]},
        "offset_channel_states": states,
        "completed_stream_lengths": completed,
        "missing_tail_stranded": missing_tail[3],
        "missing_prologue_unsafe_events": missing_prologue[2],
        "boundary": (
            "Retained scalar masks and bounded abstract cross-channel schedule"
            "s; native coordinates, TMEM visibility, cross-work retirement and"
            " empty rows open."
        ),
    }
    (STUDY / "checks/fmha-masks-result.json").write_text(json.dumps(result, indent=2) + "\n")
    sys.stdout.write(
        "PASS: FMHA scalar masks and offset channels; classification findings retained.\n"
    )


if __name__ == "__main__":
    main()
