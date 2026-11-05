"""Standalone experimental guides for eight unchanged source proposals."""

import json


def supplemental() -> dict[str, dict[str, str]]:
    return {
        "operand_precision_conversion@1": {
            "when": "An intermediate contraction operand is wider than a legal hardware operand.",
            "binding": "Caller fixes the accepted error, original dtype and accumulator type. "
            "Choose to_dtype, rounding and site only when the consumer and compiler implement "
            "them. Bind from_dtype to the actual produced value. Keep FP32 accumulator and "
            "normalization state when required; converting an operand does not convert that state.",
            "example": "Local 2x2 FP32 operand [[0.1,0.2],[0.3,0.4]] can be rounded once "
            "to FP16 at registers_before_mma with round_to_nearest, then multiplied into "
            "FP32 accumulators. The saved FP32 denominator must still use its declared values. "
            "Rounding values and rounding the accumulator are different transformations.",
            "cost": "Narrow operands may enable matrix instructions and reduce live bytes; "
            "conversion costs instructions and introduces input-dependent rounding error.",
            "boundary": "The declared format menu is not a compiler guarantee. Check long "
            "reductions, cancellation and dynamic range. No error bound is "
            "supplied by this proposal.",
        },
        "approximate_math_rewrite@1": {
            "when": "A transcendental expression has a cheaper base conversion or "
            "native approximation.",
            "binding": "Caller fixes scale units, log base and numerical acceptance. Select "
            "rewrite and fused_fma, then preserve every unit conversion. For "
            "exp_to_exp2_folded_scale "
            "use exp(y)=exp2(y*log2(e)). If y=s*(a-m), convert the entire scaled difference "
            "consistently; if a and m are already natural scaled logits, do not apply s again.",
            "example": "For y=ln(2), exp2(y*log2(e))=2 in real arithmetic. A natural-log "
            "normalizer from a base2 exponent state must convert back: log(L)=log2(L)*ln(2). "
            "Returning a base2 normalizer where natural log is required is a semantic error.",
            "cost": "Can lower transcendental cost and fold constants, at the cost of "
            "approximation "
            "and contraction differences. A faster expression still has to pass"
            " fixed O and LSE checks.",
            "boundary": "No arbitrary host tensor arithmetic is introduced by a scalar constant. "
            "Establish finite range and first-valid-reference initialization "
            "before exponentiation; "
            "an arbitrary zero reference can underflow uniformly negative logits.",
        },
        "cross_lane_reduction@1": {
            "when": "One logical reduction has partials in a complete group of lanes "
            "within a warp.",
            "binding": "Caller fixes the operator and which lane owns each partial. Choose lanes "
            "from the power-of-two menu, result_to and combine_op. Use aligned contiguous groups "
            "for the stated XOR tree; establish a converged participating mask at every exchange. "
            "A lane with no valid elements contributes the operator identity rather than exiting.",
            "example": "For an aligned group of eight lanes carrying 1..8, XOR offsets 4,2,1 "
            "produce 36 in each lane for add/all_lanes. Lane IDs in another aligned eight-lane "
            "group must never enter that sum. For a two-component state, exchange and merge both "
            "components with the declared combine, rather than summing references independently.",
            "cost": "Avoids shared-memory traffic and block barriers for an intra-warp join; "
            "adds logarithmic shuffle/combine steps and changes floating-point grouping.",
            "boundary": "A declared lane0 result requires an actual reduction-to-owner "
            "implementation; "
            "the shown XOR allreduce alone replicates the result. Native lane membership and masks "
            "remain obligations, not consequences of choosing a legal lane count.",
        },
        "accumulator_operand_reuse@1": {
            "when": "One contraction's accumulator tile is immediately consumed as "
            "another contraction's left operand.",
            "binding": "Fix both instruction ABIs and the common logical row axis. Choose the "
            "supported producer_instruction/consumer_operand pair. Enumerate lane, register and "
            "logical coordinates before reinterpreting a fragment. Apply the separately admitted "
            "dtype conversion; keep the producer's other live statistics untouched.",
            "example": "For a local intermediate X with distinct coordinate tags, associate each "
            "producer register with (row,col), then require the consumer A slot"
            " for that coordinate "
            "to read the same tag. A register tensor with the right element count but exchanged "
            "row/column digits fails this test. No physical register "
            "permutation is implied by a view.",
            "cost": "May eliminate a shared-memory round trip and synchronization. It constrains "
            "instruction/layout choices and can extend register lifetimes.",
            "boundary": "The proposal lists two ABI pairs; neither a pair label nor the weak "
            "same-row predicate establishes the native lane mapping. A TileLang fragment copy or "
            "GEMM lowering needs independent mapping evidence; otherwise retain the safe transfer.",
        },
        "collective_fragment_load@1": {
            "when": "A shared tile must be distributed into the lane/register ABI of a "
            "collective compute instruction.",
            "binding": "Caller fixes the consumer fragment coordinates and source byte layout. "
            "Choose instruction and transpose only from native supported pairs. Establish source "
            "completion and visibility before the collective. Validate all participant addresses, "
            "alignment, convergence and the full destination mapping.",
            "example": "Use an 8x8 b16 tile tagged by logical value 8*i+j. A transposed load "
            "must supply the consumer's slot for (j,i) with tag 8*i+j; it does not change the "
            "caller's logical tensor shape. A tagged-coordinate replay distinguishes transposition "
            "from a same-size but incorrect register arrangement.",
            "cost": "Collective loads can replace scalar addressing and lane exchange, but require "
            "a suitable shared layout and a fixed collective ABI.",
            "boundary": "The source's c_trans predicate is weaker than native legality: lds128 "
            "does not become a transposing collective merely because the predicate accepts it. "
            "Use transpose only for a supported ldmatrix variant and supply its"
            " actual ISA mapping.",
        },
        "staged_relayout@1": {
            "when": "A correct register fragment layout would produce scattered "
            "destination accesses.",
            "binding": "Caller fixes logical element coordinates and final tensor strides. Choose "
            "chunks, reader and write_instruction from a supported implementation. Derive slice "
            "bytes and check capacity. Map every writer and reader to the same shared coordinate, "
            "and order writes before reads and all reads before the next slice overwrites storage.",
            "example": "For 16 tagged elements split into chunks=2, each slice owns eight "
            "elements. "
            "Write those eight through the fragment-to-shared map, synchronize, then read them "
            "through a coalesced map and store to their original logical "
            "positions. A barrier before "
            "writing cannot replace the barrier between writing and reading.",
            "cost": "Trades shared traffic and synchronization for coalesced output access; "
            "more slices reduce peak scratch at the cost of more phase transitions.",
            "boundary": "Aliasing prior storage is legal only after its last required read. "
            "TMA readers require the appropriate asynchronous publication and completion protocol; "
            "ordinary CTA barriers alone do not establish those events.",
        },
        "boundary_predication@1": {
            "when": "A transfer tile or vector can cross a logical extent or an "
            "alignment boundary.",
            "binding": "Caller fixes allocation extents, strides and downstream semantics. Choose "
            "boundary_kind, fill and specialize_even. Derive an address predicate for each actual "
            "transaction, including every lane of a vector. Supply zero for "
            "additive padded values, "
            "negative infinity for masked maxima, or preserve destination bytes only when all "
            "consumers demonstrably exclude them.",
            "example": "Length 19 with an eight-element tile has a final origin 16: positions "
            "16..18 are loaded and positions 19..23 are filled neutrally without global reads. "
            "Output positions 19..23 must never be stored. Masking a later score does not make "
            "an earlier out-of-bounds address legal.",
            "cost": "Predicates and fills cost instructions but enable safe tails. An even-shape "
            "specialization may remove them only after its runtime guard proves"
            " the exact alignment.",
            "boundary": "Untouched stale data must not enter products before masking, especially "
            "when a padded value can be NaN. Keys, values, output rows and normalization stores "
            "can need different fill/store policies.",
        },
        "vectorized_access@1": {
            "when": "A logical transfer has aligned contiguous spans that can be "
            "assigned to wide thread accesses.",
            "binding": "Caller fixes element size, contiguous extent, base alignment and strides. "
            "Choose vector_bytes, threads and thread_arrangement. Derive elements_per_vector, "
            "vectors_per_row and iteration count. Prove exact-once element coverage and every "
            "actual vector address alignment, not just the allocation base.",
            "example": "A contiguous 32-element FP16 span has 64 bytes. With vector_bytes=8 "
            "there are eight vectors of four elements, each starting on an eight-byte boundary. "
            "A 32-thread cooperative map needs an explicit predicate for the 24 nonowning threads; "
            "they must not read extra vectors just because the block has 32 threads.",
            "cost": "Fewer memory instructions and coalesced spans can improve traffic efficiency. "
            "Wider accesses increase alignment/tail restrictions and may be "
            "unsupported as one instruction.",
            "boundary": "A listed 32-byte width is an abstract access choice, not a guarantee "
            "of one native instruction. Row/head strides and partial spans require legal smaller "
            "vectors or guarded scalar handling without out-of-bounds transactions.",
        },
    }


def proposal_guide(body: dict, details: dict[str, str]) -> str:
    text = f"# {body['name']}@{body['version']}\n\n"
    text += "Status: experimental source proposal; paired ProofOutline absent. "
    text += "This exposure does not promote the rule or establish proof assurance.\n\n"
    for title, value in [
        ("When to consider", details["when"]),
        ("Before and after", body["program"]["description"]),
        ("Binding steps", details["binding"]),
    ]:
        text += f"## {title}\n\n{value}\n\n"
    text += "Caller facts and derived quantities are fixed by the workload and selected "
    text += "layout. The table lists declared specialization choices; native implementation "
    text += "support can narrow that domain.\n\n"
    text += "| Variable | Binding / role | Domain | Meaning |\n|---|---|---|---|\n"
    for v in body["variables"]:
        values = [v["name"], f"{v['binding']}/{v['role']}", json.dumps(v["domain"]), v["meaning"]]
        text += "| " + " | ".join(str(x).replace("|", "\\|") for x in values) + " |\n"
    text += f"\n## Scope and coupled legality\n\n{body['scope']['text']}\n\n"
    text += f"Scope: `{body['scope']['predicate']}`.\n\n"
    for expression in body["preconditions"] + body["constraints"]:
        text += f"- `{expression['id']}`: `{expression['expr']}` — {expression['meaning']}\n"
    text += f"\nNative boundary: {details['boundary']}\n\n"
    text += f"## Local example\n\n{details['example']}\n\n"
    text += f"## Benefit and cost\n\n{details['cost']}\n\n"
    text += "## Obligations\n\n"
    for ob in body["obligations"]:
        text += f"- `{ob['name']}` ({ob['method']}): {ob['statement']}\n"
    text += f"\nNumerical risk: {body.get('numeric_risk') or 'No numerical change declared.'}\n\n"
    text += "No outline is supplied. Establish the listed assumptions and run implementation "
    text += "checks; reporting an unresolved obligation is required when its mapping is missing. "
    text += "Do not create a proof record from this guide.\n\n"
    text += "[Structured proposal](template.yaml). Related names: "
    text += ", ".join(f"`{r['target']}`" for r in body.get("relations", [])) or "none"
    return text + ". Select versions explicitly from the supplied menu.\n"
