"""Published API-equivalent token estimates, separate from unavailable native billing."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .report import summarize


def estimate(run: Path, rate_file: Path) -> dict:
    rates = json.loads(rate_file.read_text())
    report = summarize(run)
    model = report["model"]
    rule = rates[model]
    if model == "gpt-6.1-sol":
        seen = set()
        components = {
            "uncached_input": 0.0,
            "cached_input": 0.0,
            "cache_write_input": 0.0,
            "output": 0.0,
        }
        totals = {"input": 0, "cached_input": 0, "output": 0, "cache_write_input": 0}
        max_input = 0
        for source in report["native_transcripts"]:
            for index, line in enumerate(Path(source).read_text().splitlines()):
                entry = json.loads(line)
                if entry.get("type") != "token_usage_record":
                    continue
                payload = entry["payload"]
                key = payload.get("response_id") or f"{source}:{index}"
                if key in seen:
                    continue
                seen.add(key)
                usage = payload["usage"]
                counts = {
                    "cached_input": usage["cached_input_tokens"],
                    "cache_write_input": usage.get("cache_write_input_tokens", 0),
                    "output": usage["output_tokens"],
                }
                counts["uncached_input"] = (
                    usage["input_tokens"] - counts["cached_input"] - counts["cache_write_input"]
                )
                if min(counts.values()) < 0:
                    raise ValueError("Inconsistent native cache-inclusive input counts")
                long = usage["input_tokens"] > rule["long_context_input_threshold"]
                for kind, count in counts.items():
                    multiplier = (
                        rule["long_context_output_multiplier"]
                        if kind == "output"
                        else rule["long_context_input_multiplier"]
                    )
                    components[kind] += (
                        count * rule["per_million"][kind] * (multiplier if long else 1) / 1_000_000
                    )
                totals["input"] += usage["input_tokens"]
                totals["cached_input"] += counts["cached_input"]
                totals["output"] += counts["output"]
                totals["cache_write_input"] += counts["cache_write_input"]
                max_input = max(max_input, usage["input_tokens"])
        for key in ("input", "cached_input", "output"):
            if totals[key] != report["usage"][key]:
                raise ValueError(f"Native per-request sum differs from retained session {key}")
        value = {
            "estimated_usd": sum(components.values()),
            "components_usd": components,
            "native_request_count": len(seen),
            "max_request_input_tokens": max_input,
            "verified_token_sums": totals,
        }
    elif model == "deepseek-flash":
        usage = report["usage"]
        if any(usage.get(key) is None for key in ("input", "cached_input", "output")):
            raise ValueError("Cannot estimate with missing native token accounting")
        counts = {
            "uncached_input": usage["input"] - usage["cached_input"],
            "cached_input": usage["cached_input"],
            "output": usage["output"],
        }
        value = {
            label.replace("_per_million", "_estimated_usd"): sum(
                count * rule[label][kind] / 1_000_000 for kind, count in counts.items()
            )
            for label in ("off_peak_per_million", "peak_per_million")
        }
        value["verified_token_sums"] = usage
    else:
        raise ValueError(f"No pinned price record for {model}")
    return {
        "model": model,
        "native_billed_cost": None,
        "native_cost_status": "not reported",
        "basis": rates["basis"],
        "currency": rates["currency"],
        "as_of": rates["as_of"],
        "source": rule["source"],
        "rate_file": str(rate_file.resolve()),
        **value,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument(
        "--rates",
        type=Path,
        default=Path("docs/runbooks/learndemo-1/replay-cases/pricing-2026-11-05.json"),
    )
    args = parser.parse_args()
    result = estimate(args.run, args.rates)
    with (args.output or args.run / "api-equivalent-estimate.json").open("x") as stream:
        json.dump(result, stream, indent=2)
        stream.write("\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
