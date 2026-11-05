# Attention tuning: reproducible figures

These figures visualize the nine recorded agent coding/tuning sessions. No MIP
solver or performance-prediction model was used. Plotting reads existing records;
it does not run kernels or call models.

All figures are available as PNG, PDF and SVG in
[/Users/dingyi/kta-data/dev/source-attention-control-v1/plots/v1](/Users/dingyi/kta-data/dev/source-attention-control-v1/plots/v1).
The [figure gallery](/Users/dingyi/kta-data/dev/source-attention-control-v1/plots/v1/README.md)
contains previews and export links.

| Figure | What it shows | How to read it |
|---|---|---|
| [Final performance](/Users/dingyi/kta-data/dev/source-attention-control-v1/plots/v1/01_final_performance.png) | All nine generated implementations beside their own remeasured FlashAttention original, at each sequence length | Gray is original; color is generated. Lower GPU microseconds are faster. Points are medians of three window medians; whiskers show their minimum and maximum. |
| [Timing variation](/Users/dingyi/kta-data/dev/source-attention-control-v1/plots/v1/02_timing_variation.png) | Original and generated timing distributions | Each box pools 90 samples: three windows with 30 blocks each. Box = middle 50%; whiskers = 10th–90th percentiles; small dots = minimum/maximum. Each sample averages 100 graph replays. |
| [Full iteration history](/Users/dingyi/kta-data/dev/source-attention-control-v1/plots/v1/03_iteration_history.png) | Every development submission, including regressions and unusable timing | Zero is equal time to the same-window original. Negative means faster. Dashed line is the +11.1% allowance. Open points are unresolved timings; red crosses below panels mean no usable timing, including compile failures and unfinished submissions. |
| [Improvement versus active time](/Users/dingyi/kta-data/dev/source-attention-control-v1/plots/v1/04_improvement_vs_active_time.png) | How quickly each session improved its weakest shape | For each valid submission, take the largest time increase across the three shapes, then retain the best value seen so far. Reading, reasoning, compilation and tests count as active time; verified GPU waiting does not. |
| [Time and estimated cost](/Users/dingyi/kta-data/dev/source-attention-control-v1/plots/v1/05_time_and_cost.png) | Per-session active minutes, GPU waiting and model cost | Colored time is active; gray is waiting. Cost is the retained-price API-equivalent estimate, not an account charge. |

Shaded rows retain timing-unresolved observations. Within-window sample spread
and the range of window medians are different quantities; neither is presented
as a confidence interval. A sub-percent difference is not claimed as a
demonstrated performance advantage.

Final plots use the fixed-source follow-up for all four eligible unresolved
measurements, and initial confirmation for the other five, matching the
[final findings](conclusion.md). Development curves remain separate from final
confirmation. No-Template repeat 1 was stopped by a tool error before its active
budget was exhausted, which is marked in the figure captions.

## Reproduce

From the KTA repository root:

```sh
UV_CACHE_DIR=.uv/plot-cache uv run --locked --script tools/plot_source_replay.py \
  /Users/dingyi/kta-data/dev/source-attention-control-v1 \
  --output /Users/dingyi/kta-data/dev/source-attention-control-v1/plots/v1
```

[Plotting script](/Users/dingyi/Projects/agentic-codesign/kta/tools/plot_source_replay.py),
[record reader](/Users/dingyi/Projects/agentic-codesign/kta/tools/source_replay_plot_data.py),
and the script-specific lock pin the rendering environment. The application lock
and dependencies are unchanged. Package and font caches stay under `.uv/`.

The output directory also contains `final_windows.csv`, `paired_samples.csv`,
`iterations.csv`, `submissions.csv` and `provenance.json`. The provenance file
identifies the exact source records, script, helper and dependency lock used.
