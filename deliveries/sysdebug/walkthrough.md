# SYSDEBUG-1 visual walkthrough (operator check, not user acceptance)

The browser and API show the real SYSDEBUG1 store, with no fixture data seeded. Expected startup/reconnect latency through the tunnel is a few seconds. No release or acceptance button was pressed.

| Question | View / evidence | Observed answer | Gap |
| --- | --- | --- | --- |
| Is the worker on the intended code and host? | [Settings → System](http://127.0.0.1:14173/settings?tab=system) | Spark HK, local transport; current commit visible | Login stays absent until the user chooses isolation |
| What ran, and where did time go? | [Pipeline](http://127.0.0.1:14173/pipeline) | Live event indicator, real worker/host spans, stage capacities | CLI self-tests do not create case-loop WIP; zero stage depth is expected |
| Why was a measurement rejected? | [Rejected attempt](http://127.0.0.1:14173/pipeline?rec=acd658ebd5791db166ccab9408777febedc09239dd51d7a790ce797402fd5dcd) | REAL Report; observed 0.1625 against threshold 0.15; other counters visible | Raw JSON needs a concise calibration view; no automatic Inbox alert for these Reports |
| Did the corrected checker distinguish good and bad kernels? | [Checker summary](http://127.0.0.1:14173/pipeline?rec=c907edb357877d2497ca01f7cfa23eef3959b58fa910ecf82323394837f1b530) | Four expected verdicts, with linked Check records | No S0 Solution yet |
| Are profiler counters available? | [Profile](http://127.0.0.1:14173/pipeline?rec=39fb0f72242945e6588af2a26a95eb09973b8b082974d944afe107dff5445c3b) | Real duration and SM throughput | Metric units/availability require CSV inspection; DRAM n/a not summarised by Profile |
| Does model-internal agreement imply GPU accuracy? | [Consistency Report](http://127.0.0.1:14173/pipeline?rec=586fc117df8b73c37baf43480602f6dd8f4443efc6d30ed6ad7bac29f2961f60) | 3 shapes × 70 points, rho and bound violations | Model-only calculation; O3 still pending |
| Will an agent start autonomously? | [Release](http://127.0.0.1:14173/release) | S0, no agent calls permitted | User has not approved S1 or a task |

The rejected-attempt drawer was opened in the browser and visibly displayed its REAL origin, exact reason, observed values, thresholds and immutable record ID. A later joint walkthrough should measure the user's answer time; this operator inspection is not that study.
