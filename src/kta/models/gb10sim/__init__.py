"""The GB10 simulator copied from AI-Infra-Co-Design agentic/gb10 (commit 0eef497; D680, D681).

`sm.SM` runs a kernel's real SASS warp by warp under the compiler's control bits; `ws.WSSM` adds
TileLang's
warp-specialized TMA lowering; `kernel` maps a whole grid onto the busiest SM with its L2/DRAM
share; `model` keeps
upstream's L2/DRAM traffic functions the simulator uses. Parameters:
`gb10.json`, a copy
of upstream calibration/spark-6951/gb10.json. Provenance: README.md in this directory.
"""
