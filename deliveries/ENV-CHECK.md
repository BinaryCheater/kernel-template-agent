# Environment check

Read-only check, 2026-10-30 (UTC 04:00). Nothing was installed or changed on any machine.

## Control plane (local)

| Item | Value |
| --- | --- |
| Host | MacBook Pro, macOS 27.2 (26B123), arm64 |
| uv | 0.11.28 (`~/.local/bin/uv`) |
| Python 3.12 | 3.12.13, uv-managed (`~/.local/share/uv/python/cpython-3.12-macos-aarch64-none`) |
| Other Pythons | Homebrew 3.15.0 (default `python3`), 3.14.8; system 3.9.6 |
| Node / npm | v26.12.1 / 12.2.1 (Homebrew) |
| Docker CLI | present (`/opt/homebrew/bin/docker`) |

## spark (GPU worker)

| Item | Value |
| --- | --- |
| ssh | `ssh spark` works (BatchMode), user `memx`, in group `docker` |
| Host | `spark-b6d9`, Ubuntu 24.04.5 LTS, aarch64, 20 CPUs, 119 GiB unified memory |
| Disk | `/` 3.7 TB, 2.1 TB free (no separate data disk) |
| GPU | 1 × NVIDIA GB10, compute capability 12.1 (sm_121) |
| Driver | 580.95.05; nvidia-smi reports CUDA 13.0 |
| CUDA toolkit | `/usr/local/cuda` → cuda-13.0, SDK 13.0.3; nvcc 13.0.88 (not on non-interactive PATH; use `/usr/local/cuda/bin`) |
| Tools in `/usr/local/cuda/bin` | nvcc, ptxas, cuobjdump, nvdisasm, compute-sanitizer, ncu, nsys |
| Python | system 3.12.3; uv-managed 3.12.13 |
| uv | 0.11.2 in `~/.local/bin` (pre-existing, user-level; not on non-interactive PATH) |
| Node | not found |
| Docker | 29.1.3 with `nvidia` runtime; NVIDIA Container Toolkit 1.20.1 |
| gcc / cmake / git | 13.3.0 / 3.28.3 / 2.43.0 |
| Load | idle (load average 0.00) |

## slai (CPU worker, backup)

| Item | Value |
| --- | --- |
| ssh | `ssh slai` works (BatchMode), user `dy` |
| Host | `linux-slai`, Ubuntu 24.04.4 LTS, x86_64, 32 CPUs, 31 GiB RAM |
| Disk | **`/` is 100% full (778 MB free)**; `/storage` 466 GB, 101 GB free, writable |
| GPU | NVIDIA RTX A1000 (sm_86, 8 GiB), driver 610.43.02, CUDA UMD 13.3 |
| CUDA toolkit | no `/usr/local/cuda`, no nvcc |
| Python | system 3.12.3 |
| uv / Node / Docker | not found |
| gcc / cmake / git | 13.3.0 / 3.28.3 / 2.43.0 |
| Load | ~0.6 |

## Consequences for construction

- Remote commands run non-interactive shells; scripts must use absolute paths (`/usr/local/cuda/bin`, `~/.local/bin/uv`) or set PATH explicitly.
- spark has a usable uv. Version differs from local (0.11.2 vs 0.11.28); `uv.lock` must be readable by both, and the EnvManifest records each.
- slai has no uv and its root disk is full. Any slai worker directory must live under `/storage`. Installing uv there (project-local binary) is a WP2 step; it is not a system install, but it is deferred until WP2 needs slai.
- slai has a GPU (A1000, sm_86) but no CUDA toolkit; it is a CPU worker per the spec and should not report a `gpu=` capability for KTA jobs unless WP2 decides otherwise.
