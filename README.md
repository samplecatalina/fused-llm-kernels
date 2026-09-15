# fused-llm-kernels

An FP32 SGEMM optimization ladder for Ada (sm_89), from a naive kernel to warp
tiling, plus fused Triton operators. Every step up the ladder is backed by
Nsight Compute data rather than by an explanation of what should have happened.

The technical route itself is well-trodden. What this repository tries to get
right is the engineering around it:

- **Correctness on arbitrary shapes.** Every kernel accepts any `M/N/K`,
  including sizes that are not a multiple of the tile shape. Three shapes are
  checked on every run: `4096³`, `4097×513×129`, and `64³`.
- **Measurement that survives an unsteady GPU clock.** Time-based warmup until
  the SM clock settles, per-iteration CUDA-event timing, median with p10/p90,
  and clock, power and clock-event state recorded with every result row.
- **A roofline built from micro-benchmarks**, not from datasheet numbers.

> **Status: in progress.** The harness, the cuBLAS baseline and the measurement
> methodology are in place. `k1` is a stub, so `make test` currently fails on it
> by design. The only measurement so far is the cuBLAS baseline on the
> development GPU (`results/rtx4060-laptop/gemm_4096.csv`). This README will not
> carry a number that cannot be traced to a row in `results/` or to a report in
> `profiling/reports/`.

## Quick start

```bash
bash scripts/env_check.sh      # environment acceptance check
make build
make test                      # correctness: main / non-divisible / tiny shapes
make bench                     # headline 4096³ number -> results/<device>/gemm_4096.csv
make bench KERNELS=k0          # a single rung
make sweep                     # 512..8192 -> results/<device>/gemm_sweep.csv
make profile K=k1              # ncu report -> profiling/reports/<device>/
```

`DEVICE` (default `rtx4060-laptop`) names the directory results are written to;
`ARCH` (default `sm_89`) selects the build target.

`scripts/install_cuda_wsl.sh` installs the CUDA toolkit inside WSL2 if needed.

## The ladder

| Rung | What it adds | Bottleneck it targets | Status |
|---|---|---|---|
| K0 | cuBLAS baseline, timed through the same harness | - | done |
| K1 | naive: one thread per element of C | uncoalesced global access | stub |
| K2 | coalesced access (swap the thread-to-data mapping) | DRAM bandwidth | planned |
| K3 | shared-memory tiling (BM×BK / BK×BN) | shared bandwidth, low compute ratio | planned |
| K4 | 1D thread tiling (TM results per thread) | register reuse | planned |
| K5 | 2D thread tiling (TM×TN register block) | instruction scheduling | planned |
| K6 | float4 vectorized loads, transposed A tile | shared-memory bank conflicts | planned |
| K7 | warp tiling (a second blocking level per warp) | scheduling efficiency | planned |
| K8 | double buffering (optional) | latency hiding | optional |

Boundary handling is done with guard branches rather than padding, so a kernel
that passes `4097×513×129` is correct by construction rather than by luck.

## Measurement

- CUDA events, timed per iteration; **median with p10/p90**, never a mean.
  Dividing a total by a count hides clock drift; the distribution does not.
- **Warmup is measured in time, not iterations.** The GPU is kept loaded for at
  least 30 s and then until its SM clock stops trending: the mean over the last
  five one-second samples must be within 1% of the mean over the five before
  (capped at 4× the minimum). Whether it settled is recorded, together with the
  clock's mean and peak-to-peak range over that final window. A fixed count is
  meaningless across shapes: 25 iterations of a 3 ms kernel end while the GPU is
  still on a boost clock it cannot sustain, so the first timed repetitions come
  out faster than the rest. Means are compared rather than individual samples
  because a power-capped GPU keeps dithering by several percent around its
  steady operating point; that dither is the floor under the per-iteration
  spread.
- ≥ 100 repetitions. When (max − min) / median exceeds 5%, the run is flagged
  and the median is not quoted without an explanation from the clock data.
- Every row records SM clock, memory clock, temperature, power draw and active
  clock-event reasons before warmup, at the start of timing and after it, plus
  how long SW power capping and thermal or power-brake slowdown were active
  during the timed region (the driver advances these counters in coarse steps,
  so they are only meaningful over timed regions of several seconds). It also records the device tag, the source revision
  and a UTC timestamp. The runner refuses to write CSV rows without these, and
  rows from kernels that fail the correctness check are never written.
  Clocks cannot be locked from user space, so a speedup reported without the
  clock it was measured at is not a speedup.
- Correctness is checked against cuBLAS under two tolerances that must both
  hold: element-wise relative error < 1e-3 **and** relative Frobenius error
  < 1e-5. Neither catches what the other misses.
- Benchmarks are only valid on AC power, with the host power plan set to high
  performance and no other GPU load running.

`GFLOP/s = 2·M·N·K / t`.

## Hardware this was developed on

| | |
|---|---|
| GPU | RTX 4060 Laptop GPU (AD107, sm_89), 24 SMs, 8 GB |
| L2 | 32 MB |
| Shared memory per block | 48 KB by default (Ada allows opting in to ~99 KB) |
| Toolchain | CUDA 13.3, driver 610.47 |
| Host | WSL2 (Ubuntu) |

Peak FP32 throughput and peak achievable bandwidth are deliberately **not**
taken from the datasheet: this is the mobile part, whose clocks and power
budget move with temperature. Both roofline ceilings will be measured with
micro-benchmarks before any roofline plot is published.

Profiling under WSL2 requires GPU performance counters to be opened up on the
Windows host: NVIDIA Control Panel → Developer → Manage GPU Performance
Counters → allow access to all users.

## Layout

```
csrc/kernels/     one file per rung; add one = declare in kernels.h + a row in registry.cu
csrc/harness/     runner.cu: correctness, timing, CSV
csrc/common/      error-checking macros, input generation, the two-tolerance check
docs/             design notes and the optimization log
results/          benchmark CSVs, one directory per device; every published number comes from here
profiling/        Nsight Compute reports and the roofline scripts
triton_kernels/   fused Triton operators
```

## References

- siboehm, [How to Optimize a CUDA Matmul Kernel](https://siboehm.com/articles/22/CUDA-MMM)
- The Triton [layer-norm tutorial](https://triton-lang.org/main/getting-started/tutorials/05-layer-norm.html)

## License

MIT
