# Optimization log

One entry per rung, written when the rung is finished. Each entry states what
was predicted before running anything, what was actually measured, and where
the two disagree. The prediction is written first, on purpose: a number without
a prediction to compare it against teaches nothing about the hardware.

Every number here traces to a row in `results/` or to a report in
`profiling/reports/`. Entries with placeholders are not yet measured.

Entry template:

- **Hypothesis** - which bottleneck this rung targets, and which ncu metric
  should move as a result.
- **Prediction** - GFLOP/s and the reasoning behind it, recorded before the
  first benchmark run.
- **Measured** - median GFLOP/s at `4096³`, with the source CSV row.
- **Difference** - if prediction and measurement disagree, whether the wrong
  part was the assumption or the implementation.
- **Evidence** - the ncu metric that settles it.

---

## Environment baseline

Measured by `scripts/env_check.sh` on 2026-08-25.

| | |
|---|---|
| GPU | RTX 4060 Laptop GPU (AD107, sm_89) |
| SMs | 24 |
| L2 | 32 MB |
| Shared memory per block | 48 KB default (Ada allows opting in to ~99 KB) |
| Memory | 8 GB |
| Driver / CUDA | 610.47 / 13.3 |
| Nsight Compute counters | available (requires opening GPU performance counters to all users on the Windows host) |

This is the **mobile** part, not the desktop 4060: the power budget floats and
the clocks are unstable, so datasheet throughput and bandwidth figures are not
usable as roofline ceilings. Both ceilings will be measured with
micro-benchmarks instead.

## K0 - cuBLAS baseline

Not a rung to optimize, but the reference. Timed through the same harness, on
the same inputs, in the same device buffers, with no transposes or copies: the
baseline has to see exactly the memory layout the kernels see, or the
comparison is not fair. Also serves as the correctness reference for every
other kernel.

- **Measured**: 7151.8 GFLOP/s median at `4096³` (per-iteration p10/p90
  18.44/19.82 ms; `results/rtx4060-laptop/gemm_4096.csv`, row 1). The spread
  (max - min) / median is 11.7%: the GPU ran under SW power capping for the
  whole timed region (939 ms of capping counted over 2.0 s) and its SM clock
  dithered by 8.6% over the final warmup window, so the spread is clock
  dither under a power cap rather than noise.

## K1 - naive

One thread per element of C, no shared memory, no blocking. This is the floor
of the ladder and the reference ncu report that every later rung is read
against.

- **Hypothesis**: uncoalesced global access. threadIdx.x selects the row, so
  threads scheduled together in a warp read rows of A and write rows of C
  that are K and N elements apart. Each inner iteration reads 8 bytes for 2
  FLOPs, about 0.25 FLOP/byte, so the kernel should be heavily memory-bound:
  SpeedOfLight should name memory as the bottleneck and MemoryWorkloadAnalysis
  should show DRAM traffic dominating.
- **Prediction**: 45 GFLOP/s (range 25-70), recorded before the first
  benchmark run. At 0.25 FLOP/byte the roof is 0.25 x 256 GB/s = 64 GFLOP/s,
  where 256 GB/s is the theoretical bandwidth derived from the
  driver-reported memory clock and bus width; uncoalesced access should not
  reach it. A result above 64 would mean many accesses never reach DRAM (the
  L2 hit rate should show it); below 20 would point at occupancy or launch
  overhead rather than bandwidth.
- **Measured**: pending.
- **Difference**: pending.
- **Evidence**: pending.

## K2 - coalesced access

The same computation and launch shape as K1 with one change: threadIdx.x
selects the column instead of the row, and the grid dimensions swap with it.

- **Hypothesis**: for every k, the threads of a warp now read the same element
  of A, adjacent elements of one row of B, and write adjacent elements of one
  row of C, so their accesses can be coalesced. The amount of computation is
  unchanged; any speedup over K1 has to come from memory traffic.
- **Prediction**: 150 GFLOP/s (range 90-450), recorded before the first
  benchmark run. Two models disagree on the size of the gain. Counting bytes,
  a warp reads 132 bytes per 64 FLOPs instead of 256, about 2x less traffic,
  which would cap K2 near 124 GFLOP/s. Counting accesses, 32 scattered reads
  of A per warp per k become one, which would allow a gain well above 5x. The
  measured K2/K1 ratio decides between them: at or below 2.5x the kernel is
  still bandwidth-bound; at or above 5x access coalescing dominates, and
  MemoryWorkloadAnalysis should show uncoalesced and DRAM accesses collapsing.
- **Measured**: pending.
- **Difference**: pending.
- **Evidence**: pending.
