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

## K3 - shared-memory tiling

Keeps the coalesced mapping of K2 and adds one level of blocking: for every
tile of 32 values of k, the block copies the matching 32x32 pieces of A and
B into shared memory, and the inner products read only those copies. Each
thread loads one cell of each shared array, which covers both only because
the tile is square. Threads outside the matrix still load and synchronize;
two synchronizations per tile keep loading and reading apart.

- **Hypothesis**: after K2, every inner iteration still reads global memory,
  and A and B are each larger than L2, so the kernel should be bound by DRAM
  bandwidth. Reading global memory in batches moves that bound: a tile reads
  8 KiB from DRAM and then performs 65,536 FLOPs on the shared copy, so DRAM
  traffic per FLOP drops by more than an order of magnitude. The new limits
  should be shared-memory bandwidth and the cost of 256 thread
  synchronizations per block at `4096³`. MemoryWorkloadAnalysis should show
  the DRAM share falling; SchedulerStats and WarpStateStats should show time
  spent waiting at synchronization.
- **Prediction**: 250 GFLOP/s (range 120-600), recorded before the first
  benchmark run. The range is wide because shared-memory bandwidth on this
  part has not been measured. If K3 is not faster than K2, synchronization or
  shared-memory conflicts ate the gain; if it is more than 4x faster, K2 was
  firmly bandwidth-bound, which the K2/K1 ratio should confirm independently.
- **Measured**: pending.
- **Difference**: pending.
- **Evidence**: pending.

## K4 - 1D thread tiling

Same 32x32x32 shared-memory tiles as K3. Each thread owns 8 consecutive rows
in one column of C: a block runs 32x4 threads instead of 32x32, and for
every k the thread reads the shared B value into a register once and reuses
it for all 8 results.

- **Hypothesis**: K3 still runs one thread per result, and for every k the 8
  results stacked in a column each re-read the same shared value of B. Reading
  it once cuts shared-memory traffic per FLOP from 4 to 2.25 bytes, and an
  eighth of the threads doing eight times the work dilutes per-thread
  scheduling and synchronization overhead. What remains is the A side, which
  is still read once per result.
- **Prediction**: 450 GFLOP/s (range 200-1100), recorded before the first
  benchmark run: roughly 1.8x the K3 prediction from the traffic reduction,
  plus the diluted overhead. If K4 is not faster than K3, four warps per block
  are too few to keep the SMs occupied (Occupancy's block limits should show
  it); if it is more than 3x faster, K3 was bound by thread scheduling rather
  than by shared-memory bandwidth.
- **Measured**: pending.
- **Difference**: pending.
- **Evidence**: pending.

## K5 - 2D thread tiling

Each thread owns an 8x8 block of C. For every k it reads the 8 values of A
and the 8 values of B it needs into registers and accumulates their outer
product into 64 result registers. Tiles change to 128x128 results with 16
values of k per tile: a 2D split divides the threads per block by 8 in both
directions, and 32x32 tiles would leave 16 threads per block, fewer than one
warp. This rung therefore changes two things at once, which the profile has
to separate.

- **Hypothesis**: K4 reuses registers only on the B side; the A values are
  still read once per result. With an 8x8 register block, shared-memory
  traffic falls from 2.25 to 0.5 bytes per FLOP and the inner multiply-add
  runs entirely in registers. The limit should start moving to the compute
  side: 80 register floats per thread and 8x fewer, heavier threads than K4,
  so Occupancy's register and warp limits become relevant. Accessing A down a
  column and B along a row in the same loop is left for the next rung.
- **Prediction**: 750 GFLOP/s (range 350-2000), recorded before the first
  benchmark run, about 1.7x the K4 prediction. That is roughly 10% of the
  measured cuBLAS baseline, six times below an estimate of 60-80% made before
  the ladder was started. The gap is traceable to one assumption: every
  prediction from K1 on compounds the claim that K1 is bandwidth-bound near
  0.25 FLOP/byte. If K1 measures at 150 GFLOP/s or more, the starting point
  is wrong and only the rung-to-rung ratios remain testable; if K5 reaches
  60% of cuBLAS, the earlier estimate was right. If K5 is within 10% of K4,
  register or occupancy limits cancelled the 2D split, or the tile change
  cost something, which Occupancy and LaunchStats should separate.
- **Measured**: pending.
- **Difference**: pending.
- **Evidence**: pending.
