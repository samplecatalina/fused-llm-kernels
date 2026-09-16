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

- **Measured**: the baseline was re-established across four independent runs
  once the host power configuration was corrected (rows 2, 8, 9, 10): 9115.1,
  9045.0, 9010.9 and 8918.5 GFLOP/s, a **median of 9028.0 with a 2.18%
  run-to-run range**. The four runs fall monotonically as the part heats
  (75 to 82 C, settled SM clock 2415 down to 2346 MHz), so back-to-back runs
  drift downward and a baseline has to be quoted as a median with its
  run-to-run range rather than as a single number. Note that within one run
  the per-iteration spread (5.6% to 16.4%) is larger than the spread between
  runs.
- Each rung below is quoted against the cuBLAS run inside its own benchmark
  (9115.1 GFLOP/s, row 2), which is what the `pct_cublas` column records;
  those percentages carry the 2.2% baseline uncertainty.
- **Measured earlier, under a misconfigured host power setting**: 7151.8
  GFLOP/s (row 1, spread 11.7%), 26.2% below the corrected baseline. That run sat under SW power capping for its whole timed
  region (939 ms of capping over 2.0 s) with the SM clock dithering 8.6%
  across the final warmup window; the later run had no capping at all and
  0.0% dither. The rows are not comparable, which is why the power limit is
  now recorded in every row - the older row predates the column and leaves it
  empty. Power budget alone moved the baseline by 27%.

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
- **Measured**: 115.8 GFLOP/s, 1.27% of the cuBLAS baseline (1186.6 ms
  median, spread 2.7%; row 3).
- **Difference**: 2.6x above the predicted 45 and outside the predicted range.
  The prediction assumed every inner iteration reaches DRAM, which would cap
  the kernel at 64 GFLOP/s; measuring above that cap falsified the assumption
  exactly the way the prediction said it would.
- **Evidence** (`profiling/reports/rtx4060-laptop/k1_20260915_221324.details.csv`):
  DRAM throughput is 5.29% of peak while the L1/TEX cache path sits at 99.07%
  and its hit rate at 99.08%. The kernel is not bandwidth-bound at all; it is
  bound by the number of memory requests. The uncoalesced mapping turns one
  warp instruction into 32 separate sector requests, and each warp spends
  210.3 cycles stalled on the queue for global memory instructions, with
  226.2 warp cycles per issued instruction and no eligible warp 96.5% of the
  time. Bytes were the wrong unit for this rung; requests were the right one.

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
- **Measured**: 845.5 GFLOP/s, 9.28% of the baseline, **7.30x over K1**
  (162.5 ms median, spread 1.4%; row 4).
- **Difference**: above the predicted 150. Of the two competing models in the
  prediction, the byte model (which expected at most 2.5x) is falsified and
  the access-count model is confirmed.
- **Evidence** (`k2_20260915_221400.details.csv`): stall cycles on the global
  memory queue fall from 210.3 to 18.4 per warp and warp cycles per issued
  instruction from 226.2 to 30.2, while DRAM throughput only rises from 5.3%
  to 20.1% - the traffic in bytes barely changed, the number of requests
  collapsed. Compute and memory now sit at the same 90.6% of peak, which the
  profiler reports as a balanced workload.

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
- **Measured**: 757.6 GFLOP/s, 8.31% of the baseline, **0.90x of K2: this
  rung is 10% slower than the one before it** (181.4 ms median; row 5).
- **Difference**: the prediction's falsification clause fired - "if K3 is not
  faster than K2, synchronization or shared-memory conflicts ate the gain".
  The cause turned out to be more specific than that.
- **Evidence** (`k3_20260915_221407.details.csv`): the L1/TEX hit rate drops
  from 94.98% in K2 to **0.39%**. K2 was already being served almost entirely
  out of L1: the hardware cache was doing, for free, what this rung does by
  hand. Copying the same data into shared memory replaces those hits with an
  explicit copy plus two block-wide barriers per tile, and moves the queue
  pressure from the global-memory queue to the MIO queue (27.0 stall cycles
  per warp). DRAM throughput does fall, 20.1% to 17.7%, so the tiling does
  what it was supposed to do - there was simply nothing left to win, because
  L1 had already won it. Occupancy is unchanged at 66.7%, and with 8.19 KB of
  shared memory per 1024-thread block only one block fits per SM.
  The lesson generalizes: a manual cache only pays once the automatic one
  stops working, which is why the next rung keeps the tiles and changes what
  each thread does with them.

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
- **Measured**: 1541.4 GFLOP/s, 16.91% of the baseline, **2.03x over K3**
  (89.2 ms median; row 6).
- **Difference**: inside the predicted range of 200-1100 - the only rung whose
  absolute prediction held - and the measured 2.03x matches the predicted 1.8x
  ratio. This is also the rung whose gain could be computed in advance from
  traffic alone, without knowing which cache would catch what.
- **Evidence** (`k4_20260915_221414.details.csv`): achieved occupancy rises
  from 66.7% to 82.8% (128-thread blocks at 48 registers per thread allow 10
  blocks per SM instead of one), compute and memory both reach 98.0% of peak,
  and effective memory throughput rises from 45.1 to 81.3 GB/s. Stall cycles
  on the MIO queue fall from 27.0 to 23.6 per warp.

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
- **Measured**: 4062.9 GFLOP/s, **44.57% of the cuBLAS baseline**, 2.64x over
  K4 (33.8 ms median; row 7).
- **Difference**: above the predicted range of 350-2000, and the ratio 2.64x
  is above the predicted 1.7x. The absolute prediction chain was too low at
  every rung because its first link assumed K1 was DRAM-bound; the profile
  showed it was request-bound instead, and every later prediction inherited
  the error. The rung-to-rung ratios held up better than the absolute values:
  2.03x predicted 1.8x for K4, while K2 and K5 both beat their predicted
  ratios and K3 went the other way entirely.
- **Evidence** (`k5_20260915_221418.details.csv`): the bottleneck has moved.
  Compute sits at 59.6% of peak while memory sits at 96.4%, so the limit is
  now the memory pipe feeding the registers, not the arithmetic. Warp cycles
  per issued instruction fall from 36.3 to 10.4 and MIO stalls from 23.6 to
  3.6 cycles. This happens despite occupancy halving to 33.0%: at 121
  registers per thread only 2 blocks fit per SM, and the work per thread makes
  up for the missing warps. Shared-memory traffic per FLOP is what the next
  rung has to attack.

## K6 - float4 vectorized loads, transposed A tile

Same geometry as K5. Three changes, all about instruction count rather than
bytes: the A tile is stored transposed so the values a thread needs for one k
are contiguous, the inner loop reads both operands as float4 (16 scalar shared
loads per k become 4 vector loads), and the global loads are vectorized as
well. float4 needs 16-byte alignment, so the vector path is taken only when K
and N are multiples of 4 and the group of four stays inside the matrix;
otherwise the loads fall back to element at a time, which is what the
`4097x513x129` shape exercises.

- **Hypothesis**: K5 was limited by the memory pipe feeding the registers
  (96.4% against 59.6% compute). The same bytes, fetched with a quarter of the
  instructions, should relieve it. The transposing store costs four scalar
  stores per chunk, but it runs once per tile while the inner loop runs 16
  times, so the trade should pay.
- **Prediction**: 7534 GFLOP/s (range 5795-8577), recorded before the
  benchmark ran. The lower end, 5795, is a floor derived from K5's own profile:
  at 59.59% compute throughput the arithmetic alone cannot take less than
  33.828 ms x 0.5959 = 20.158 ms, which is 6818 GFLOP/s at perfect efficiency.
  Exceeding 6818 would mean that ceiling was not a ceiling.
- **Measured**: 6902.8 GFLOP/s, **76.82% of the cuBLAS run inside the same
  benchmark**, 1.706x over K5 (19.911 ms median, spread 6.3%).
- **Difference**: inside the predicted range but 8% under the point estimate,
  and 1.2% **past** the supposed compute ceiling: 19.911 ms against 20.158 ms.
  The ceiling assumed the arithmetic work was irreducible, but vectorizing also
  removes address arithmetic, which is compute-side work - registers per thread
  fall from 121 to 107. A ceiling derived from a profile is only a ceiling for
  changes that leave the instruction mix alone.
- **Evidence** (`profiling/reports/rtx4060-laptop/k6_20260915_234804.details.csv`):
  the memory pipe drops from 96.35% to 71.37% of peak and the L1/TEX path from
  97.33% to 73.13%, warp cycles per issued instruction from 10.43 to 6.97, and
  the share of cycles with no eligible warp from 62.01% to 41.95%. Achieved
  occupancy is unchanged at 33%. Same bytes, same occupancy, 1.7x the
  throughput: on this part the memory path is limited by requests, not by
  bytes - the same conclusion K1 and K2 reached from the other direction.
- **Where the limit is now**: memory 71.37%, compute 56.54%, neither
  saturated, occupancy 33.16% with 107 registers per thread allowing two
  blocks per SM. The limit has moved from a saturated pipe to latency and
  parallelism.
