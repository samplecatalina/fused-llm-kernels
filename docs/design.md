# Design notes

## 1. Goals and non-goals

**Goals.** A kernel repository that demonstrates three things: that the kernels
are correct (a harness that checks arbitrary shapes), that the numbers are
trustworthy (a reproducible benchmark methodology), and that each speedup is
understood (an Nsight Compute measurement behind every claim).

Concretely:

1. An FP32 SGEMM ladder from naive to double buffering, measured at `4096³` against
   cuBLAS through the same harness.
2. One fused epilogue on the GEMM - `D = SiLU(alpha*A*B + bias)` - measured
   against the same GEMM followed by a separate element-wise pass, over the
   inner dimension K.
3. Fused Triton operators - bias + SiLU, RMSNorm and online softmax -
   compared against PyTorch eager, the native fused module where one exists,
   and `torch.compile`.
4. Stretch: a simplified fused attention forward pass.

**Non-goals**, stated explicitly to bound the scope:

- No backward passes (RMSNorm backward is optional).
- No FP16/BF16 GEMM and no Tensor Core WMMA on the main line. TF32 via WMMA is
  a possible side branch once the ladder is finished; it is a different
  baseline, not another rung.
- No CUTLASS-level generality (arbitrary layouts and epilogues); the fused
  epilogue is one fixed form, bias + SiLU, forward only.
- No Hopper-specific features on the portable FP32 main line.
- No attempt to beat cuBLAS.

**Definition of done.** `make test && make bench` reproduces every published
number from a clean checkout; every kernel has an archived ncu report; the
README carries a GFLOP/s ladder chart, a measured roofline, and one paragraph
per rung explaining its measured gain or regression.

## 2. Requirements and constraints

Every GEMM kernel has the same signature:

```
sgemm_kN(int M, int N, int K, float alpha, const float* A, const float* B,
         float beta, float* C)
```

Row-major throughout. Non-square and non-tile-divisible shapes must work.
Triton kernels must handle arbitrary hidden sizes via masking.

Correctness: compared against cuBLAS on identical inputs and identical device
buffers, requiring element-wise relative error < 1e-3 **and** relative
Frobenius error < 1e-5. Both are required because each one is blind to a
failure mode the other catches - `max_rel` is inflated by elements near zero,
while `fro_rel` averages away the handful of badly wrong elements that a
boundary bug actually produces.

`max_rel` divides by `|ref|` plus a floor of 5% of the RMS of the reference.
A fixed absolute floor rejects correct kernels: two implementations that sum
in a different order differ by rounding of up to about 2e-6 of the RMS, and on
an element that happens to be near zero that reads as a large relative error.
cuBLAS switches to a differently ordered kernel below 2048, and with a fixed
1e-5 floor every kernel in this repository failed at 1024^3 while matching
each other bit for bit. On synthetic 4096^2-element matrices the 5% floor
keeps rounding noise 5x under the tolerance at every size (a 1% floor left
only 1.1x), and a single wrong A*B term is still rejected - by 1.5x in the
worst case, on the largest element of an 8192^3 result.

Benchmarking: CUDA events, time-based warmup (at least 30 s, then until the SM
clock settles), ≥ 100 repetitions, timed per iteration, reported as median with
p10/p90. Clock, power and clock-event state are recorded before warmup, at the
start of timing and after it, together with how long each slowdown reason was
active during the timed region.

Environment constraint: clocks cannot be locked on this consumer part
(`nvidia-smi -lgc` is unavailable, and on the mobile part even `power.limit` is
not readable). This is mitigated rather than solved: time-based warmups, a per-
iteration distribution instead of an average, and the actual clock recorded
alongside every row of results. `enforced.power.limit` is readable and checked.

The device baseline is the median of repeated independent cuBLAS runs, with
the run-to-run range recorded separately from each run's p10/p90. Headline
runs cool down for at least 120 seconds first. Every CSV row records
`source_rev`, `kernel_desc`, and `baseline_check`. Uncommitted source is
rejected; an out-of-band cuBLAS baseline is flagged, so that run is used only
for comparisons within the run, not as a headline result.

## 3. The GEMM ladder

K1-K8 are implemented, measured and profiled. The evidence below is detailed in
[the optimization log](optimization-log.md), with source CSVs under
`results/rtx4060-laptop/` and counters under `profiling/reports/rtx4060-laptop/`.

| Rung | Implemented change | Observed limit or tradeoff |
|---|---|---|
| K0 | cuBLAS reference | Device and temperature dependent baseline |
| K1 | One thread per C element, scattered access | L1/TEX request path saturates despite low DRAM throughput |
| K2 | Coalesced access | Request pressure falls; DRAM bandwidth is not saturated |
| K3 | Shared-memory tiles | Manual caching costs more than the L1 reuse it replaces |
| K4 | 1D thread tiles | More resident blocks and reuse improve throughput |
| K5 | 2D register tiles | Memory path saturates; register use limits occupancy |
| K6 | float4 loads and transposed A tile | Fewer instructions, but limited parallelism |
| K7 | Warp tiles and parameter search | Smaller tiles improve occupancy at the cost of reuse |
| K8 | Double buffering, next tile stored directly into the alternate buffer | Hides some load latency, but interleaved buffers halve the L1 hit rate; net equal to K7 |

K3/K4 use 32-cubed tiles. In K4 a 128-by-128 output tile would need too
many threads with only 1D thread tiling. K5/K6 use 128-by-128-by-16 tiles
with 8-by-8 results per thread. K7's search selects 128-by-64-by-16,
8-by-4 results per thread, and 32-by-32 warp tiles. K8 keeps K7's geometry, so the
comparison isolates double buffering; its register-prefetch form (`k8c1`) and
four prefetch variants (`k8c2`-`k8c5`) remain registered.

**Negative result: K3.** Explicit shared-memory staging regresses relative
to K2. The profiles show that K2 already benefits from high L1 hit rates;
extra staging and synchronization do not pay for themselves. A completed
rung can be slower than its predecessor.

**Separate structure from tuning: K7.** The `k7c1` control keeps K6's tile
and thread count and changes only warp mapping. That change alone regresses;
the selected smaller tile supplies the gain. Search results are preserved in
`tuning_k7.csv`, rather than attributing the entire speedup to warp tiling.

**A mechanism that works but does not pay: K8.** Double buffering removes one
barrier per tile and lets global loads for the next tile proceed while the
current one is computed. The paired profiles confirm the mechanism (less time
with no eligible warp at unchanged occupancy), and also its cost: reads of one
buffer interleaved with stores to the other halve the L1 hit rate. Elapsed
cycles match K7's, and three order-balanced runs give K8/K7 = 0.994. Holding
prefetched values in registers until after the compute keeps the hit rate,
but then the no-eligible share does not fall either, and the extra register
state crosses the limit that sets four resident blocks per SM.

Boundary handling uses guards rather than padding. Every registered
configuration is checked at `4096³`, `4097×513×129`, and `64³` against cuBLAS.
Reported throughput is `2*M*N*K/time`; the size sweep covers 512 through 8192.

## 4. Profiling and roofline

Every measured rung has an ncu export covering SpeedOfLight,
MemoryWorkloadAnalysis, Occupancy, SchedulerStats, WarpStateStats and
LaunchStats. Counters explain throughput; profiler durations are not used
as headline benchmark timings.

The ceilings are measured with `make roofline`: **200.6 GB/s** from `copy_f4`
and **12.63 TFLOP/s** from `fma_f4`, meeting at **63.0 FLOP/byte**
([source CSV](../results/rtx4060-laptop/roofline.csv)). The copy uses 256 MiB
arrays; the FMA loop uses 262144 iterations per thread after testing for a
throughput plateau. These are achieved micro-benchmark rates, not universal
upper bounds. K7 reaches 73.9% of the roof at its measured intensity; see
the Roofline section of the optimization log for all rung positions.

Arithmetic intensity uses the duration and DRAM throughput from the same
ncu report. The ideal one-read-per-input traffic estimate does not describe
the actual DRAM traffic of every implementation. Each device needs its own
measured ceilings.

The K2 size sweep shows a transition at 2880-3072, consistent with **one
matrix** filling L2 (`4*N*N` bytes, N approximately 2896), rather than the
original A+B hypothesis at 2048. The descending pass and L2 counters support
this interpretation; data are in `gemm_sweep.csv`. K0 and K7 show no analogous
knee in that sweep.

## 5. Fused epilogue

**Kernels.** A separate signature, `(M, N, K, alpha, A, B, bias, D)`, with bias
broadcast over columns and no beta. `e1` applies bias and SiLU in the register
block of a K7-structured tiled GEMM; `e0` runs the same template writing
`alpha*A*B` into an intermediate buffer (allocated once, outside the timed
region) and then launches an element-wise kernel: four elements of one row per
thread, float4 access when the row start is 4-aligned. SiLU uses
`exp(-|x|)`, so it cannot overflow. The reference is the cuBLAS product with
bias and SiLU applied on the host in double precision; `make test-epilogue`
adds row tails, partial tiles, K = 1, large `alpha` (both SiLU tails) and a
nonzero initial D that neither path may read.

**Model.** With `T_fused(K)` the time of `e1` and `T_pass` the time of the
element-wise pass (independent of K), the gain is `1 + (T_pass - T_act) / T_fused(K)`,
where `T_act` is the inline activation cost. At large K, `T_fused` grows as
`2MNK/P` and the gain is `1 + c/K` with `c = P*T_pass/(2MN)`: M and N cancel.
At small K, `T_fused` is dominated by its own fixed cost (storing D, the
launch), and the gain levels off at `1 + T_pass/T_fixed`. The two asymptotes
cross near `K = P*T_fixed/(2MN)`, but the measured curve is smooth.

**Measurement.** `make epilogue`: M = N = 4096, K from 32 to 8192, order
`k0, e0, e1, e1, e0` per shape, ratio of the two-slot means. Profiles at
K = 4096 and K = 64 for both paths.

## 6. Triton operators

**Harness.** `triton_kernels/bench.py` follows the rules of the C++ runner:
float64 reference check (same two tolerances), time-based warmup until the
SM clock settles, per-iteration CUDA-event timing, clock and power state per
row, refusal of rows from uncommitted code. Result allocation is inside the
timed region for every implementation; `torch.compile` is compiled before
warmup. Each implementation runs in an early and a late slot of every shape.
Besides time ratios, every row carries an effective bandwidth - the fused
operator's minimum traffic over the median time - and its share of the
measured bandwidth roof. `make triton-profile` collects ncu for the
handwritten Triton kernel, inductor's kernel or PyTorch's eager kernels.

**Fused bias + SiLU (measured).** One program per row, masked power-of-two
block, bias indexed by column. Compared against eager `silu(x + b)`, eager
`t = x + b; t * sigmoid(t)` and `torch.compile`, for `rows = 4096`,
`hidden ∈ {1024, 2048, 4096, 8192}`. `num_warps = 4` from a bounded search.

**RMSNorm forward (measured).** One program per row,
`y = x * rsqrt(mean(x²) + ε) * w`, with `BLOCK_SIZE = next_pow2(hidden)` and
masking; the row is loaded once and scaled after the reduction. Compared
against the eager expression (six launches), `torch.nn.RMSNorm` (one fused
kernel) and `torch.compile`, over `hidden ∈ {1024, 2048, 4096, 8192}` with
`rows = 4096`; `ε = 1e-6` everywhere. `num_warps = 4` from a bounded search.
Optional extension: the backward pass.

**Row-wise softmax (measured).** One program per row: the row is loaded once
with masked positions as `-inf`, and the maximum and normalizer are parallel
reductions over the loaded values - what the online normalizer achieves with
a recurrence, without the sequential dependency. Compared against the naive
eager form (five launches, about four times the traffic of one pass),
`torch.softmax` (one kernel, with a dispatch switch between hidden 2048 and
2049) and `torch.compile`, reported both as time ratios and as effective
bandwidth against the measured roof. `num_warps = 4` from a bounded search.

## 7. Fused causal attention (measured)

Scope as fixed up front: forward only, causal, no dropout,
`head_dim ∈ {64, 128}`, FP16 inputs with FP32 accumulation, compared against
the math and flash backends of `scaled_dot_product_attention`. One program per
block of 64 queries walks the key blocks it may see, keeping a running maximum
and normalizer; the score matrix never reaches memory. Tile sizes come from a
bounded search at seq 2048.

The tolerance was fixed at 2e-2 before measuring and did not survive it: the
FP16 rounding of a single output against an FP32 reference is 2.4e-2 at seq
1024 and 6.2e-2 at seq 4096, for PyTorch's flash backend as much as for this
kernel. Acceptance is therefore the Frobenius error against the FP32 reference
(2e-3, measured 2.4e-4) plus an element-wise comparison against the flash
backend, which produces output within one FP16 ulp of this kernel.

Being compute-bound, this kernel is reported against a measured FP16 matmul
ceiling (29.8 TFLOP/s) rather than the FP32 roof.
