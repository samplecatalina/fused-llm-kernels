# Design notes

## 1. Goals and non-goals

**Goals.** A kernel repository that demonstrates three things: that the kernels
are correct (a harness that checks arbitrary shapes), that the numbers are
trustworthy (a reproducible benchmark methodology), and that each speedup is
understood (an Nsight Compute measurement behind every claim).

Concretely:

1. An FP32 SGEMM ladder from naive to warp tiling, measured at `4096³` against
   cuBLAS through the same harness.
2. Fused Triton operators - RMSNorm and online softmax - compared against both
   PyTorch eager and `torch.compile`.
3. Stretch: a simplified fused attention forward pass.

**Non-goals**, stated explicitly to bound the scope:

- No backward passes (RMSNorm backward is optional).
- No FP16/BF16 GEMM and no Tensor Core WMMA on the main line. TF32 via WMMA is
  a possible side branch once the ladder is finished; it is a different
  baseline, not another rung.
- No CUTLASS-level generality (arbitrary layouts and epilogues).
- No Hopper-specific features - no access to the hardware.
- No attempt to beat cuBLAS.

**Definition of done.** `make test && make bench` reproduces every published
number from a clean checkout; every kernel has an archived ncu report; the
README carries a GFLOP/s ladder chart, a measured roofline, and one paragraph
per rung explaining why it is faster.

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
alongside every row of results.

## 3. The GEMM ladder

| Rung | Content | Expected bottleneck shift |
|---|---|---|
| K0 | cuBLAS baseline, timed through the same harness | - |
| K1 | naive: one thread per element of C | uncoalesced global access |
| K2 | coalesced access (swap the thread index mapping) | DRAM bandwidth |
| K3 | shared-memory tiling (BM×BK / BK×BN) | shared bandwidth, compute ratio too low |
| K4 | 1D thread tiling (TM results per thread) | insufficient register reuse |
| K5 | 2D thread tiling (TM×TN register block) | instruction scheduling |
| K6 | float4 vectorized loads + transposed A tile | shared-memory bank conflicts |
| K7 | warp tiling (a second blocking level per warp) | scheduling efficiency |
| K8 | double buffering / shared prefetch pipeline (optional) | latency hiding |

Starting parameters: `BM = BN = 128`, `BK = 8..16`, `TM = TN = 8`. From K7 on,
a small scripted grid search over `(BM, BN, BK, TM, TN, WM, WN)`; results go to
CSV, and the README reports the best configuration together with the search
range that produced it.

Boundary handling uses guard branches rather than padding, so correctness holds
for any `M, N, K`.

Every rung is checked on three shapes - `4096³` (the headline), `4097×513×129`
(non-divisible), `64³` (small enough that launch overhead dominates) - against
K0. Reported as `GFLOP/s = 2·M·N·K / t`, plus a 512→8192 sweep. With 32 MB of
L2 on this part, the sweep is expected to show a knee where the combined
footprint of A and B crosses the L2 capacity.

## 4. Profiling and roofline

At least one ncu collection per rung, over these sections: SpeedOfLight,
MemoryWorkloadAnalysis, Occupancy, SchedulerStats, WarpStateStats. The "why is
it faster" paragraph for each rung must cite the corresponding measurement;
theory alone does not count.

**Roofline ceilings are measured, not quoted** (`make roofline`): a float4
copy over 256 MiB arrays for achievable bandwidth (200.6 GB/s on the
development GPU), a register-only FMA loop for sustained FP32 compute (12.63
TFLOP/s), and a triad as a check that sits on the slanted roof. The earlier
plan follows. A triad/copy micro-benchmark
for achievable DRAM bandwidth, an FMA saturation micro-benchmark for achievable
FP32 throughput, and the ridge point computed from those two. The reason is
specific to this machine: the mobile 4060's power budget and clocks move with
temperature, so datasheet figures would shift the entire plot. The README will
state that the ceilings are locally measured and will ship the micro-benchmark
code alongside them.

## 5. Triton operators

**RMSNorm forward.** One program per row, `y = x * rsqrt(mean(x²) + ε) * w`,
with `BLOCK_SIZE = next_pow2(hidden)` and masking. Benchmarked over
`hidden ∈ {1024, 2048, 4096, 8192}` with `rows = 4096`, against both PyTorch
eager and `torch.compile`. Optional extension: the backward pass.

**Online softmax.** Single-pass running-max formulation compared against
`torch.softmax`. The point of interest is the derivation of the reduction in
memory traffic from three passes to one, checked against measured bandwidth.

## 6. Simplified fused attention (stretch)

Scope fixed up front: forward only, causal, no dropout, `head_dim ∈ {64, 128}`,
FP16 inputs with FP32 accumulation. Triton implementation, compared against the
math and flash backends of `scaled_dot_product_attention`, with the tolerance
relaxed to 2e-2. With 8 GB of memory on this part, the maximum sequence length
has to be determined empirically.
