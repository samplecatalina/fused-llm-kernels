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

- **Measured**: pending.

## K1 - naive

One thread per element of C, no shared memory, no blocking. This is the floor
of the ladder and the reference ncu report that every later rung is read
against.

- **Hypothesis**: pending.
- **Prediction**: pending.
- **Measured**: pending.
- **Difference**: pending.
- **Evidence**: pending.
