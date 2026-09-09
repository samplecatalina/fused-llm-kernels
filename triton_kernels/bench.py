"""Correctness check, timing and CSV output for the Triton operators.

The same discipline as csrc/harness/runner.cu:
  - every implementation is checked against a double-precision host
    reference before it is timed, and a failing one writes no row;
  - warmup is measured in time: at least --warmup-seconds of load, then until
    the mean SM clock of the last 5 s is within 1% of the 5 s before, capped;
  - each repetition is timed on its own with CUDA events, and the median,
    p10 and p90 are reported (--timing per-call, the default), or the
    repetitions are submitted back to back with one synchronise at the end
    and the mean per call is reported (--timing pipeline): the first measures
    the latency of one call, the second the throughput of a stream of them,
    and they differ once a call is short enough for the launch to matter;
  - clock, temperature, power, enforced power limit and clock-event state
    are recorded around every timed region;
  - rows are refused from uncommitted code (--allow-dirty marks them) and
    without --device-tag, --log-clocks, time-based warmup and >= 100 reps.

Output allocation is inside the timed region for every implementation: eager
allocates its intermediates and its result, and the fused paths allocate
their result, as they would in use. Every implementation is called once per
shape before its check and warmup, so compilation and module construction
are never timed.

Operators are registered in OPS; each brings its input generator, host
reference, implementations, preset shapes and minimum traffic per element.

Usage:
  python -m triton_kernels.bench --op bias_silu --impl triton --shape 4096x4096
  python -m triton_kernels.bench --op bias_silu --preset main --impl eager_composite,eager,compile,triton,triton,compile,eager,eager_composite ...
  python -m triton_kernels.bench --op bias_silu --preset correctness --impl all --no-bench
"""
import argparse
import csv
import dataclasses
import os
import statistics
import sys
import time

import torch
import triton

from . import bias_silu as bs
from . import gpu_monitor as gm
from . import provenance as pv
from . import rmsnorm as rn
from . import rmsnorm_backward as rb
from . import softmax as sm
from . import verify as vf

SETTLE_WINDOW = 5
SETTLE_TOL = 0.01
SAMPLE_INTERVAL_S = 1.0
SPREAD_WARN_PCT = 5.0
MIN_PUBLISH_REPS = 100

HEADER = [
    "timestamp_utc", "source_rev", "device_tag", "gpu", "op", "impl", "rows",
    "hidden", "num_warps", "check", "max_rel", "fro_rel", "warmup_s_min",
    "warmup_s", "warmup_iters", "warmup_settled", "warmup_sm_mean_mhz",
    "warmup_sm_range_pct", "reps", "ms_median", "ms_p10", "ms_p90", "ms_min",
    "ms_max", "ms_max_rep", "spread_pct", "bytes_min", "eff_bandwidth_gbs",
    "pct_bandwidth_roof", "flops_per_call", "tflops", "pre_sm_mhz", "pre_temp_c", "pre_power_w",
    "pre_power_limit_w", "start_sm_mhz", "start_mem_mhz", "start_temp_c",
    "start_power_w", "start_power_limit_w", "start_reasons", "end_sm_mhz",
    "end_mem_mhz", "end_temp_c", "end_power_w", "end_power_limit_w",
    "end_reasons", "timed_s", "sw_power_cap_ms", "sw_thermal_ms",
    "hw_thermal_ms", "hw_power_brake_ms", "tag", "torch", "triton", "timing",
    "submit_ms", "desc",
]

@dataclasses.dataclass
class Op:
    """One operator under test.

    By default the inputs are x of shape (rows, hidden) and a per-column
    parameter p of shape (hidden,), and an implementation is called as
    f(x, p). An operator with other inputs supplies make_inputs, which
    returns the argument tuple; implementations and the reference take the
    same tuple. The reference returns a host float64 tensor, or a tuple of
    them when the operator produces several outputs.
    """
    impls: object              # () -> {name: (description, factory, num_warps)}
    reference: object          # (*args) -> float64 host tensor(s)
    presets: dict              # "main" / "correctness" -> [(rows, hidden)]
    bytes_per_element: int = 0  # minimum traffic; 0 leaves the column empty
    make_inputs: object = None  # (rows, hidden, gen) -> tuple of arguments
    flops: object = None        # (rows, hidden) -> FLOPs per call, or None
    labels: tuple = ("rows", "hidden")   # what the two shape numbers mean


def bias_silu_impls():
    impls = {
        "eager_composite": ("eager: t = x + b; t * sigmoid(t) (3 launches)",
                            lambda: bs.eager_composite, None),
        "eager": ("eager: silu(x + b) (2 launches)", lambda: bs.eager_native, None),
        "compile": ("torch.compile of silu(x + b)",
                    lambda: torch.compile(bs.eager_native, dynamic=False), None),
        "triton": (f"Triton, one program per row, num_warps={bs.DEFAULT_NUM_WARPS}",
                   lambda: bs.bias_silu, bs.DEFAULT_NUM_WARPS),
    }
    for w in (2, 4, 8, 16):
        impls[f"triton_w{w}"] = (
            f"Triton, one program per row, num_warps={w}",
            lambda w=w: (lambda x, b: bs.bias_silu(x, b, num_warps=w)), w)
    return impls


def rmsnorm_impls():
    def native():
        # The module holds its own copy of the weight; build it on first call
        # (outside every timed region) for the weight passed in.
        cache = {}

        def run(x, w):
            key = (w.data_ptr(), w.shape[0])
            if key not in cache:
                cache.clear()
                cache[key] = rn.native_module(w)
            return cache[key](x)
        return run

    impls = {
        "eager": ("eager: x * rsqrt(pow(x, 2).mean(-1) + eps) * w (6 launches)",
                  lambda: rn.eager, None),
        "native": ("torch.nn.RMSNorm (1 launch)", native, None),
        "compile": ("torch.compile of the eager expression",
                    lambda: torch.compile(rn.eager, dynamic=False), None),
        "triton": (f"Triton, one program per row, num_warps={rn.DEFAULT_NUM_WARPS}",
                   lambda: rn.rmsnorm, rn.DEFAULT_NUM_WARPS),
    }
    for w in (2, 4, 8, 16):
        impls[f"triton_w{w}"] = (
            f"Triton, one program per row, num_warps={w}",
            lambda w=w: (lambda x, p: rn.rmsnorm(x, p, num_warps=w)), w)
    return impls


def rmsnorm_bwd_impls():
    impls = {
        "eager": ("autograd through the eager expression",
                  lambda: rb.autograd_backward_fn(rb.eager_forward), None),
        "native": ("autograd through PyTorch's fused RMSNorm",
                   lambda: rb.autograd_backward_fn(rb.native_forward), None),
        "compile": ("autograd through the compiled eager expression",
                    lambda: rb.autograd_backward_fn(
                        torch.compile(rb.eager_forward, dynamic=False)), None),
        "triton": (f"Triton, dx and partial dw in one kernel, "
                   f"num_warps={rb.DEFAULT_NUM_WARPS}",
                   lambda: rb.rmsnorm_backward, rb.DEFAULT_NUM_WARPS),
    }
    for w in (2, 4, 8, 16):
        impls[f"triton_w{w}"] = (
            f"Triton, dx and partial dw in one kernel, num_warps={w}",
            lambda w=w: (lambda x, p, dy: rb.rmsnorm_backward(x, p, dy, num_warps=w)), w)
    return impls


def rmsnorm_bwd_inputs(rows, hidden, gen):
    x = (torch.rand(rows, hidden, generator=gen) * 2 - 1).cuda()
    w = (torch.rand(hidden, generator=gen) * 2 - 1).cuda()
    dy = (torch.rand(rows, hidden, generator=gen) * 2 - 1).cuda()
    return x, w, dy


def softmax_impls():
    # Softmax has no per-column parameter: every callable ignores it.
    impls = {
        "eager": ("eager: max, x - max, exp, sum, divide (5 launches)",
                  lambda: (lambda x, _p: sm.eager(x)), None),
        "native": ("torch.softmax (1 launch)", lambda: (lambda x, _p: sm.native(x)), None),
        "compile": ("torch.compile of the eager expression",
                    lambda: (lambda f: (lambda x, _p: f(x)))(
                        torch.compile(sm.eager, dynamic=False)), None),
        "triton": (f"Triton, one program per row, num_warps={sm.DEFAULT_NUM_WARPS}",
                   lambda: (lambda x, _p: sm.softmax(x)), sm.DEFAULT_NUM_WARPS),
    }
    for w in (2, 4, 8, 16):
        impls[f"triton_w{w}"] = (
            f"Triton, one program per row, num_warps={w}",
            lambda w=w: (lambda x, _p: sm.softmax(x, num_warps=w)), w)
    return impls


OPS = {
    "bias_silu": Op(
        impls=bias_silu_impls,
        reference=vf.reference_bias_silu,
        presets={
            # Benchmark matrix: rows = 4096, hidden over 1024..8192.
            "main": [(4096, h) for h in (1024, 2048, 4096, 8192)],
            # Tails, a single row, non-power-of-two widths, the block limit.
            "correctness": [(1, 1), (3, 5), (4097, 513), (64, 64), (7, 8191),
                            (2, bs.MAX_COLS), (4096, 4096)],
        },
        bytes_per_element=8,  # load x, store y; the bias is served from L1
    ),
    "rmsnorm": Op(
        impls=rmsnorm_impls,
        reference=rn.reference,
        presets={
            "main": [(4096, h) for h in (1024, 2048, 4096, 8192)],
            "correctness": [(1, 1), (3, 5), (4097, 513), (64, 64), (7, 8191),
                            (2, rn.MAX_COLS), (4096, 4096)],
        },
        bytes_per_element=8,  # load x, store y; the weight is served from L1
    ),
    "rmsnorm_bwd": Op(
        impls=rmsnorm_bwd_impls,
        reference=rb.reference,
        make_inputs=rmsnorm_bwd_inputs,
        presets={
            "main": [(4096, h) for h in (1024, 2048, 4096, 8192)],
            # hidden 1 is excluded: dx is a difference of two nearly equal
            # terms there and no float32 implementation is accurate (see
            # rmsnorm_backward.conditioning).
            "correctness": [(3, 5), (4097, 513), (64, 64), (7, 8191),
                            (2, rb.MAX_COLS), (4096, 4096)],
        },
        bytes_per_element=12,  # read dy, read x, write dx
    ),
    "softmax": Op(
        impls=softmax_impls,
        reference=lambda x, _p: sm.reference(x),
        presets={
            # torch.softmax switches implementation between hidden 2048 and
            # 2049, so 2048 and 4096 exercise its two paths.
            "main": [(4096, h) for h in (1024, 2048, 4096, 8192)],
            "correctness": [(1, 1), (3, 5), (4097, 513), (64, 64), (7, 8191),
                            (5, 2048), (5, 2049), (2, sm.MAX_COLS), (4096, 4096)],
        },
        bytes_per_element=8,  # load x, store y
    ),
}


def default_inputs(rows, hidden, gen):
    """x and the per-column parameter (bias, or weight), uniform in [-1, 1)."""
    x = (torch.rand(rows, hidden, generator=gen) * 2 - 1).cuda()
    p = (torch.rand(hidden, generator=gen) * 2 - 1).cuda()
    return x, p


def verify_outputs(ref, got):
    """Worst max_rel and fro_rel over one or several outputs."""
    if isinstance(ref, tuple):
        pairs = [vf.verify(r, g) for r, g in zip(ref, got)]
        return max(p[0] for p in pairs), max(p[1] for p in pairs)
    return vf.verify(ref, got)


def parse_shape(s):
    try:
        r, h = (int(v) for v in s.lower().split("x"))
    except ValueError:
        raise argparse.ArgumentTypeError("shape must be ROWSxHIDDEN")
    if r <= 0 or h <= 0:
        raise argparse.ArgumentTypeError("shape must be positive")
    return r, h


def bandwidth_roof(device_tag):
    """copy_f4 row of results/<device>/roofline.csv, in GB/s, or None."""
    path = pv.REPO / "results" / device_tag / "roofline.csv"
    try:
        with open(path, newline="") as f:
            for r in csv.DictReader(f):
                if r["bench"] == "copy_f4":
                    return float(r["gb_per_s"])
    except (OSError, KeyError, ValueError):
        pass
    return None


def warm_up(fn, args, opt, selector):
    if opt.warmup_seconds <= 0:
        for _ in range(opt.warmup_iters):
            fn(*args)
        torch.cuda.synchronize()
        return dict(seconds=0.0, iters=opt.warmup_iters, settled="n/a",
                    last=gm.GpuSample(), mean=-1.0, range_pct=-1.0)
    cap = opt.warmup_max_seconds or 4 * opt.warmup_seconds
    sampler = gm.ClockSampler(selector, SAMPLE_INTERVAL_S)
    sampler.start()
    t0 = time.monotonic()
    last_check = t0
    iters = 0
    settled = False
    while True:
        fn(*args)
        torch.cuda.synchronize()
        iters += 1
        now = time.monotonic()
        if now - t0 < opt.warmup_seconds or now - last_check < 0.25:
            continue
        last_check = now
        settled = gm.clock_settled(sampler.snapshot(), SETTLE_WINDOW, SETTLE_TOL)
        if settled or now - t0 >= cap:
            break
    seconds = time.monotonic() - t0
    sampler.stop()
    samples = sampler.snapshot()
    mean, rng = gm.clock_window(samples, SETTLE_WINDOW)
    return dict(seconds=seconds, iters=iters,
                settled=("unknown" if not samples else "yes" if settled else "no"),
                last=samples[-1] if samples else gm.GpuSample(),
                mean=mean, range_pct=rng)


def time_pipeline(fn, args, reps):
    """Submit `reps` calls back to back, synchronise once. Reported as the
    mean per call: individual calls cannot be separated in this regime."""
    torch.cuda.synchronize()
    t0 = time.monotonic()
    for _ in range(reps):
        fn(*args)
    submit_s = time.monotonic() - t0
    torch.cuda.synchronize()
    wall = time.monotonic() - t0
    per_call = wall / reps * 1e3
    return dict(median=per_call, p10=per_call, p90=per_call, min=per_call,
                max=per_call, max_rep=-1, start_s=t0, wall_s=wall,
                submit_ms=submit_s / reps * 1e3)


def time_it(fn, args, reps):
    start = torch.cuda.Event(enable_timing=True)
    stop = torch.cuda.Event(enable_timing=True)
    times = []
    t0 = time.monotonic()
    for _ in range(reps):
        start.record()
        fn(*args)
        stop.record()
        stop.synchronize()
        times.append(start.elapsed_time(stop))
    wall = time.monotonic() - t0
    max_rep = max(range(reps), key=times.__getitem__)
    q = statistics.quantiles(times, n=10, method="inclusive") if reps > 1 else times * 9
    return dict(median=statistics.median(times), p10=q[0], p90=q[8],
                min=min(times), max=max(times), max_rep=max_rep,
                start_s=t0, wall_s=wall, submit_ms=-1.0)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--op", choices=sorted(OPS), default="bias_silu")
    ap.add_argument("--impl", default="",
                    help="comma separated; repeat a name to time it in two slots; "
                         "'all' for every registered implementation (default: all)")
    ap.add_argument("--shape", type=parse_shape, action="append", default=[],
                    help="ROWSxHIDDEN, repeatable (default 4096x4096)")
    ap.add_argument("--preset", choices=["main", "correctness"])
    ap.add_argument("--reps", type=int, default=100)
    ap.add_argument("--timing", choices=["per-call", "pipeline"],
                    default="per-call",
                    help="per-call: CUDA events around every call (latency); "
                         "pipeline: back-to-back submission, one synchronise, "
                         "mean per call (throughput)")
    ap.add_argument("--warmup-seconds", type=float, default=30.0)
    ap.add_argument("--warmup-max-seconds", type=float, default=0.0)
    ap.add_argument("--warmup", dest="warmup_iters", type=int, default=0,
                    help="count-based warmup, only with --warmup-seconds 0")
    ap.add_argument("--log-clocks", action="store_true")
    ap.add_argument("--device-tag", default="")
    ap.add_argument("--min-power-limit", type=float, default=0.0)
    ap.add_argument("--allow-dirty", action="store_true")
    ap.add_argument("--no-check", dest="check", action="store_false")
    ap.add_argument("--no-bench", dest="bench", action="store_false")
    ap.add_argument("--csv", default="")
    ap.add_argument("--tag", default="")
    opt = ap.parse_args(argv)
    # Progress lines and refusals must interleave correctly in a log file.
    sys.stdout.reconfigure(line_buffering=True)

    op = OPS[opt.op]
    impls = op.impls()
    names = [n for n in opt.impl.split(",") if n]
    if not names or names == ["all"]:
        names = list(impls)
    unknown = [n for n in names if n not in impls]
    if unknown:
        ap.error(f"unknown impl: {', '.join(unknown)} (known: {', '.join(impls)})")
    shapes = list(op.presets[opt.preset]) if opt.preset else []
    shapes += opt.shape
    if not shapes:
        shapes = [(4096, 4096)]
    if opt.reps < 1:
        ap.error("--reps must be >= 1")
    if opt.warmup_seconds > 0 and opt.warmup_iters > 0:
        ap.error("--warmup N only applies with --warmup-seconds 0")
    if opt.device_tag and not pv.valid_device_tag(opt.device_tag):
        ap.error("--device-tag must match [a-z0-9-]+")
    if opt.csv and opt.bench:
        missing = [m for m, bad in (
            ("--device-tag", not opt.device_tag),
            ("--log-clocks", not opt.log_clocks),
            ("--warmup-seconds > 0", opt.warmup_seconds <= 0),
            (f"--reps >= {MIN_PUBLISH_REPS}", opt.reps < MIN_PUBLISH_REPS)) if bad]
        if missing:
            ap.error("--csv requires: " + " ".join(missing))

    if not torch.cuda.is_available():
        print("CUDA is not available to torch", file=sys.stderr)
        return 1
    props = torch.cuda.get_device_properties(0)
    selector = pv.gpu_selector(props)
    rev = pv.source_revision()
    print(f"GPU: {props.name}  sm_{props.major}{props.minor}  torch {torch.__version__}  "
          f"triton {triton.__version__}")
    print(f"device tag: {opt.device_tag or '(none)'}  source: {rev}")
    if opt.csv and opt.bench and pv.is_untraceable(rev) and not opt.allow_dirty:
        print(f"refusing to write CSV rows from uncommitted code (source {rev}): "
              "commit first, or pass --allow-dirty for an exploratory run",
              file=sys.stderr)
        return 1
    if opt.bench and opt.min_power_limit > 0:
        s = gm.sample_gpu(selector)
        if not s.valid or s.power_limit_w < 0:
            print("cannot read the enforced power limit; refusing to benchmark "
                  "with --min-power-limit set", file=sys.stderr)
            return 1
        if s.power_limit_w < opt.min_power_limit:
            print(f"enforced power limit {s.power_limit_w:.1f} W is below the "
                  f"required {opt.min_power_limit:.1f} W: benchmark conditions "
                  "not met", file=sys.stderr)
            return 1
        print(f"power limit: {s.power_limit_w:.1f} W "
              f"(required >= {opt.min_power_limit:.1f} W)")
    roof = bandwidth_roof(opt.device_tag) if opt.device_tag else None

    writer = None
    csv_file = None
    if opt.csv and opt.bench:
        exists = os.path.exists(opt.csv) and os.path.getsize(opt.csv) > 0
        if exists:
            with open(opt.csv, newline="") as f:
                if next(csv.reader(f), None) != HEADER:
                    print(f"CSV header mismatch in {opt.csv}: use a new file",
                          file=sys.stderr)
                    return 1
        os.makedirs(os.path.dirname(opt.csv) or ".", exist_ok=True)
        csv_file = open(opt.csv, "a", newline="")
        writer = csv.writer(csv_file, lineterminator="\n")
        if not exists:
            writer.writerow(HEADER)

    failures = 0
    gen = torch.Generator(device="cpu").manual_seed(1234)
    for rows, hidden in shapes:
        args = (op.make_inputs(rows, hidden, gen) if op.make_inputs
                else default_inputs(rows, hidden, gen))
        if opt.check:
            ref = op.reference(*args)
        print(f"\n=== {opt.op} {op.labels[0]}={rows} {op.labels[1]}={hidden} ===")
        print(f"{'impl':<16} {'ms(median)':>10} {'p10':>9} {'p90':>9} "
              f"{'GB/s eff':>9} {'%roof':>7} {'TFLOP/s':>8} {'max_rel':>10} "
              f"{'fro_rel':>10}")
        compiled = {}
        for name in names:
            desc, factory, warps = impls[name]
            if name not in compiled:
                compiled[name] = factory()
                # One untimed call per shape: compiles (torch.compile, Triton
                # JIT) and builds module state (nn.RMSNorm) outside every
                # timed region.
                compiled[name](*args)
                torch.cuda.synchronize()
            fn = compiled[name]

            max_rel = fro_rel = 0.0
            ok = True
            if opt.check:
                got = fn(*args)
                torch.cuda.synchronize()
                max_rel, fro_rel = verify_outputs(ref, got)
                del got
                ok = vf.passed(max_rel, fro_rel)
                failures += 0 if ok else 1

            t = dict(median=0.0, p10=0.0, p90=0.0, min=0.0, max=0.0, max_rep=-1,
                     start_s=0.0, wall_s=0.0, submit_ms=-1.0)
            w = dict(seconds=0.0, iters=0, settled="n/a", last=gm.GpuSample(),
                     mean=-1.0, range_pct=-1.0)
            pre = start = end = gm.GpuSample()
            if opt.bench:
                if opt.log_clocks or opt.min_power_limit > 0:
                    pre = gm.sample_gpu(selector)
                w = warm_up(fn, args, opt, selector)
                start = w["last"]
                if opt.log_clocks and not start.valid:
                    start = gm.sample_gpu(selector)
                t = (time_it(fn, args, opt.reps) if opt.timing == "per-call"
                     else time_pipeline(fn, args, opt.reps))
                if opt.log_clocks or opt.min_power_limit > 0:
                    end = gm.sample_gpu(selector)
            power_ok = (not opt.bench or opt.min_power_limit <= 0 or
                        (end.valid and end.power_limit_w >= opt.min_power_limit))
            if not power_ok:
                print(f"       ! enforced power limit dropped to "
                      f"{end.power_limit_w:.1f} W during the run")
                failures += 1

            bytes_min = op.bytes_per_element * rows * hidden
            eff = (bytes_min / (t["median"] * 1e-3) / 1e9) if t["median"] > 0 else 0.0
            pct = 100.0 * eff / roof if (roof and eff > 0) else -1.0
            flops = op.flops(rows, hidden) if op.flops else 0
            tflops = (flops / (t["median"] * 1e-3) / 1e12) if (flops and t["median"] > 0) else -1.0
            spread = (100.0 * (t["max"] - t["min"]) / t["median"]
                      if t["median"] > 0 else 0.0)
            print(f"{name:<16} {t['median']:>10.3f} {t['p10']:>9.3f} {t['p90']:>9.3f} "
                  f"{(f'{eff:.1f}' if bytes_min else '-'):>9} "
                  f"{(f'{pct:.1f}%' if pct >= 0 else '-'):>7} "
                  f"{(f'{tflops:.2f}' if tflops >= 0 else '-'):>8} "
                  f"{max_rel:>10.2e} {fro_rel:>10.2e}"
                  f"{'' if ok else '  << FAIL'}")
            if opt.bench:
                if w["settled"] != "n/a":
                    print(f"       warmup {w['seconds']:.1f} s, {w['iters']} iters, "
                          f"SM clock {w['settled']}: mean {w['mean']:.0f} MHz, "
                          f"range {w['range_pct']:.1f}%")
                if opt.log_clocks and start.valid and end.valid:
                    print(f"       timed {t['wall_s']:.1f} s, SM clock "
                          f"{start.sm_mhz} -> {end.sm_mhz} MHz, "
                          f"{start.temp_c} -> {end.temp_c} C")
                if spread > SPREAD_WARN_PCT:
                    print(f"       ! spread (max-min)/median = {spread:.1f}% "
                          f"(slowest: rep {t['max_rep']} of {opt.reps})")

            if writer and not ok:
                print("       row not written: correctness check failed")
            if writer and ok and not power_ok:
                print("       row not written: power limit below the minimum")
            if writer and ok and power_ok:
                writer.writerow([
                    pv.utc_timestamp(), rev, opt.device_tag, pv.csv_safe(props.name),
                    opt.op, name, rows, hidden, warps or "",
                    "pass" if opt.check else "skipped",
                    f"{max_rel:.3e}" if opt.check else "",
                    f"{fro_rel:.3e}" if opt.check else "",
                    f"{opt.warmup_seconds:.1f}", f"{w['seconds']:.2f}", w["iters"],
                    w["settled"], pv.fmt_num(w["mean"], ".0f"),
                    pv.fmt_num(w["range_pct"], ".2f"), opt.reps,
                    f"{t['median']:.6f}", f"{t['p10']:.6f}", f"{t['p90']:.6f}",
                    f"{t['min']:.6f}", f"{t['max']:.6f}", t["max_rep"],
                    f"{spread:.2f}", bytes_min or "", f"{eff:.3f}" if bytes_min else "",
                    pv.fmt_num(pct, ".2f"), flops or "", pv.fmt_num(tflops, ".4f"),
                    pv.fmt_int(pre.sm_mhz), pv.fmt_int(pre.temp_c),
                    pv.fmt_num(pre.power_w, ".2f"), pv.fmt_num(pre.power_limit_w, ".2f"),
                    pv.fmt_int(start.sm_mhz), pv.fmt_int(start.mem_mhz),
                    pv.fmt_int(start.temp_c), pv.fmt_num(start.power_w, ".2f"),
                    pv.fmt_num(start.power_limit_w, ".2f"), start.reasons,
                    pv.fmt_int(end.sm_mhz), pv.fmt_int(end.mem_mhz),
                    pv.fmt_int(end.temp_c), pv.fmt_num(end.power_w, ".2f"),
                    pv.fmt_num(end.power_limit_w, ".2f"), end.reasons,
                    f"{t['wall_s']:.2f}",
                    pv.fmt_num(gm.active_ms(start.us_sw_power_cap, end.us_sw_power_cap), ".1f"),
                    pv.fmt_num(gm.active_ms(start.us_sw_thermal, end.us_sw_thermal), ".1f"),
                    pv.fmt_num(gm.active_ms(start.us_hw_thermal, end.us_hw_thermal), ".1f"),
                    pv.fmt_num(gm.active_ms(start.us_hw_power_brake, end.us_hw_power_brake), ".1f"),
                    pv.csv_safe(opt.tag), torch.__version__, triton.__version__,
                    opt.timing, pv.fmt_num(t["submit_ms"], ".6f"),
                    pv.csv_safe(desc),
                ])
                csv_file.flush()
        del args
        torch.cuda.empty_cache()

    if csv_file:
        csv_file.close()
    if failures:
        print(f"\n!! {failures} check(s) failed")
        return 1
    print("\nall checks passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
