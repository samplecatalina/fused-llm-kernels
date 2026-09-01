"""Correctness check, timing and CSV output for the Triton operators.

The same discipline as csrc/harness/runner.cu:
  - every implementation is checked against a double-precision host
    reference before it is timed, and a failing one writes no row;
  - warmup is measured in time: at least --warmup-seconds of load, then until
    the mean SM clock of the last 5 s is within 1% of the 5 s before, capped;
  - each repetition is timed on its own with CUDA events, and the median,
    p10 and p90 are reported;
  - clock, temperature, power, enforced power limit and clock-event state
    are recorded around every timed region;
  - rows are refused from uncommitted code (--allow-dirty marks them) and
    without --device-tag, --log-clocks, time-based warmup and >= 100 reps.

Output allocation is inside the timed region for every implementation: eager
allocates its intermediates and its result, and the fused paths allocate
their result, as they would in use. torch.compile is compiled before warmup,
once per shape.

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
    "pct_bandwidth_roof", "pre_sm_mhz", "pre_temp_c", "pre_power_w",
    "pre_power_limit_w", "start_sm_mhz", "start_mem_mhz", "start_temp_c",
    "start_power_w", "start_power_limit_w", "start_reasons", "end_sm_mhz",
    "end_mem_mhz", "end_temp_c", "end_power_w", "end_power_limit_w",
    "end_reasons", "timed_s", "sw_power_cap_ms", "sw_thermal_ms",
    "hw_thermal_ms", "hw_power_brake_ms", "tag", "torch", "triton", "desc",
]

@dataclasses.dataclass
class Op:
    """One operator under test: y = f(x, p) with x of shape (rows, hidden) and
    a per-column parameter p of shape (hidden,)."""
    impls: object              # () -> {name: (description, factory, num_warps)}
    reference: object          # (x, p) -> float64 host tensor
    presets: dict              # "main" / "correctness" -> [(rows, hidden)]
    bytes_per_element: int     # the fused operator's minimum traffic


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
}


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
                start_s=t0, wall_s=wall)


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
        # x and the per-column parameter (bias, or weight) are uniform in [-1, 1).
        x = (torch.rand(rows, hidden, generator=gen) * 2 - 1).cuda()
        b = (torch.rand(hidden, generator=gen) * 2 - 1).cuda()
        if opt.check:
            ref = op.reference(x, b)
        print(f"\n=== {opt.op} {rows}x{hidden} ===")
        print(f"{'impl':<16} {'ms(median)':>10} {'p10':>9} {'p90':>9} "
              f"{'GB/s eff':>9} {'%roof':>7} {'max_rel':>10} {'fro_rel':>10}")
        compiled = {}
        for name in names:
            desc, factory, warps = impls[name]
            if name not in compiled:
                compiled[name] = factory()
                if name == "compile":  # compile outside every timed region
                    compiled[name](x, b)
                    torch.cuda.synchronize()
            fn = compiled[name]

            max_rel = fro_rel = 0.0
            ok = True
            if opt.check:
                got = fn(x, b)
                torch.cuda.synchronize()
                max_rel, fro_rel = vf.verify(ref, got)
                del got
                ok = vf.passed(max_rel, fro_rel)
                failures += 0 if ok else 1

            t = dict(median=0.0, p10=0.0, p90=0.0, min=0.0, max=0.0, max_rep=-1,
                     start_s=0.0, wall_s=0.0)
            w = dict(seconds=0.0, iters=0, settled="n/a", last=gm.GpuSample(),
                     mean=-1.0, range_pct=-1.0)
            pre = start = end = gm.GpuSample()
            if opt.bench:
                if opt.log_clocks or opt.min_power_limit > 0:
                    pre = gm.sample_gpu(selector)
                w = warm_up(fn, (x, b), opt, selector)
                start = w["last"]
                if opt.log_clocks and not start.valid:
                    start = gm.sample_gpu(selector)
                t = time_it(fn, (x, b), opt.reps)
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
            spread = (100.0 * (t["max"] - t["min"]) / t["median"]
                      if t["median"] > 0 else 0.0)
            print(f"{name:<16} {t['median']:>10.3f} {t['p10']:>9.3f} {t['p90']:>9.3f} "
                  f"{eff:>9.1f} {(f'{pct:.1f}%' if pct >= 0 else '-'):>7} "
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
                    f"{spread:.2f}", bytes_min, f"{eff:.3f}", pv.fmt_num(pct, ".2f"),
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
                    pv.csv_safe(desc),
                ])
                csv_file.flush()
        del x, b
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
