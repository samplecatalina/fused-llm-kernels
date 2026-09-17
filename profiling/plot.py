"""Figures for the README: the size sweep and the roofline.

Every value plotted is read from results/ or from an exported ncu report;
nothing is typed in by hand. Run from the repository root:

    python3 profiling/plot.py [--device rtx4060-laptop]
"""
import argparse
import csv
import glob
import statistics
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.ticker import FuncFormatter, NullFormatter  # noqa: E402

# Validated categorical slots (light surface), text and chrome tokens.
SERIES = ["#2a78d6", "#eb6834", "#1baf7a"]
SURFACE = "#fcfcfb"
TEXT = "#0b0b0b"
TEXT2 = "#52514e"
GRID = "#e6e5e1"
REF = "#b9b8b2"  # reference lines: visible, but quieter than data

FLOP_4096 = 2 * 4096**3
# Candidate knee positions: A + B = 8 N^2 bytes equal to the L2 size.
L2_BYTES = 33_554_432
L2_PERSIST_BYTES = 23_068_672
N_L2 = (L2_BYTES / 8) ** 0.5
N_L2_PERSIST = (L2_PERSIST_BYTES / 8) ** 0.5
# Formed after the first sweep: a single matrix (4 N^2 bytes) filling L2.
N_L2_ONE = (L2_BYTES / 4) ** 0.5


def style(ax):
    ax.set_facecolor(SURFACE)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)
        ax.spines[side].set_linewidth(1)
    ax.tick_params(which="both", colors=TEXT2, labelsize=8, length=0)
    ax.grid(True, color=GRID, linewidth=0.8, linestyle="-")
    ax.set_axisbelow(True)


def read(path):
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def knee(points):
    """First N >= 1280 whose throughput is at least 5% under the best of all
    smaller N, with the next grid point under that bar too."""
    points = sorted(points)
    for i in range(1, len(points) - 1):
        n, g = points[i]
        if n < 1280:
            continue
        best = max(v for _, v in points[:i])
        best_next = max(v for _, v in points[: i + 1])
        if g <= 0.95 * best and points[i + 1][1] <= 0.95 * best_next:
            return n
    return None


def plot_sweep(rows, out):
    by = {}
    for r in rows:
        # "refine" rows are extra ascending-order sizes added around the step.
        key = (r["kernel"], "descending" if r["tag"] == "desc" else "ascending")
        by.setdefault(key, []).append((int(r["M"]), float(r["gflops_median"])))
    panels = [("k0", "cuBLAS"), ("k2", "K2 coalesced, untiled"), ("k7", "K7 tuned")]
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.8), facecolor=SURFACE)
    knees = {}
    for ax, (k, title) in zip(axes, panels):
        style(ax)
        passes = [p for p in ("ascending", "descending") if (k, p) in by]
        for j, p in enumerate(passes):
            pts = sorted(by[(k, p)])
            xs, ys = zip(*pts)
            ax.plot(xs, ys, color=SERIES[j], linewidth=2, solid_capstyle="round",
                    marker="o", markersize=5, markeredgecolor=SURFACE,
                    markeredgewidth=1.5, label=f"{p} sizes", zorder=3)
            kn = knee(pts)
            knees[(k, p)] = kn
            if kn is not None and j == 0:
                y = dict(pts)[kn]
                ax.annotate(f"knee: {kn}", (kn, y), xytext=(28, -46),
                            textcoords="offset points", fontsize=8, color=TEXT,
                            arrowprops=dict(arrowstyle="-", color=TEXT2, lw=0.8))
        refs = ((N_L2_PERSIST, "A+B fills persisting L2", 0.42),
                (N_L2, "A+B fills L2", 0.30),
                (N_L2_ONE, "one matrix fills L2", 0.18))
        for x, lab, y in refs:
            ax.axvline(x, color=REF, linewidth=1, zorder=1)
            if k == "k2":  # label once, on the panel the lines are about
                ax.text(x * 1.03, y, f"{lab}\nN = {x:.0f}",
                        transform=ax.get_xaxis_transform(), fontsize=7,
                        color=TEXT2, va="top")
        ax.set_xscale("log", base=2)
        ax.set_xticks([512, 1024, 2048, 4096, 8192])
        ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{int(v)}"))
        ax.xaxis.set_minor_formatter(NullFormatter())
        ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:,.0f}"))
        ax.set_ylim(bottom=0)
        ax.set_title(title, loc="left", fontsize=10, color=TEXT)
        ax.set_xlabel("N (square A, B, C)", fontsize=8, color=TEXT2)
        if len(passes) > 1:
            ax.legend(frameon=False, fontsize=8, labelcolor=TEXT2, loc="lower left")
    axes[0].set_ylabel("GFLOP/s", fontsize=8, color=TEXT2)
    fig.suptitle("Throughput against size. Vertical lines: where the data would fill "
                 "the part's 32 MiB L2 (and its 22 MiB persisting share)",
                 x=0.01, ha="left", fontsize=11, color=TEXT)
    fig.tight_layout()
    fig.savefig(out, dpi=150, facecolor=SURFACE)
    plt.close(fig)
    return knees


# Epilogue prediction, committed before the first measurement: c = P *
# T_pass / (2 M N) with P = 7900 GFLOP/s and T_pass = 1.0 ms.
EPILOGUE_C_PREDICTED = 235.0
EPILOGUE_MN = 4096


def epilogue_bound(k, n=EPILOGUE_MN):
    """Byte-ratio ceiling: (MK + KN + 3MN) / (MK + KN + MN) with M = N = n."""
    return (2 * k + 3 * n) / (2 * k + n)


def epilogue_speedups(rows):
    """Per K: mean e0 time over mean e1 time (each kernel runs in an early
    and a late slot of the same shape)."""
    ms = {}
    for r in rows:
        if r["kernel"] in ("e0", "e1"):
            ms.setdefault(int(r["K"]), {}).setdefault(r["kernel"], []).append(
                float(r["ms_median"]))
    return {k: (sum(v["e0"]) / len(v["e0"])) / (sum(v["e1"]) / len(v["e1"]))
            for k, v in sorted(ms.items()) if "e0" in v and "e1" in v}


def plot_epilogue(rows, out):
    speedups = epilogue_speedups(rows)
    ks = list(speedups)
    fig, ax = plt.subplots(figsize=(8, 4.6), facecolor=SURFACE)
    style(ax)
    xs = [2 ** (e / 8) for e in range(8 * 5, 8 * 13 + 1)]
    ax.plot(xs, [epilogue_bound(x) for x in xs], color=REF, linewidth=1,
            label="byte-ratio ceiling", zorder=1)
    ax.plot(xs, [min(1 + EPILOGUE_C_PREDICTED / x, epilogue_bound(x)) for x in xs],
            color=SERIES[1], linewidth=1.5, linestyle="--",
            label=f"prediction: min(1 + {EPILOGUE_C_PREDICTED:.0f}/K, ceiling)",
            zorder=2)
    ax.plot(ks, [speedups[k] for k in ks], color=SERIES[0], linewidth=2,
            marker="o", markersize=5, markeredgecolor=SURFACE,
            markeredgewidth=1.5, label="measured", zorder=3)
    for k in ks:
        ax.annotate(f"{speedups[k]:.2f}x", (k, speedups[k]), xytext=(4, 6),
                    textcoords="offset points", fontsize=7, color=TEXT)
    ax.set_xscale("log", base=2)
    ax.set_xticks(ks)
    ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{int(v)}"))
    ax.xaxis.set_minor_formatter(NullFormatter())
    ax.set_ylim(0.9, 3.2)
    ax.set_xlabel("K (inner dimension), M = N = 4096", fontsize=8, color=TEXT2)
    ax.set_ylabel("unfused time / fused time", fontsize=8, color=TEXT2)
    ax.set_title("Fusing bias + SiLU into the GEMM: gain against reduction depth",
                 loc="left", fontsize=11, color=TEXT)
    ax.legend(frameon=False, fontsize=8, labelcolor=TEXT2, loc="upper right")
    fig.tight_layout()
    fig.savefig(out, dpi=150, facecolor=SURFACE)
    plt.close(fig)
    return speedups


TRITON_PLOTS = {
    "bias_silu": ("SiLU(x + bias): fused kernels move the minimum traffic at the roof",
                  [("triton", "Triton, fused (1 launch)"),
                   ("compile", "torch.compile (1 launch)"),
                   ("eager", "eager silu(x + b) (2 launches)"),
                   ("eager_composite", "eager x + b, sigmoid, mul (3 launches)")]),
    "rmsnorm": ("RMSNorm: three single-pass kernels at the roof, eager 3-3.5x slower",
                [("triton", "Triton, fused (1 launch)"),
                 ("native", "torch.nn.RMSNorm (1 launch)"),
                 ("compile", "torch.compile (1 launch)"),
                 ("eager", "eager x * rsqrt(mean(x^2) + eps) * w (6 launches)")]),
    "softmax": ("Softmax: three single-pass kernels at the roof",
                [("triton", "Triton, fused (1 launch)"),
                 ("native", "torch.softmax (1 launch)"),
                 ("compile", "torch.compile (1 launch)"),
                 ("eager", "eager max, sub, exp, sum, div (5 launches)")]),
}


def plot_triton(rows, roof_gbs, out, op):
    """Effective bandwidth (8 bytes per element over the median time) per
    implementation against hidden, each point the mean of two slots."""
    title, impls = TRITON_PLOTS[op]
    by = {}
    for r in rows:
        if r["tag"] != "triton-final":
            continue
        by.setdefault(r["impl"], {}).setdefault(int(r["hidden"]), []).append(
            float(r["eff_bandwidth_gbs"]))
    fig, ax = plt.subplots(figsize=(8, 4.6), facecolor=SURFACE)
    style(ax)
    colors = SERIES + [TEXT2]
    means = {}
    for (impl, label), color in zip(impls, colors):
        pts = sorted((h, sum(v) / len(v)) for h, v in by.get(impl, {}).items())
        if not pts:
            continue
        means[impl] = dict(pts)
        xs, ys = zip(*pts)
        ax.plot(xs, ys, color=color, linewidth=2, marker="o", markersize=5,
                markeredgecolor=SURFACE, markeredgewidth=1.5, label=label, zorder=3)
    ax.axhline(roof_gbs, color=REF, linewidth=1, zorder=1)
    ax.text(8192, roof_gbs * 0.96, f"bandwidth roof {roof_gbs:.0f} GB/s",
            fontsize=7, color=TEXT2, ha="right", va="top")
    ax.set_xscale("log", base=2)
    ax.set_xticks([1024, 2048, 4096, 8192])
    ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{int(v)}"))
    ax.xaxis.set_minor_formatter(NullFormatter())
    ax.set_yscale("log")
    ax.set_yticks([50, 100, 200, 400, 600, 800])
    ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:,.0f}"))
    ax.yaxis.set_minor_formatter(NullFormatter())
    ax.set_xlabel("hidden (rows = 4096)", fontsize=8, color=TEXT2)
    ax.set_ylabel("effective bandwidth, GB/s (8 bytes per element)", fontsize=8,
                  color=TEXT2)
    ax.set_title(title, loc="left", fontsize=11, color=TEXT)
    ax.legend(frameon=False, fontsize=8, labelcolor=TEXT2, loc="upper right")
    fig.tight_layout()
    fig.savefig(out, dpi=150, facecolor=SURFACE)
    plt.close(fig)
    return means


def headline_report(reports, kernel, shape="4096x4096x4096"):
    """Latest exported ncu report for a kernel at the headline shape.

    Reports are named <kernel>_<shape>_<timestamp>; older ones carry no shape
    and were all taken at the headline shape."""
    found = []
    for path in glob.glob(str(reports / f"{kernel}_*.details.csv")):
        stem = Path(path).name[len(kernel) + 1:]
        first = stem.split("_", 1)[0]
        taken_at = first if "x" in first else shape
        if taken_at == shape:
            found.append(path)
    if not found:
        raise SystemExit(f"no {shape} report for {kernel} in {reports}")
    return sorted(found, key=lambda f: Path(f).name.rsplit("_", 2)[-2:])[-1]


def ncu_intensity(report):
    """FLOP per byte moved, both measured inside the same ncu run."""
    duration = mem = None
    for r in read(report):
        if r["Metric Name"] == "Duration" and duration is None:
            scale = {"s": 1.0, "ms": 1e-3, "us": 1e-6}[r["Metric Unit"]]
            duration = float(r["Metric Value"]) * scale
        if (r["Section Name"] == "Memory Workload Analysis"
                and r["Metric Name"] == "Memory Throughput" and mem is None):
            mem = float(r["Metric Value"])
    return FLOP_4096 / duration / 1e9 / mem


def plot_roofline(results, reports, out):
    roof = {r["bench"]: r for r in read(results / "roofline.csv")}
    bandwidth = float(roof["copy_f4"]["gb_per_s"])
    compute = float(roof["fma_f4"]["gflops"])
    ridge = compute / bandwidth

    headline = read(results / "gemm_4096.csv")
    clean = [r for r in headline if not r["source_rev"].endswith("-dirty")
             and r.get("baseline_check") != "out-of-band"]
    rungs = []
    for k in ["k1", "k2", "k3", "k4", "k5", "k6", "k7", "k8"]:
        # Keep the original published K1-K7 runs; later K7 controls belong
        # to the K8 experiment. K8 uses only its final runs.
        selected = [r for r in clean if r["kernel"] == k
                    and (r["tag"] == "k8-final" if k == "k8"
                         else not r["tag"].startswith("k8"))]
        vals = [float(r["gflops_median"]) for r in selected]
        if k == "k8":
            # Each final run measures K8 in an early and a late slot; a run's
            # value is the mean of the two, as in the published table.
            vals = [(a + b) / 2 for a, b in zip(vals[::2], vals[1::2])]
        if not vals:
            continue
        # Later K7 profiles are controls for the K8 experiment. Pair the
        # published K7 timing with its original profile on this device.
        original_k7 = reports / "k7_20260916_002316.details.csv"
        report = (str(original_k7) if k == "k7" and original_k7.exists()
                  else headline_report(reports, k))
        rungs.append((k.upper(), ncu_intensity(report), statistics.median(vals)))

    fig, ax = plt.subplots(figsize=(8, 5), facecolor=SURFACE)
    style(ax)
    xs = [10 ** (e / 20) for e in range(-30, 111)]
    ax.plot(xs, [min(compute, bandwidth * x) for x in xs], color=TEXT,
            linewidth=2, label="measured roof", zorder=2)
    ax.plot(xs, [256.0 * x for x in xs], color=TEXT2, linewidth=1,
            label="theoretical bandwidth (256 GB/s)", zorder=1)
    ax.plot([x for _, x, _ in rungs], [y for _, _, y in rungs], color=SERIES[0],
            linewidth=1, alpha=0.5, zorder=3)
    ax.scatter([x for _, x, _ in rungs], [y for _, _, y in rungs], s=40,
               color=SERIES[0], edgecolor=SURFACE, linewidth=1.5,
               label="SGEMM rungs at 4096³", zorder=4)
    # K2 and K3 sit almost on top of each other; nudge their labels apart.
    offsets = {"K2": (-20, 6), "K3": (6, -10), "K6": (6, -6), "K7": (-22, 4)}
    for name, x, y in rungs:
        ax.annotate(name, (x, y), xytext=offsets.get(name, (6, -3)),
                    textcoords="offset points", fontsize=8, color=TEXT)
    micro = [("triad", float(roof["triad_f4"]["arith_intensity"]), float(roof["triad_f4"]["gflops"])),
             ("FMA", float(roof["fma_f4"]["arith_intensity"]), compute)]
    ax.scatter([x for _, x, _ in micro], [y for _, _, y in micro], s=40,
               color=SERIES[1], edgecolor=SURFACE, linewidth=1.5,
               label="micro-benchmarks", zorder=4)
    for name, x, y in micro:
        ax.annotate(name, (x, y), xytext=(6, 4), textcoords="offset points",
                    fontsize=8, color=TEXT)
    ax.annotate(f"ridge: {ridge:.0f} FLOP/byte", (ridge, compute), xytext=(8, 8),
                textcoords="offset points", fontsize=8, color=TEXT2)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(0.03, 3e5)
    ax.set_ylim(1, 3e4)
    ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:,.0f}"))
    ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}"))
    ax.set_xlabel("arithmetic intensity (FLOP per byte moved)", fontsize=8, color=TEXT2)
    ax.set_ylabel("GFLOP/s", fontsize=8, color=TEXT2)
    ax.set_title(f"Roofline, measured: {bandwidth:.0f} GB/s bandwidth, "
                 f"{compute / 1000:.1f} TFLOP/s compute",
                 loc="left", fontsize=11, color=TEXT)
    ax.legend(frameon=False, fontsize=8, labelcolor=TEXT2, loc="lower right")
    fig.tight_layout()
    fig.savefig(out, dpi=150, facecolor=SURFACE)
    plt.close(fig)
    return bandwidth, compute, ridge, rungs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="rtx4060-laptop")
    args = ap.parse_args()
    results = Path("results") / args.device
    reports = Path("profiling/reports") / args.device
    img = Path("docs/img")
    img.mkdir(parents=True, exist_ok=True)
    if (results / "gemm_sweep.csv").exists():
        knees = plot_sweep(read(results / "gemm_sweep.csv"), img / "sweep.png")
        for (k, p), n in knees.items():
            print(f"sweep {k} {p}: knee {n if n else 'none'}")
    if (results / "epilogue_sweep.csv").exists():
        speedups = plot_epilogue(read(results / "epilogue_sweep.csv"),
                                 img / "epilogue.png")
        for k, v in speedups.items():
            print(f"epilogue K={k}: {v:.3f}x")
    for op in TRITON_PLOTS:
        if not (results / f"triton_{op}.csv").exists() or not (results / "roofline.csv").exists():
            continue
        roof = {r["bench"]: r for r in read(results / "roofline.csv")}
        means = plot_triton(read(results / f"triton_{op}.csv"),
                            float(roof["copy_f4"]["gb_per_s"]), img / f"triton_{op}.png", op)
        for impl, pts in means.items():
            print(f"triton {op} {impl}: " + ", ".join(f"{h}: {g:.1f}" for h, g in pts.items()))
    if (results / "roofline.csv").exists():
        b, c, r, rungs = plot_roofline(results, reports, img / "roofline.png")
        print(f"roofline: bandwidth {b:.1f} GB/s, compute {c:.1f} GFLOP/s, ridge {r:.1f} FLOP/byte")
        for name, x, y in rungs:
            print(f"  {name}: {x:.2f} FLOP/byte, {y:.1f} GFLOP/s")


if __name__ == "__main__":
    main()
