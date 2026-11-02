"""Overview figure comparing GPU speedups across machines.

usage: python scripts/compare_machines.py results/rtx3060ti results/jetson_orin_nano [--out results/overview.png]

For every dataset, takes the lowest minimum support that both fpgrowth_cpu and
fpgrowth_gpu completed on a machine and plots the speedup CPU time / GPU time,
for the total (build + mine) time and for the mining phase alone.
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path


def lowest_support_speedups(bench: Path) -> dict[str, tuple[float, float, float, int]]:
    rows = list(csv.DictReader(open(bench, newline="", encoding="utf-8")))
    by_key = {(r["dataset"], float(r["minsup"]), r["binary"]): r for r in rows}
    result = {}
    for ds in dict.fromkeys(r["dataset"] for r in rows):
        sups = [s for (d, s, b) in by_key if d == ds and b == "fpgrowth_cpu" and (d, s, "fpgrowth_gpu") in by_key]
        if not sups:
            continue
        s = min(sups)
        cpu, gpu = by_key[(ds, s, "fpgrowth_cpu")], by_key[(ds, s, "fpgrowth_gpu")]
        total = float(cpu["algo_ms"]) / float(gpu["algo_ms"])
        mine = float(cpu["mine_ms"]) / max(1e-6, float(gpu["mine_ms"]))
        result[ds] = (total, mine, s, int(cpu["itemsets"]))
    return result


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("machines", nargs="+", help="result folders containing bench.csv")
    ap.add_argument("--out", default="results/overview.png")
    args = ap.parse_args()

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    data = {Path(m).name: lowest_support_speedups(Path(m) / "bench.csv") for m in args.machines}
    first = next(iter(data.values()))
    datasets = sorted(first, key=lambda d: first[d][0])
    n_m = len(data)
    height = 0.8 / n_m

    fig, axes = plt.subplots(1, 2, figsize=(12, 0.42 * len(datasets) + 1.6), sharey=True)
    for ax, idx, title in ((axes[0], 0, "total time (tree build + mining)"), (axes[1], 1, "mining phase only")):
        for k, (machine, res) in enumerate(data.items()):
            ys = [i + (k - (n_m - 1) / 2) * height for i in range(len(datasets))]
            vals = [res[d][idx] if d in res else 0 for d in datasets]
            bars = ax.barh(ys, vals, height=height, label=machine)
            for b, v in zip(bars, vals):
                ax.text(v * 1.04, b.get_y() + b.get_height() / 2, f"{v:.1f}×", va="center", fontsize=7)
        ax.axvline(1.0, color="k", lw=0.8)
        ax.set_xscale("log")
        ax.set_xlim(0.2, 40)
        ticks = [0.25, 0.5, 1, 2, 4, 8, 16, 32]
        ax.set_xticks(ticks, [f"{t:g}×" for t in ticks])
        ax.minorticks_off()
        ax.set_title(title)
        ax.set_xlabel("speedup = CPU time / GPU time (log scale, >1 means GPU faster)")
        ax.grid(True, axis="x", which="both", alpha=0.3)
    labels = [f"{d}  ({100 * first[d][2]:g}%, {first[d][3]:,} sets)" for d in datasets]
    axes[0].set_yticks(range(len(datasets)), labels, fontsize=8)
    axes[1].legend(loc="lower right", fontsize=8)
    fig.suptitle("FP-growth: CUDA vs sequential C++ at the lowest support of each sweep")
    fig.tight_layout()
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=140)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
