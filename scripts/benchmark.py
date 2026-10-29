"""Support sweeps on the UCI benchmarks: runtime, memory and itemset counts.

usage: python scripts/benchmark.py [--datasets mushroom chess ...] [--bins fpgrowth_cpu fpgrowth_gpu]
                                   [--repeat 3] [--max-seconds 60] [--out results]

Writes <out>/bench.csv, <out>/summary.md and one figure per dataset
(<out>/<dataset>.png: runtime, frequent itemsets, peak memory and, when both
CPU and GPU results exist, GPU speedup versus minimum support).
Itemsets are counted but not written to disk, so timings exclude output I/O.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import statistics
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

SWEEPS = {
    "mushroom": [0.5, 0.4, 0.3, 0.2, 0.15, 0.1, 0.05, 0.03, 0.02, 0.01],
    "chess": [0.9, 0.85, 0.8, 0.75, 0.7, 0.65, 0.6, 0.55, 0.5],
    "connect4": [0.95, 0.9, 0.85, 0.8, 0.75, 0.7, 0.65],
    "adult": [0.3, 0.2, 0.1, 0.05, 0.02, 0.01, 0.005],
    "tictactoe": [0.3, 0.2, 0.1, 0.05, 0.02, 0.01, 0.005],
    "votes": [0.5, 0.4, 0.3, 0.2, 0.1, 0.05],
    "splice": [0.3, 0.2, 0.1, 0.05, 0.03, 0.02],
    "nursery": [0.1, 0.05, 0.02, 0.01, 0.005, 0.002, 0.001],
    "letter": [0.1, 0.05, 0.02, 0.01, 0.005, 0.002, 0.001],
    "krkopt": [0.05, 0.02, 0.01, 0.005, 0.002, 0.001, 0.0005],
    "poker": [0.05, 0.01, 0.005, 0.001, 0.0005, 0.0002, 0.0001],
    "covertype": [0.2, 0.1, 0.05, 0.02, 0.01, 0.005, 0.002, 0.001, 0.0005],
    "census": [0.8, 0.7, 0.6, 0.5, 0.4, 0.35, 0.3, 0.25],
    "retail": [0.05, 0.03, 0.02, 0.01, 0.0075, 0.005],
}
FIELDS = ["dataset", "binary", "minsup", "min_count", "transactions", "frequent_items", "tree_nodes", "itemsets",
          "max_len", "build_ms", "mine_ms", "algo_ms", "read_ms", "peak_host_mb", "peak_device_mb", "batches",
          "device"]


def run_once(binary: Path, data: Path, minsup: float, timeout: float, extra: list[str]) -> dict:
    cmd = [str(binary), str(data), str(minsup), "--json", "--quiet"] + extra
    proc = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
    if proc.returncode != 0:
        raise RuntimeError(proc.stderr.strip())
    return json.loads(proc.stdout.strip().splitlines()[-1])


def plot(rows: list[dict], out: Path) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not installed; skipping plots")
        return
    for ds in sorted({r["dataset"] for r in rows}):
        sub = [r for r in rows if r["dataset"] == ds]
        bins = sorted({r["binary"] for r in sub})
        has_speedup = "fpgrowth_cpu" in bins and "fpgrowth_gpu" in bins
        n = 4 if has_speedup else 3
        fig, axes = plt.subplots(1, n, figsize=(4.6 * n, 3.8))
        for b in bins:
            pts = sorted((r for r in sub if r["binary"] == b), key=lambda r: r["minsup"])
            x = [100 * r["minsup"] for r in pts]
            axes[0].plot(x, [r["algo_ms"] for r in pts], "o-", label=b)
            axes[2].plot(x, [r["peak_host_mb"] for r in pts], "o-", label=f"{b} host")
            if b != "fpgrowth_cpu":
                axes[2].plot(x, [r["peak_device_mb"] for r in pts], "s--", label=f"{b} device")
        cpu_pts = sorted((r for r in sub if r["binary"] == bins[0]), key=lambda r: r["minsup"])
        axes[1].plot([100 * r["minsup"] for r in cpu_pts], [r["itemsets"] for r in cpu_pts], "o-", color="k")
        if has_speedup:
            cpu = {r["minsup"]: r for r in sub if r["binary"] == "fpgrowth_cpu"}
            gpu = {r["minsup"]: r for r in sub if r["binary"] == "fpgrowth_gpu"}
            common = sorted(set(cpu) & set(gpu))
            axes[3].plot([100 * s for s in common], [cpu[s]["algo_ms"] / gpu[s]["algo_ms"] for s in common], "o-")
            axes[3].axhline(1.0, color="gray", lw=0.8)
            axes[3].set_title("GPU speedup (CPU time / GPU time)")
            axes[3].set_xlabel("minimum support (%)")
        axes[0].set_yscale("log")
        axes[0].set_title("runtime (build + mine)")
        axes[0].set_ylabel("ms")
        axes[1].set_yscale("log")
        axes[1].set_title("frequent itemsets")
        axes[2].set_title("peak memory")
        axes[2].set_ylabel("MB")
        axes[2].set_ylim(bottom=0, top=max(1.0, axes[2].get_ylim()[1] * 1.15))
        for ax in axes:
            ax.set_xlabel("minimum support (%)")
            ax.invert_xaxis()
            ax.grid(True, which="both", alpha=0.3)
        axes[0].legend(fontsize=8)
        axes[2].legend(fontsize=7)
        tx = sub[0]["transactions"]
        fig.suptitle(f"{ds}  ({tx} transactions)")
        fig.tight_layout()
        fig.savefig(out / f"{ds}.png", dpi=130)
        plt.close(fig)
        print(f"wrote {out / f'{ds}.png'}")


def summary(rows: list[dict], out: Path) -> None:
    lines = ["| dataset | min sup | itemsets | max size | binary | build ms | mine ms | total ms | host MB | device MB |",
             "|---|---|---|---|---|---|---|---|---|---|"]
    for r in sorted(rows, key=lambda r: (r["dataset"], -r["minsup"], r["binary"])):
        dev = f"{r['peak_device_mb']:.1f}" if r["binary"] != "fpgrowth_cpu" else "-"
        lines.append(f"| {r['dataset']} | {100 * r['minsup']:g}% ({r['min_count']}) | {r['itemsets']:,} | "
                     f"{r['max_len']} | {r['binary']} | {r['build_ms']:.1f} | {r['mine_ms']:.1f} | "
                     f"{r['algo_ms']:.1f} | {r['peak_host_mb']:.1f} | {dev} |")
    (out / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"wrote {out / 'summary.md'}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--datasets", nargs="+", default=[d for d in SWEEPS if d != "retail"])
    ap.add_argument("--bins", nargs="+", default=["fpgrowth_cpu", "fpgrowth_gpu"])
    ap.add_argument("--bin-dir", default=str(ROOT / "bin"))
    ap.add_argument("--repeat", type=int, default=3, help="runs per point; the median time is reported")
    ap.add_argument("--max-seconds", type=float, default=60, help="stop lowering support once a run exceeds this")
    ap.add_argument("--timeout", type=float, default=900)
    ap.add_argument("--out", default=str(ROOT / "results"))
    ap.add_argument("--plot-only", action="store_true", help="redraw figures and summary from <out>/bench.csv")
    ap.add_argument("--extra", nargs=argparse.REMAINDER, default=[], help="extra arguments for every binary")
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    if args.plot_only:
        with open(out / "bench.csv", newline="", encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
        for r in rows:
            for k in FIELDS:
                if k not in ("dataset", "binary", "device"):
                    r[k] = float(r[k]) if "." in r[k] else int(r[k])
        summary(rows, out)
        plot(rows, out)
        return
    exe = ".exe" if os.name == "nt" else ""
    rows: list[dict] = []
    for ds in args.datasets:
        data = ROOT / "data" / "uci" / f"{ds}.txt"
        if not data.exists():
            print(f"{data} missing; run scripts/fetch_uci.py {ds}")
            continue
        for b in args.bins:
            binary = Path(args.bin_dir) / f"{b}{exe}"
            if not binary.exists():
                print(f"{binary} not built; skipping")
                continue
            for s in SWEEPS[ds]:
                try:
                    runs = [run_once(binary, data, s, args.timeout, args.extra)]
                    if runs[0]["algo_ms"] < 10_000:
                        runs += [run_once(binary, data, s, args.timeout, args.extra) for _ in range(args.repeat - 1)]
                except (subprocess.TimeoutExpired, RuntimeError) as e:
                    print(f"{ds} {b} minsup={s}: {type(e).__name__} {e}")
                    break
                r = dict(runs[0])
                for k in ("build_ms", "mine_ms", "algo_ms", "read_ms"):
                    r[k] = statistics.median(x[k] for x in runs)
                r.update(dataset=ds, binary=b, minsup=s, max_len=len(r["by_length"]))
                rows.append({k: r[k] for k in FIELDS})
                print(f"{ds:9s} {b:18s} minsup={s:<6} itemsets={r['itemsets']:>12,}  "
                      f"{r['algo_ms']:10.1f} ms  host {r['peak_host_mb']:.1f} MB  device {r['peak_device_mb']:.1f} MB")
                if r["algo_ms"] > 1000 * args.max_seconds:
                    break

    with open(out / "bench.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        w.writeheader()
        w.writerows(rows)
    print(f"wrote {out / 'bench.csv'}")
    if rows:
        summary(rows, out)
        plot(rows, out)


if __name__ == "__main__":
    main()
