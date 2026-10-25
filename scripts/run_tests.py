"""Correctness tests for the FP-growth implementations.

  1. Hand-checked textbook examples (data/examples/*.expected).
  2. Randomized datasets checked against an independent Apriori miner,
     including degenerate cases and tiny GPU batch budgets that force the
     batch split / merge paths.
  3. UCI benchmarks: Apriori vs CPU at high support, CPU vs GPU at lower
     support (run scripts/fetch_uci.py first).

usage: python scripts/run_tests.py [--bin-dir bin] [--quick]
Every binary found among fpgrowth_cpu, fpgrowth_gpu and fpgrowth_gpu_emu is tested.
"""
from __future__ import annotations

import argparse
import os
import random
import subprocess
import sys
import tempfile
from pathlib import Path

from bruteforce import apriori
from compare import compare
from itemsets import read_itemsets, read_transactions, resolve_min_count, write_itemsets

ROOT = Path(__file__).resolve().parent.parent


def find_binaries(bin_dir: Path) -> dict[str, Path]:
    exe = ".exe" if os.name == "nt" else ""
    found = {}
    for name in ("fpgrowth_cpu", "fpgrowth_gpu", "fpgrowth_gpu_emu"):
        p = bin_dir / f"{name}{exe}"
        if p.exists():
            found[name] = p
    return found


def run_miner(binary: Path, data: Path, minsup: str, out: Path, extra: list[str] | None = None) -> dict:
    cmd = [str(binary), str(data), str(minsup), "-o", str(out), "--quiet"] + (extra or [])
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        raise RuntimeError(f"{' '.join(cmd)} failed ({proc.returncode}):\n{proc.stderr}")
    return read_itemsets(out)


class Results:
    def __init__(self) -> None:
        self.passed = 0
        self.failed: list[str] = []

    def check(self, name: str, got: dict, want: dict) -> None:
        problems = compare(got, want, max_show=3)
        if problems:
            self.failed.append(name)
            print(f"  FAIL {name}")
            for p in problems:
                print(f"       {p}")
        else:
            self.passed += 1
            print(f"  ok   {name}  ({len(want)} itemsets)")


def random_dataset(rng: random.Random, n_tx: int, n_items: int, avg_len: float, skew: float) -> list[list[str]]:
    weights = [1.0 / (i + 1) ** skew for i in range(n_items)]
    txs = []
    for _ in range(n_tx):
        k = max(1, min(n_items, int(rng.expovariate(1.0 / avg_len)) + 1))
        txs.append(sorted(set(rng.choices(range(n_items), weights=weights, k=k))))
    return [[f"i{x}" for x in t] for t in txs]


def write_dataset(path: Path, txs: list[list[str]]) -> None:
    path.write_text("".join(" ".join(t) + "\n" for t in txs), encoding="utf-8")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--bin-dir", default=str(ROOT / "bin"))
    ap.add_argument("--quick", action="store_true", help="fewer random cases, skip large UCI runs")
    args = ap.parse_args()

    bins = find_binaries(Path(args.bin_dir))
    if not bins:
        print(f"no binaries in {args.bin_dir}; build first (make / cmake)")
        return 2
    print("testing:", ", ".join(bins))
    res = Results()
    tmp = Path(tempfile.mkdtemp(prefix="fpg_tests_"))

    # GPU variants are also run with tiny batch budgets.
    variants: list[tuple[str, Path, list[str]]] = []
    for name, path in bins.items():
        variants.append((name, path, []))
        if name != "fpgrowth_cpu":
            variants.append((f"{name}[tiny-batches]", path, ["--max-entries", "7", "--max-cands", "5"]))

    print("\n[1] textbook examples")
    for exp in sorted((ROOT / "data" / "examples").glob("*.expected")):
        data = exp.with_suffix(".txt")
        minsup = {"han_textbook": "2", "fp_paper": "3"}.get(exp.stem, "2")
        want = read_itemsets(exp)
        for name, path, extra in variants:
            res.check(f"{exp.stem} minsup={minsup} {name}", run_miner(path, data, minsup, tmp / "out.txt", extra), want)

    print("\n[2] randomized datasets vs Apriori")
    rng = random.Random(327)
    cases = [
        ("empty-result", [["a", "b"], ["c"]], "3"),
        ("single-tx", [["a", "b", "c", "d"]], "1"),
        ("identical-tx", [["a", "b", "c", "d", "e"]] * 6, "2"),
        ("nested-chain", [list("abcdefgh"[:k]) for k in range(1, 9)] * 2, "2"),
        ("duplicate-items", [["a", "a", "b"], ["b", "b", "a"], ["a", "c", "c"]], "2"),
    ]
    n_random = 12 if args.quick else 40
    for i in range(n_random):
        n_tx = rng.choice([20, 50, 200, 500])
        n_items = rng.choice([5, 10, 20, 40])
        avg = rng.choice([2.0, 4.0, 8.0])
        skew = rng.choice([0.0, 0.8, 1.5])
        txs = random_dataset(rng, n_tx, n_items, avg, skew)
        minsup = rng.choice(["0.02", "0.05", "0.1", "0.2", "3"])
        cases.append((f"rand{i:02d}(T={n_tx},I={n_items},L={avg},s={skew})", txs, minsup))
    for name_c, txs, minsup in cases:
        data = tmp / "rand.txt"
        write_dataset(data, txs)
        loaded = read_transactions(data)
        want = apriori(loaded, resolve_min_count(minsup, len(loaded)))
        if len(want) > 200000:
            continue
        for name, path, extra in variants:
            res.check(f"{name_c} minsup={minsup} {name}", run_miner(path, data, minsup, tmp / "out.txt", extra), want)

    print("\n[3] UCI benchmarks")
    uci = ROOT / "data" / "uci"
    apriori_runs = [("mushroom", "0.4"), ("chess", "0.85"), ("adult", "0.4")]
    cross_runs = [("mushroom", "0.15"), ("chess", "0.75"), ("connect4", "0.9"), ("adult", "0.1")]
    if args.quick:
        cross_runs = cross_runs[:2]
    if not uci.exists():
        print("  skipped (run: python scripts/fetch_uci.py)")
    else:
        for ds, minsup in apriori_runs:
            data = uci / f"{ds}.txt"
            if not data.exists():
                continue
            ref = tmp / f"{ds}_apriori.txt"
            loaded = read_transactions(data)
            want = apriori(loaded, resolve_min_count(minsup, len(loaded)))
            write_itemsets(ref, want)
            for name, path, extra in variants:
                if "tiny" in name:
                    continue
                res.check(f"{ds} minsup={minsup} {name} vs apriori", run_miner(path, data, minsup, tmp / "out.txt", extra), want)
        if "fpgrowth_cpu" in bins:
            for ds, minsup in cross_runs:
                data = uci / f"{ds}.txt"
                if not data.exists():
                    continue
                want = run_miner(bins["fpgrowth_cpu"], data, minsup, tmp / "cpu.txt")
                for name, path, extra in variants:
                    if name == "fpgrowth_cpu" or "tiny" in name:
                        continue
                    res.check(f"{ds} minsup={minsup} {name} vs cpu", run_miner(path, data, minsup, tmp / "out.txt", extra), want)
                    res.check(f"{ds} minsup={minsup} {name}[small-batches] vs cpu",
                              run_miner(path, data, minsup, tmp / "out.txt", ["--max-entries", "4096", "--max-cands", "4096"]),
                              want)

    print(f"\n{res.passed} passed, {len(res.failed)} failed")
    for f in res.failed:
        print("  failed:", f)
    return 1 if res.failed else 0


if __name__ == "__main__":
    sys.exit(main())
