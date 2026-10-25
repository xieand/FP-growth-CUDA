"""Shared helpers: reading transaction files and frequent-itemset output files."""
from __future__ import annotations

import math
from pathlib import Path


def read_transactions(path: str | Path) -> list[frozenset[str]]:
    txs = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            items = line.split()
            if items:
                txs.append(frozenset(items))
    return txs


def read_itemsets(path: str | Path) -> dict[frozenset[str], int]:
    """Parses "a b c #SUP: n" lines into {frozenset: support}."""
    result: dict[frozenset[str], int] = {}
    with open(path, encoding="utf-8") as f:
        for lineno, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            items, sep, sup = line.rpartition("#SUP:")
            if not sep:
                raise ValueError(f"{path}:{lineno}: missing '#SUP:'")
            key = frozenset(items.split())
            if key in result:
                raise ValueError(f"{path}:{lineno}: duplicate itemset {sorted(key)}")
            result[key] = int(sup)
    return result


def write_itemsets(path: str | Path, itemsets: dict[frozenset[str], int]) -> None:
    rows = sorted(itemsets.items(), key=lambda kv: (len(kv[0]), sorted(kv[0])))
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        for items, sup in rows:
            f.write(" ".join(sorted(items)) + f" #SUP: {sup}\n")


def resolve_min_count(arg: str, n_transactions: int) -> int:
    """Same rule as the C++ programs: '0.3' / '30%' relative, '3' absolute."""
    s = str(arg)
    percent = s.endswith("%")
    v = float(s[:-1]) / 100 if percent else float(s)
    if percent or v < 1.0:
        return max(1, math.ceil(v * n_transactions - 1e-9))
    return math.ceil(v)
