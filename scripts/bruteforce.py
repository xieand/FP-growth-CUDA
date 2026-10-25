"""Independent reference miner (Apriori with exhaustive support counting).

It shares no code with the C++ implementations and is used to validate them on
small and medium inputs.

usage: python scripts/bruteforce.py transactions.txt minsup [-o output.txt]
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from itertools import combinations
from math import comb

from itemsets import read_transactions, resolve_min_count, write_itemsets


def apriori(txs: list[frozenset[str]], min_count: int) -> dict[frozenset[str], int]:
    counts = Counter(i for t in txs for i in t)
    frequent_items = {i for i, c in counts.items() if c >= min_count}
    rows = [tuple(sorted(t & frequent_items)) for t in txs]
    level = {(i,): c for i, c in counts.items() if c >= min_count}
    result = {frozenset(k): c for k, c in level.items()}
    k = 1
    while level:
        # Join k-itemsets (sorted tuples) that share their first k-1 items,
        # then prune candidates having an infrequent k-subset.
        by_prefix: dict[tuple, list[str]] = defaultdict(list)
        for s in level:
            by_prefix[s[:-1]].append(s[-1])
        cands: dict[tuple, int] = {}
        for prefix, lasts in by_prefix.items():
            lasts.sort()
            for a, b in combinations(lasts, 2):
                c = prefix + (a, b)
                if all(c[:j] + c[j + 1:] in level for j in range(k - 1)):
                    cands[c] = 0
        if not cands:
            break
        cand_list = list(cands)
        for r in rows:
            if len(r) <= k:
                continue
            if comb(len(r), k + 1) <= len(cand_list):
                for sub in combinations(r, k + 1):
                    if sub in cands:
                        cands[sub] += 1
            else:
                rs = set(r)
                for c in cand_list:
                    if rs.issuperset(c):
                        cands[c] += 1
        level = {c: n for c, n in cands.items() if n >= min_count}
        result.update((frozenset(c), n) for c, n in level.items())
        k += 1
    return result


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("input")
    ap.add_argument("minsup")
    ap.add_argument("-o", "--output")
    args = ap.parse_args()
    txs = read_transactions(args.input)
    min_count = resolve_min_count(args.minsup, len(txs))
    result = apriori(txs, min_count)
    print(f"{len(txs)} transactions, min count {min_count}: {len(result)} frequent itemsets")
    if args.output:
        write_itemsets(args.output, result)


if __name__ == "__main__":
    main()
