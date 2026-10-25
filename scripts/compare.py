"""Compare two frequent-itemset files (order of lines and of items is ignored).

usage: python scripts/compare.py result.txt reference.txt
Exit status 0 when both contain exactly the same itemsets with equal supports.
"""
from __future__ import annotations

import sys

from itemsets import read_itemsets


def compare(a: dict, b: dict, max_show: int = 10) -> list[str]:
    problems = []
    only_a = [k for k in a if k not in b]
    only_b = [k for k in b if k not in a]
    wrong = [k for k in a if k in b and a[k] != b[k]]
    for k in only_a[:max_show]:
        problems.append(f"only in first : {' '.join(sorted(k))} #SUP: {a[k]}")
    for k in only_b[:max_show]:
        problems.append(f"only in second: {' '.join(sorted(k))} #SUP: {b[k]}")
    for k in wrong[:max_show]:
        problems.append(f"support differs: {' '.join(sorted(k))}: {a[k]} vs {b[k]}")
    if only_a or only_b or wrong:
        problems.append(f"totals: {len(only_a)} only-first, {len(only_b)} only-second, {len(wrong)} support mismatches")
    return problems


def main() -> int:
    if len(sys.argv) != 3:
        print(__doc__)
        return 2
    a, b = read_itemsets(sys.argv[1]), read_itemsets(sys.argv[2])
    problems = compare(a, b)
    if problems:
        print("\n".join(problems))
        return 1
    print(f"identical: {len(a)} itemsets")
    return 0


if __name__ == "__main__":
    sys.exit(main())
