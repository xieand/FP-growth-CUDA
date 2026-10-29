"""Download benchmark datasets from the UCI Machine Learning Repository and
convert them to transaction files (one transaction per line, items separated
by spaces).

Each record of a categorical dataset becomes one transaction whose items are
"attribute=value" tokens; missing values ('?') are skipped. Numeric Adult and
Census-KDD attributes are discretized into ranges, numeric Covertype attributes
into quintiles. Poker hands become baskets of their five cards plus the hand
class, and Online Retail invoices become baskets of stock codes.

usage: python scripts/fetch_uci.py [names...]     (default: all except retail)
       names: mushroom chess connect4 adult tictactoe votes splice nursery
              letter krkopt poker covertype census retail
Output: data/uci/<name>.txt   (downloads cached in data/raw/)
"""
from __future__ import annotations

import io
import sys
import urllib.request
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RAW = ROOT / "data" / "raw"
OUT = ROOT / "data" / "uci"
BASE = "https://archive.ics.uci.edu/static/public"

MUSHROOM_ATTRS = [
    "class", "cap-shape", "cap-surface", "cap-color", "bruises", "odor", "gill-attachment", "gill-spacing",
    "gill-size", "gill-color", "stalk-shape", "stalk-root", "stalk-surface-above-ring",
    "stalk-surface-below-ring", "stalk-color-above-ring", "stalk-color-below-ring", "veil-type", "veil-color",
    "ring-number", "ring-type", "spore-print-color", "population", "habitat",
]
CHESS_ATTRS = [
    "bkblk", "bknwy", "bkon8", "bkona", "bkspr", "bkxbq", "bkxcr", "bkxwp", "blxwp", "bxqsq", "cntxt", "dsopp",
    "dwipd", "hdchk", "katri", "mulch", "qxmsq", "r2ar8", "reskd", "reskr", "rimmx", "rkxwp", "rxmsq", "simpl",
    "skach", "skewr", "skrxp", "spcop", "stlmt", "thrsk", "wkcti", "wkna8", "wknck", "wkovl", "wkpos", "wtoeg",
    "class",
]
CONNECT4_ATTRS = [f"{c}{r}" for c in "abcdefg" for r in range(1, 7)] + ["class"]
TICTACTOE_ATTRS = ["tl", "tm", "tr", "ml", "mm", "mr", "bl", "bm", "br", "class"]
VOTES_ATTRS = [
    "party", "handicapped-infants", "water-project-cost-sharing", "adoption-of-the-budget-resolution",
    "physician-fee-freeze", "el-salvador-aid", "religious-groups-in-schools", "anti-satellite-test-ban",
    "aid-to-nicaraguan-contras", "mx-missile", "immigration", "synfuels-corporation-cutback", "education-spending",
    "superfund-right-to-sue", "crime", "duty-free-exports", "export-administration-act-south-africa",
]
NURSERY_ATTRS = ["parents", "has_nurs", "form", "children", "housing", "finance", "social", "health", "class"]
LETTER_ATTRS = [
    "letter", "x-box", "y-box", "width", "high", "onpix", "x-bar", "y-bar", "x2bar", "y2bar", "xybar", "x2ybr",
    "xy2br", "x-ege", "xegvy", "y-ege", "yegvx",
]
KRKOPT_ATTRS = ["wk-file", "wk-rank", "wr-file", "wr-rank", "bk-file", "bk-rank", "result"]
COVTYPE_NUMERIC = [
    "elevation", "aspect", "slope", "hdist-hydro", "vdist-hydro", "hdist-road", "hillshade-9am", "hillshade-noon",
    "hillshade-3pm", "hdist-fire",
]
# census-income.names order; numeric columns are handled in census().
CENSUS_COLS = [
    "age", "class-of-worker", "industry-code", "occupation-code", "education", "wage-per-hour", "enrolled-in-edu",
    "marital-status", "major-industry", "major-occupation", "race", "hispanic-origin", "sex", "labor-union",
    "unemployment-reason", "employment-status", "capital-gains", "capital-losses", "dividends", "tax-filer-status",
    "prev-region", "prev-state", "household-stat", "household-summary", "instance-weight", "migration-msa",
    "migration-reg", "migration-within-reg", "same-house-1yr", "migration-sunbelt", "num-persons-worked-for-employer",
    "family-under-18", "father-birth-country", "mother-birth-country", "birth-country", "citizenship",
    "own-business", "veterans-questionnaire", "veterans-benefits", "weeks-worked", "year", "income",
]


def download(url: str, name: str) -> bytes:
    RAW.mkdir(parents=True, exist_ok=True)
    path = RAW / name
    if not path.exists():
        print(f"  downloading {url}")
        with urllib.request.urlopen(url) as r:
            path.write_bytes(r.read())
    return path.read_bytes()


def zip_member(archive: bytes, member: str) -> bytes:
    with zipfile.ZipFile(io.BytesIO(archive)) as z:
        return z.read(member)


def unlzw(data: bytes) -> bytes:
    """Decompress Unix 'compress' (.Z) data."""
    if data[:2] != b"\x1f\x9d":
        raise ValueError("not a .Z file")
    max_bits = data[2] & 0x1F
    block_mode = bool(data[2] & 0x80)
    first = 257 if block_mode else 256
    table = [bytes([i]) for i in range(256)] + ([b""] if block_mode else [])
    out = bytearray()
    total_bits = len(data) * 8
    bits = 9
    seg_start = 24      # codes are written in groups of 8; a width change or
    n_in_seg = 0        # CLEAR pads to the end of the current group
    prev = None
    while True:
        if len(table) > (1 << bits) - 1 and bits < max_bits and prev is not None:
            seg_start += ((n_in_seg + 7) // 8) * 8 * bits
            n_in_seg = 0
            bits += 1
        pos = seg_start + n_in_seg * bits
        if pos + bits > total_bits:
            break
        byte = pos >> 3
        chunk = int.from_bytes(data[byte:byte + 4], "little")
        code = (chunk >> (pos & 7)) & ((1 << bits) - 1)
        n_in_seg += 1
        if block_mode and code == 256:
            seg_start += ((n_in_seg + 7) // 8) * 8 * bits
            n_in_seg = 0
            bits = 9
            del table[first:]
            prev = None
            continue
        if code < len(table):
            entry = table[code]
        elif code == len(table) and prev is not None:
            entry = prev + prev[:1]
        else:
            raise ValueError("corrupt .Z stream")
        out += entry
        if prev is not None and len(table) < (1 << max_bits):
            table.append(prev + entry[:1])
        prev = entry
    return bytes(out)


def categorical_rows(text: str, attrs: list[str], skip_cols: set[int] = frozenset()) -> list[list[str]]:
    txs = []
    for line in text.splitlines():
        vals = [v.strip() for v in line.strip().split(",")]
        if len(vals) != len(attrs):
            continue
        txs.append([f"{a}={v}" for i, (a, v) in enumerate(zip(attrs, vals)) if v != "?" and i not in skip_cols])
    return txs


def write(name: str, txs: list[list[str]]) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / f"{name}.txt"
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        for t in txs:
            f.write(" ".join(t) + "\n")
    n_items = len({i for t in txs for i in t})
    avg = sum(map(len, txs)) / max(1, len(txs))
    print(f"  {path.relative_to(ROOT)}: {len(txs)} transactions, {n_items} items, avg length {avg:.1f}")


def mushroom() -> None:
    z = download(f"{BASE}/73/mushroom.zip", "mushroom.zip")
    write("mushroom", categorical_rows(zip_member(z, "agaricus-lepiota.data").decode(), MUSHROOM_ATTRS))


def chess() -> None:
    z = download(f"{BASE}/22/chess+king+rook+vs+king+pawn.zip", "chess.zip")
    write("chess", categorical_rows(zip_member(z, "kr-vs-kp.data").decode(), CHESS_ATTRS))


def connect4() -> None:
    z = download(f"{BASE}/26/connect+4.zip", "connect4.zip")
    text = unlzw(zip_member(z, "connect-4.data.Z")).decode()
    write("connect4", categorical_rows(text, CONNECT4_ATTRS))


def _bin(v: int, edges: list[int], labels: list[str]) -> str:
    for e, lab in zip(edges, labels):
        if v < e:
            return lab
    return labels[-1]


def adult() -> None:
    z = download(f"{BASE}/2/adult.zip", "adult.zip")
    cols = ["age", "workclass", "fnlwgt", "education", "education-num", "marital-status", "occupation",
            "relationship", "race", "sex", "capital-gain", "capital-loss", "hours-per-week", "native-country",
            "income"]
    txs = []
    for member in ("adult.data", "adult.test"):
        for line in zip_member(z, member).decode().splitlines():
            vals = [v.strip() for v in line.split(",")]
            if len(vals) != len(cols):
                continue
            rec = dict(zip(cols, vals))
            items = [
                "age=" + _bin(int(rec["age"]), [25, 35, 45, 55, 65], ["<25", "25-34", "35-44", "45-54", "55-64", "65+"]),
                "capital-gain=" + ("none" if rec["capital-gain"] == "0" else "positive"),
                "capital-loss=" + ("none" if rec["capital-loss"] == "0" else "positive"),
                "hours-per-week=" + _bin(int(rec["hours-per-week"]), [40, 41], ["<40", "40", ">40"]),
                "income=" + rec["income"].rstrip("."),
            ]
            for c in ("workclass", "education", "marital-status", "occupation", "relationship", "race", "sex",
                      "native-country"):
                if rec[c] != "?":
                    items.append(f"{c}={rec[c].replace(' ', '_')}")
            txs.append(items)
    write("adult", txs)


def tictactoe() -> None:
    z = download(f"{BASE}/101/tic+tac+toe+endgame.zip", "tictactoe.zip")
    write("tictactoe", categorical_rows(zip_member(z, "tic-tac-toe.data").decode(), TICTACTOE_ATTRS))


def votes() -> None:
    z = download(f"{BASE}/105/congressional+voting+records.zip", "votes.zip")
    write("votes", categorical_rows(zip_member(z, "house-votes-84.data").decode(), VOTES_ATTRS))


def splice() -> None:
    z = download(f"{BASE}/69/molecular+biology+splice+junction+gene+sequences.zip", "splice.zip")
    txs = []
    for line in zip_member(z, "splice.data").decode().splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) != 3 or len(parts[2]) != 60:
            continue
        txs.append([f"class={parts[0]}"] + [f"p{i + 1:02d}={b}" for i, b in enumerate(parts[2])])
    write("splice", txs)


def nursery() -> None:
    z = download(f"{BASE}/76/nursery.zip", "nursery.zip")
    write("nursery", categorical_rows(zip_member(z, "nursery.data").decode(), NURSERY_ATTRS))


def letter() -> None:
    z = download(f"{BASE}/59/letter+recognition.zip", "letter.zip")
    write("letter", categorical_rows(zip_member(z, "letter-recognition.data").decode(), LETTER_ATTRS))


def krkopt() -> None:
    z = download(f"{BASE}/23/chess+king+rook+vs+king.zip", "krkopt.zip")
    write("krkopt", categorical_rows(zip_member(z, "krkopt.data").decode(), KRKOPT_ATTRS))


def poker() -> None:
    z = download(f"{BASE}/158/poker+hand.zip", "poker.zip")
    suits = "HSDC"
    ranks = ["A", "2", "3", "4", "5", "6", "7", "8", "9", "10", "J", "Q", "K"]
    txs = []
    for member in ("poker-hand-training-true.data", "poker-hand-testing.data"):
        for line in zip_member(z, member).decode().splitlines():
            v = line.strip().split(",")
            if len(v) != 11:
                continue
            cards = [f"{ranks[int(v[i + 1]) - 1]}{suits[int(v[i]) - 1]}" for i in range(0, 10, 2)]
            txs.append(cards + [f"hand={v[10]}"])
    write("poker", txs)


def covertype() -> None:
    import bisect
    import gzip
    z = download(f"{BASE}/31/covertype.zip", "covertype.zip")
    rows = [line.split(",") for line in gzip.decompress(zip_member(z, "covtype.data.gz")).decode().splitlines()]
    rows = [[int(x) for x in r] for r in rows if len(r) == 55]
    cuts = []
    for c in range(len(COVTYPE_NUMERIC)):
        col = sorted(r[c] for r in rows)
        cuts.append(sorted({col[len(col) * q // 5] for q in range(1, 5)}))
    txs = []
    for r in rows:
        items = [f"{a}=q{bisect.bisect_right(cuts[c], r[c]) + 1}" for c, a in enumerate(COVTYPE_NUMERIC)]
        items += [f"wilderness={k + 1}" for k in range(4) if r[10 + k]]
        items += [f"soil={k + 1}" for k in range(40) if r[14 + k]]
        items.append(f"cover={r[54]}")
        txs.append(items)
    write("covertype", txs)


def census() -> None:
    import tarfile
    z = download(f"{BASE}/117/census+income+kdd.zip", "census_kdd.zip")
    tar = tarfile.open(fileobj=io.BytesIO(zip_member(z, "census.tar.gz")))
    zero_or_pos = {"wage-per-hour", "capital-gains", "capital-losses", "dividends"}
    txs = []
    for member in ("census-income.data", "census-income.test"):
        for line in tar.extractfile(member).read().decode().splitlines():
            vals = [v.strip() for v in line.split(",")]
            if len(vals) != len(CENSUS_COLS):
                continue
            items = []
            for col, v in zip(CENSUS_COLS, vals):
                if col == "instance-weight" or v == "?":
                    continue
                if col == "age":
                    v = _bin(int(v), [16, 25, 35, 45, 55, 65], ["<16", "16-24", "25-34", "35-44", "45-54", "55-64", "65+"])
                elif col in zero_or_pos:
                    v = "none" if float(v) == 0 else "positive"
                elif col == "weeks-worked":
                    v = _bin(int(v), [1, 52], ["0", "1-51", "52"])
                elif col == "income":
                    v = v.rstrip(".").replace(" ", "")
                items.append(f"{col}={v.replace(' ', '_')}")
            txs.append(items)
    write("census", txs)


def retail() -> None:
    try:
        import pandas as pd
        import openpyxl  # noqa: F401  (pandas needs it for .xlsx)
    except ImportError:
        print("  retail needs pandas + openpyxl:  pip install pandas openpyxl")
        return
    download(f"{BASE}/352/online+retail.zip", "online_retail.zip")
    with zipfile.ZipFile(RAW / "online_retail.zip") as zf:
        df = pd.read_excel(io.BytesIO(zf.read("Online Retail.xlsx")), dtype={"InvoiceNo": str, "StockCode": str})
    df = df[~df["InvoiceNo"].str.startswith("C") & (df["Quantity"] > 0)]
    df["StockCode"] = df["StockCode"].str.strip().str.replace(" ", "_")
    baskets = df.groupby("InvoiceNo")["StockCode"].apply(lambda s: sorted(set(s)))
    write("retail", [b for b in baskets if b])


DATASETS = {
    "mushroom": mushroom, "chess": chess, "connect4": connect4, "adult": adult, "tictactoe": tictactoe,
    "votes": votes, "splice": splice, "nursery": nursery, "letter": letter, "krkopt": krkopt, "poker": poker,
    "covertype": covertype, "census": census, "retail": retail,
}


def main() -> None:
    names = sys.argv[1:] or [n for n in DATASETS if n != "retail"]
    for n in names:
        if n not in DATASETS:
            sys.exit(f"unknown dataset {n}; choose from {', '.join(DATASETS)}")
        print(n)
        DATASETS[n]()


if __name__ == "__main__":
    main()
