"""Phase 2 COLD normalization CLI.

Usage:
  python scripts/phase2_normalize.py \\
      --in data/processed/cold/test_processed.csv \\
      --out data/processed/v2/cold_test.jsonl
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from cyberpuppy.data.phase2 import Phase2Normalizer


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", required=True)
    ap.add_argument("--out", dest="out", required=True)
    ap.add_argument("--dedup", action="store_true")
    ap.add_argument("--min-chars", type=int, default=3)
    ap.add_argument("--max-chars", type=int, default=512)
    args = ap.parse_args()

    df = pd.read_csv(args.inp)
    print(f"Loaded {len(df)} rows from {args.inp}")

    n = Phase2Normalizer(min_chars=args.min_chars, max_chars=args.max_chars)
    out = n.process_cold_dataframe(df, dedup=args.dedup)
    print(f"Normalized -> {len(out)} valid records (dedup={args.dedup})")

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        for rec in out:
            f.write(json.dumps(rec.to_dict(), ensure_ascii=False) + "\n")
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
