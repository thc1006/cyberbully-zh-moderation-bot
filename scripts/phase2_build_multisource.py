"""Build multisource v2 training JSONL: COLD + STATE-ToxiCN + SCCD + CHNCI.

Usage:
  python scripts/phase2_build_multisource.py
  python scripts/phase2_build_multisource.py --out-train data/processed/v2/multisource_train.jsonl

CPU-only — produces JSONL the trainer can consume.
"""
from __future__ import annotations

import argparse
import json
import random
from collections import Counter
from pathlib import Path

import pandas as pd

from cyberpuppy.data.phase2 import Phase2Normalizer

COLD_TRAIN = "data/processed/cold/train_processed.csv"
COLD_DEV = "data/processed/cold/dev_processed.csv"
COLD_TEST = "data/processed/cold/test_processed.csv"
STATE_DIR = "data/external/STATE-ToxiCN/data"
SCCD_DIR = "data/external/SCCD"
CHNCI_DIR = "data/external/CHNCI/dataset"


def write_jsonl(records: list, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r.to_dict(), ensure_ascii=False) + "\n")


def stats(records: list) -> dict:
    sources = Counter(r.metadata["source"] for r in records)
    tox = Counter(r.label["toxicity"] for r in records)
    bull = Counter(r.label["bullying"] for r in records)
    role = Counter(r.label["role"] for r in records)
    emo = Counter(r.label["emotion"] for r in records)
    qual = Counter(r.metadata.get("annotation_quality", "?") for r in records)
    return {
        "total": len(records),
        "by_source": dict(sources),
        "toxicity": dict(tox),
        "bullying": dict(bull),
        "role": dict(role),
        "emotion": dict(emo),
        "quality": dict(qual),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-train", default="data/processed/v2/multisource_train.jsonl")
    ap.add_argument("--out-dev", default="data/processed/v2/multisource_dev.jsonl")
    ap.add_argument("--out-test", default="data/processed/v2/multisource_test.jsonl")
    ap.add_argument("--out-stats", default="reports/phase2_multisource_stats.json")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--include-chnci", action="store_true", default=False,
                    help="Include CHNCI (~220K rows; majority is silver via majority vote on 3 annotators)")
    args = ap.parse_args()

    random.seed(args.seed)
    n = Phase2Normalizer(min_chars=3, max_chars=512)

    train_recs: list = []
    dev_recs: list = []
    test_recs: list = []

    # ---- COLD ----
    print("[COLD] processing 3 splits...", flush=True)
    for path, target in [(COLD_TRAIN, train_recs), (COLD_DEV, dev_recs), (COLD_TEST, test_recs)]:
        df = pd.read_csv(path)
        target.extend(n.process_cold_dataframe(df, dedup=True))
    print(f"  train={sum(1 for r in train_recs if r.metadata['source']=='cold')} "
          f"dev={sum(1 for r in dev_recs if r.metadata['source']=='cold')} "
          f"test={sum(1 for r in test_recs if r.metadata['source']=='cold')}", flush=True)

    # ---- STATE-ToxiCN ----
    print("[STATE-ToxiCN] processing train.json + test.json...", flush=True)
    state_train = json.loads(Path(STATE_DIR, "train.json").read_text(encoding="utf-8"))
    state_test = json.loads(Path(STATE_DIR, "test.json").read_text(encoding="utf-8"))
    state_train_recs = n.process_state_toxicn_records(state_train, dedup=True)
    state_test_recs = n.process_state_toxicn_records(state_test, dedup=True)
    # 90/10 split of the train into train/dev (reserve official test as test)
    random.shuffle(state_train_recs)
    split = int(len(state_train_recs) * 0.9)
    train_recs.extend(state_train_recs[:split])
    dev_recs.extend(state_train_recs[split:])
    test_recs.extend(state_test_recs)
    print(f"  state_train -> train={split} dev={len(state_train_recs)-split} test={len(state_test_recs)}", flush=True)

    # ---- SCCD ----
    print("[SCCD] processing posts + comments...", flush=True)
    posts_df = pd.read_csv(Path(SCCD_DIR, "posts.csv"))
    comments_df = pd.read_csv(Path(SCCD_DIR, "comments.csv"))
    sev_map = {str(p["post_id"]): str(p.get("cyberbullying_severity", "low") or "low")
               for _, p in posts_df.iterrows()}
    sccd_all = n.process_sccd_comments(comments_df.to_dict(orient="records"), sev_map, dedup=True)
    # 80/10/10 split by post_id (avoid leakage)
    post_ids = sorted(set(c.metadata.get("post_id", "") for c in sccd_all))
    random.shuffle(post_ids)
    n_posts = len(post_ids)
    train_pids = set(post_ids[:int(0.8 * n_posts)])
    dev_pids = set(post_ids[int(0.8 * n_posts):int(0.9 * n_posts)])
    test_pids = set(post_ids[int(0.9 * n_posts):])
    sc_train = [c for c in sccd_all if c.metadata.get("post_id", "") in train_pids]
    sc_dev = [c for c in sccd_all if c.metadata.get("post_id", "") in dev_pids]
    sc_test = [c for c in sccd_all if c.metadata.get("post_id", "") in test_pids]
    train_recs.extend(sc_train); dev_recs.extend(sc_dev); test_recs.extend(sc_test)
    print(f"  sccd train={len(sc_train)} dev={len(sc_dev)} test={len(sc_test)}", flush=True)

    # ---- CHNCI (optional, large) ----
    if args.include_chnci:
        print("[CHNCI] processing incident folders...", flush=True)
        chnci_all = n.process_chnci_dir(CHNCI_DIR, dedup=True)
        random.shuffle(chnci_all)
        n_total = len(chnci_all)
        ch_train = chnci_all[:int(0.8 * n_total)]
        ch_dev = chnci_all[int(0.8 * n_total):int(0.9 * n_total)]
        ch_test = chnci_all[int(0.9 * n_total):]
        train_recs.extend(ch_train); dev_recs.extend(ch_dev); test_recs.extend(ch_test)
        print(f"  chnci train={len(ch_train)} dev={len(ch_dev)} test={len(ch_test)}", flush=True)
    else:
        print("[CHNCI] skipped (use --include-chnci to add ~220K rows)", flush=True)

    random.shuffle(train_recs)

    print("\n=== Writing JSONL ===", flush=True)
    write_jsonl(train_recs, Path(args.out_train))
    write_jsonl(dev_recs, Path(args.out_dev))
    write_jsonl(test_recs, Path(args.out_test))
    print(f"  train -> {args.out_train} ({len(train_recs)})")
    print(f"  dev   -> {args.out_dev} ({len(dev_recs)})")
    print(f"  test  -> {args.out_test} ({len(test_recs)})")

    summary = {
        "train": stats(train_recs),
        "dev": stats(dev_recs),
        "test": stats(test_recs),
    }
    Path(args.out_stats).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out_stats).write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nStats -> {args.out_stats}")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
