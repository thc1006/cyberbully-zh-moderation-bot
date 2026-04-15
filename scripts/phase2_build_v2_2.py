"""Build v2.2 training data = v2.1 multisource + ToxiCloakCN adversarial.

Splits:
- train: v2.1 train + ToxiCloakCN rows [0, train_end)
- dev:   v2.1 dev unchanged (pure clean data so we can track regression)
- test:  v2.1 test unchanged
- toxicloak_heldout: ToxiCloakCN rows [train_end, n_rows) across all 3 variants

Each ToxiCloakCN record carries cloak_pair_id so the trainer can compute
consistency loss between base/homo/emoji of the same pair.

CPU-only.
"""
from __future__ import annotations

import argparse
import json
import random
from collections import Counter
from pathlib import Path

import pandas as pd

from cyberpuppy.data.phase2 import Phase2Normalizer

V2_1_TRAIN = "data/processed/v2/multisource_train.jsonl"
V2_1_DEV = "data/processed/v2/multisource_dev.jsonl"
V2_1_TEST = "data/processed/v2/multisource_test.jsonl"
TOXICLOAK_DIR = Path("data/external/ToxiCloakCN/Datasets")


def write_jsonl(records: list, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for r in records:
            obj = r.to_dict() if hasattr(r, "to_dict") else r
            f.write(json.dumps(obj, ensure_ascii=False) + "\n")


def load_jsonl(path: str) -> list[dict]:
    with Path(path).open() as f:
        return [json.loads(line) for line in f]


def stats(records: list) -> dict:
    dicts = [r if isinstance(r, dict) else r.to_dict() for r in records]
    sources = Counter(r["metadata"]["source"] for r in dicts)
    tox = Counter(r["label"]["toxicity"] for r in dicts)
    bull = Counter(r["label"]["bullying"] for r in dicts)
    role = Counter(r["label"]["role"] for r in dicts)
    emo = Counter(r["label"]["emotion"] for r in dicts)
    variants = Counter(
        r["metadata"].get("cloak_variant", "—") for r in dicts if r["metadata"].get("source") == "toxicloak"
    )
    return {
        "total": len(dicts),
        "by_source": dict(sources),
        "toxicity": dict(tox),
        "bullying": dict(bull),
        "role": dict(role),
        "emotion": dict(emo),
        "cloak_variants": dict(variants),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--train-split", type=float, default=0.8,
                    help="Fraction of ToxiCloakCN rows used for training (rest -> held-out eval)")
    ap.add_argument("--out-train", default="data/processed/v2/v2_2_train.jsonl")
    ap.add_argument("--out-dev", default="data/processed/v2/v2_2_dev.jsonl")
    ap.add_argument("--out-test", default="data/processed/v2/v2_2_test.jsonl")
    ap.add_argument("--out-heldout", default="data/processed/v2/toxicloak_heldout.jsonl")
    ap.add_argument("--out-stats", default="reports/v2_2_dataset_stats.json")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    random.seed(args.seed)
    n = Phase2Normalizer(min_chars=3, max_chars=512)

    # --- Copy v2.1 base splits (clean, no ToxiCloakCN yet) ---
    print("[v2.1 base] loading splits...", flush=True)
    train_recs_dicts = load_jsonl(V2_1_TRAIN)
    dev_recs_dicts = load_jsonl(V2_1_DEV)
    test_recs_dicts = load_jsonl(V2_1_TEST)
    print(f"  train={len(train_recs_dicts)}  dev={len(dev_recs_dicts)}  test={len(test_recs_dicts)}", flush=True)

    # --- ToxiCloakCN: shuffle-then-split (base_data.tsv is sorted by label!) ---
    print("[ToxiCloakCN] SHUFFLE-then-split 80/20 (raw file is sorted by label)...", flush=True)
    base_df = pd.read_csv(TOXICLOAK_DIR / "base_data.tsv", sep="\t")
    n_pairs = len(base_df)
    all_pair_ids = list(range(n_pairs))
    rng = random.Random(args.seed)
    rng.shuffle(all_pair_ids)
    train_count = int(n_pairs * args.train_split)
    train_pair_ids = all_pair_ids[:train_count]
    heldout_pair_ids = all_pair_ids[train_count:]
    # Sanity: verify class balance survived the split
    train_toxic = sum(base_df.iloc[i]["toxic"] for i in train_pair_ids)
    heldout_toxic = sum(base_df.iloc[i]["toxic"] for i in heldout_pair_ids)
    print(f"  n_pairs={n_pairs}  train_pairs={len(train_pair_ids)} (toxic={train_toxic}/{len(train_pair_ids)})"
          f"  heldout_pairs={len(heldout_pair_ids)} (toxic={heldout_toxic}/{len(heldout_pair_ids)})", flush=True)

    train_cloak_recs = n.process_toxicloak_tsvs(
        base_path=TOXICLOAK_DIR / "base_data.tsv",
        homo_path=TOXICLOAK_DIR / "Full_Perturbation" / "homo_full.tsv",
        emoji_path=TOXICLOAK_DIR / "Full_Perturbation" / "emoji_full.tsv",
        keep_pair_ids=train_pair_ids,
    )
    heldout_cloak_recs = n.process_toxicloak_tsvs(
        base_path=TOXICLOAK_DIR / "base_data.tsv",
        homo_path=TOXICLOAK_DIR / "Full_Perturbation" / "homo_full.tsv",
        emoji_path=TOXICLOAK_DIR / "Full_Perturbation" / "emoji_full.tsv",
        keep_pair_ids=heldout_pair_ids,
    )
    print(f"  cloak_train={len(train_cloak_recs)}  cloak_heldout={len(heldout_cloak_recs)}", flush=True)

    # --- Merge + shuffle train ---
    train_recs_dicts.extend(r.to_dict() for r in train_cloak_recs)
    random.shuffle(train_recs_dicts)

    # --- Write ---
    print("\n=== Writing v2.2 JSONL ===", flush=True)
    write_jsonl(train_recs_dicts, Path(args.out_train))
    write_jsonl(dev_recs_dicts, Path(args.out_dev))
    write_jsonl(test_recs_dicts, Path(args.out_test))
    write_jsonl([r.to_dict() for r in heldout_cloak_recs], Path(args.out_heldout))
    for label, rlist, path in [
        ("train", train_recs_dicts, args.out_train),
        ("dev", dev_recs_dicts, args.out_dev),
        ("test", test_recs_dicts, args.out_test),
        ("heldout", [r.to_dict() for r in heldout_cloak_recs], args.out_heldout),
    ]:
        print(f"  {label:<8} -> {path} ({len(rlist)})", flush=True)

    summary = {
        "train": stats(train_recs_dicts),
        "dev": stats(dev_recs_dicts),
        "test": stats(test_recs_dicts),
        "heldout": stats([r.to_dict() for r in heldout_cloak_recs]),
    }
    Path(args.out_stats).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out_stats).write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"\nStats -> {args.out_stats}")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
