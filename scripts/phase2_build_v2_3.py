"""Build v2.3 training data = v2.2 + lexicon-driven homophone augmentation.

Adds ~10K synthetic (base, homo) pairs derived from existing toxic samples
in v2.2 that overlap the STATE-ToxiCN 829-word hate lexicon. Pairs share a
new range of cloak_pair_id (>= 1_000_000) so they don't collide with the
real ToxiCloakCN ids (0..4585).

CPU-only.
"""
from __future__ import annotations

import argparse
import json
import random
from collections import Counter
from pathlib import Path

from cyberpuppy.data.homophone_aug import HomophoneAugmenter

V2_2_TRAIN = "data/processed/v2/v2_2_train.jsonl"
V2_2_DEV = "data/processed/v2/v2_2_dev.jsonl"
V2_2_TEST = "data/processed/v2/v2_2_test.jsonl"
LEXICON = "data/external/STATE-ToxiCN/data/annotated lexicon.json"
PAIR_ID_OFFSET = 1_000_000  # synthetic pair ids start here


def write_jsonl(records: list, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for r in records:
            obj = r.to_dict() if hasattr(r, "to_dict") else r
            f.write(json.dumps(obj, ensure_ascii=False) + "\n")


def stats(records: list) -> dict:
    dicts = [r if isinstance(r, dict) else r.to_dict() for r in records]
    by_source = Counter(r["metadata"]["source"] for r in dicts)
    by_variant = Counter(
        r["metadata"].get("cloak_variant", "—") for r in dicts
    )
    by_tox = Counter(r["label"]["toxicity"] for r in dicts)
    return {
        "total": len(dicts),
        "by_source": dict(by_source),
        "by_cloak_variant": dict(by_variant),
        "toxicity": dict(by_tox),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-aug-pairs", type=int, default=10_000,
                    help="Cap on synthetic homo pairs added")
    ap.add_argument("--out-train", default="data/processed/v2/v2_3_train.jsonl")
    ap.add_argument("--out-dev", default="data/processed/v2/v2_3_dev.jsonl")
    ap.add_argument("--out-test", default="data/processed/v2/v2_3_test.jsonl")
    ap.add_argument("--out-stats", default="reports/v2_3_dataset_stats.json")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    rng = random.Random(args.seed)
    print(f"[1/3] Loading v2.2 splits...", flush=True)
    train_recs = [json.loads(l) for l in open(V2_2_TRAIN)]
    dev_recs = [json.loads(l) for l in open(V2_2_DEV)]
    test_recs = [json.loads(l) for l in open(V2_2_TEST)]
    print(f"  v2.2 train={len(train_recs)} dev={len(dev_recs)} test={len(test_recs)}",
          flush=True)

    print(f"[2/3] Loading lexicon + scanning toxic candidates...", flush=True)
    aug = HomophoneAugmenter(lexicon_path=Path(LEXICON), seed=args.seed)
    print(f"  lexicon terms: {len(aug.terms)}", flush=True)

    # Pool: toxic / severe samples from v2.2 train that DON'T already have
    # a cloak_pair_id (avoid double-pairing real ToxiCloakCN samples).
    candidates = [
        r for r in train_recs
        if r["label"]["toxicity"] in {"toxic", "severe"}
        and r["metadata"].get("cloak_pair_id", -1) < 0
    ]
    print(f"  candidate toxic samples (no existing pair): {len(candidates)}", flush=True)
    rng.shuffle(candidates)

    pid = PAIR_ID_OFFSET
    aug_records: list = []
    skipped_no_overlap = 0
    for r in candidates:
        pair = aug.build_pair_records(r, pair_id=pid)
        if not pair:
            skipped_no_overlap += 1
            continue
        aug_records.extend(pair)
        pid += 1
        if (pid - PAIR_ID_OFFSET) >= args.max_aug_pairs:
            break
    n_pairs = (pid - PAIR_ID_OFFSET)
    print(f"  synthesized {n_pairs} pairs ({len(aug_records)} records); "
          f"skipped {skipped_no_overlap} samples with no lexicon overlap",
          flush=True)

    print(f"[3/3] Writing v2.3 splits (train += synthetic; dev/test unchanged)...",
          flush=True)
    train_v2_3 = train_recs + [r.to_dict() for r in aug_records]
    rng.shuffle(train_v2_3)
    write_jsonl(train_v2_3, Path(args.out_train))
    write_jsonl(dev_recs, Path(args.out_dev))
    write_jsonl(test_recs, Path(args.out_test))
    print(f"  train -> {args.out_train} ({len(train_v2_3)})", flush=True)
    print(f"  dev   -> {args.out_dev} ({len(dev_recs)})", flush=True)
    print(f"  test  -> {args.out_test} ({len(test_recs)})", flush=True)

    summary = {
        "train": stats(train_v2_3),
        "dev": stats(dev_recs),
        "test": stats(test_recs),
        "synthetic_pairs_added": n_pairs,
        "skipped_no_overlap": skipped_no_overlap,
    }
    Path(args.out_stats).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out_stats).write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"\nStats -> {args.out_stats}", flush=True)
    print(json.dumps(summary["train"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
