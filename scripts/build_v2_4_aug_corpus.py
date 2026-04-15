"""Build v2.4 training corpus: v2.2 base + JioNLP homophone-augmented pairs.

Strategy (per ADR Phase 3.5 plan):
  - Source pool: v2_2_train.jsonl toxic+severe samples that are NOT already
    paired (i.e., not from ToxiCloakCN). Targets: cold + state_toxicn + sccd
    toxic content.
  - Augment each via ToxiCloakStyleAugmenter (JioNLP-backed, in-distribution
    with ToxiCloakCN heldout by construction).
  - For each successful augmentation, the original toxic record gets
    cloak_pair_id assigned and a new "homo_jionlp" sibling is added.
  - All other v2.2 records (clean text + ToxiCloakCN triplets) pass through
    untouched.

Output: data/processed/v2/v2_4_train.jsonl + reports/v2_4_corpus_stats.json
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from collections import Counter
from pathlib import Path

# Ensure we can import cyberpuppy from src/
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from cyberpuppy.data.toxicloak_style_aug import ToxiCloakStyleAugmenter

# Pair IDs assigned to JioNLP-augmented pairs start above the existing
# ToxiCloakCN/lexicon-aug ranges (those used pid <~ 1.5M per phase2_build_v2_3.py).
PAIR_ID_BASE = 3_000_000


def is_toxic(record: dict) -> bool:
    return record.get("label", {}).get("toxicity") in ("toxic", "severe")


def already_paired(record: dict) -> bool:
    pid = record.get("metadata", {}).get("cloak_pair_id", -1)
    return pid is not None and pid >= 0


def is_augmentable_source(record: dict) -> bool:
    """Skip ToxiCloakCN (already paired) and any prior homo_aug attempts."""
    src = record.get("metadata", {}).get("source", "")
    return src not in ("toxicloak", "homo_aug", "jionlp_homo_aug")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default="data/processed/v2/v2_2_train.jsonl",
                    help="Base training file to extend (default: v2_2_train).")
    ap.add_argument("--out", default="data/processed/v2/v2_4_train.jsonl")
    ap.add_argument("--stats", default="reports/v2_4_corpus_stats.json")
    ap.add_argument("--homo-ratio", type=float, default=0.3,
                    help="JioNLP per-word substitution probability (paper §3.1.2 = 0.3).")
    ap.add_argument("--max-pairs", type=int, default=30000,
                    help="Cap on number of augmented pairs to add (default 30K).")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--limit", type=int, default=None,
                    help="(debug) Process only first N source records.")
    args = ap.parse_args()

    src_path = Path(args.src)
    out_path = Path(args.out)
    stats_path = Path(args.stats)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    stats_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"Building v2.4 corpus", flush=True)
    print(f"  src        : {src_path}")
    print(f"  out        : {out_path}")
    print(f"  homo_ratio : {args.homo_ratio}")
    print(f"  max_pairs  : {args.max_pairs}", flush=True)

    aug = ToxiCloakStyleAugmenter(homo_ratio=args.homo_ratio, seed=args.seed)
    print(f"Augmenter ready (JioNLP backend)", flush=True)

    n_in = 0
    n_passthrough = 0
    n_aug_attempted = 0
    n_aug_success = 0
    n_aug_skipped_not_toxic = 0
    n_aug_skipped_already_paired = 0
    n_aug_skipped_source = 0
    n_aug_failed = 0
    n_out_records = 0
    src_counter = Counter()
    aug_src_counter = Counter()

    pair_id = PAIR_ID_BASE
    t0 = time.time()
    last_log = t0

    with src_path.open() as fin, out_path.open("w") as fout:
        for line in fin:
            n_in += 1
            if args.limit and n_in > args.limit:
                break
            r = json.loads(line)
            src_counter[r.get("metadata", {}).get("source", "?")] += 1

            # Decide if this record is a candidate for augmentation
            if not is_toxic(r):
                n_aug_skipped_not_toxic += 1
            elif already_paired(r):
                n_aug_skipped_already_paired += 1
            elif not is_augmentable_source(r):
                n_aug_skipped_source += 1
            elif n_aug_success >= args.max_pairs:
                # Hit pair cap; pass through remaining toxics unchanged
                pass
            else:
                # Try augmentation
                n_aug_attempted += 1
                pair_records = aug.build_pair_records(r, pair_id=pair_id)
                if pair_records:
                    n_aug_success += 1
                    aug_src_counter[r.get("metadata", {}).get("source", "?")] += 1
                    pair_id += 1
                    for pr in pair_records:
                        fout.write(json.dumps(pr.to_dict(), ensure_ascii=False) + "\n")
                        n_out_records += 1
                    # Don't ALSO emit the original (it's been replaced by the paired base)
                    continue
                else:
                    n_aug_failed += 1
                    # Fall through to pass-through

            # Pass-through (non-augmentable or augmentation failed)
            fout.write(json.dumps(r, ensure_ascii=False) + "\n")
            n_out_records += 1
            n_passthrough += 1

            now = time.time()
            if now - last_log >= 30:
                rate = n_in / max(1.0, now - t0)
                print(f"  [{int(now-t0):4d}s] in={n_in:6d} aug_ok={n_aug_success:5d} "
                      f"aug_fail={n_aug_failed:4d} out={n_out_records:6d} "
                      f"({rate:.1f} src/s)", flush=True)
                last_log = now

    elapsed = time.time() - t0
    stats = {
        "src": str(src_path),
        "out": str(out_path),
        "homo_ratio": args.homo_ratio,
        "max_pairs_cap": args.max_pairs,
        "elapsed_s": elapsed,
        "n_in_records": n_in,
        "n_out_records": n_out_records,
        "n_passthrough": n_passthrough,
        "n_aug_attempted": n_aug_attempted,
        "n_aug_success": n_aug_success,
        "n_aug_failed": n_aug_failed,
        "n_skipped_not_toxic": n_aug_skipped_not_toxic,
        "n_skipped_already_paired": n_aug_skipped_already_paired,
        "n_skipped_source": n_aug_skipped_source,
        "src_distribution": dict(src_counter),
        "augmented_src_distribution": dict(aug_src_counter),
    }
    stats_path.write_text(json.dumps(stats, ensure_ascii=False, indent=2))

    print(f"\n=== DONE in {elapsed:.1f}s ===", flush=True)
    print(f"  in records         : {n_in}")
    print(f"  out records        : {n_out_records}")
    print(f"  augmentation pairs : {n_aug_success}  (target: {args.max_pairs})")
    print(f"  augmentation fails : {n_aug_failed}")
    print(f"  pass-through       : {n_passthrough}")
    print(f"  stats saved        : {stats_path}", flush=True)


if __name__ == "__main__":
    main()
