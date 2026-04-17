"""Build v5 bilingual training dataset: Traditional + Simplified + CNTP.

Fixes 3 gaps vs PCR-ToxiCN:
  1. Script: adds Simplified copies of all v2.2 records
  2. Length: adds CNTP short texts (~10 chars avg)
  3. Domain: adds CNTP internet-style perturbed texts

Output: data/processed/v2/v5_bilingual_train.jsonl + v5_bilingual_dev.jsonl
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from opencc import OpenCC


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--v2-train", default="data/processed/v2/v2_2_train.jsonl")
    ap.add_argument("--v2-dev", default="data/processed/v2/v2_2_dev.jsonl")
    ap.add_argument("--cntp-dir", default="data/external/ToxiBenchCN/CNTP_dataset")
    ap.add_argument("--out-train", default="data/processed/v2/v5_bilingual_train.jsonl")
    ap.add_argument("--out-dev", default="data/processed/v2/v5_bilingual_dev.jsonl")
    args = ap.parse_args()

    cc_t2s = OpenCC("tw2sp")   # Traditional → Simplified
    cc_s2t = OpenCC("s2twp")   # Simplified → Traditional

    t0 = time.time()

    # ================================================================
    # Part 1: v2.2 Traditional (original) + Simplified copies
    # ================================================================
    print("[1/4] Loading v2.2 Traditional train...", flush=True)
    trad_records = []
    with open(args.v2_train, encoding="utf-8") as f:
        for line in f:
            trad_records.append(json.loads(line))
    print(f"  {len(trad_records)} Traditional records", flush=True)

    print("[2/4] Creating Simplified copies...", flush=True)
    simp_records = []
    for r in trad_records:
        simp = dict(r)
        simp["text"] = cc_t2s.convert(r["text"])
        simp["metadata"] = dict(r.get("metadata", {}))
        simp["metadata"]["script"] = "simplified"
        simp["metadata"]["is_copy"] = True
        # Remove cloak_pair_id from copies (they're independent training samples)
        if "cloak_pair_id" in simp["metadata"]:
            simp["metadata"]["cloak_pair_id"] = -1
        simp_records.append(simp)

    # Tag originals
    for r in trad_records:
        r.setdefault("metadata", {})["script"] = "traditional"

    print(f"  {len(simp_records)} Simplified copies", flush=True)

    # ================================================================
    # Part 2: CNTP data (toxic + non-toxic + homo pairs)
    # ================================================================
    cntp_dir = Path(args.cntp_dir)
    cntp_records = []

    # 2a: Toxic base (Simplified, convert to both)
    print("[3/4] Loading CNTP data...", flush=True)
    toxic_base = json.loads((cntp_dir / "toxic_base.json").read_text())
    for r in toxic_base:
        text_simp = r["original_text"]
        text_trad = cc_s2t.convert(text_simp)
        base_meta = {"source": "cntp_toxic", "script": "simplified",
                     "sentence_id": r.get("sentence_id")}
        cntp_records.append({
            "text": text_simp,
            "label": {"toxicity": "toxic", "bullying": "none", "role": "none", "emotion": "neg"},
            "metadata": base_meta,
        })
        cntp_records.append({
            "text": text_trad,
            "label": {"toxicity": "toxic", "bullying": "none", "role": "none", "emotion": "neg"},
            "metadata": {**base_meta, "script": "traditional", "is_copy": True},
        })
    print(f"  CNTP toxic base: {len(toxic_base)} → {len(toxic_base)*2} (simp+trad)", flush=True)

    # 2b: Non-toxic base
    nontox_path = cntp_dir / "non_toxic_base.json"
    if nontox_path.exists():
        nontox_base = json.loads(nontox_path.read_text())
        for r in nontox_base:
            text_simp = r.get("original_text", r.get("text", ""))
            if not text_simp:
                continue
            text_trad = cc_s2t.convert(text_simp)
            base_meta = {"source": "cntp_nontoxic", "script": "simplified"}
            cntp_records.append({
                "text": text_simp,
                "label": {"toxicity": "none", "bullying": "none", "role": "none", "emotion": "neu"},
                "metadata": base_meta,
            })
            cntp_records.append({
                "text": text_trad,
                "label": {"toxicity": "none", "bullying": "none", "role": "none", "emotion": "neu"},
                "metadata": {**base_meta, "script": "traditional", "is_copy": True},
            })
        print(f"  CNTP non-toxic: {len(nontox_base)} → {len(nontox_base)*2}", flush=True)

    # 2c: Homo perturbed pairs (with cloak_pair_id for consistency loss)
    homo_path = cntp_dir / "perturbed_data" / "homo_toxi_transform.json"
    if homo_path.exists():
        homo_data = json.loads(homo_path.read_text())
        if isinstance(homo_data, dict):
            homo_data = homo_data.get("results", [])
        pair_id_base = 5_000_000  # above existing ranges
        n_pairs = 0
        for i, r in enumerate(homo_data):
            orig = r.get("original_text", "")
            pert = r.get("perturbed_text", "")
            if not orig or not pert or orig == pert:
                continue
            pid = pair_id_base + i
            # Original (base)
            cntp_records.append({
                "text": orig,
                "label": {"toxicity": "toxic", "bullying": "none", "role": "none", "emotion": "neg"},
                "metadata": {"source": "cntp_homo", "script": "simplified",
                             "cloak_pair_id": pid, "cloak_variant": "base"},
            })
            # Perturbed (homo)
            cntp_records.append({
                "text": pert,
                "label": {"toxicity": "toxic", "bullying": "none", "role": "none", "emotion": "neg"},
                "metadata": {"source": "cntp_homo", "script": "simplified",
                             "cloak_pair_id": pid, "cloak_variant": "homo"},
            })
            n_pairs += 1
        print(f"  CNTP homo pairs: {n_pairs} pairs → {n_pairs*2} records", flush=True)

    print(f"  Total CNTP: {len(cntp_records)} records", flush=True)

    # ================================================================
    # Part 3: Merge and write
    # ================================================================
    print("[4/4] Writing output...", flush=True)
    all_train = trad_records + simp_records + cntp_records
    Path(args.out_train).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out_train, "w", encoding="utf-8") as f:
        for r in all_train:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    # Dev: just copy v2.2 dev (Traditional only — eval consistency)
    import shutil
    shutil.copy2(args.v2_dev, args.out_dev)

    elapsed = time.time() - t0
    print(f"\n=== Done in {elapsed:.1f}s ===")
    print(f"  Train: {len(all_train)} records → {args.out_train}")
    print(f"  Dev: copied from {args.v2_dev}")
    print(f"  Breakdown:")
    print(f"    Traditional (v2.2): {len(trad_records)}")
    print(f"    Simplified copies:  {len(simp_records)}")
    print(f"    CNTP:               {len(cntp_records)}")


if __name__ == "__main__":
    main()
