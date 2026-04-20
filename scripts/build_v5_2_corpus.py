"""Build v5.2 training corpus: v5.1 bilingual (179K) + HED-COLD (6.3K) + CHNCI (220K).

Output: data/processed/v2/v5_2_train.jsonl
        data/processed/v2/v5_2_dev.jsonl

HED-COLD: simplified Chinese, homophone-perturbed offensive language (EMNLP 2025)
CHNCI: simplified Chinese, cyberbullying incidents from Weibo/Douyin/XHS/Bilibili

Both are converted to our standard format:
  {"text": ..., "label": {"toxicity": ..., "bullying": ..., "role": ..., "emotion": ...},
   "metadata": {"source": ..., ...}}
"""
from __future__ import annotations

import csv
import json
import re
from pathlib import Path
from collections import Counter

import opencc

# Traditional conversion for HED-COLD and CHNCI (both are simplified)
s2tw = opencc.OpenCC('s2twp')


def load_hed_cold(split='train'):
    """Load HED-COLD CSV. Labels: 0=non-offensive, 1=offensive."""
    path = Path(f'data/external/HED-COLD/dataset/{split}.csv')
    records = []
    with open(path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            text = row['TEXT'].strip()
            if not text:
                continue
            label_int = int(row['label'])
            toxicity = 'toxic' if label_int == 1 else 'none'
            records.append({
                'text': text,
                'label': {
                    'toxicity': toxicity,
                    'bullying': 'harassment' if label_int == 1 else 'none',
                    'role': 'perpetrator' if label_int == 1 else 'none',
                    'emotion': 'neg' if label_int == 1 else 'neu',
                },
                'metadata': {
                    'source': 'HED-COLD',
                    'topic': row.get('topic', ''),
                    'split': split,
                }
            })
    return records


def load_chnci():
    """Load CHNCI CSVs (GBK encoding). Labels: majority vote of label1/label2/label3."""
    base = Path('data/external/CHNCI/dataset')
    records = []

    for category in ['cyberbullying', 'non-cyberbullying']:
        cat_dir = base / category
        if not cat_dir.exists():
            continue
        for csv_path in sorted(cat_dir.glob('*.csv')):
            try:
                with open(csv_path, 'r', encoding='gbk', errors='replace') as f:
                    reader = csv.DictReader(f)
                    for row in reader:
                        content = row.get('content', '').strip()
                        if not content or len(content) < 2:
                            continue
                        # Skip emoji-only entries
                        if re.match(r'^\[.*\]$', content):
                            continue

                        # Majority vote: label1, label2, label3
                        try:
                            votes = [
                                int(float(row.get('label1', 0))),
                                int(float(row.get('label2', 0))),
                                int(float(row.get('label3', 0))),
                            ]
                            label_int = 1 if sum(votes) >= 2 else 0
                        except (ValueError, TypeError):
                            continue

                        toxicity = 'toxic' if label_int == 1 else 'none'
                        records.append({
                            'text': content,
                            'label': {
                                'toxicity': toxicity,
                                'bullying': 'harassment' if label_int == 1 else 'none',
                                'role': 'perpetrator' if label_int == 1 else 'none',
                                'emotion': 'neg' if label_int == 1 else 'neu',
                            },
                            'metadata': {
                                'source': 'CHNCI',
                                'platform': row.get('platform', ''),
                                'incident': csv_path.stem,
                            }
                        })
            except Exception as e:
                print(f"  Warning: {csv_path.name}: {e}")

    return records


def to_traditional(records):
    """Convert simplified records to traditional Chinese."""
    trad_records = []
    for r in records:
        trad_text = s2tw.convert(r['text'])
        trad_r = {
            'text': trad_text,
            'label': dict(r['label']),
            'metadata': {**r['metadata'], 'variant': 'traditional'},
        }
        trad_records.append(trad_r)
    return trad_records


def main():
    print("=" * 60)
    print("Building v5.2 corpus")
    print("=" * 60)

    # 1. Load existing v5.1 train/dev
    print("\n[1/4] Loading v5.1 bilingual train...")
    v5_train = [json.loads(l) for l in
                Path('data/processed/v2/v2_2_train.jsonl').open()]
    v5_dev = [json.loads(l) for l in
              Path('data/processed/v2/v2_2_dev.jsonl').open()]
    print(f"  v5.1 train: {len(v5_train)}, dev: {len(v5_dev)}")

    # 2. Load HED-COLD
    print("\n[2/4] Loading HED-COLD...")
    hed_train = load_hed_cold('train')
    hed_dev = load_hed_cold('dev')
    print(f"  HED-COLD train: {len(hed_train)}, dev: {len(hed_dev)}")

    # Convert to traditional (bilingual)
    hed_train_trad = to_traditional(hed_train)
    hed_dev_trad = to_traditional(hed_dev)
    print(f"  + Traditional copies: train +{len(hed_train_trad)}, dev +{len(hed_dev_trad)}")

    # 3. Load CHNCI
    print("\n[3/4] Loading CHNCI...")
    chnci_all = load_chnci()
    print(f"  CHNCI total: {len(chnci_all)}")

    # Stats
    toxic_count = sum(1 for r in chnci_all if r['label']['toxicity'] == 'toxic')
    print(f"  CHNCI toxic: {toxic_count}/{len(chnci_all)} ({100*toxic_count/max(1,len(chnci_all)):.1f}%)")

    # Split CHNCI: 95% train, 5% dev (by incident to avoid data leakage)
    incidents = sorted(set(r['metadata']['incident'] for r in chnci_all))
    n_dev_incidents = max(5, len(incidents) // 20)  # 5% of incidents
    dev_incidents = set(incidents[:n_dev_incidents])

    chnci_train = [r for r in chnci_all if r['metadata']['incident'] not in dev_incidents]
    chnci_dev = [r for r in chnci_all if r['metadata']['incident'] in dev_incidents]
    print(f"  CHNCI split: train {len(chnci_train)}, dev {len(chnci_dev)} "
          f"({n_dev_incidents} dev incidents)")

    # Convert to traditional
    chnci_train_trad = to_traditional(chnci_train)
    chnci_dev_trad = to_traditional(chnci_dev)
    print(f"  + Traditional copies: train +{len(chnci_train_trad)}, dev +{len(chnci_dev_trad)}")

    # 4. Merge
    print("\n[4/4] Merging...")
    final_train = v5_train + hed_train + hed_train_trad + chnci_train + chnci_train_trad
    final_dev = v5_dev + hed_dev + hed_dev_trad + chnci_dev + chnci_dev_trad

    print(f"\n  Final train: {len(final_train)}")
    print(f"  Final dev: {len(final_dev)}")

    # Source breakdown
    sources = Counter(r.get('metadata', {}).get('source', 'v5.1') for r in final_train)
    print(f"\n  Source breakdown (train):")
    for src, cnt in sources.most_common():
        print(f"    {src}: {cnt}")

    # Toxicity distribution
    tox_dist = Counter(r['label']['toxicity'] for r in final_train)
    print(f"\n  Toxicity distribution (train):")
    for label, cnt in tox_dist.most_common():
        print(f"    {label}: {cnt} ({100*cnt/len(final_train):.1f}%)")

    # Write
    out_dir = Path('data/processed/v2')
    out_train = out_dir / 'v5_2_train.jsonl'
    out_dev = out_dir / 'v5_2_dev.jsonl'

    with open(out_train, 'w', encoding='utf-8') as f:
        for r in final_train:
            f.write(json.dumps(r, ensure_ascii=False) + '\n')

    with open(out_dev, 'w', encoding='utf-8') as f:
        for r in final_dev:
            f.write(json.dumps(r, ensure_ascii=False) + '\n')

    print(f"\n  Written: {out_train} ({len(final_train)} records)")
    print(f"  Written: {out_dev} ({len(final_dev)} records)")
    print("\nDone!")


if __name__ == '__main__':
    main()
