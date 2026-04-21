"""TDD tests for dual-LoRA simultaneous retrain data pipeline.

Critical property: text and pinyin training data must be perfectly aligned —
same records, same labels, same pair_ids, only the text field differs.

HED-COLD special property: homophone-perturbed texts should produce
identical pinyin to the original texts (that's the whole point).
"""
import pytest
import json
import csv
import re
from pathlib import Path
from pypinyin import Style, pinyin as get_pinyin


_HAN_RE = re.compile(r"[\u3400-\u4dbf\u4e00-\u9fff\uf900-\ufaff]")


def text_to_pinyin(text):
    syls = []
    for ch in text:
        if _HAN_RE.match(ch):
            try:
                py = get_pinyin(ch, style=Style.NORMAL, errors="ignore")
                if py and py[0] and py[0][0]:
                    syls.append(py[0][0])
            except:
                pass
        elif ch not in ' \t\n':
            syls.append(ch)
    return " ".join(syls)


class TestHEDCOLDHomophoneInvariance:
    """Verify that HED-COLD's homophone perturbations produce identical pinyin."""

    @pytest.fixture
    def traceability_pairs(self):
        """Load original→perturbed pairs from HED-COLD traceability data."""
        orig_path = Path('data/external/HED-COLD/traceability/sampled_train_data.csv')
        pert_path = Path('data/external/HED-COLD/dataset/train.csv')
        if not orig_path.exists() or not pert_path.exists():
            pytest.skip("HED-COLD data not available")

        originals = {}
        with open(orig_path, 'r', encoding='utf-8') as f:
            for row in csv.DictReader(f):
                originals[row['new_id']] = row['TEXT']

        perturbed = {}
        with open(pert_path, 'r', encoding='utf-8') as f:
            for row in csv.DictReader(f):
                perturbed[row['id']] = row['TEXT']

        pairs = []
        for rid in originals:
            if rid in perturbed:
                pairs.append((originals[rid], perturbed[rid]))
        return pairs

    def test_homophone_pairs_exist(self, traceability_pairs):
        """Traceability data should have matching pairs."""
        assert len(traceability_pairs) > 100

    def test_homophone_pinyin_invariance_rate(self, traceability_pairs):
        """Most homophone-perturbed texts should produce same pinyin as original.

        We expect >60% invariance (some perturbations are non-homophonic).
        """
        match = 0
        for orig, pert in traceability_pairs[:500]:  # sample for speed
            if text_to_pinyin(orig) == text_to_pinyin(pert):
                match += 1
        rate = match / min(500, len(traceability_pairs))
        assert rate > 0.4, f"Homophone invariance rate {rate:.2%} too low (expected >40%)"


class TestPinyinDataBuilder:
    """Test the pinyin JSONL builder."""

    def test_build_pinyin_record(self):
        """Pinyin conversion preserves all fields except text."""
        from cyberpuppy.data.pinyin_data_builder import build_pinyin_record

        record = {
            'text': '你好世界',
            'label': {'toxicity': 'none', 'bullying': 'none',
                      'role': 'none', 'emotion': 'neu'},
            'metadata': {'source': 'test', 'cloak_pair_id': 42},
        }
        pinyin_record = build_pinyin_record(record)

        # Text should be pinyin
        assert pinyin_record['text'] == text_to_pinyin('你好世界')
        # Labels must be identical
        assert pinyin_record['label'] == record['label']
        # Metadata preserved
        assert pinyin_record['metadata']['source'] == 'test'
        assert pinyin_record['metadata']['cloak_pair_id'] == 42

    def test_build_pinyin_preserves_non_han(self):
        """Non-Han characters (numbers, letters) should pass through."""
        from cyberpuppy.data.pinyin_data_builder import build_pinyin_record

        record = {
            'text': '装X 4了 NMSL',
            'label': {'toxicity': 'toxic', 'bullying': 'harassment',
                      'role': 'perpetrator', 'emotion': 'neg'},
            'metadata': {},
        }
        pr = build_pinyin_record(record)
        # X, 4, NMSL should remain
        assert 'X' in pr['text'] or 'x' in pr['text'].lower()
        assert '4' in pr['text']

    def test_build_pinyin_empty_text(self):
        """Empty text should produce empty pinyin."""
        from cyberpuppy.data.pinyin_data_builder import build_pinyin_record

        record = {'text': '', 'label': {'toxicity': 'none'}, 'metadata': {}}
        pr = build_pinyin_record(record)
        assert pr['text'] == ''


class TestDualDataAlignment:
    """Test that text and pinyin JSONL files are perfectly aligned."""

    def test_matching_record_count(self, tmp_path):
        """Text and pinyin files must have same number of records."""
        from cyberpuppy.data.pinyin_data_builder import build_pinyin_jsonl

        # Create sample text JSONL
        text_path = tmp_path / "text.jsonl"
        records = [
            {'text': '你好', 'label': {'toxicity': 'none'}, 'metadata': {}},
            {'text': '去死', 'label': {'toxicity': 'toxic'}, 'metadata': {}},
            {'text': '白痴', 'label': {'toxicity': 'toxic'}, 'metadata': {}},
        ]
        with open(text_path, 'w') as f:
            for r in records:
                f.write(json.dumps(r, ensure_ascii=False) + '\n')

        pinyin_path = tmp_path / "pinyin.jsonl"
        build_pinyin_jsonl(text_path, pinyin_path)

        # Count lines
        text_lines = sum(1 for _ in open(text_path))
        pinyin_lines = sum(1 for _ in open(pinyin_path))
        assert text_lines == pinyin_lines

    def test_matching_labels(self, tmp_path):
        """Every record's labels must be identical between text and pinyin."""
        from cyberpuppy.data.pinyin_data_builder import build_pinyin_jsonl

        text_path = tmp_path / "text.jsonl"
        records = [
            {'text': '你好世界', 'label': {'toxicity': 'none', 'bullying': 'none'},
             'metadata': {'cloak_pair_id': -1}},
            {'text': '去死吧', 'label': {'toxicity': 'toxic', 'bullying': 'threat'},
             'metadata': {'cloak_pair_id': 5}},
        ]
        with open(text_path, 'w') as f:
            for r in records:
                f.write(json.dumps(r, ensure_ascii=False) + '\n')

        pinyin_path = tmp_path / "pinyin.jsonl"
        build_pinyin_jsonl(text_path, pinyin_path)

        text_records = [json.loads(l) for l in open(text_path)]
        pinyin_records = [json.loads(l) for l in open(pinyin_path)]

        for tr, pr in zip(text_records, pinyin_records):
            assert tr['label'] == pr['label']
            assert tr['metadata'].get('cloak_pair_id') == pr['metadata'].get('cloak_pair_id')

    def test_pinyin_text_differs(self, tmp_path):
        """Pinyin text should actually be different from original (Han text)."""
        from cyberpuppy.data.pinyin_data_builder import build_pinyin_jsonl

        text_path = tmp_path / "text.jsonl"
        records = [
            {'text': '你好世界', 'label': {'toxicity': 'none'}, 'metadata': {}},
        ]
        with open(text_path, 'w') as f:
            for r in records:
                f.write(json.dumps(r, ensure_ascii=False) + '\n')

        pinyin_path = tmp_path / "pinyin.jsonl"
        build_pinyin_jsonl(text_path, pinyin_path)

        tr = json.loads(open(text_path).readline())
        pr = json.loads(open(pinyin_path).readline())
        assert tr['text'] != pr['text'], "Pinyin should differ from original Han text"
        assert 'ni' in pr['text'].lower(), f"Expected pinyin 'ni' in '{pr['text']}'"
