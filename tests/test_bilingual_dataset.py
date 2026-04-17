"""TDD contract for v5 bilingual + short-text training dataset.

Three gaps to fix:
  1. Script: training 95% Traditional → add Simplified copies
  2. Length: training avg 35 chars → add CNTP short texts (~10 chars)
  3. Domain: formal offensive → add CNTP internet-style perturbed texts

Dataset should contain:
  - Original v2.2 Traditional records (70K, with ToxiCloakCN triplets)
  - Simplified copies of v2.2 (70K, no cloak_pair_id)
  - CNTP toxic + non-toxic (both simp + trad, ~10K)
  - CNTP homo pairs with cloak_pair_id (for consistency loss)
"""
from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

import pytest


V5_TRAIN = Path("data/processed/v2/v5_bilingual_train.jsonl")
V5_DEV = Path("data/processed/v2/v5_bilingual_dev.jsonl")


@pytest.fixture(scope="module")
def train_records():
    if not V5_TRAIN.exists():
        pytest.skip("v5 bilingual dataset not built yet")
    records = []
    with V5_TRAIN.open() as f:
        for line in f:
            records.append(json.loads(line))
    return records


class TestDatasetSize:
    def test_train_has_enough_records(self, train_records):
        """Should be ~140-160K (70K trad + 70K simp + ~10K CNTP)."""
        assert 130_000 <= len(train_records) <= 170_000, \
            f"Expected ~150K, got {len(train_records)}"


class TestBilingualCoverage:
    def test_contains_traditional_markers(self, train_records):
        """Dataset must contain Traditional Chinese (這/個/們/還)."""
        trad_count = sum(1 for r in train_records[:1000]
                         if any(c in r['text'] for c in '這個們還學認國對說經過開進從東'))
        assert trad_count > 100, f"Too few Traditional: {trad_count}/1000"

    def test_contains_simplified_markers(self, train_records):
        """Dataset must contain Simplified Chinese (这/个/们/还)."""
        simp_count = sum(1 for r in train_records[:1000]
                         if any(c in r['text'] for c in '这个们还学认国对说经过开进从东'))
        # Check in the second half (simplified copies are after traditional)
        mid = len(train_records) // 2
        simp_count2 = sum(1 for r in train_records[mid:mid+1000]
                          if any(c in r['text'] for c in '这个们还学认国对说经过开进从东'))
        total = max(simp_count, simp_count2)
        assert total > 100, f"Too few Simplified: {total}/1000"


class TestShortTextCoverage:
    def test_contains_short_texts(self, train_records):
        """Dataset must contain short texts (< 15 chars) from CNTP."""
        short = sum(1 for r in train_records if len(r['text']) < 15)
        assert short > 1000, f"Too few short texts: {short}"

    def test_short_text_has_toxic_labels(self, train_records):
        """Short texts must include toxic examples (not all none)."""
        short_toxic = sum(1 for r in train_records
                          if len(r['text']) < 15
                          and r['label'].get('toxicity') in ('toxic', 'severe'))
        assert short_toxic > 200, f"Too few short toxic: {short_toxic}"


class TestCNTPIntegration:
    def test_cntp_source_present(self, train_records):
        """Records from CNTP should have source='cntp' in metadata."""
        cntp = sum(1 for r in train_records
                   if r.get('metadata', {}).get('source', '').startswith('cntp'))
        assert cntp > 3000, f"Too few CNTP records: {cntp}"

    def test_cntp_homo_pairs_have_cloak_pair_id(self, train_records):
        """CNTP homo perturbed pairs should have cloak_pair_id for consistency loss."""
        cntp_paired = sum(1 for r in train_records
                          if r.get('metadata', {}).get('source', '') == 'cntp_homo'
                          and r.get('metadata', {}).get('cloak_pair_id', -1) >= 0)
        assert cntp_paired > 2000, f"Too few CNTP homo pairs: {cntp_paired}"


class TestToxiCloakCNPreserved:
    def test_toxicloak_triplets_preserved(self, train_records):
        """Original ToxiCloakCN triplets must still be present with cloak_pair_ids."""
        tc = sum(1 for r in train_records
                 if r.get('metadata', {}).get('source') == 'toxicloak')
        assert tc >= 11000, f"ToxiCloakCN records missing: {tc} (need ≥11000)"


class TestLabelIntegrity:
    def test_all_records_have_toxicity_label(self, train_records):
        """Every record must have a toxicity label."""
        missing = sum(1 for r in train_records
                      if r.get('label', {}).get('toxicity') is None)
        assert missing == 0, f"{missing} records missing toxicity label"

    def test_no_empty_text(self, train_records):
        """No record should have empty text."""
        empty = sum(1 for r in train_records if not r.get('text', '').strip())
        assert empty == 0, f"{empty} records have empty text"

    def test_label_values_valid(self, train_records):
        """Toxicity labels must be none/toxic/severe."""
        valid = {'none', 'toxic', 'severe'}
        invalid = sum(1 for r in train_records
                      if r['label'].get('toxicity') not in valid)
        assert invalid == 0, f"{invalid} records have invalid toxicity label"
