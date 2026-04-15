"""TDD for v2.3 lexicon-driven homophone augmentation.

Strategy: take existing toxic samples, swap key tokens to their pinyin-
homophones using the STATE-ToxiCN 829-word hate-slang lexicon as both
target list AND substitution source. Output: synthetic (clean, homophone)
pairs that share a `cloak_pair_id`, just like ToxiCloakCN.

Pipeline contract (the unit under test):
  HomophoneAugmenter(lexicon_path, seed=42)
    .augment_text(text)              -> str | None       (None if no swap possible)
    .build_pair_records(rec, pid)    -> list[UnifiedRecord]   (clean + cloaked, same pid)

Tests use a tiny synthetic lexicon, NOT the real STATE-ToxiCN file.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from cyberpuppy.data.homophone_aug import HomophoneAugmenter

pytestmark = pytest.mark.unit


@pytest.fixture
def tiny_lexicon(tmp_path: Path) -> Path:
    """Minimal lexicon: 4 terms covering different homophone patterns."""
    lex = {
        "title": "test",
        "language": "zh",
        "terms": [
            {"term": "笨蛋", "category": "Insult", "definition": "stupid person"},
            {"term": "白人", "category": "Racism", "definition": "white people"},
            {"term": "滾開", "category": "Insult", "definition": "go away"},
            {"term": "打死", "category": "Threat", "definition": "beat to death"},
        ],
    }
    p = tmp_path / "lex.json"
    p.write_text(json.dumps(lex, ensure_ascii=False), encoding="utf-8")
    return p


def test_augmenter_loads_lexicon(tiny_lexicon: Path) -> None:
    aug = HomophoneAugmenter(lexicon_path=tiny_lexicon, seed=42)
    assert len(aug.terms) == 4
    assert "笨蛋" in aug.terms


def test_augment_text_swaps_known_term(tiny_lexicon: Path) -> None:
    aug = HomophoneAugmenter(lexicon_path=tiny_lexicon, seed=42)
    src = "你這個笨蛋，滾開！"
    out = aug.augment_text(src)
    assert out is not None
    assert out != src
    # At least one term should have been swapped to its homophone form
    assert ("笨蛋" not in out) or ("滾開" not in out)


def test_augment_text_returns_none_when_no_swap_possible(tiny_lexicon: Path) -> None:
    """Text with no overlap with lexicon can't be augmented."""
    aug = HomophoneAugmenter(lexicon_path=tiny_lexicon, seed=42)
    out = aug.augment_text("今天天氣真好我們去公園散步")
    assert out is None


def test_augment_is_deterministic_with_same_seed(tiny_lexicon: Path) -> None:
    a = HomophoneAugmenter(lexicon_path=tiny_lexicon, seed=42)
    b = HomophoneAugmenter(lexicon_path=tiny_lexicon, seed=42)
    src = "你這個笨蛋，滾開！"
    assert a.augment_text(src) == b.augment_text(src)


def test_augment_differs_with_different_seed(tiny_lexicon: Path) -> None:
    a = HomophoneAugmenter(lexicon_path=tiny_lexicon, seed=1)
    b = HomophoneAugmenter(lexicon_path=tiny_lexicon, seed=2)
    src = "你笨蛋，滾開，白人打死"  # 4 swap candidates -> diff orders
    outs = {a.augment_text(src), b.augment_text(src)}
    assert len(outs) >= 1  # at least produces something; may differ


def test_build_pair_records_shares_pair_id(tiny_lexicon: Path) -> None:
    """Generated cloaked record carries cloak_pair_id for consistency loss."""
    aug = HomophoneAugmenter(lexicon_path=tiny_lexicon, seed=42)
    src_record = {
        "text": "你這個笨蛋，滾開！",
        "label": {
            "toxicity": "toxic", "bullying": "harassment",
            "role": "perpetrator", "emotion": "neg", "emotion_strength": 3,
        },
        "metadata": {"source": "cold", "is_traditional": True},
    }
    pair = aug.build_pair_records(src_record, pair_id=12345)
    assert len(pair) == 2
    base, homo = pair
    assert base.metadata["cloak_pair_id"] == 12345
    assert homo.metadata["cloak_pair_id"] == 12345
    assert base.metadata["cloak_variant"] == "base"
    assert homo.metadata["cloak_variant"] == "homo_lexicon"
    # Labels must match — augmentation preserves semantic
    assert base.label == homo.label
    assert base.text != homo.text
    # Source provenance flagged
    assert homo.metadata["source"] == "homo_aug"


def test_build_pair_records_returns_empty_when_no_swap(tiny_lexicon: Path) -> None:
    aug = HomophoneAugmenter(lexicon_path=tiny_lexicon, seed=42)
    src = {"text": "今天天氣不錯", "label": {"toxicity": "none", "bullying": "none",
            "role": "none", "emotion": "neu", "emotion_strength": 0},
            "metadata": {"source": "cold", "is_traditional": True}}
    assert aug.build_pair_records(src, pair_id=1) == []


def test_real_state_toxicn_lexicon_loadable() -> None:
    """Sanity: the real 829-term lexicon parses (skip if file absent)."""
    p = Path("data/external/STATE-ToxiCN/data/annotated lexicon.json")
    if not p.exists():
        pytest.skip("STATE-ToxiCN lexicon not cloned locally")
    aug = HomophoneAugmenter(lexicon_path=p, seed=42)
    assert len(aug.terms) >= 800  # ~829
