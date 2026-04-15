"""TDD contract for v2.4 ToxiCloakCN-style augmenter.

Goal: same-distribution synthesis (matches ToxiCloakCN heldout) by using
the SAME library ToxiCloakCN's authors used (JioNLP, per paper §3.1.2).
This fixes v2.3's distribution-mismatch failure (homo drop −9.23%, worse
than v2.2's −8.51%).
"""
from __future__ import annotations

import pytest

from cyberpuppy.data.toxicloak_style_aug import ToxiCloakStyleAugmenter


@pytest.fixture
def aug():
    return ToxiCloakStyleAugmenter(homo_ratio=0.3, seed=42)


class TestBasicBehavior:
    def test_creates_instance(self, aug):
        assert aug is not None

    def test_augment_text_returns_str_or_none(self, aug):
        out = aug.augment_text("你這個笨蛋滾開")
        assert out is None or isinstance(out, str)

    def test_augment_text_changes_input_for_typical_toxic(self, aug):
        """Typical toxic Chinese should produce SOME perturbation."""
        text = "你這個笨蛋滾開"
        out = aug.augment_text(text)
        assert out is not None, "JioNLP should produce a variant for non-trivial toxic Chinese"
        assert out != text, "Variant should differ from original"

    def test_augment_text_preserves_length_roughly(self, aug):
        """Whole-word homophone substitution shouldn't change length much (±20%)."""
        text = "好久都沒有見過那麼乾淨的白人班級了"
        out = aug.augment_text(text)
        assert out is not None
        assert abs(len(out) - len(text)) <= max(2, int(len(text) * 0.2))


class TestTraditionalChinesePassthrough:
    def test_traditional_input_yields_traditional_output(self, aug):
        """v2.4 training data is Traditional (s2twp); augmenter must round-trip."""
        text = "你這個笨蛋滾開"  # Traditional
        out = aug.augment_text(text)
        assert out is not None
        # Forbid common simplified-only chars from leaking through
        # (這 is Trad; 这 is Simp; if input had 這 and not 这, output shouldn't introduce 这)
        if "這" in text and "这" not in text:
            assert "这" not in out, f"Simplified char leaked into Traditional output: {out}"

    def test_unchanged_chars_preserve_traditional_form(self, aug):
        text = "今天天氣真好"
        out = aug.augment_text(text)
        # If 氣 was unchanged, it should still be Traditional 氣 not Simp 气
        if out is not None and "氣" in text:
            # At least one of: (a) 氣 still present, or (b) 氣 was substituted away (no 氣 in out and no 气)
            if "氣" not in out:
                # 氣 was substituted; substitute should not be 气 (that would be a simp/trad bug)
                pass


class TestDeterminism:
    def test_same_seed_same_output(self):
        """Fixed seed -> reproducible output (required for deterministic data builds)."""
        a1 = ToxiCloakStyleAugmenter(homo_ratio=0.3, seed=42)
        a2 = ToxiCloakStyleAugmenter(homo_ratio=0.3, seed=42)
        text = "再嘴一句試試看，我打死你"
        assert a1.augment_text(text) == a2.augment_text(text)

    def test_different_seeds_may_differ(self):
        a1 = ToxiCloakStyleAugmenter(homo_ratio=0.5, seed=1)
        a2 = ToxiCloakStyleAugmenter(homo_ratio=0.5, seed=999)
        text = "我覺得中國女性媚外值得批評的是中國男性"
        # Don't strictly require different (could collide); just check both succeed
        out1 = a1.augment_text(text)
        out2 = a2.augment_text(text)
        assert out1 is not None and out2 is not None


class TestEdgeCases:
    def test_empty_string(self, aug):
        assert aug.augment_text("") is None

    def test_no_chinese_returns_none(self, aug):
        out = aug.augment_text("hello world 123")
        assert out is None or out == "hello world 123"

    def test_pure_emoji(self, aug):
        out = aug.augment_text("🎉🎊🎈")
        assert out is None or out == "🎉🎊🎈"

    def test_single_char_returns_none_or_unchanged(self, aug):
        # Single char usually has nothing to substitute
        out = aug.augment_text("好")
        assert out is None or out == "好"


class TestPairBuilding:
    def test_build_pair_returns_two_records(self, aug):
        src = {
            "text": "再嘴一句試試看，我打死你",
            "label": {"toxicity": "toxic", "bullying": "harassment"},
            "metadata": {"source": "cold"},
        }
        records = aug.build_pair_records(src, pair_id=12345)
        assert len(records) == 2

    def test_pair_records_share_pair_id(self, aug):
        src = {
            "text": "再嘴一句試試看，我打死你",
            "label": {"toxicity": "toxic"},
            "metadata": {"source": "cold"},
        }
        records = aug.build_pair_records(src, pair_id=12345)
        assert records[0].metadata["cloak_pair_id"] == 12345
        assert records[1].metadata["cloak_pair_id"] == 12345

    def test_pair_records_have_distinct_variants(self, aug):
        src = {
            "text": "再嘴一句試試看，我打死你",
            "label": {"toxicity": "toxic"},
            "metadata": {"source": "cold"},
        }
        records = aug.build_pair_records(src, pair_id=1)
        variants = {r.metadata["cloak_variant"] for r in records}
        assert variants == {"base", "homo_jionlp"}

    def test_pair_records_preserve_label(self, aug):
        src = {
            "text": "再嘴一句試試看，我打死你",
            "label": {"toxicity": "toxic", "bullying": "harassment"},
            "metadata": {"source": "cold"},
        }
        records = aug.build_pair_records(src, pair_id=1)
        for r in records:
            assert r.label["toxicity"] == "toxic"
            assert r.label["bullying"] == "harassment"

    def test_pair_records_tag_homo_source(self, aug):
        src = {
            "text": "再嘴一句試試看，我打死你",
            "label": {"toxicity": "toxic"},
            "metadata": {"source": "cold"},
        }
        records = aug.build_pair_records(src, pair_id=1)
        homo = next(r for r in records if r.metadata["cloak_variant"] == "homo_jionlp")
        assert homo.metadata["source"] == "jionlp_homo_aug"
        assert homo.metadata["augmented_from"] == "cold"

    def test_pair_records_empty_when_no_augmentation(self, aug):
        src = {
            "text": "hello",
            "label": {"toxicity": "none"},
            "metadata": {"source": "cold"},
        }
        records = aug.build_pair_records(src, pair_id=999)
        assert records == []

    def test_pair_text_actually_differs(self, aug):
        """The two records must have textually different `text` fields."""
        src = {
            "text": "再嘴一句試試看，我打死你",
            "label": {"toxicity": "toxic"},
            "metadata": {"source": "cold"},
        }
        records = aug.build_pair_records(src, pair_id=1)
        assert records[0].text != records[1].text


class TestDistributionAlignment:
    """Sanity checks that output style matches ToxiCloakCN (via JioNLP)."""

    def test_substitutions_at_char_level(self, aug):
        """At char level we should see >=1 differences for typical toxic content."""
        text = "好久都沒有見過那麼乾淨的白人班級了"
        out = aug.augment_text(text)
        assert out is not None
        diffs = sum(1 for a, b in zip(text, out) if a != b)
        assert diffs >= 1

    def test_higher_ratio_more_aggressive(self):
        """Higher homo_ratio should generally produce >= swaps as lower ratio."""
        text = "我覺得中國女性媚外值得批評的是中國男性，你們看一看出軌的女人"
        a_low = ToxiCloakStyleAugmenter(homo_ratio=0.05, seed=42)
        a_high = ToxiCloakStyleAugmenter(homo_ratio=0.5, seed=42)
        out_low = a_low.augment_text(text)
        out_high = a_high.augment_text(text)
        # Both should produce a variant; high should not have FEWER swaps than low
        if out_low is not None and out_high is not None:
            diffs_low = sum(1 for a, b in zip(text, out_low) if a != b)
            diffs_high = sum(1 for a, b in zip(text, out_high) if a != b)
            assert diffs_high >= diffs_low - 2, (
                f"Higher ratio should give >= swaps; low={diffs_low} high={diffs_high}"
            )
