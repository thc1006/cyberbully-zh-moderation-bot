"""TDD contract for toneless pinyin prompt preprocessor.

Approach: prepend toneless pinyin of Chinese characters to input text.
Validated by PCR-ToxiCN (arXiv 2507.07640) on LLMs.

Core invariant: true homophones get IDENTICAL pinyin prefix.
Known limitation: near-homophones (嗨/黑 = hai/hei) will differ.
"""
from __future__ import annotations

import pytest

from cyberpuppy.data.pinyin_prompt import add_pinyin_prompt, extract_pinyin_prefix


class TestBasicBehavior:
    def test_returns_string(self):
        assert isinstance(add_pinyin_prompt("白人都該死"), str)

    def test_format_has_prefix_marker(self):
        result = add_pinyin_prompt("白人都該死")
        assert result.startswith("[拼音 ")
        assert "] " in result

    def test_original_text_preserved(self):
        text = "白人都該死"
        result = add_pinyin_prompt(text)
        assert text in result

    def test_pinyin_is_toneless(self):
        result = add_pinyin_prompt("白")
        # Should be "bai" not "bái" or "bai2"
        prefix = extract_pinyin_prefix(result)
        assert prefix == "bai", f"Expected toneless 'bai', got '{prefix}'"

    def test_pinyin_space_separated(self):
        result = add_pinyin_prompt("白人")
        prefix = extract_pinyin_prefix(result)
        assert prefix == "bai ren"


class TestHomophoneInvariant:
    """Core design invariant: true homophones produce identical prefix."""

    HOMO_PAIRS = [
        ("智商", "之上"),
        ("白人都該死", "拜仁都該死"),
        ("歧視黑皮", "其實黑披"),
        ("狗貨給私信關了", "溝或給私信館了"),
        ("非洲黑人智商70", "非洲黑人之上70"),
    ]

    @pytest.mark.parametrize("clean,homo", HOMO_PAIRS)
    def test_homophone_same_pinyin_prefix(self, clean, homo):
        prefix_clean = extract_pinyin_prefix(add_pinyin_prompt(clean))
        prefix_homo = extract_pinyin_prefix(add_pinyin_prompt(homo))
        assert prefix_clean == prefix_homo, (
            f"Homophone prefix mismatch:\n"
            f"  clean: '{prefix_clean}'\n"
            f"  homo:  '{prefix_homo}'"
        )


class TestEdgeCases:
    def test_empty_string(self):
        result = add_pinyin_prompt("")
        assert isinstance(result, str)

    def test_pure_english(self):
        result = add_pinyin_prompt("hello world")
        # No Chinese chars → empty pinyin but text preserved
        assert "hello world" in result

    def test_pure_numbers(self):
        result = add_pinyin_prompt("12345")
        assert "12345" in result

    def test_emoji_skipped_in_pinyin(self):
        result = add_pinyin_prompt("好👍壞")
        prefix = extract_pinyin_prefix(result)
        # Only Chinese chars get pinyin; emoji skipped
        assert "hao" in prefix
        assert "huai" in prefix

    def test_mixed_chinese_english(self):
        result = add_pinyin_prompt("hello白人world")
        prefix = extract_pinyin_prefix(result)
        assert "bai" in prefix
        assert "ren" in prefix
        assert "hello" not in prefix  # English not in pinyin

    def test_punctuation_not_in_pinyin(self):
        result = add_pinyin_prompt("你好，世界！")
        prefix = extract_pinyin_prefix(result)
        # Punctuation should not appear in pinyin prefix
        assert "，" not in prefix
        assert "！" not in prefix


class TestTraditionalChinese:
    def test_traditional_chars_get_pinyin(self):
        """Traditional Chinese chars (our training data) should work."""
        result = add_pinyin_prompt("該死")
        prefix = extract_pinyin_prefix(result)
        assert prefix == "gai si"

    def test_traditional_equals_simplified_pinyin(self):
        """繁體 and 簡體 of same word should have same pinyin."""
        trad = extract_pinyin_prefix(add_pinyin_prompt("該死"))
        simp = extract_pinyin_prefix(add_pinyin_prompt("该死"))
        assert trad == simp


class TestBatchProcessing:
    def test_process_records(self):
        """Simulate processing training records."""
        records = [
            {"text": "白人都該死", "label": {"toxicity": "toxic"}},
            {"text": "今天天氣真好", "label": {"toxicity": "none"}},
        ]
        processed = [
            {**r, "text": add_pinyin_prompt(r["text"])} for r in records
        ]
        assert all("[拼音" in r["text"] for r in processed)
        assert processed[0]["label"]["toxicity"] == "toxic"  # label preserved
