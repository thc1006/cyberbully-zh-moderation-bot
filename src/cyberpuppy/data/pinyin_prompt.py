"""Toneless pinyin prompt preprocessor for homophone-robust classification.

Prepends a toneless pinyin prefix to input text so that homophones
produce IDENTICAL prefixed input. Validated by PCR-ToxiCN (arXiv
2507.07640) on LLMs; adapted here for fine-tuned classification.

Example:
  add_pinyin_prompt("白人都該死")
  → "[拼音 bai ren dou gai si] 白人都該死"

  add_pinyin_prompt("拜仁都該死")
  → "[拼音 bai ren dou gai si] 拜仁都該死"   ← SAME prefix!

Known limitation: near-homophones with different base pinyin (e.g.,
嗨/黑 = hai/hei) will produce different prefixes. These account for
~17% of ToxiCloakCN substitutions.
"""
from __future__ import annotations

import re

from pypinyin import Style, pinyin

# CJK Unified Ideographs (basic + Extension A + Compatibility)
_HAN_RE = re.compile(r"[\u3400-\u4dbf\u4e00-\u9fff\uf900-\ufaff]")

_PREFIX_TAG = "[拼音 "
_SUFFIX_TAG = "] "


def _chinese_chars_to_pinyin(text: str) -> str:
    """Extract toneless pinyin for Chinese characters only, space-separated.

    Non-Chinese characters (English, numbers, emoji, punctuation) are
    silently skipped — they don't contribute to the pinyin prefix.
    """
    syllables: list[str] = []
    for ch in text:
        if _HAN_RE.match(ch):
            try:
                py = pinyin(ch, style=Style.NORMAL, errors="ignore")
                if py and py[0] and py[0][0]:
                    syllables.append(py[0][0])
            except Exception:
                pass
    return " ".join(syllables)


def add_pinyin_prompt(text: str) -> str:
    """Prepend toneless pinyin prefix to text.

    Format: "[拼音 <pinyin syllables>] <original text>"

    Args:
        text: Input text (typically Traditional Chinese).

    Returns:
        Text with pinyin prefix prepended.
    """
    py = _chinese_chars_to_pinyin(text)
    return f"{_PREFIX_TAG}{py}{_SUFFIX_TAG}{text}"


def extract_pinyin_prefix(prompted_text: str) -> str:
    """Extract the pinyin portion from a prompted text.

    Inverse of add_pinyin_prompt — returns just the space-separated
    pinyin syllables (without the [拼音 ] markers).
    """
    if not prompted_text.startswith(_PREFIX_TAG):
        return ""
    end = prompted_text.index(_SUFFIX_TAG)
    return prompted_text[len(_PREFIX_TAG):end]
