"""TDD tests for Phase 2 dataset normalization (ADR 0001 §3.2)."""
import json

import pandas as pd
import pytest

from cyberpuppy.data.phase2 import (LABELS, Phase2Normalizer, UnifiedRecord)


@pytest.fixture
def normalizer() -> Phase2Normalizer:
    return Phase2Normalizer(target="traditional", min_chars=3, max_chars=512)


# ---- 1. OpenCC 繁化 -------------------------------------------------------

def test_opencc_simplified_to_traditional(normalizer: Phase2Normalizer) -> None:
    out = normalizer.opencc_convert("简体中文测试")
    assert "簡" in out and "體" in out and "測" in out and "試" in out


def test_opencc_idempotent_on_traditional(normalizer: Phase2Normalizer) -> None:
    src = "我打死你"
    assert normalizer.opencc_convert(src) == src


def test_opencc_preserves_punctuation(normalizer: Phase2Normalizer) -> None:
    out = normalizer.opencc_convert("你好！怎么样？")
    assert "！" in out and "？" in out


# ---- 2. PII 去除 ----------------------------------------------------------

def test_scrub_pii_phone(normalizer: Phase2Normalizer) -> None:
    assert "[PHONE]" in normalizer.scrub_pii("聯絡 0912345678")
    assert "0912345678" not in normalizer.scrub_pii("聯絡 0912345678")


def test_scrub_pii_email(normalizer: Phase2Normalizer) -> None:
    assert "[EMAIL]" in normalizer.scrub_pii("a@b.com")
    assert "a@b.com" not in normalizer.scrub_pii("a@b.com 記得回覆")


def test_scrub_pii_id(normalizer: Phase2Normalizer) -> None:
    assert "[ID]" in normalizer.scrub_pii("身分證 A123456789 請保管")


def test_scrub_pii_no_false_positive(normalizer: Phase2Normalizer) -> None:
    benign = "今天天氣真好"
    assert normalizer.scrub_pii(benign) == benign


# ---- 3. 長度過濾 ----------------------------------------------------------

@pytest.mark.parametrize("text,expected", [
    ("", False),
    ("a", False),
    ("aa", False),
    ("aaa", True),
    ("a" * 512, True),
    ("a" * 513, False),
])
def test_is_valid_length(normalizer: Phase2Normalizer, text: str, expected: bool) -> None:
    assert normalizer.is_valid_length(text) is expected


# ---- 4. COLD 標籤對映 -----------------------------------------------------

def test_normalize_cold_row_toxic(normalizer: Phase2Normalizer) -> None:
    rec = normalizer.normalize_cold_row({
        "TEXT": "笨蛋滾開", "label": 1, "fine-grained-label": 1
    })
    assert isinstance(rec, UnifiedRecord)
    assert rec.label["toxicity"] == "toxic"
    assert rec.label["bullying"] == "harassment"
    assert rec.label["role"] == "none"
    assert rec.label["emotion"] == "neu"
    assert rec.metadata["source"] == "cold"


def test_normalize_cold_row_none(normalizer: Phase2Normalizer) -> None:
    rec = normalizer.normalize_cold_row({
        "TEXT": "今天天氣真好", "label": 0, "fine-grained-label": 0
    })
    assert rec.label["toxicity"] == "none"
    assert rec.label["bullying"] == "none"


def test_normalize_cold_row_invalid_returns_none(normalizer: Phase2Normalizer) -> None:
    # text below min_chars -> drop
    assert normalizer.normalize_cold_row({"TEXT": "hi", "label": 0, "fine-grained-label": 0}) is None


def test_cold_row_traditional_after_normalize(normalizer: Phase2Normalizer) -> None:
    rec = normalizer.normalize_cold_row({
        "TEXT": "简体测试", "label": 0, "fine-grained-label": 0
    })
    assert rec is not None
    assert rec.metadata["is_traditional"] is True
    assert "簡" in rec.text


# ---- 5. Schema validation ------------------------------------------------

def test_unified_record_label_vocab(normalizer: Phase2Normalizer) -> None:
    rec = normalizer.normalize_cold_row({"TEXT": "笨蛋滾開", "label": 1, "fine-grained-label": 1})
    d = rec.to_dict()
    assert d["label"]["toxicity"] in LABELS["toxicity"]
    assert d["label"]["bullying"] in LABELS["bullying"]
    assert d["label"]["role"] in LABELS["role"]
    assert d["label"]["emotion"] in LABELS["emotion"]
    assert isinstance(d["label"]["emotion_strength"], int)


def test_unified_record_round_trip_json(normalizer: Phase2Normalizer) -> None:
    rec = normalizer.normalize_cold_row({"TEXT": "笨蛋滾開", "label": 1, "fine-grained-label": 1})
    json_str = json.dumps(rec.to_dict(), ensure_ascii=False)
    parsed = json.loads(json_str)
    assert parsed["label"]["toxicity"] == "toxic"


# ---- 6. Hashing for dedup -------------------------------------------------

def test_text_hash_stable(normalizer: Phase2Normalizer) -> None:
    assert normalizer.text_hash("hello") == normalizer.text_hash("hello")


def test_text_hash_distinct(normalizer: Phase2Normalizer) -> None:
    assert normalizer.text_hash("hello") != normalizer.text_hash("world")


def test_text_hash_length(normalizer: Phase2Normalizer) -> None:
    assert len(normalizer.text_hash("x")) == 16


# ---- 7. End-to-end COLD dataframe ----------------------------------------

def test_process_cold_dataframe_basic(normalizer: Phase2Normalizer) -> None:
    df = pd.DataFrame([
        {"TEXT": "今天天氣真好", "label": 0, "fine-grained-label": 0},
        {"TEXT": "笨蛋你滾開", "label": 1, "fine-grained-label": 1},
        {"TEXT": "ab", "label": 0, "fine-grained-label": 0},  # too short, dropped
    ])
    out = normalizer.process_cold_dataframe(df)
    assert len(out) == 2
    assert all(r.label["toxicity"] in {"none", "toxic"} for r in out)


def test_process_cold_dataframe_dedup(normalizer: Phase2Normalizer) -> None:
    df = pd.DataFrame([
        {"TEXT": "完全相同句", "label": 0, "fine-grained-label": 0},
        {"TEXT": "完全相同句", "label": 0, "fine-grained-label": 0},
    ])
    out = normalizer.process_cold_dataframe(df, dedup=True)
    assert len(out) == 1
