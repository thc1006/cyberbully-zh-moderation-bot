"""TDD tests for Phase 2 dataset normalization (ADR 0001 §3.2)."""
import json

import pandas as pd
import pytest

from cyberpuppy.data.phase2 import (LABELS, Phase2Normalizer, UnifiedRecord)

pytestmark = pytest.mark.unit


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


# ---- 8. STATE-ToxiCN normalizer --------------------------------------

def test_normalize_state_toxicn_hate_row(normalizer: Phase2Normalizer) -> None:
    row = {
        "id": 3346,
        "content": "話是這樣說，但是像我們這樣的普信男國女又看不上",
        "platform": "zhihu",
        "topic": "race",
        "sen_hate": "1",
        "Q1 Target": "普信男",
        "Q1 Argument": "國女又看不上",
        "Q1 Group": "Sexism",
        "Q1 hateful": "hate",
    }
    rec = normalizer.normalize_state_toxicn_row(row)
    assert rec is not None
    assert rec.label["toxicity"] == "toxic"
    assert rec.label["bullying"] == "harassment"
    assert rec.label["emotion"] == "neg"
    assert rec.metadata["source"] == "state_toxicn"
    assert "Sexism" in rec.metadata.get("hate_groups", "")


def test_normalize_state_toxicn_non_hate(normalizer: Phase2Normalizer) -> None:
    row = {
        "id": 2448,
        "content": "我的天啊，求求你坦白吧，放过人家",
        "platform": "zhihu",
        "topic": "race",
        "sen_hate": "0",
        "Q1 Target": "你",
        "Q1 Argument": "坦白吧",
        "Q1 Group": "non-hate",
        "Q1 hateful": "non-hate",
    }
    rec = normalizer.normalize_state_toxicn_row(row)
    assert rec is not None
    assert rec.label["toxicity"] == "none"
    assert rec.label["bullying"] == "none"
    assert rec.label["emotion"] == "neu"


def test_normalize_state_toxicn_short_dropped(normalizer: Phase2Normalizer) -> None:
    row = {"id": 1, "content": "ok", "sen_hate": "0",
           "platform": "x", "topic": "x", "Q1 Target": "", "Q1 Argument": "",
           "Q1 Group": "non-hate", "Q1 hateful": "non-hate"}
    assert normalizer.normalize_state_toxicn_row(row) is None


def test_state_toxicn_severe_when_multiple_targets(normalizer: Phase2Normalizer) -> None:
    """Records with 3+ hate quadruples are treated as 'severe'/'threat'."""
    row = {
        "id": 5, "content": "包含多目標仇恨的長文本，超過三個族群被攻擊",
        "platform": "x", "topic": "x", "sen_hate": "1",
        "Q1 Target": "A", "Q1 Argument": "x", "Q1 Group": "Sexism", "Q1 hateful": "hate",
        "Q2 Target": "B", "Q2 Argument": "y", "Q2 Group": "Racism", "Q2 hateful": "hate",
        "Q3 Target": "C", "Q3 Argument": "z", "Q3 Group": "LGBTQ", "Q3 hateful": "hate",
    }
    rec = normalizer.normalize_state_toxicn_row(row)
    assert rec is not None
    assert rec.label["toxicity"] == "severe"
    assert rec.label["bullying"] == "threat"


def test_process_state_toxicn_records_basic(normalizer: Phase2Normalizer) -> None:
    rows = [
        {"id": 1, "content": "我的天啊，求求你坦白吧，放过人家", "platform": "x", "topic": "x",
         "sen_hate": "0", "Q1 Target": "", "Q1 Argument": "", "Q1 Group": "non-hate",
         "Q1 hateful": "non-hate"},
        {"id": 2, "content": "話是這樣說，但是像我們這樣的普信男國女又看不上",
         "platform": "x", "topic": "x", "sen_hate": "1",
         "Q1 Target": "普信男", "Q1 Argument": "國女又看不上",
         "Q1 Group": "Sexism", "Q1 hateful": "hate"},
        {"id": 3, "content": "ab", "platform": "x", "topic": "x",
         "sen_hate": "0", "Q1 Target": "", "Q1 Argument": "",
         "Q1 Group": "non-hate", "Q1 hateful": "non-hate"},  # too short
    ]
    out = normalizer.process_state_toxicn_records(rows)
    assert len(out) == 2
    assert {r.label["toxicity"] for r in out} == {"none", "toxic"}


# ---- 9. SCCD normalizer (Yang et al., COLING 2025) -------------------

def test_normalize_sccd_comment_cb_low(normalizer: Phase2Normalizer) -> None:
    rec = normalizer.normalize_sccd_comment(
        comment_row={"comment_id": "x1", "label": "CB-Threat", "comment_content": "我打死你",
                     "post_id": "p1", "to_id": ""},
        post_severity="low",
    )
    assert rec is not None
    assert rec.label["toxicity"] == "toxic"
    # session severity low/med → toxic; high → severe
    rec2 = normalizer.normalize_sccd_comment(
        comment_row={"comment_id": "x2", "label": "CB-Insult", "comment_content": "笨蛋滾開",
                     "post_id": "p1", "to_id": ""},
        post_severity="high",
    )
    assert rec2.label["toxicity"] == "severe"


def test_normalize_sccd_comment_non_cb(normalizer: Phase2Normalizer) -> None:
    rec = normalizer.normalize_sccd_comment(
        comment_row={"comment_id": "x3", "label": "Non-CB", "comment_content": "今天天氣真好",
                     "post_id": "p1", "to_id": ""},
        post_severity="low",
    )
    assert rec.label["toxicity"] == "none"
    assert rec.label["bullying"] == "none"


def test_sccd_short_comment_dropped(normalizer: Phase2Normalizer) -> None:
    rec = normalizer.normalize_sccd_comment(
        comment_row={"comment_id": "x", "label": "Non-CB", "comment_content": "ok",
                     "post_id": "p", "to_id": ""},
        post_severity="low",
    )
    assert rec is None


def test_sccd_threat_label_maps_to_threat(normalizer: Phase2Normalizer) -> None:
    rec = normalizer.normalize_sccd_comment(
        comment_row={"comment_id": "x", "label": "CB-Threat", "comment_content": "我會找到你並且傷害你",
                     "post_id": "p", "to_id": ""},
        post_severity="medium",
    )
    assert rec.label["bullying"] == "threat"


# ---- 10. CHNCI normalizer (Zhu/Zou/Wu, May 2025) --------------------

def test_normalize_chnci_row_cyberbullying(normalizer: Phase2Normalizer) -> None:
    row = {"timestamp": "2024/9/26 6:02", "platform": "dy",
           "label1": 1, "label2": 1, "label3": 0, "diff": 1,
           "content": "這群人就該被掛在牆上示眾，國家不管他們真是太可悲了"}
    rec = normalizer.normalize_chnci_row(row, incident_label="cyberbullying", incident_name="x")
    assert rec is not None
    assert rec.label["toxicity"] == "toxic"  # majority vote 2/3 says CB
    assert rec.metadata["source"] == "chnci"
    assert rec.metadata["incident_label"] == "cyberbullying"


def test_normalize_chnci_row_non_cb(normalizer: Phase2Normalizer) -> None:
    row = {"timestamp": "2024/9/26", "platform": "wb",
           "label1": 0, "label2": 0, "label3": 0, "diff": 0,
           "content": "希望大家都平安健康，加油加油加油"}
    rec = normalizer.normalize_chnci_row(row, incident_label="non-cyberbullying", incident_name="x")
    assert rec is not None
    assert rec.label["toxicity"] == "none"


def test_chnci_split_vote_treated_as_toxic(normalizer: Phase2Normalizer) -> None:
    """Diff > 0 means annotators disagreed; if majority CB, still toxic."""
    row = {"timestamp": "x", "platform": "wb",
           "label1": 1, "label2": 1, "label3": 0, "diff": 1,
           "content": "這個世界真是糟糕透頂讓人無法理解"}
    rec = normalizer.normalize_chnci_row(row, incident_label="cyberbullying", incident_name="x")
    assert rec.label["toxicity"] == "toxic"
    assert rec.metadata["annotation_quality"] in ("silver", "weak")  # disagreement


# ---- 11. ToxiCloakCN normalizer (EMNLP 2024) ------------------------

def test_normalize_toxicloak_row_base(normalizer: Phase2Normalizer) -> None:
    rec = normalizer.normalize_toxicloak_row(
        content="好久都没有见过那么干净的白人班级了，一个黑乐色都没有",
        toxic=1, pair_id=42, variant="base",
    )
    assert rec is not None
    assert rec.label["toxicity"] == "toxic"
    assert rec.label["bullying"] == "harassment"
    assert rec.metadata["source"] == "toxicloak"
    assert rec.metadata["cloak_pair_id"] == 42
    assert rec.metadata["cloak_variant"] == "base"


def test_normalize_toxicloak_row_homo(normalizer: Phase2Normalizer) -> None:
    rec = normalizer.normalize_toxicloak_row(
        content="好久都没有间国那么干净的拜仁班级了，一个黑了铯都没有",
        toxic=1, pair_id=42, variant="homo",
    )
    assert rec.label["toxicity"] == "toxic"
    assert rec.metadata["cloak_pair_id"] == 42
    assert rec.metadata["cloak_variant"] == "homo"


def test_normalize_toxicloak_row_nontoxic(normalizer: Phase2Normalizer) -> None:
    rec = normalizer.normalize_toxicloak_row(
        content="今天天氣真好，我們去公園散步吧",
        toxic=0, pair_id=1, variant="base",
    )
    assert rec.label["toxicity"] == "none"
    assert rec.label["bullying"] == "none"
    assert rec.label["emotion"] == "neu"


def test_toxicloak_pair_shares_id_across_variants(normalizer: Phase2Normalizer) -> None:
    """base/homo/emoji of same row_index must carry identical cloak_pair_id."""
    recs = [
        normalizer.normalize_toxicloak_row("白人班級", toxic=1, pair_id=5, variant="base"),
        normalizer.normalize_toxicloak_row("拜仁班級", toxic=1, pair_id=5, variant="homo"),
        normalizer.normalize_toxicloak_row("👌人班級", toxic=1, pair_id=5, variant="emoji"),
    ]
    assert all(r is not None for r in recs)
    assert len({r.metadata["cloak_pair_id"] for r in recs}) == 1
    assert {r.metadata["cloak_variant"] for r in recs} == {"base", "homo", "emoji"}


def test_toxicloak_invalid_variant_raises(normalizer: Phase2Normalizer) -> None:
    with pytest.raises(ValueError, match="variant"):
        normalizer.normalize_toxicloak_row("x" * 10, toxic=0, pair_id=1, variant="BAD")


def test_toxicloak_short_text_dropped(normalizer: Phase2Normalizer) -> None:
    assert normalizer.normalize_toxicloak_row("ok", toxic=1, pair_id=1, variant="base") is None
