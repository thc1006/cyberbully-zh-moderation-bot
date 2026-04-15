"""Lexicon-driven homophone augmentation for v2.3 adversarial training.

Goal: programmatically generate (clean, homophone-cloaked) text pairs
from existing toxic samples, using the STATE-ToxiCN 829-word hate-slang
lexicon as both target list AND substitution source.

This complements the human-curated ToxiCloakCN dataset by extending
adversarial coverage to the long tail of homophone patterns the v2.2
training set under-represents.
"""
from __future__ import annotations

import json
import random
import re
from collections import defaultdict
from pathlib import Path
from typing import Any, List, Optional

from pypinyin import Style, pinyin

from .phase2 import UnifiedRecord

_HAN_RE = re.compile(r"[\u4e00-\u9fff]")


def _pinyin_of(ch: str) -> Optional[str]:
    py = pinyin(ch, style=Style.NORMAL, errors="ignore")
    if not py or not py[0]:
        return None
    return py[0][0]


class HomophoneAugmenter:
    def __init__(
        self,
        lexicon_path: Path,
        seed: int = 42,
        extra_char_pool: Optional[str] = None,
    ) -> None:
        self.lexicon_path = Path(lexicon_path)
        self.rng = random.Random(seed)
        raw = json.loads(self.lexicon_path.read_text(encoding="utf-8"))
        self.terms: List[str] = sorted(
            {t["term"] for t in raw.get("terms", []) if t.get("term")},
            key=len, reverse=True,  # longest-first match
        )
        char_pool: defaultdict[str, set[str]] = defaultdict(set)
        seed_chars = set("".join(self.terms))
        if extra_char_pool:
            seed_chars |= set(extra_char_pool)
        # Frequent-char corpus includes (a) generic high-freq chars and
        # (b) deliberate homophone clusters so swaps actually have alternatives.
        # Each cluster is a set of distinct characters sharing a Mandarin
        # pinyin (without tone). Expand here as v2.3 training surfaces gaps.
        _HOMO_CLUSTERS = (
            "笨本苯畚奔 蛋但淡彈氮丹單 滾棍輥磙 開揩凱慨愾鎧 "
            "白百佰柏拜捌掰 人壬仁忍認任 打答搭瘩噠 死絲思司私撕嘶 "
            "黑嘿 殺沙紗砂煞 害嗨亥還孩 毀悔晦會匯惠 "
            "噁哦惡鄂俄峨蛾額 臭醜抽仇酬綢 爛濫覽攬纜 廢費肺吠 "
            "笨蛋豬狗賤騷婊婊 操草艹肏 幹乾杆甘竿肝桿趕敢"
        )
        _COMMON = (
            "的一是不了人我在有他這為之大來以個中上們到說國和地也子時道出而要於就下"
            "得可你年生自會那後能對他著看起發當沒成只如事把還用第樣前所定其去行過家"
        )
        seed_chars |= set(_COMMON.replace(" ", ""))
        seed_chars |= set(_HOMO_CLUSTERS.replace(" ", ""))
        for ch in seed_chars:
            if not _HAN_RE.match(ch):
                continue
            py = _pinyin_of(ch)
            if py:
                char_pool[py].add(ch)
        self._char_pool = {k: sorted(v) for k, v in char_pool.items()}

    def _swap_char(self, ch: str) -> str:
        py = _pinyin_of(ch)
        if not py:
            return ch
        candidates = [c for c in self._char_pool.get(py, []) if c != ch]
        if not candidates:
            return ch
        return self.rng.choice(candidates)

    def _swap_term(self, term: str) -> str:
        return "".join(self._swap_char(c) for c in term)

    def augment_text(self, text: str) -> Optional[str]:
        out: List[str] = []
        i = 0
        n = len(text)
        any_swap = False
        while i < n:
            matched = False
            for term in self.terms:
                if term and text.startswith(term, i):
                    swapped = self._swap_term(term)
                    if swapped != term:
                        any_swap = True
                    out.append(swapped)
                    i += len(term)
                    matched = True
                    break
            if not matched:
                out.append(text[i])
                i += 1
        return "".join(out) if any_swap else None

    def build_pair_records(
        self,
        src_record: dict[str, Any],
        pair_id: int,
    ) -> List[UnifiedRecord]:
        cloaked = self.augment_text(src_record["text"])
        if cloaked is None:
            return []
        base_meta = {
            **src_record.get("metadata", {}),
            "cloak_pair_id": int(pair_id),
            "cloak_variant": "base",
            "augmented_from": src_record.get("metadata", {}).get("source", "unknown"),
        }
        homo_meta = {
            **src_record.get("metadata", {}),
            "source": "homo_aug",
            "cloak_pair_id": int(pair_id),
            "cloak_variant": "homo_lexicon",
            "augmented_from": src_record.get("metadata", {}).get("source", "unknown"),
        }
        return [
            UnifiedRecord(
                text=src_record["text"],
                label=dict(src_record["label"]),
                metadata=base_meta,
            ),
            UnifiedRecord(
                text=cloaked,
                label=dict(src_record["label"]),
                metadata=homo_meta,
            ),
        ]
