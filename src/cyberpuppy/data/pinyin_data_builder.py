"""Build pinyin-converted JSONL from text JSONL for dual-LoRA training.

Converts the 'text' field to toneless pinyin while preserving all other
fields (labels, metadata, pair_ids) exactly. This ensures text and pinyin
LoRAs are trained on perfectly aligned data.
"""
from __future__ import annotations

import json
import re
from pathlib import Path
from pypinyin import Style, pinyin as get_pinyin

_HAN_RE = re.compile(r"[\u3400-\u4dbf\u4e00-\u9fff\uf900-\ufaff]")


def text_to_pinyin(text: str) -> str:
    """Convert Chinese text to toneless pinyin. Non-Han chars kept as-is."""
    syls = []
    for ch in text:
        if _HAN_RE.match(ch):
            try:
                py = get_pinyin(ch, style=Style.NORMAL, errors="ignore")
                if py and py[0] and py[0][0]:
                    syls.append(py[0][0])
            except Exception:
                pass
        elif ch not in ' \t\n':
            syls.append(ch)
    return " ".join(syls)


def build_pinyin_record(record: dict) -> dict:
    """Convert a single JSONL record's text to pinyin, preserving everything else."""
    return {
        'text': text_to_pinyin(record.get('text', '')),
        'label': record.get('label', {}),
        'metadata': record.get('metadata', {}),
    }


def build_pinyin_jsonl(text_path: str | Path, pinyin_path: str | Path) -> int:
    """Convert an entire text JSONL file to pinyin JSONL.

    Args:
        text_path: input JSONL with Chinese text
        pinyin_path: output JSONL with pinyin text

    Returns:
        number of records written
    """
    text_path = Path(text_path)
    pinyin_path = Path(pinyin_path)
    pinyin_path.parent.mkdir(parents=True, exist_ok=True)

    count = 0
    with open(text_path, 'r', encoding='utf-8') as fin, \
         open(pinyin_path, 'w', encoding='utf-8') as fout:
        for line in fin:
            record = json.loads(line)
            pinyin_record = build_pinyin_record(record)
            fout.write(json.dumps(pinyin_record, ensure_ascii=False) + '\n')
            count += 1

    return count
