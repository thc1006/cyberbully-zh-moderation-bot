"""LLM cascade for confidence-gated toxicity classification.

When the dual-LoRA ensemble is uncertain (confidence < threshold),
falls back to Qwen3-8B Instruct with few-shot chat template.

Usage:
    cascade = LLMCascade("Qwen/Qwen3-8B")
    results = cascade.classify(["密碼4了", "大廈避風了"])
    # {0: 1, 1: 1}  (both toxic)
"""
from __future__ import annotations

from typing import Dict, List, Optional

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

SYSTEM_PROMPT = """你是一位中文網路安全專家，專門辨識各種隱藏式有毒內容。用戶會使用以下手法：
- 諧音字：「勾史」=「狗屎」，「四調」=「死掉」
- 數字代替：「4了」=「死了」，「13」=「B（逼）」
- 字母代替：「装X」=「装逼」，「G8」=「雞巴」，「NMSL」=「你媽死了」
- 詞級暗語：「密碼」=「你媽」，「春竹」=「蠢豬」
- 英文諧音：「funny mud pee」=「放你媽屁」
- 句級暗語：「大廈避風」=「大傻逼」，「採九蓮」=「操你臉」

你必須只回答「有毒」或「安全」兩個字，不要解釋。"""

FEW_SHOT = [
    ("密碼4了", "有毒"), ("装X", "有毒"),
    ("今天天氣真好", "安全"), ("勾史", "有毒"),
    ("你真太6了", "安全"), ("這大廈避風了", "有毒"),
    ("四調的黑人是好人", "有毒"), ("我對你一見粽情", "安全"),
]


class LLMCascade:
    """Qwen3-8B Instruct cascade for uncertain samples."""

    def __init__(
        self,
        model_name: str = "Qwen/Qwen3-8B",
        device: Optional[torch.device] = None,
        dtype: torch.dtype = torch.bfloat16,
    ):
        self.device = device or torch.device("cuda")
        self.tok = AutoTokenizer.from_pretrained(model_name)
        if self.tok.pad_token_id is None:
            self.tok.pad_token_id = self.tok.eos_token_id
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name, dtype=dtype, device_map=self.device,
            low_cpu_mem_usage=True,
        )
        self.model.eval()

    def _build_prompt(self, text: str) -> str:
        messages = [{"role": "system", "content": SYSTEM_PROMPT}]
        for ex_text, ex_label in FEW_SHOT:
            messages.append({"role": "user", "content": f"「{ex_text}」"})
            messages.append({"role": "assistant", "content": ex_label})
        messages.append({"role": "user", "content": f"「{text}」"})
        return self.tok.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True,
            enable_thinking=False,
        )

    @torch.inference_mode()
    def classify_one(self, text: str) -> int:
        """Classify a single text. Returns 1=toxic, 0=safe, -1=unparseable."""
        prompt = self._build_prompt(text)
        inputs = self.tok(prompt, return_tensors="pt").to(self.device)
        out = self.model.generate(
            **inputs, max_new_tokens=10, do_sample=False,
            pad_token_id=self.tok.eos_token_id,
        )
        gen = self.tok.decode(
            out[0, inputs["input_ids"].shape[-1]:], skip_special_tokens=True
        ).strip()
        if "有毒" in gen:
            return 1
        elif "安全" in gen:
            return 0
        return -1

    def classify(self, texts: List[str]) -> Dict[int, int]:
        """Classify multiple texts. Returns {index: prediction}."""
        results = {}
        for i, text in enumerate(texts):
            results[i] = self.classify_one(text)
        return results

    def free(self):
        """Free GPU memory."""
        del self.model, self.tok
        torch.cuda.empty_cache()
