"""Real inference smoke test against bullying_a100_best."""
import json
import sys
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoTokenizer, BertModel

MODEL_DIR = Path("models/bullying_a100_best")
LABELS = ["none", "toxic/harassment", "severe/threat"]


class MultiTaskBert(nn.Module):
    def __init__(self, base_name="hfl/chinese-macbert-base"):
        super().__init__()
        self.bert = BertModel.from_pretrained(base_name)
        self.bullying_head = nn.Linear(768, 3)
        self.toxicity_head = nn.Linear(768, 3)

    def forward(self, input_ids, attention_mask):
        out = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        pooled = out.pooler_output
        return {
            "bullying": self.bullying_head(pooled),
            "toxicity": self.toxicity_head(pooled),
        }


def load_model():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_bf16 = device.type == "cuda" and torch.cuda.is_bf16_supported()
    dtype = torch.bfloat16 if use_bf16 else torch.float32
    print(f"Device: {device}  dtype: {dtype}")

    tokenizer = AutoTokenizer.from_pretrained(str(MODEL_DIR))
    model = MultiTaskBert()
    state = torch.load(MODEL_DIR / "best_model.pt", map_location="cpu", weights_only=False)
    missing, unexpected = model.load_state_dict(state, strict=False)
    print(f"Loaded weights. Missing: {len(missing)}  Unexpected: {len(unexpected)}")
    if missing:
        print("  first missing:", missing[:5])
    if unexpected:
        print("  first unexpected:", unexpected[:5])
    model.eval().to(device=device, dtype=dtype)
    return tokenizer, model, device


def predict(tokenizer, model, device, texts):
    enc = tokenizer(
        texts, padding=True, truncation=True, max_length=256, return_tensors="pt"
    ).to(device)
    with torch.inference_mode():
        logits = model(enc["input_ids"], enc["attention_mask"])
    bull = F.softmax(logits["bullying"].float(), dim=-1).cpu()
    tox = F.softmax(logits["toxicity"].float(), dim=-1).cpu()
    return bull.numpy(), tox.numpy()


def fmt(probs):
    return ", ".join(f"{LABELS[i]}={p:.3f}" for i, p in enumerate(probs))


def main():
    tokenizer, model, device = load_model()
    print(f"Tokenizer: {tokenizer.__class__.__name__}; vocab={tokenizer.vocab_size}")

    samples = [
        "今天天氣真好，我們去公園玩吧！",
        "你這個笨蛋，滾開！",
        "我恨你，希望你去死，沒人喜歡你",
        "謝謝你的幫忙，你真的是個很棒的人",
        "再嘴一句試試看，我打死你",
        "我覺得這部電影還不錯，演員演技也很到位",
    ]
    bull, tox = predict(tokenizer, model, device, samples)
    print("\n=== 即時推論結果 ===")
    for s, b, t in zip(samples, bull, tox):
        print(f"\nText: {s}")
        print(f"  Bullying : {fmt(b)}  -> pred={LABELS[b.argmax()]}")
        print(f"  Toxicity : {fmt(t)}  -> pred={LABELS[t.argmax()]}")


if __name__ == "__main__":
    main()
