#!/usr/bin/env python3
"""
Quick Inference Script
快速推論腳本

A simple script to quickly test toxicity detection on Chinese text.
"""

import sys
import torch
from pathlib import Path
from transformers import AutoTokenizer, AutoModel

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

def quick_toxicity_check(text: str):
    """Quick toxicity check for Chinese text"""

    print(f"[ANALYZE] {text}")

    # Load model and tokenizer
    model_name = "hfl/chinese-macbert-base"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)

    # Tokenize
    inputs = tokenizer(text, return_tensors="pt", max_length=256, truncation=True, padding=True)

    # Get embeddings
    with torch.no_grad():
        outputs = model(**inputs)
        cls_embedding = outputs.last_hidden_state[:, 0, :]  # [CLS] token

    # Simple heuristic analysis (placeholder for actual model)
    embedding_stats = {
        'mean': cls_embedding.mean().item(),
        'std': cls_embedding.std().item(),
        'max': cls_embedding.max().item(),
        'min': cls_embedding.min().item(),
    }

    print(f"[STATS] {embedding_stats}")

    # Placeholder analysis
    if "白痴" in text or "笨" in text or "垃圾" in text:
        risk_level = "HIGH"
        toxicity = "toxic"
    elif "差" in text or "失望" in text:
        risk_level = "MEDIUM"
        toxicity = "mild"
    else:
        risk_level = "LOW"
        toxicity = "none"

    print(f"[RISK] {risk_level}")
    print(f"[TOXICITY] {toxicity}")
    print(f"[TOKENS] {len(inputs['input_ids'][0])} tokens")

    return {
        'text': text,
        'risk_level': risk_level,
        'toxicity': toxicity,
        'embedding_stats': embedding_stats,
        'token_count': len(inputs['input_ids'][0])
    }

def main():
    """Main function for quick testing"""

    test_texts = [
        "你好，這是一個正常的句子。",
        "今天天氣很好。",
        "這個產品質量很差。",
        "你是白痴嗎？",
        "完全是垃圾！",
    ]

    print("="*50)
    print("CYBERPUPPY QUICK TOXICITY CHECK")
    print("="*50)

    results = []

    for i, text in enumerate(test_texts, 1):
        print(f"\n[{i}/{len(test_texts)}] Testing:")
        result = quick_toxicity_check(text)
        results.append(result)
        print("-" * 30)

    # Summary
    print("\n" + "="*50)
    print("SUMMARY")
    print("="*50)

    high_risk = [r for r in results if r['risk_level'] == 'HIGH']
    medium_risk = [r for r in results if r['risk_level'] == 'MEDIUM']
    low_risk = [r for r in results if r['risk_level'] == 'LOW']

    print(f"[HIGH] {len(high_risk)} texts")
    print(f"[MEDIUM] {len(medium_risk)} texts")
    print(f"[LOW] {len(low_risk)} texts")

    if high_risk:
        print(f"\n[WARNING] High risk texts detected:")
        for r in high_risk:
            print(f"   - {r['text']}")

    print(f"\n[DONE] Analysis complete!")

if __name__ == "__main__":
    main()