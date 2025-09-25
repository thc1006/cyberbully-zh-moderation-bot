#!/usr/bin/env python3
"""
Minimal Model Inference Example
"""

import torch
from pathlib import Path
import sys

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / "src"))

from cyberpuppy.models.baselines import BaselineModel

def main():
    # Load model
    model_path = "models/toxicity_only_demo"
    model = BaselineModel.load_model(model_path)

    # Test text
    text = "這是一個測試句子。"

    # Tokenize
    inputs = model.tokenizer(
        text,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=256
    )

    # Inference
    model.eval()
    with torch.no_grad():
        predictions = model.predict(
            inputs["input_ids"],
            inputs["attention_mask"],
            inputs.get("token_type_ids")
        )

    # Print results
    print(f"Input text: {text}")
    print("Predictions:")
    for key, value in predictions.items():
        print(f"  {key}: {value}")

if __name__ == "__main__":
    main()
