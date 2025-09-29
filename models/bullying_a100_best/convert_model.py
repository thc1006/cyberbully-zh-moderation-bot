#!/usr/bin/env python3
"""
轉換 .pt 模型到標準 HuggingFace 格式
"""
import json
import torch
import os
from transformers import AutoTokenizer

def convert_model():
    """轉換模型格式"""
    print("開始轉換模型格式...")

    # 載入原始模型
    model_data = torch.load('best_model.pt', map_location='cpu')
    print(f"載入 {len(model_data)} 個參數")

    # 檢查分類頭
    bullying_head_weight = model_data.get('bullying_head.weight')
    toxicity_head_weight = model_data.get('toxicity_head.weight')

    if bullying_head_weight is not None:
        num_bullying_labels = bullying_head_weight.shape[0]
        print(f"  - 霸凌分類: {num_bullying_labels} 類")

    if toxicity_head_weight is not None:
        num_toxicity_labels = toxicity_head_weight.shape[0]
        print(f"  - 毒性分類: {num_toxicity_labels} 類")

    # 創建 config.json
    config = {
        "architectures": ["BertForSequenceClassification"],
        "attention_probs_dropout_prob": 0.1,
        "classifier_dropout": None,
        "hidden_act": "gelu",
        "hidden_dropout_prob": 0.1,
        "hidden_size": 768,
        "initializer_range": 0.02,
        "intermediate_size": 3072,
        "layer_norm_eps": 1e-12,
        "max_position_embeddings": 512,
        "model_type": "bert",
        "num_attention_heads": 12,
        "num_hidden_layers": 12,
        "pad_token_id": 0,
        "position_embedding_type": "absolute",
        "transformers_version": "4.46.3",
        "type_vocab_size": 2,
        "use_cache": True,
        "vocab_size": 21128,
        "num_labels": max(num_bullying_labels, num_toxicity_labels) if 'num_bullying_labels' in locals() else 3,
        "id2label": {
            "0": "none",
            "1": "toxic/harassment",
            "2": "severe/threat"
        },
        "label2id": {
            "none": 0,
            "toxic/harassment": 1,
            "severe/threat": 2
        }
    }

    # 保存 config.json
    with open('config.json', 'w', encoding='utf-8') as f:
        json.dump(config, f, indent=2, ensure_ascii=False)
    print("創建 config.json")

    # 重新映射權重 (如果需要)
    # 對於我們的多任務模型，我們需要選擇一個主要任務
    # 假設使用 bullying_head 作為主要分類頭
    if 'bullying_head.weight' in model_data:
        model_data['classifier.weight'] = model_data['bullying_head.weight']
        model_data['classifier.bias'] = model_data['bullying_head.bias']
        print("使用 bullying_head 作為主分類器")

    # 保存為標準 PyTorch 格式
    torch.save(model_data, 'pytorch_model.bin')
    print("保存 pytorch_model.bin")

    # 檢查 tokenizer 檔案
    tokenizer_files = ['tokenizer_config.json', 'vocab.txt', 'tokenizer.json']
    for file in tokenizer_files:
        if os.path.exists(file):
            print(f"{file} 存在")
        else:
            print(f"{file} 不存在")

    print("\n轉換完成！")
    print("現在可以使用 HuggingFace 標準方式載入模型:")
    print("from transformers import AutoModelForSequenceClassification")
    print("model = AutoModelForSequenceClassification.from_pretrained('.')")

if __name__ == "__main__":
    convert_model()