#!/usr/bin/env python3
"""
GPU 加速訓練腳本
使用 NVIDIA RTX 3050 進行模型訓練
"""
import torch
import logging
from pathlib import Path
import sys

# 設定路徑
sys.path.insert(0, str(Path(__file__).parent / "src"))

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def check_gpu():
    """檢查 GPU 可用性"""
    print("="*60)
    print("GPU 配置檢查")
    print("="*60)

    if torch.cuda.is_available():
        print(f"[OK] CUDA Available")
        print(f"  CUDA 版本: {torch.version.cuda}")
        print(f"  PyTorch 版本: {torch.__version__}")
        print(f"  GPU 數量: {torch.cuda.device_count()}")
        print(f"  當前 GPU: {torch.cuda.get_device_name(0)}")
        print(f"  GPU 記憶體: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")

        # 測試 GPU 運算
        x = torch.randn(100, 100).cuda()
        y = torch.randn(100, 100).cuda()
        z = torch.matmul(x, y)
        print(f"[OK] GPU Computation Test Success")

        return "cuda"
    else:
        print("[ERROR] CUDA not available, will use CPU")
        return "cpu"

def train_with_gpu():
    """使用 GPU 訓練模型"""
    device = check_gpu()

    if device == "cuda":
        print("\n" + "="*60)
        print("開始 GPU 加速訓練")
        print("="*60)

        # 導入必要模組
        from transformers import AutoTokenizer, AutoModelForSequenceClassification
        from torch.utils.data import DataLoader, Dataset
        import torch.nn as nn
        from torch.optim import AdamW

        # 載入預訓練模型到 GPU
        model_name = "hfl/chinese-macbert-base"
        print(f"\n載入模型: {model_name}")

        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForSequenceClassification.from_pretrained(
            model_name,
            num_labels=3  # none, toxic, severe
        ).to(device)

        print(f"[OK] Model loaded to GPU")
        print(f"  模型參數: {sum(p.numel() for p in model.parameters())/1e6:.1f}M")

        # 準備訓練數據
        train_texts = [
            "你好朋友",
            "今天天氣真好",
            "謝謝你的幫助",
            "你這個笨蛋",
            "去死吧",
            "我討厭你"
        ]
        labels = [0, 0, 0, 1, 2, 1]  # 0=none, 1=toxic, 2=severe

        # 批次訓練範例
        print("\n開始訓練...")
        model.train()
        optimizer = AdamW(model.parameters(), lr=2e-5)
        loss_fn = nn.CrossEntropyLoss()

        for epoch in range(3):
            total_loss = 0
            for text, label in zip(train_texts, labels):
                # Tokenize
                inputs = tokenizer(
                    text,
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                    max_length=128
                ).to(device)

                # Forward pass
                outputs = model(**inputs)
                logits = outputs.logits

                # Calculate loss
                label_tensor = torch.tensor([label]).to(device)
                loss = loss_fn(logits, label_tensor)

                # Backward pass
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                total_loss += loss.item()

            print(f"Epoch {epoch+1}/3, Loss: {total_loss/len(train_texts):.4f}")

        print("\n[SUCCESS] GPU Training Complete!")

        # 保存模型
        save_dir = Path("models/gpu_trained_model")
        save_dir.mkdir(parents=True, exist_ok=True)

        model.save_pretrained(save_dir)
        tokenizer.save_pretrained(save_dir)

        print(f"[OK] Model saved to: {save_dir}")

        # 測試推理
        print("\n測試 GPU 推理...")
        model.eval()
        test_text = "你真是太棒了"

        with torch.no_grad():
            inputs = tokenizer(test_text, return_tensors="pt").to(device)
            outputs = model(**inputs)
            probs = torch.softmax(outputs.logits, dim=-1)
            pred = torch.argmax(probs, dim=-1)

            print(f"文本: {test_text}")
            print(f"預測: {['正常', '有毒', '嚴重'][pred.item()]}")
            print(f"機率: {probs[0].cpu().numpy()}")

        return True

    else:
        print("無法使用 GPU，請檢查 CUDA 安裝")
        return False

if __name__ == "__main__":
    success = train_with_gpu()
    if success:
        print("\n" + "="*60)
        print("[SUCCESS] GPU Accelerated Training Completed Successfully!")
        print("="*60)