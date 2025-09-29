"""
驗證所有可用模型的實際性能
測試真實推論結果並與聲稱的性能比較
"""

import json
import logging
from pathlib import Path
from typing import Dict, List, Any

import numpy as np
import torch
from sklearn.metrics import classification_report, f1_score
from transformers import AutoModelForSequenceClassification, AutoTokenizer

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)


def load_test_data(data_path: str) -> tuple:
    """載入測試資料"""
    with open(data_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    texts = [item['text'] for item in data]

    # 根據模型類型選擇標籤
    # 霸凌模型: none=0, harassment=1, threat=2
    # 毒性模型: none=0, toxic=1, severe=2
    label_map = {"none": 0, "harassment": 1, "threat": 2, "toxic": 1, "severe": 2}

    labels = []
    for item in data:
        label_value = item['label'].get('bullying', item['label'].get('toxicity', 'none'))
        labels.append(label_map.get(label_value, 0))

    return texts, labels


def verify_model(model_path: str, test_data_path: str, device: str = 'cpu') -> Dict[str, Any]:
    """驗證單個模型的性能"""
    logger.info(f"\n{'='*80}")
    logger.info(f"驗證模型: {model_path}")
    logger.info(f"{'='*80}")

    model_path = Path(model_path)

    # 檢查模型檔案是否存在
    has_safetensors = (model_path / "model.safetensors").exists()
    has_bin = (model_path / "pytorch_model.bin").exists()
    has_config = (model_path / "config.json").exists()

    logger.info(f"檔案檢查:")
    logger.info(f"  - config.json: {'✅' if has_config else '❌'}")
    logger.info(f"  - model.safetensors: {'✅' if has_safetensors else '❌'}")
    logger.info(f"  - pytorch_model.bin: {'✅' if has_bin else '❌'}")

    if not has_config or (not has_safetensors and not has_bin):
        logger.error("❌ 缺少必要的模型檔案")
        return {"error": "Missing model files", "model_path": str(model_path)}

    try:
        # 載入模型和 tokenizer
        logger.info("載入模型...")
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        model = AutoModelForSequenceClassification.from_pretrained(model_path)
        model.to(device)
        model.eval()

        # 檢查模型配置
        num_labels = model.config.num_labels
        logger.info(f"模型配置: {num_labels} 個類別")

        # 載入測試資料
        logger.info("載入測試資料...")
        texts, true_labels = load_test_data(test_data_path)
        logger.info(f"測試樣本數: {len(texts)}")

        # 批次推論
        logger.info("進行推論...")
        predictions = []
        batch_size = 32

        with torch.no_grad():
            for i in range(0, len(texts), batch_size):
                batch_texts = texts[i:i+batch_size]

                inputs = tokenizer(
                    batch_texts,
                    padding=True,
                    truncation=True,
                    max_length=512,
                    return_tensors="pt"
                )
                inputs = {k: v.to(device) for k, v in inputs.items()}

                outputs = model(**inputs)
                logits = outputs.logits
                preds = torch.argmax(logits, dim=-1).cpu().numpy()
                predictions.extend(preds)

        # 計算指標
        logger.info("計算評估指標...")

        # 確保標籤範圍正確
        true_labels = np.array(true_labels)
        predictions = np.array(predictions)

        # 處理標籤映射（如果模型是3類但標籤只有0/1）
        if num_labels == 3:
            # 對於毒性模型: none=0, toxic=1, severe=2
            # 但測試資料可能是二元的，需要處理
            pass

        # 計算 F1 分數
        # 明確指定 labels 參數以處理類別不匹配的情況
        all_labels = list(range(num_labels))

        f1_macro = f1_score(true_labels, predictions, labels=all_labels, average='macro', zero_division=0)
        f1_weighted = f1_score(true_labels, predictions, labels=all_labels, average='weighted', zero_division=0)

        # 詳細分類報告
        target_names = [f"Class_{i}" for i in range(num_labels)]
        report = classification_report(
            true_labels,
            predictions,
            labels=all_labels,
            target_names=target_names,
            output_dict=True,
            zero_division=0
        )

        logger.info(f"\n📊 評估結果:")
        logger.info(f"  - Macro F1: {f1_macro:.4f}")
        logger.info(f"  - Weighted F1: {f1_weighted:.4f}")
        logger.info(f"  - Accuracy: {report['accuracy']:.4f}")

        logger.info(f"\n各類別 F1 分數:")
        for i in range(num_labels):
            class_name = target_names[i]
            class_f1 = report[class_name]['f1-score']
            class_support = report[class_name]['support']
            logger.info(f"  - {class_name}: {class_f1:.4f} (n={int(class_support)})")

        return {
            "model_path": str(model_path),
            "num_labels": num_labels,
            "f1_macro": f1_macro,
            "f1_weighted": f1_weighted,
            "accuracy": report['accuracy'],
            "classification_report": report,
            "test_samples": len(texts),
            "success": True
        }

    except Exception as e:
        logger.error(f"❌ 驗證失敗: {e}")
        import traceback
        traceback.print_exc()
        return {"error": str(e), "model_path": str(model_path), "success": False}


def main():
    """主函數"""
    logger.info("🔍 開始驗證所有可用模型的性能")
    logger.info("="*80)

    # 測試資料路徑
    test_data_path = "data/processed/training_dataset/test.json"

    # 檢查測試資料是否存在
    if not Path(test_data_path).exists():
        logger.error(f"❌ 測試資料不存在: {test_data_path}")
        return

    # 設定裝置
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    logger.info(f"使用裝置: {device}")

    # 要測試的模型列表
    models_to_test = [
        "models/working_toxicity_model",
        "models/bullying_a100_best",  # 生產級模型 (F1=0.826)
    ]

    results = []

    for model_path in models_to_test:
        if Path(model_path).exists():
            result = verify_model(model_path, test_data_path, device)
            results.append(result)
        else:
            logger.warning(f"⚠️ 模型目錄不存在: {model_path}")

    # 生成總結報告
    logger.info(f"\n{'='*80}")
    logger.info("📊 驗證總結")
    logger.info(f"{'='*80}")

    for result in results:
        if result.get('success'):
            logger.info(f"\n✅ {result['model_path']}")
            logger.info(f"  - Macro F1: {result['f1_macro']:.4f}")
            logger.info(f"  - Weighted F1: {result['f1_weighted']:.4f}")
            logger.info(f"  - Accuracy: {result['accuracy']:.4f}")
            logger.info(f"  - 類別數: {result['num_labels']}")
        else:
            logger.info(f"\n❌ {result['model_path']}")
            logger.info(f"  - 錯誤: {result.get('error', 'Unknown error')}")

    # 保存結果
    output_file = "models/performance_verification_results.json"
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    logger.info(f"\n✅ 結果已保存到: {output_file}")


if __name__ == "__main__":
    main()