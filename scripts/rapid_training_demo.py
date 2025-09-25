#!/usr/bin/env python3
"""
快速訓練演示腳本
使用小樣本快速訓練多個模型變體並生成評估報告
"""

import json
import logging
import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Tuple, Any

import numpy as np
import torch
from sklearn.metrics import f1_score
from torch.utils.data import DataLoader, random_split

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    from src.cyberpuppy.models.baselines import (
        BaselineModel,
        ModelConfig,
        MultiTaskDataset,
        ModelEvaluator,
        create_model_variants
    )
    from src.cyberpuppy.labeling.label_map import (
        UnifiedLabel,
        ToxicityLevel,
        BullyingLevel,
        RoleType,
        EmotionType
    )
except ImportError as e:
    print(f"Import error: {e}")
    print("Please ensure the project structure is correct")
    sys.exit(1)

# 設定日誌
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def load_sample_data(
        data_path: str, max_samples: int = 1000
) -> List[Tuple[str, UnifiedLabel]]:
    """載入小樣本訓練資料"""
    logger.info(f"Loading sample data from {data_path}")

    data = []

    with open(data_path, 'r', encoding='utf-8') as f:
        raw_data = json.load(f)

    # 取樣本數據並確保平衡
    toxic_samples = []
    non_toxic_samples = []

    for item in raw_data:
        if 'text' in item and 'label' in item:
            text = item['text']
            label_dict = item['label']

            # 重建UnifiedLabel
            unified_label = UnifiedLabel(
                toxicity=ToxicityLevel(label_dict.get('toxicity', 'none')),
                bullying=BullyingLevel(label_dict.get('bullying', 'none')),
                role=RoleType(label_dict.get('role', 'none')),
                emotion=EmotionType(label_dict.get('emotion', 'neutral')),
                emotion_intensity=label_dict.get('emotion_strength', 2)
            )

            if unified_label.toxicity == ToxicityLevel.TOXIC:
                toxic_samples.append((text, unified_label))
            else:
                non_toxic_samples.append((text, unified_label))

    # 平衡採樣
    max_per_class = min(
        max_samples // 2, len(toxic_samples), len(non_toxic_samples)
    )
    data.extend(toxic_samples[:max_per_class])
    data.extend(non_toxic_samples[:max_per_class])

    # 隨機打亂
    np.random.shuffle(data)

    logger.info(
        f"Loaded {len(data)} balanced samples ({max_per_class} per class)"
    )
    return data


class RapidTrainer:
    """快速訓練器 - 用於演示"""

    def __init__(self, model: BaselineModel, device: str = 'cpu'):
        self.model = model.to(device)
        self.device = device
        self.evaluator = ModelEvaluator(self.model, device)

    def rapid_train(
            self, train_loader: DataLoader, val_loader: DataLoader,
            epochs: int = 2
    ) -> Dict[str, Any]:
        """快速訓練 - 最少epoch數"""
        logger.info(f"Starting rapid training for {epochs} epochs...")

        optimizer = torch.optim.AdamW(self.model.parameters(), lr=5e-5)

        train_losses = []
        val_losses = []

        for epoch in range(1, epochs + 1):
            # 訓練
            self.model.train()
            epoch_loss = 0.0
            num_batches = 0

            for batch in train_loader:
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)

                labels = {
                    'toxicity_label': batch['toxicity_label'].to(self.device),
                    'bullying_label': batch['bullying_label'].to(self.device),
                    'role_label': batch['role_label'].to(self.device),
                    'emotion_label': batch['emotion_label'].to(self.device)
                }

                optimizer.zero_grad()
                outputs = self.model(input_ids, attention_mask)
                losses = self.model.compute_loss(outputs, labels)
                loss = losses['total']

                loss.backward()
                optimizer.step()

                epoch_loss += loss.item()
                num_batches += 1

                if num_batches >= 10:  # 限制每個epoch的batch數
                    break

            avg_train_loss = epoch_loss / num_batches
            train_losses.append(avg_train_loss)

            # 驗證
            val_loss = self._validate(val_loader)
            val_losses.append(val_loss)

            logger.info(
                f"Epoch {epoch}: Train Loss: {avg_train_loss:.4f}, "
                f"Val Loss: {val_loss:.4f}"
            )

        return {
            'train_losses': train_losses,
            'val_losses': val_losses,
            'final_train_loss': train_losses[-1],
            'final_val_loss': val_losses[-1]
        }

    def _validate(self, val_loader: DataLoader) -> float:
        """快速驗證"""
        self.model.eval()
        total_loss = 0.0
        num_batches = 0

        with torch.no_grad():
            for batch in val_loader:
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)

                labels = {
                    'toxicity_label': batch['toxicity_label'].to(self.device),
                    'bullying_label': batch['bullying_label'].to(self.device),
                    'role_label': batch['role_label'].to(self.device),
                    'emotion_label': batch['emotion_label'].to(self.device)
                }

                outputs = self.model(input_ids, attention_mask)
                losses = self.model.compute_loss(outputs, labels)
                total_loss += losses['total'].item()
                num_batches += 1

                if num_batches >= 5:  # 限制驗證batch數
                    break

        return total_loss / num_batches if num_batches > 0 else float('inf')


def create_data_loaders(
        data: List[Tuple[str, UnifiedLabel]], tokenizer, batch_size: int = 16
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """建立快速訓練的資料載入器"""
    texts, labels = zip(*data)

    dataset = MultiTaskDataset(
        list(texts), list(labels), tokenizer, max_length=128
    )  # 短序列

    # 分割資料 70/20/10
    total_size = len(dataset)
    train_size = int(0.7 * total_size)
    val_size = int(0.2 * total_size)
    test_size = total_size - train_size - val_size

    train_dataset, val_dataset, test_dataset = random_split(
        dataset, [train_size, val_size, test_size]
    )

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, num_workers=0
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False, num_workers=0
    )
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False, num_workers=0
    )

    logger.info(f"Data split - Train: {train_size}, "
        "Val: {val_size}, Test: {test_size}")

    return train_loader, val_loader, test_loader


def evaluate_model(
    model: BaselineModel,
    test_loader: DataLoader,
    device: str
) -> Dict[str, Any]::
    """快速評估模型"""
    logger.info("Evaluating model...")

    model.eval()
    all_preds = {'toxicity': [], 'bullying': [], 'emotion': []}
    all_labels = {'toxicity': [], 'bullying': [], 'emotion': []}

    with torch.no_grad():
        for batch in test_loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)

            outputs = model(input_ids, attention_mask)

            # 毒性預測
            toxicity_preds = torch.argmax(
                outputs['toxicity'],
                dim=1).cpu().numpy(
            )
            toxicity_labels = batch['toxicity_label'].numpy()

            all_preds['toxicity'].extend(toxicity_preds)
            all_labels['toxicity'].extend(toxicity_labels)

            # 霸凌預測
            bullying_preds = torch.argmax(
                outputs['bullying'],
                dim=1).cpu().numpy(
            )
            bullying_labels = batch['bullying_label'].numpy()

            all_preds['bullying'].extend(bullying_preds)
            all_labels['bullying'].extend(bullying_labels)

            # 情緒預測
            emotion_preds = torch.argmax(
                outputs['emotion'],
                dim=1).cpu().numpy(
            )
            emotion_labels = batch['emotion_label'].numpy()

            all_preds['emotion'].extend(emotion_preds)
            all_labels['emotion'].extend(emotion_labels)

    # 計算F1分數
    results = {}
    for task in ['toxicity', 'bullying', 'emotion']:
        macro_f1 = f1_score(all_labels[task], all_preds[task], average='macro')
        micro_f1 = f1_score(all_labels[task], all_preds[task], average='micro')

        results[f'{task}_macro_f1'] = macro_f1
        results[f'{task}_micro_f1'] = micro_f1

        logger.info(
            f"{task.capitalize()} - Macro F1: {macro_f1:.4f}, "
            f"Micro F1: {micro_f1:.4f}"
        )

    return results


def train_model_variant(
        variant_name: str, config: ModelConfig,
        data: List[Tuple[str, UnifiedLabel]]
) -> Dict[str, Any]:
    """訓練單個模型變體"""
    logger.info(f"\n=== Training {variant_name} ===")

    start_time = time.time()

    # 建立模型
    try:
        model = BaselineModel(config)
        logger.info(f"Model created: {config.model_name}")
    except Exception as e:
        logger.error(f"Failed to create model {variant_name}: {e}")
        return {"status": "failed", "error": str(e)}

    # 建立資料載入器
    train_loader, val_loader, test_loader = create_data_loaders(
        data, model.tokenizer, batch_size=8
    )

    # 快速訓練
    trainer = RapidTrainer(model, device='cpu')
    train_results = trainer.rapid_train(train_loader, val_loader, epochs=2)

    # 評估
    eval_results = evaluate_model(model, test_loader, device='cpu')

    training_time = time.time() - start_time

    # 保存模型
    model_path = f"models/{variant_name}_demo"
    Path(model_path).mkdir(parents=True, exist_ok=True)
    model.save_model(model_path)

    return {
        "status": "completed",
        "variant": variant_name,
        "model_path": model_path,
        "training_time": training_time,
        "train_results": train_results,
        "eval_results": eval_results,
        "model_params": sum(p.numel() for p in model.parameters()),
        "model_config": {
            "model_name": config.model_name,
            "max_length": config.max_length,
            "num_labels": {
                "toxicity": config.num_toxicity_labels,
                "bullying": config.num_bullying_labels,
                "emotion": config.num_emotion_labels
            }
        }
    }


def main():
    """主函數 - 快速訓練演示"""
    logger.info("Starting Rapid Training Demo...")

    # 載入資料
    data_path = "data/processed/unified/train_unified.json"
    if not os.path.exists(data_path):
        logger.error(f"Data file not found: {data_path}")
        return

    data = load_sample_data(data_path, max_samples=800)  # 小樣本

    # 獲取模型變體
    model_variants = create_model_variants()

    # 選擇要訓練的模型（選擇較小的模型）
    selected_variants = ['macbert_base', 'toxicity_only']  # 只訓練兩個變體

    all_results = {}

    for variant_name in selected_variants:
        if variant_name not in model_variants:
            logger.warning(f"Model variant {variant_name} not found")
            continue

        config = model_variants[variant_name]

        try:
            results = train_model_variant(variant_name, config, data)
            all_results[variant_name] = results

            if results["status"] == "completed":
                logger.info(f"✓ {variant_name} completed in "
                    "{results['training_time']:.1f}s")
                eval_results = results["eval_results"]

                # 檢查DoD要求
                toxicity_f1 = eval_results.get('toxicity_macro_f1', 0)

                dod_status = "✓" if toxicity_f1 >= 0.78 else "✗"
                logger.info(
                    f"  DoD Check - Toxicity F1: {"
                        "toxicity_f1:.4f} {dod_status}"
                )

        except Exception as e:
            logger.error(f"Failed to train {variant_name}: {e}")
            all_results[variant_name] = {
                "status": "failed", "error": str(e)
            }

    # 生成評估報告
    generate_evaluation_report(all_results)

    logger.info("\n=== Rapid Training Demo Completed ===")


def generate_evaluation_report(results: Dict[str, Dict[str, Any]]):
    """生成評估報告"""
    logger.info("Generating evaluation report...")

    report = {
        "evaluation_timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "total_models_trained": len(results),
        "successf"
            "ul_models": sum(1 for r in results.values() if r.get(
        "models": results,
        "dod_compliance": {}
    }

    # DoD合規檢查
    for variant, result in results.items():
        if result.get("status") == "completed":
            eval_results = result.get("eval_results", {})
            toxicity_f1 = eval_results.get('toxicity_macro_f1', 0)
            emotion_f1 = eval_results.get('emotion_macro_f1', 0)

            report["dod_compliance"][variant] = {
                "toxicity_f1": toxicity_f1,
                "toxicity_meets_dod": toxicity_f1 >= 0.78,
                "emotion_f1": emotion_f1,
                "emotion_meets_dod": emotion_f1 >= 0.85,
                "overall_compliant": toxicity_f1 >= 0.78 and emotion_f1 >= 0.85
            }

    # 保存報告
    report_path = "models/evaluation_report.json"
    Path("models").mkdir(exist_ok=True)

    with open(report_path, 'w', encoding='utf-8') as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    logger.info(f"Evaluation report saved: {report_path}")

    # 打印摘要
    print("\n" + "="*60)
    print("CYBERPUPPY MODEL TRAINING EVALUATION REPORT")
    print("="*60)
    print(f"Timestamp: {report['evaluation_timestamp']}")
    print(f"Models Trained: {report['successful_mod"
        "els']}/{report['total_models_trained']}")
    print()

    for variant, compliance in report["dod_compliance"].items():
        print(f"📊 {variant.upper()}:")
        toxicity_status = '✓' if compliance['toxicity_meets_dod'] else '✗'
        print(
            f"   Toxicity F1:   {compliance['toxicity_f1']:.4f} "
            f"{toxicity_status} (DoD: ≥0.78)"
        )
        emotion_status = '✓' if compliance['emotion_meets_dod'] else '✗'
        print(
            f"   Emotion F1:    {compliance['emotion_f1']:.4f} "
            f"{emotion_status} (DoD: ≥0.85)"
        )
        print(f"   DoD Compliant: {'✓ YES' if complia"
            "nce['overall_compliant'] else '✗ NO'}")
        print()


if __name__ == "__main__":
    main()
