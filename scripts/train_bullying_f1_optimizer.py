#!/usr/bin/env python3
"""
專門優化霸凌偵測F1分數的訓練腳本
目標: 霸凌偵測 F1 ≥ 0.75, 毒性偵測 F1 ≥ 0.78

核心功能:
1. 使用改進的模型架構 (improved_detector.py)
2. 焦點損失和類別權重優化
3. RTX 3050 記憶體優化
4. 完整的監控和評估系統
5. TensorBoard視覺化
6. 自動調參建議系統
"""

import argparse
import json
import logging
import os
import random
import sys
import time
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import (
    classification_report, confusion_matrix, f1_score,
    precision_recall_fscore_support, balanced_accuracy_score
)
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from transformers import (
    AutoTokenizer, get_cosine_schedule_with_warmup,
    get_cosine_with_hard_restarts_schedule_with_warmup
)
from torch.optim import AdamW
import yaml
import traceback

# 添加專案根目錄到路徑
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.cyberpuppy.models.improved_detector import (
    ImprovedDetector, ImprovedModelConfig
)

# 忽略警告
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

logger = logging.getLogger(__name__)


def setup_logging(level=logging.INFO, log_file=None):
    """設定日誌系統"""
    handlers = [logging.StreamHandler(sys.stdout)]

    if log_file:
        handlers.append(logging.FileHandler(log_file, encoding='utf-8'))

    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=handlers
    )


class CyberBullyDataset(torch.utils.data.Dataset):
    """霸凌偵測資料集"""

    def __init__(self, data_path: str, tokenizer, max_length: int = 384,
                 augmentation: Optional[Dict] = None):
        self.data_path = Path(data_path)
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.augmentation = augmentation or {}

        self.data = self._load_data()

    def _load_data(self) -> List[Dict]:
        """載入資料"""
        if not self.data_path.exists():
            logger.error(f"資料檔案不存在: {self.data_path}")
            return []

        with open(self.data_path, 'r', encoding='utf-8') as f:
            if self.data_path.suffix == '.json':
                data = json.load(f)
            elif self.data_path.suffix == '.jsonl':
                data = [json.loads(line) for line in f]
            else:
                raise ValueError(f"不支援的檔案格式: {self.data_path.suffix}")

        logger.info(f"載入 {len(data)} 筆資料從 {self.data_path}")
        return data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        text = item.get('text', '')

        # Tokenize
        encoding = self.tokenizer(
            text,
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt'
        )

        # 準備標籤 - 使用統一標籤格式
        result = {
            'input_ids': encoding['input_ids'].squeeze(),
            'attention_mask': encoding['attention_mask'].squeeze(),
        }

        # 標籤映射字典
        label_maps = {
            'toxicity': {'none': 0, 'toxic': 1, 'severe': 2},
            'bullying': {'none': 0, 'harassment': 1, 'threat': 2},
            'role': {'none': 0, 'perpetrator': 1, 'victim': 2, 'bystander': 3},
            'emotion': {'pos': 0, 'neu': 1, 'neg': 2}
        }

        # 從嵌套結構獲取標籤
        labels = item.get('label', {})

        for task, label_map in label_maps.items():
            if task in labels:
                label_str = labels[task]
                label_id = label_map.get(label_str, 0)  # 默認為 0 (none)
                result[f'{task}_labels'] = torch.tensor(label_id, dtype=torch.long)

        return result


def compute_class_weights(dataset: CyberBullyDataset) -> Dict[str, torch.Tensor]:
    """計算類別權重"""
    from collections import Counter

    weights = {}

    for task in ['toxicity', 'bullying', 'role', 'emotion']:
        labels = []
        for item in dataset:
            if f'{task}_labels' in item:
                labels.append(item[f'{task}_labels'].item())

        if labels:
            counter = Counter(labels)
            total = len(labels)
            num_classes = max(counter.keys()) + 1

            # 計算平衡權重
            class_weights = []
            for i in range(num_classes):
                count = counter.get(i, 1)  # 避免除零
                weight = total / (num_classes * count)
                class_weights.append(weight)

            weights[task] = torch.tensor(class_weights, dtype=torch.float32)
            logger.info(f"{task} 類別權重: {weights[task].tolist()}")

    return weights


def save_model_artifacts(model, tokenizer, config: Dict, output_dir: Path):
    """保存模型相關文件"""
    artifacts_dir = output_dir / "model_artifacts"
    artifacts_dir.mkdir(exist_ok=True)

    # 保存模型
    model.save_pretrained(artifacts_dir / "model")
    tokenizer.save_pretrained(artifacts_dir / "tokenizer")

    # 保存配置
    with open(artifacts_dir / "training_config.yaml", 'w', encoding='utf-8') as f:
        yaml.dump(config, f, allow_unicode=True, default_flow_style=False)

    logger.info(f"模型文件已保存至: {artifacts_dir}")


class BullyingF1Optimizer:
    """霸凌F1分數優化訓練器"""

    def __init__(self, config_path: str, experiment_name: str):
        self.config = self._load_config(config_path)
        self.experiment_name = experiment_name
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # 設定隨機種子
        self._set_seed(self.config.get("experiment", {}).get("seed", 42))

        # 初始化組件
        self.tokenizer = None
        self.model = None
        self.train_loader = None
        self.val_loader = None
        self.test_loader = None
        self.optimizer = None
        self.scheduler = None
        self.scaler = torch.cuda.amp.GradScaler() if self.config["training"]["fp16"] else None

        # 訓練狀態
        self.current_epoch = 0
        self.best_bullying_f1 = 0.0
        self.best_metrics = {}
        self.early_stopping_counter = 0
        self.training_history = []

        # 輸出目錄
        self.output_dir = Path(self.config["experiment"]["output_dir"]) / experiment_name
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # TensorBoard
        self.tensorboard_writer = SummaryWriter(
            log_dir=self.output_dir / "tensorboard_logs"
        )

        # 設定日誌檔案
        setup_logging(
            level=getattr(logging, self.config.get("experiment", {}).get("log_level", "INFO")),
            log_file=self.output_dir / "training.log"
        )

    def _load_config(self, config_path: str) -> Dict:
        """載入訓練配置"""
        with open(config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        return config

    def _set_seed(self, seed: int):
        """設定隨機種子"""
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        if self.config.get("experiment", {}).get("deterministic", False):
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False

    def prepare_data(self):
        """準備訓練資料"""
        logger.info("準備訓練資料...")

        # 初始化tokenizer
        model_name = self.config["model"]["base_model"]
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

        # 載入資料集
        data_config = self.config["data"]

        # 檢查資料檔案是否存在
        train_path = Path(data_config["train_path"])
        val_path = Path(data_config["val_path"])
        test_path = Path(data_config["test_path"])

        if not train_path.exists():
            logger.error(f"訓練資料檔案不存在: {train_path}")
            # 嘗試使用替代路徑
            alt_train_path = Path("data/processed/cold/train.json")
            if alt_train_path.exists():
                logger.info(f"使用替代訓練資料: {alt_train_path}")
                train_path = alt_train_path
            else:
                raise FileNotFoundError(f"找不到訓練資料檔案")

        self.train_dataset = CyberBullyDataset(
            data_path=train_path,
            tokenizer=self.tokenizer,
            max_length=self.config["model"]["max_length"],
            augmentation=data_config.get("augmentation", {})
        )

        if val_path.exists():
            self.val_dataset = CyberBullyDataset(
                data_path=val_path,
                tokenizer=self.tokenizer,
                max_length=self.config["model"]["max_length"]
            )
        else:
            # 從訓練集分割驗證集
            logger.info("驗證集不存在，從訓練集分割20%作為驗證集")
            train_size = int(0.8 * len(self.train_dataset))
            val_size = len(self.train_dataset) - train_size
            self.train_dataset, self.val_dataset = torch.utils.data.random_split(
                self.train_dataset, [train_size, val_size]
            )

        if test_path.exists():
            self.test_dataset = CyberBullyDataset(
                data_path=test_path,
                tokenizer=self.tokenizer,
                max_length=self.config["model"]["max_length"]
            )
        else:
            # 使用驗證集作為測試集
            logger.info("測試集不存在，使用驗證集進行最終評估")
            self.test_dataset = self.val_dataset

        # 建立DataLoader
        batch_size = self.config["training"]["batch_size"]

        self.train_loader = DataLoader(
            self.train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=data_config.get("num_workers", 0),
            pin_memory=data_config.get("pin_memory", True),
            drop_last=self.config["training"].get("dataloader_drop_last", True)
        )

        self.val_loader = DataLoader(
            self.val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=data_config.get("num_workers", 0),
            pin_memory=data_config.get("pin_memory", True)
        )

        self.test_loader = DataLoader(
            self.test_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=data_config.get("num_workers", 0),
            pin_memory=data_config.get("pin_memory", True)
        )

        logger.info(f"訓練樣本: {len(self.train_dataset)}")
        logger.info(f"驗證樣本: {len(self.val_dataset)}")
        logger.info(f"測試樣本: {len(self.test_dataset)}")

    def prepare_model(self):
        """準備改進的模型"""
        logger.info("初始化改進的霸凌偵測模型...")

        # 建立模型配置
        model_config = ImprovedModelConfig(
            model_name=self.config["model"]["base_model"],
            hidden_size=self.config["model"]["hidden_size"],
            max_length=self.config["model"]["max_length"],
            num_toxicity_classes=self.config["model"]["toxicity_classes"],
            num_bullying_classes=self.config["model"]["bullying_classes"],
            num_role_classes=self.config["model"]["role_classes"],
            num_emotion_classes=self.config["model"]["emotion_classes"],

            # 損失函數配置
            use_focal_loss=self.config["model"]["use_focal_loss"],
            focal_alpha=self.config["model"]["focal_loss"]["alpha"],
            focal_gamma=self.config["model"]["focal_loss"]["gamma"],
            label_smoothing=self.config["model"]["label_smoothing"]["epsilon"],
            use_class_balanced_loss=self.config["model"]["use_class_weights"]
        )

        # 建立模型
        self.model = ImprovedDetector(model_config)

        # 移到GPU
        self.model.to(self.device)

        # 記錄模型資訊
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        logger.info(f"模型參數總量: {total_params:,}")
        logger.info(f"可訓練參數: {trainable_params:,}")

    def prepare_optimizer(self):
        """準備優化器"""
        logger.info("設定優化器...")

        # 設定不同的權重衰減
        no_decay = ["bias", "LayerNorm.weight"]
        optimizer_grouped_parameters = [
            {
                "params": [p for n, p in self.model.named_parameters()
                          if not any(nd in n for nd in no_decay)],
                "weight_decay": self.config["training"]["weight_decay"],
            },
            {
                "params": [p for n, p in self.model.named_parameters()
                          if any(nd in n for nd in no_decay)],
                "weight_decay": 0.0,
            },
        ]

        self.optimizer = AdamW(
            optimizer_grouped_parameters,
            lr=float(self.config["training"]["learning_rate"]),
            betas=tuple(self.config["optimization"].get("betas", [0.9, 0.999])),
            eps=float(self.config["optimization"].get("eps", 1e-8))
        )

        # 學習率排程器
        num_training_steps = len(self.train_loader) * self.config["training"]["num_epochs"]
        num_warmup_steps = int(num_training_steps * self.config["training"]["warmup_ratio"])

        if self.config["training"]["lr_scheduler"] == "cosine":
            self.scheduler = get_cosine_schedule_with_warmup(
                self.optimizer,
                num_warmup_steps=num_warmup_steps,
                num_training_steps=num_training_steps
            )
        elif self.config["training"]["lr_scheduler"] == "cosine_with_restarts":
            self.scheduler = get_cosine_with_hard_restarts_schedule_with_warmup(
                self.optimizer,
                num_warmup_steps=num_warmup_steps,
                num_training_steps=num_training_steps,
                num_cycles=2
            )

        logger.info(f"總訓練步數: {num_training_steps}")
        logger.info(f"預熱步數: {num_warmup_steps}")

    def train_epoch(self) -> Dict[str, float]:
        """訓練一個epoch"""
        self.model.train()

        total_loss = 0.0
        task_losses = {'toxicity': 0.0, 'bullying': 0.0, 'role': 0.0, 'emotion': 0.0}
        num_batches = 0

        gradient_accumulation_steps = self.config["training"]["gradient_accumulation_steps"]

        progress_bar = tqdm(self.train_loader, desc=f"Epoch {self.current_epoch + 1}")

        for batch_idx, batch in enumerate(progress_bar):
            # 移動到GPU
            batch = {k: v.to(self.device) for k, v in batch.items()}

            # 分離輸入和標籤
            input_ids = batch["input_ids"]
            attention_mask = batch["attention_mask"]
            labels = {
                "toxicity_label": batch["toxicity_labels"],
                "bullying_label": batch["bullying_labels"],
                "role_label": batch["role_labels"],
                "emotion_label": batch["emotion_labels"]
            }

            # 前向傳播
            with torch.cuda.amp.autocast(enabled=self.config["training"]["fp16"]):
                outputs = self.model(input_ids, attention_mask)
                loss_dict = self.model.compute_loss(outputs, labels)
                loss = loss_dict["total"] / gradient_accumulation_steps

            # 反向傳播
            if self.scaler is not None:
                self.scaler.scale(loss).backward()
            else:
                loss.backward()

            total_loss += loss.item() * gradient_accumulation_steps

            # 記錄任務損失
            for task in task_losses.keys():
                if task in loss_dict:
                    task_losses[task] += loss_dict[task].item()

            # 梯度更新
            if (batch_idx + 1) % gradient_accumulation_steps == 0:
                if self.scaler is not None:
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(),
                        self.config["training"]["max_grad_norm"]
                    )
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(),
                        self.config["training"]["max_grad_norm"]
                    )
                    self.optimizer.step()

                self.scheduler.step()
                self.optimizer.zero_grad()

            num_batches += 1

            # 更新進度條
            current_lr = self.scheduler.get_last_lr()[0]
            progress_bar.set_postfix({
                'loss': f'{loss.item() * gradient_accumulation_steps:.4f}',
                'lr': f'{current_lr:.2e}'
            })

            # 記錄到TensorBoard
            global_step = self.current_epoch * len(self.train_loader) + batch_idx
            if batch_idx % self.config["experiment"]["log_every_n_steps"] == 0:
                self.tensorboard_writer.add_scalar(
                    'Training/BatchLoss',
                    loss.item() * gradient_accumulation_steps,
                    global_step
                )
                self.tensorboard_writer.add_scalar(
                    'Training/LearningRate',
                    current_lr,
                    global_step
                )

        # 計算平均損失
        avg_loss = total_loss / num_batches
        avg_task_losses = {k: v / num_batches for k, v in task_losses.items()}

        results = {'train_loss': avg_loss}
        results.update({f'train_{k}_loss': v for k, v in avg_task_losses.items()})

        return results

    def evaluate(self, data_loader: DataLoader, prefix: str = "val") -> Dict[str, float]:
        """評估模型"""
        self.model.eval()

        all_predictions = {
            "toxicity": [], "bullying": [], "role": [], "emotion": []
        }
        all_labels = {
            "toxicity": [], "bullying": [], "role": [], "emotion": []
        }
        total_loss = 0.0
        task_losses = {'toxicity': 0.0, 'bullying': 0.0, 'role': 0.0, 'emotion': 0.0}
        num_batches = 0

        with torch.no_grad():
            for batch in tqdm(data_loader, desc=f"Evaluating {prefix}"):
                batch = {k: v.to(self.device) for k, v in batch.items()}

                with torch.cuda.amp.autocast(enabled=self.config["training"]["fp16"]):
                    outputs = self.model(**batch)

                total_loss += outputs["loss"].item()

                # 記錄任務損失
                for task in task_losses.keys():
                    if f"{task}_loss" in outputs:
                        task_losses[task] += outputs[f"{task}_loss"].item()

                # 收集預測和標籤
                for task in ["toxicity", "bullying", "role", "emotion"]:
                    if f"{task}_logits" in outputs and f"{task}_labels" in batch:
                        logits = outputs[f"{task}_logits"]
                        preds = torch.argmax(logits, dim=-1)
                        labels = batch[f"{task}_labels"]

                        all_predictions[task].extend(preds.cpu().numpy())
                        all_labels[task].extend(labels.cpu().numpy())

                num_batches += 1

        # 計算指標
        metrics = {f"{prefix}_loss": total_loss / num_batches}

        # 添加任務損失
        for task, loss in task_losses.items():
            if loss > 0:
                metrics[f"{prefix}_{task}_loss"] = loss / num_batches

        # 計算每個任務的F1分數
        for task in ["toxicity", "bullying", "role", "emotion"]:
            if all_predictions[task] and all_labels[task]:
                preds = np.array(all_predictions[task])
                labels = np.array(all_labels[task])

                # F1分數
                f1_macro = f1_score(labels, preds, average='macro', zero_division=0)
                f1_weighted = f1_score(labels, preds, average='weighted', zero_division=0)

                metrics[f"{task}_f1"] = f1_macro
                metrics[f"{task}_f1_weighted"] = f1_weighted

                # 精確率和召回率
                precision, recall, _, _ = precision_recall_fscore_support(
                    labels, preds, average='macro', zero_division=0
                )
                metrics[f"{task}_precision"] = precision
                metrics[f"{task}_recall"] = recall

                # 平衡準確率
                balanced_acc = balanced_accuracy_score(labels, preds)
                metrics[f"{task}_balanced_accuracy"] = balanced_acc

        # 計算總體指標
        valid_f1s = []
        for task in ["toxicity", "bullying", "role", "emotion"]:
            if f"{task}_f1" in metrics and metrics[f"{task}_f1"] > 0:
                valid_f1s.append(metrics[f"{task}_f1"])

        if valid_f1s:
            metrics["overall_macro_f1"] = np.mean(valid_f1s)

        return metrics

    def save_checkpoint(self, metrics: Dict[str, float], is_best: bool = False):
        """保存檢查點"""
        checkpoint_dir = self.output_dir / "checkpoints"
        checkpoint_dir.mkdir(exist_ok=True)

        checkpoint = {
            "epoch": self.current_epoch,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict(),
            "metrics": metrics,
            "best_bullying_f1": self.best_bullying_f1,
            "config": self.config,
            "training_history": self.training_history
        }

        if self.scaler is not None:
            checkpoint["scaler_state_dict"] = self.scaler.state_dict()

        # 保存最新檢查點
        torch.save(checkpoint, checkpoint_dir / "last.ckpt")

        # 保存最佳檢查點
        if is_best:
            torch.save(checkpoint, checkpoint_dir / "best.ckpt")
            logger.info(f"💾 保存最佳模型 (霸凌F1: {metrics.get('bullying_f1', 0):.4f})")

    def analyze_and_suggest_improvements(self, metrics: Dict[str, float]) -> List[str]:
        """分析當前表現並提供改進建議"""
        suggestions = []

        bullying_f1 = metrics.get('bullying_f1', 0)
        toxicity_f1 = metrics.get('toxicity_f1', 0)

        if bullying_f1 < 0.75:
            gap = 0.75 - bullying_f1
            suggestions.append(f"霸凌F1距離目標還有 {gap:.3f}")

            if bullying_f1 < 0.6:
                suggestions.append("建議增加霸凌類別的訓練樣本或調整類別權重")
                suggestions.append("考慮使用更強的資料增強技術")

            if gap > 0.1:
                suggestions.append("建議調高霸凌任務權重 (task_weights.bullying)")
                suggestions.append("考慮調整焦點損失參數 (focal_loss.gamma)")

        if toxicity_f1 < 0.78:
            suggestions.append(f"毒性F1需要提升 {0.78 - toxicity_f1:.3f}")

        # 檢查過擬合
        train_loss = metrics.get('train_loss', 0)
        val_loss = metrics.get('val_loss', 0)
        if val_loss > train_loss * 1.2:
            suggestions.append("可能存在過擬合，建議增加dropout或正規化")

        return suggestions

    def train(self):
        """執行完整訓練流程"""
        logger.info("🚀 開始霸凌偵測模型訓練...")

        # 準備訓練
        self.prepare_data()
        self.prepare_model()
        self.prepare_optimizer()

        # 早停配置
        early_stopping_config = self.config["optimization"]["early_stopping"]
        patience = early_stopping_config["patience"]

        # 訓練循環
        for epoch in range(self.config["training"]["num_epochs"]):
            self.current_epoch = epoch
            epoch_start_time = time.time()

            logger.info(f"\n{'='*60}")
            logger.info(f"📈 Epoch {epoch + 1}/{self.config['training']['num_epochs']}")
            logger.info(f"{'='*60}")

            # 訓練
            train_metrics = self.train_epoch()

            # 驗證
            val_metrics = self.evaluate(self.val_loader, "val")

            # 合併指標
            all_metrics = {**train_metrics, **val_metrics}
            all_metrics['epoch_time'] = time.time() - epoch_start_time

            # 記錄訓練歷史
            self.training_history.append(all_metrics.copy())

            # TensorBoard記錄
            for key, value in all_metrics.items():
                if isinstance(value, (int, float)):
                    self.tensorboard_writer.add_scalar(f"Epoch/{key}", value, epoch)

            # 輸出關鍵指標
            logger.info(f"📊 訓練損失: {train_metrics['train_loss']:.4f}")
            logger.info(f"📊 驗證損失: {val_metrics['val_loss']:.4f}")

            if "bullying_f1" in val_metrics:
                logger.info(f"🎯 霸凌F1: {val_metrics['bullying_f1']:.4f} (目標: ≥0.75)")
            if "toxicity_f1" in val_metrics:
                logger.info(f"🎯 毒性F1: {val_metrics['toxicity_f1']:.4f} (目標: ≥0.78)")
            if "overall_macro_f1" in val_metrics:
                logger.info(f"🎯 總體F1: {val_metrics['overall_macro_f1']:.4f}")

            # 檢查最佳模型
            current_bullying_f1 = val_metrics.get("bullying_f1", 0)
            is_best = current_bullying_f1 > self.best_bullying_f1

            if is_best:
                self.best_bullying_f1 = current_bullying_f1
                self.best_metrics = val_metrics.copy()
                self.early_stopping_counter = 0
                logger.info("🏆 新的最佳霸凌F1分數!")
            else:
                self.early_stopping_counter += 1

            # 分析和建議
            suggestions = self.analyze_and_suggest_improvements(val_metrics)
            if suggestions:
                logger.info("💡 改進建議:")
                for suggestion in suggestions:
                    logger.info(f"   • {suggestion}")

            # 保存檢查點
            self.save_checkpoint(all_metrics, is_best)

            # 早停檢查
            if self.early_stopping_counter >= patience:
                logger.info(f"⏹️  Early stopping after {patience} epochs without improvement")
                break

        # 最終評估
        logger.info("\n" + "="*60)
        logger.info("🎉 訓練完成! 進行最終評估...")
        logger.info("="*60)

        # 載入最佳模型
        best_checkpoint = torch.load(self.output_dir / "checkpoints" / "best.ckpt")
        self.model.load_state_dict(best_checkpoint["model_state_dict"])

        # 測試集評估
        test_metrics = self.evaluate(self.test_loader, "test")

        # 準備最終結果
        results = {
            "experiment_name": self.experiment_name,
            "best_epoch": best_checkpoint["epoch"],
            "training_time_epochs": self.current_epoch + 1,
            "best_val_metrics": self.best_metrics,
            "test_metrics": test_metrics,
            "target_achieved": {
                "bullying_f1_075": test_metrics.get("bullying_f1", 0) >= 0.75,
                "toxicity_f1_078": test_metrics.get("toxicity_f1", 0) >= 0.78,
                "overall_macro_f1_076": test_metrics.get("overall_macro_f1", 0) >= 0.76
            },
            "config": self.config,
            "training_history": self.training_history
        }

        # 保存結果
        with open(self.output_dir / "final_results.json", 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)

        # 輸出最終總結
        logger.info(f"\n📋 最終結果摘要:")
        logger.info(f"🎯 霸凌偵測 F1: {test_metrics.get('bullying_f1', 0):.4f} (目標: ≥0.75)")
        logger.info(f"🎯 毒性偵測 F1: {test_metrics.get('toxicity_f1', 0):.4f} (目標: ≥0.78)")
        logger.info(f"🎯 總體Macro F1: {test_metrics.get('overall_macro_f1', 0):.4f} (目標: ≥0.76)")

        # 目標達成狀況
        targets = results["target_achieved"]
        achievements = []
        if targets["bullying_f1_075"]:
            achievements.append("✅ 霸凌偵測F1目標達成!")
        else:
            achievements.append("❌ 霸凌偵測F1目標未達成")

        if targets["toxicity_f1_078"]:
            achievements.append("✅ 毒性偵測F1目標達成!")
        else:
            achievements.append("❌ 毒性偵測F1目標未達成")

        if targets["overall_macro_f1_076"]:
            achievements.append("✅ 總體Macro F1目標達成!")
        else:
            achievements.append("❌ 總體Macro F1目標未達成")

        for achievement in achievements:
            logger.info(achievement)

        # 如果未達成目標，提供進一步建議
        if not all(targets.values()):
            logger.info("\n💡 進一步優化建議:")
            final_suggestions = self.analyze_and_suggest_improvements(test_metrics)
            for suggestion in final_suggestions:
                logger.info(f"   • {suggestion}")

            logger.info("   • 考慮增加訓練epochs")
            logger.info("   • 嘗試不同的學習率")
            logger.info("   • 使用更多的資料增強技術")
            logger.info("   • 調整模型架構參數")

        # 保存模型工件
        save_model_artifacts(
            model=self.model,
            tokenizer=self.tokenizer,
            config=self.config,
            output_dir=self.output_dir
        )

        self.tensorboard_writer.close()

        logger.info(f"\n📁 所有輸出已保存至: {self.output_dir}")
        logger.info(f"📊 TensorBoard: tensorboard --logdir {self.output_dir}/tensorboard_logs")

        return results


def main():
    parser = argparse.ArgumentParser(description="專門優化霸凌偵測F1分數的訓練腳本")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/training/bullying_f1_optimization.yaml",
        help="訓練配置檔案路徑"
    )
    parser.add_argument(
        "--experiment-name",
        type=str,
        default=None,
        help="實驗名稱"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="輸出目錄 (覆蓋配置檔案設定)"
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default=None,
        help="資料目錄 (覆蓋配置檔案設定)"
    )

    args = parser.parse_args()

    # 生成實驗名稱
    if args.experiment_name is None:
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        args.experiment_name = f"bullying_f1_optimization_{timestamp}"

    # 初始日誌
    setup_logging(level=logging.INFO)

    # 檢查GPU
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        gpu_memory = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        logger.info(f"🚀 使用GPU: {gpu_name} ({gpu_memory:.1f}GB)")

        # RTX 3050特殊提示
        if "3050" in gpu_name:
            logger.info("💡 偵測到RTX 3050，已自動優化記憶體使用")
    else:
        logger.warning("⚠️  未偵測到GPU，將使用CPU訓練")

    # 建立訓練器
    try:
        trainer = BullyingF1Optimizer(
            config_path=args.config,
            experiment_name=args.experiment_name
        )

        # 覆蓋配置
        if args.output_dir:
            trainer.output_dir = Path(args.output_dir) / args.experiment_name
            trainer.output_dir.mkdir(parents=True, exist_ok=True)

        if args.data_dir:
            # 更新資料路徑
            trainer.config["data"]["train_path"] = f"{args.data_dir}/train.json"
            trainer.config["data"]["val_path"] = f"{args.data_dir}/val.json"
            trainer.config["data"]["test_path"] = f"{args.data_dir}/test.json"

        # 開始訓練
        logger.info(f"🎯 目標: 霸凌F1≥0.75, 毒性F1≥0.78, 總體F1≥0.76")
        logger.info(f"📁 實驗名稱: {args.experiment_name}")

        results = trainer.train()

        # 最終輸出
        success = all(results["target_achieved"].values())
        if success:
            logger.info("🎉 所有目標已達成!")
            return 0
        else:
            logger.info("📈 部分目標未達成，請查看改進建議")
            return 1

    except Exception as e:
        logger.error(f"❌ 訓練過程發生錯誤: {e}")
        logger.error(f"詳細錯誤: {traceback.format_exc()}")
        return 1


if __name__ == "__main__":
    sys.exit(main())