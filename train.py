#!/usr/bin/env python3
"""
模型訓練腳本
支援單任務/多任務訓練、early stopping、混合精度訓練
"""

import argparse
import json
import logging
import random
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.cuda.amp import GradScaler, autocast
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader, random_split
from torch.utils.tensorboard import SummaryWriter
from transformers import get_linear_schedule_with_warmup
from tqdm import tqdm

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

# 設定警告
warnings.filterwarnings('ignore')

# 設定日誌
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class EarlyStopping:
    """Early Stopping 機制"""

    def __init__(self,
                 patience: int = 7,
                 min_delta: float = 0.001,
                 restore_best_weights: bool = True):
        self.patience = patience
        self.min_delta = min_delta
        self.restore_best_weights = restore_best_weights

        self.best_loss = float('inf')
        self.counter = 0
        self.best_weights = None

    def __call__(self, val_loss: float, model: nn.Module) -> bool:
        """
        檢查是否應該停止訓練

        Returns:
            bool: True表示應該停止訓練
        """
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
            if self.restore_best_weights:
                self.best_weights = model.state_dict().copy()
        else:
            self.counter += 1

        if self.counter >= self.patience:
            if self.restore_best_weights and self.best_weights:
                model.load_state_dict(self.best_weights)
                logger.info(
                    f"Early stopping triggered. Restored best weights "
                    f"(loss: {self.best_loss:.4f})"
                )
            return True

        return False


class Trainer:
    """模型訓練器"""

    def __init__(self,
                 model: BaselineModel,
                 train_dataloader: DataLoader,
                 val_dataloader: DataLoader,
                 test_dataloader: Optional[DataLoader] = None,
                 config: Dict = None):
        self.model = model
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
        self.test_dataloader = test_dataloader

        # 訓練配置
        self.config = config or {}
        self.epochs = self.config.get('epochs', 10)
        self.learning_rate = self.config.get('learning_rate', 2e-5)
        self.weight_decay = self.config.get('weight_decay', 0.01)
        self.warmup_steps = self.config.get('warmup_steps', 0.1)
        self.use_amp = self.config.get('use_amp', True)
        self.gradient_clip = self.config.get('gradient_clip', 1.0)
        self.save_path = self.config.get('save_path', 'models/baseline')
        self.log_dir = self.config.get('log_dir', 'logs/baseline')

        # 設備設定
        self.device = torch.device(
            'cuda' if torch.cuda.is_available() else 'cpu'
        )
        self.model.to(self.device)

        # 優化器
        self.optimizer = self._create_optimizer()

        # 學習率調度器
        total_steps = len(train_dataloader) * self.epochs
        warmup_steps = (
            int(total_steps * self.warmup_steps)
            if self.warmup_steps < 1.0
            else int(self.warmup_steps)
        )

        self.scheduler = get_linear_schedule_with_warmup(
            self.optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=total_steps
        )

        self.lr_scheduler = ReduceLROnPlateau(
            self.optimizer,
            mode='min',
            factor=0.5,
            patience=3,
            # verbose=True
        )

        # Early Stopping
        early_stopping_config = self.config.get('early_stopping', {})
        self.early_stopping = EarlyStopping(
            patience=early_stopping_config.get('patience', 5),
            min_delta=early_stopping_config.get('min_delta', 0.001)
        )

        # 混合精度
        self.scaler = GradScaler() if self.use_amp else None

        # TensorBoard
        self.writer = SummaryWriter(self.log_dir)

        # 評估器
        self.evaluator = ModelEvaluator(self.model, self.device)

        logger.info(f"Trainer initialized on device: {self.device}")
        total_params = sum(p.numel() for p in self.model.parameters())
        logger.info(f"Model parameters: {total_params:,}")
        trainable_params = sum(
            p.numel() for p in self.model.parameters() if p.requires_grad
        )
        logger.info(f"Trainable parameters: {trainable_params:,}")

    def _create_optimizer(self) -> AdamW:
        """建立優化器"""
        # 區分不同層的學習率
        no_decay = ["bias", "LayerNorm.weight"]
        optimizer_grouped_parameters = [
            {
                "params": [
                    p for n, p in self.model.named_parameters()
                    if not any(nd in n for nd in no_decay)
                ],
                "weight_decay": self.weight_decay,
            },
            {
                "params": [
                    p for n, p in self.model.named_parameters()
                    if any(nd in n for nd in no_decay)
                ],
                "weight_decay": 0.0,
            },
        ]

        return AdamW(
            optimizer_grouped_parameters,
            lr=self.learning_rate,
            eps=1e-8
        )

    def train_epoch(self, epoch: int) -> Dict[str, float]:
        """訓練一個epoch"""
        self.model.train()
        total_loss = 0.0
        task_losses = {
            'toxicity': 0.0, 'bullying': 0.0, 'role': 0.0,
            'emotion': 0.0, 'emotion_intensity': 0.0
        }
        num_batches = len(self.train_dataloader)

        progress_bar = tqdm(self.train_dataloader, desc=f"Epoch {epoch}")

        for batch_idx, batch in enumerate(progress_bar):
            # 移動資料到設備
            input_ids = batch['input_ids'].to(self.device)
            attention_mask = batch['attention_mask'].to(self.device)
            token_type_ids = batch.get('token_type_ids', None)
            if token_type_ids is not None:
                token_type_ids = token_type_ids.to(self.device)

            # 標籤
            labels = {
                'toxicity_label': batch['toxicity_label'].to(self.device),
                'bullying_label': batch['bullying_label'].to(self.device),
                'role_label': batch['role_label'].to(self.device),
                'emotion_label': batch['emotion_label'].to(self.device)
            }

            if 'emotion_intensity' in batch:
                labels['emotion_intensity'] = batch['emotion_intensity'].to(
                    self.device
                )

            # 前向傳播
            if self.use_amp and self.scaler:
                with autocast():
                    outputs = self.model(
                        input_ids, attention_mask, token_type_ids
                    )
                    losses = self.model.compute_loss(outputs, labels)
            else:
                outputs = self.model(
                    input_ids, attention_mask, token_type_ids
                )
                losses = self.model.compute_loss(outputs, labels)

            loss = losses['total']

            # 反向傳播
            self.optimizer.zero_grad()

            if self.use_amp and self.scaler:
                self.scaler.scale(loss).backward()
                if self.gradient_clip > 0:
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(),
                        self.gradient_clip
                    )
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                loss.backward()
                if self.gradient_clip > 0:
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(),
                        self.gradient_clip
                    )
                self.optimizer.step()

            self.scheduler.step()

            # 累計損失
            total_loss += loss.item()
            for task, task_loss in losses.items():
                if task != 'total' and task in task_losses:
                    task_losses[task] += task_loss.item()

            # 更新進度條
            current_lr = self.optimizer.param_groups[0]['lr']
            progress_bar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'avg_loss': f'{total_loss/(batch_idx+1):.4f}',
                'lr': f'{current_lr:.2e}'
            })

            # 記錄到TensorBoard
            global_step = epoch * num_batches + batch_idx
            self.writer.add_scalar('train/loss', loss.item(), global_step)
            self.writer.add_scalar(
                'train/learning_rate', current_lr, global_step
            )

            for task, task_loss in losses.items():
                if task != 'total':
                    self.writer.add_scalar(
                        f'train/{task}_loss', task_loss.item(), global_step
                    )

        # 計算平均損失
        avg_total_loss = total_loss / num_batches
        avg_task_losses = {
            task: task_loss / num_batches
            for task, task_loss in task_losses.items()
        }

        return {'total': avg_total_loss, **avg_task_losses}

    def validate_epoch(self, epoch: int) -> Dict[str, float]:
        """驗證一個epoch"""
        self.model.eval()
        total_loss = 0.0
        task_losses = {
            'toxicity': 0.0, 'bullying': 0.0, 'role': 0.0,
            'emotion': 0.0, 'emotion_intensity': 0.0
        }
        num_batches = len(self.val_dataloader)

        with torch.no_grad():
            for batch in tqdm(self.val_dataloader, desc="Validation"):
                # 移動資料到設備
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                token_type_ids = batch.get('token_type_ids', None)
                if token_type_ids is not None:
                    token_type_ids = token_type_ids.to(self.device)

                # 標籤
                labels = {
                    'toxicity_label': batch['toxicity_label'].to(self.device),
                    'bullying_label': batch['bullying_label'].to(self.device),
                    'role_label': batch['role_label'].to(self.device),
                    'emotion_label': batch['emotion_label'].to(self.device)
                }

                if 'emotion_intensity' in batch:
                    labels['emotion_intensity'] = batch['emotion_intensity'].to(\n                        self.device\n                    )

                # 前向傳播
                if self.use_amp:
                    with autocast():
                        outputs = self.model(
                            input_ids, attention_mask, token_type_ids
                        )
                        losses = self.model.compute_loss(outputs, labels)
                else:
                    outputs = self.model(
                        input_ids, attention_mask, token_type_ids
                    )
                    losses = self.model.compute_loss(outputs, labels)

                # 累計損失
                total_loss += losses['total'].item()
                for task, task_loss in losses.items():
                    if task != 'total' and task in task_losses:
                        task_losses[task] += task_loss.item()

        # 計算平均損失
        avg_total_loss = total_loss / num_batches
        avg_task_losses = {
            task: task_loss / num_batches
            for task, task_loss in task_losses.items()
        }

        # 更新學習率調度器
        self.lr_scheduler.step(avg_total_loss)

        # 記錄到TensorBoard
        self.writer.add_scalar('val/loss', avg_total_loss, epoch)
        for task, task_loss in avg_task_losses.items():
            self.writer.add_scalar(f'val/{task}_loss', task_loss, epoch)

        # 計算評估指標
        metrics = self.evaluator.evaluate(self.val_dataloader)
        for metric_name, metric_value in metrics.items():
            self.writer.add_scalar(f'val/{metric_name}', metric_value, epoch)

        return {'total': avg_total_loss, **avg_task_losses, **metrics}

    def train(self) -> Dict[str, List[float]]:
        """完整訓練循環"""
        logger.info("Starting training...")

        best_val_loss = float('inf')
        history = {
            'train_loss': [], 'val_loss': [],
            'val_toxicity_macro_f1': [], 'val_bullying_macro_f1': [],
            'val_emotion_macro_f1': []
        }

        for epoch in range(1, self.epochs + 1):
            logger.info(f"\nEpoch {epoch}/{self.epochs}")

            # 訓練
            train_metrics = self.train_epoch(epoch)
            logger.info(f"Train Loss: {train_metrics['total']:.4f}")

            # 驗證
            val_metrics = self.validate_epoch(epoch)
            logger.info(f"Val Loss: {val_metrics['total']:.4f}")

            # 記錄歷史
            history['train_loss'].append(train_metrics['total'])
            history['val_loss'].append(val_metrics['total'])

            for task in ['toxicity', 'bullying', 'emotion']:
                metric_key = f'val_{task}_macro_f1'
                if metric_key in val_metrics:
                    if metric_key not in history:
                        history[metric_key] = []
                    history[metric_key].append(val_metrics[metric_key])
                    logger.info(
                        f"{task.capitalize()} Macro F1:"
                            " {val_metrics[metric_key]:.4f}"
                    )

            # 儲存最佳模型
            if val_metrics['total'] < best_val_loss:
                best_val_loss = val_metrics['total']
                self.save_checkpoint(epoch, val_metrics, is_best=True)
                logger.info(
                    f"New best model saved (val_loss: {best_val_loss:.4f})"
                )

            # Early Stopping
            if self.early_stopping(val_metrics['total'], self.model):
                logger.info(f"Early stopping triggered at epoch {epoch}")
                break

        # 最終測試
        if self.test_dataloader:
            logger.info("\nRunning final evaluation on test set...")
            test_metrics = self.evaluator.evaluate(self.test_dataloader)

            logger.info("Test Results:")
            for task in ['toxicity', 'bullying', 'role', 'emotion']:
                if f'{task}_macro_f1' in test_metrics:
                    logger.info(
                        f"  {task.capitalize()} Macro F1: "
                        f"{test_metrics[f'{task}_macro_f1']:.4f}"
                    )

            # 生成分類報告
            report = self.evaluator.generate_classification_report(
                self.test_dataloader
            )
            logger.info(f"\n{report}")

            # 儲存測試結果
            with open(
                Path(self.save_path) / 'test_results.json', 'w',
                    encoding='utf-8'
            ) as f:
                json.dump(test_metrics, f, indent=2, ensure_ascii=False)

        self.writer.close()
        logger.info(
            f"Training completed. Best model saved to {self.save_path}"
        )

        return history

    def save_checkpoint(
        self, epoch: int, metrics: Dict[str, float], is_best: bool = False
    ):
        """儲存檢查點"""
        save_path = Path(self.save_path)
        save_path.mkdir(parents=True, exist_ok=True)

        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'metrics': metrics,
            'config': self.config
        }

        # 儲存當前檢查點
        torch.save(checkpoint, save_path / 'last.ckpt')

        # 儲存最佳檢查點
        if is_best:
            self.model.save_model(str(save_path))

        # 儲存訓練日誌
        with open(save_path / 'training_log.json', 'w', encoding='utf-8') as f:
            log_data = {
                'epoch': epoch,
                'metrics': metrics,
                'config': self.config
            }
            json.dump(log_data, f, indent=2, ensure_ascii=False)


def load_data(
    data_path: str,
    tokenizer,
    max_samples: Optional[int] = None
) -> List[Tuple[str, UnifiedLabel]]:
    """載入訓練資料"""
    logger.info(f"Loading data from {data_path}")

    data = []
    data_path = Path(data_path)

    if data_path.suffix == '.json':
        with open(data_path, 'r', encoding='utf-8') as f:
            raw_data = json.load(f)

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

                data.append((text, unified_label))

                if max_samples and len(data) >= max_samples:
                    break

    logger.info(f"Loaded {len(data)} samples")
    return data


def create_data_loaders(data: List[Tuple[str, UnifiedLabel]],
                       tokenizer,
                       batch_size: int = 16,
                       val_split: float = 0.2,
                       test_split: float = 0.1,
                       max_length: int = 256) -> Tuple[DataLoader, DataLoader,
                           Optional[DataLoader]]:
    """建立資料載入器"""
    texts, labels = zip(*data)

    # 建立資料集
    full_dataset = MultiTaskDataset(
        list(texts),
        list(labels),
        tokenizer,
        max_length
    )

    # 分割資料
    total_size = len(full_dataset)
    test_size = int(total_size * test_split) if test_split > 0 else 0
    val_size = int(total_size * val_split)
    train_size = total_size - val_size - test_size

    if test_size > 0:
        train_dataset, val_dataset, test_dataset = random_split(
            full_dataset, [train_size, val_size, test_size]
        )
    else:
        train_dataset, val_dataset = random_split(
            full_dataset,
            [train_size,
            val_size]
        )
        test_dataset = None

    # 建立資料載入器
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=2
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=2
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=2
    ) if test_dataset else None

    logger.info(
        f"Data split - Train: {train_size}, Val: {val_size}, Test: {test_size}"
    )

    return train_loader, val_loader, test_loader


def set_seed(seed: int = 42):
    """設定隨機種子"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def main():
    parser = argparse.ArgumentParser(
        description="Train CyberPuppy baseline models"
    )

    # 資料參數
    parser.add_argument(
        '--data_path', type=str, required=True, help='Path to training data'
    )
    parser.add_argument(
        '--max_samples', type=int, default=None,
        help='Maximum number of samples to load'
    )

    # 模型參數
    parser.add_argument(
        '--model_variant', type=str, default='macbert_base',
        choices=[
            'macbert_base', 'macbert_focal', 'roberta_base',
            'roberta_advanced', 'toxicity_only'
        ],
        help='Model variant to use'
    )
    parser.add_argument(
        '--custom_config', type=str, default=None,
        help='Path to custom model config JSON'
    )

    # 訓練參數
    parser.add_argument(
        '--batch_size', type=int, default=16, help='Batch size'
    )
    parser.add_argument(
        '--epochs', type=int, default=10, help='Number of epochs'
    )
    parser.add_argument(
        '--learning_rate', type=float, default=2e-5, help='Learning rate'
    )
    parser.add_argument(
        '--weight_decay', type=float, default=0.01, help='Weight decay'
    )
    parser.add_argument(
        '--warmup_steps', type=float, default=0.1,
        help='Warmup steps (ratio or absolute)'
    )

    # 高級選項
    parser.add_argument(
        '--use_amp', action='store_true', default=True,
        help='Use automatic mixed precision'
    )
    parser.add_argument(
        '--gradient_clip', type=float, default=1.0,
        help='Gradient clipping norm'
    )
    parser.add_argument(
        '--early_stopping_patience', type=int, default=5,
        help='Early stopping patience'
    )

    # 資料分割
    parser.add_argument(
        '--val_split', type=float, default=0.2, help='Validation split ratio'
    )
    parser.add_argument(
        '--test_split', type=float, default=0.1, help='Test split ratio'
    )

    # 儲存路徑
    parser.add_argument(
        '--save_path', type=str, default='models/baseline', help='Model save path'
    )
    parser.add_argument(
        '--log_dir', type=str, default='logs/baseline',
        help='TensorBoard log directory'
    )

    # 其他
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument(
        '--num_workers', type=int, default=2, help='Number of data loading workers'
    )

    args = parser.parse_args()

    # 設定隨機種子
    set_seed(args.seed)

    # 載入模型配置
    if args.custom_config:
        with open(args.custom_config, 'r', encoding='utf-8') as f:
            custom_config = json.load(f)
        config = ModelConfig(**custom_config)
    else:
        model_variants = create_model_variants()
        config = model_variants[args.model_variant]

    logger.info(f"Using model variant: {args.model_variant}")
    logger.info(f"Model backbone: {config.model_name}")

    # 建立模型
    model = BaselineModel(config)

    # 載入資料
    data = load_data(args.data_path, model.tokenizer, args.max_samples)

    # 建立資料載入器
    train_loader, val_loader, test_loader = create_data_loaders(
        data,
        model.tokenizer,
        batch_size=args.batch_size,
        val_split=args.val_split,
        test_split=args.test_split,
        max_length=config.max_length
    )

    # 訓練配置
    train_config = {
        'epochs': args.epochs,
        'learning_rate': args.learning_rate,
        'weight_decay': args.weight_decay,
        'warmup_steps': args.warmup_steps,
        'use_amp': args.use_amp,
        'gradient_clip': args.gradient_clip,
        'save_path': args.save_path,
        'log_dir': args.log_dir,
        'early_stopping': {
            'patience': args.early_stopping_patience,
            'min_delta': 0.001
        }
    }

    # 建立訓練器
    trainer = Trainer(
        model=model,
        train_dataloader=train_loader,
        val_dataloader=val_loader,
        test_dataloader=test_loader,
        config=train_config
    )

    # 開始訓練
    trainer.train()

    logger.info("Training completed successfully!")


if __name__ == "__main__":
    main()
