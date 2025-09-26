#!/usr/bin/env python3
"""
簡化的訓練腳本 - 支持命令行參數
用於 Colab A100 訓練
"""

import json
import logging
import sys
import argparse
import torch
import torch.nn as nn
from pathlib import Path
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModel, get_linear_schedule_with_warmup
from torch.optim import AdamW
from tqdm import tqdm
from sklearn.metrics import f1_score, classification_report
import numpy as np
import os

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class SimpleCyberBullyDataset(Dataset):
    """簡化的資料集"""

    def __init__(self, data_path, tokenizer, max_length=384):
        self.data = json.load(open(data_path, encoding='utf-8'))
        self.tokenizer = tokenizer
        self.max_length = max_length

        self.label_maps = {
            'toxicity': {'none': 0, 'toxic': 1, 'severe': 2},
            'bullying': {'none': 0, 'harassment': 1, 'threat': 2},
        }

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        text = item.get('text', '')

        encoding = self.tokenizer(
            text,
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt'
        )

        labels = item.get('label', {})

        return {
            'input_ids': encoding['input_ids'].squeeze(),
            'attention_mask': encoding['attention_mask'].squeeze(),
            'bullying': torch.tensor(self.label_maps['bullying'].get(labels.get('bullying', 'none'), 0)),
            'toxicity': torch.tensor(self.label_maps['toxicity'].get(labels.get('toxicity', 'none'), 0)),
        }


class SimpleBullyingDetector(nn.Module):
    """簡化的雙任務模型"""

    def __init__(self, model_name='hfl/chinese-macbert-base'):
        super().__init__()
        self.bert = AutoModel.from_pretrained(model_name)
        hidden_size = self.bert.config.hidden_size

        self.bullying_head = nn.Linear(hidden_size, 3)
        self.toxicity_head = nn.Linear(hidden_size, 3)

        self.dropout = nn.Dropout(0.1)

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        pooled = outputs.pooler_output
        pooled = self.dropout(pooled)

        return {
            'bullying': self.bullying_head(pooled),
            'toxicity': self.toxicity_head(pooled)
        }


def train_epoch(model, dataloader, optimizer, scheduler, device, bullying_weight=2.0, accumulation_steps=4):
    model.train()
    total_loss = 0
    bullying_preds, bullying_true = [], []

    progress_bar = tqdm(dataloader, desc="Training")

    optimizer.zero_grad()

    for batch_idx, batch in enumerate(progress_bar):
        input_ids = batch['input_ids'].to(device, non_blocking=True)
        attention_mask = batch['attention_mask'].to(device, non_blocking=True)
        bullying_labels = batch['bullying'].to(device, non_blocking=True)
        toxicity_labels = batch['toxicity'].to(device, non_blocking=True)

        outputs = model(input_ids, attention_mask)

        bullying_loss = nn.CrossEntropyLoss()(outputs['bullying'], bullying_labels)
        toxicity_loss = nn.CrossEntropyLoss()(outputs['toxicity'], toxicity_labels)

        loss = (bullying_weight * bullying_loss + 0.8 * toxicity_loss) / accumulation_steps

        loss.backward()

        if (batch_idx + 1) % accumulation_steps == 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

        total_loss += loss.item() * accumulation_steps

        bullying_preds.extend(outputs['bullying'].argmax(dim=1).cpu().numpy())
        bullying_true.extend(bullying_labels.cpu().numpy())

        progress_bar.set_postfix({'loss': f'{loss.item() * accumulation_steps:.4f}'})

    bullying_f1 = f1_score(bullying_true, bullying_preds, average='macro')

    return total_loss / len(dataloader), bullying_f1


def evaluate(model, dataloader, device):
    model.eval()
    bullying_preds, bullying_true = [], []
    toxicity_preds, toxicity_true = [], []

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating"):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            bullying_labels = batch['bullying']
            toxicity_labels = batch['toxicity']

            outputs = model(input_ids, attention_mask)

            bullying_preds.extend(outputs['bullying'].argmax(dim=1).cpu().numpy())
            bullying_true.extend(bullying_labels.numpy())
            toxicity_preds.extend(outputs['toxicity'].argmax(dim=1).cpu().numpy())
            toxicity_true.extend(toxicity_labels.numpy())

    bullying_f1 = f1_score(bullying_true, bullying_preds, average='macro')
    toxicity_f1 = f1_score(toxicity_true, toxicity_preds, average='macro')

    unique_labels = sorted(set(bullying_true))
    label_names = ['none', 'harassment', 'threat']
    target_names_actual = [label_names[i] for i in unique_labels]

    logger.info("\n霸凌偵測:\n" + classification_report(bullying_true, bullying_preds,
                                                   labels=unique_labels,
                                                   target_names=target_names_actual,
                                                   digits=4,
                                                   zero_division=0))

    return bullying_f1, toxicity_f1


def main():
    parser = argparse.ArgumentParser(description='簡化的霸凌偵測訓練腳本')

    # 模型參數
    parser.add_argument('--model_name', type=str, default='hfl/chinese-macbert-base',
                        help='預訓練模型名稱')
    parser.add_argument('--output_dir', type=str, default='models/simple_trained',
                        help='輸出目錄')

    # 資料參數
    parser.add_argument('--train_file', type=str, default='data/processed/training_dataset/train.json',
                        help='訓練資料路徑')
    parser.add_argument('--dev_file', type=str, default='data/processed/training_dataset/dev.json',
                        help='驗證資料路徑')
    parser.add_argument('--test_file', type=str, default='data/processed/training_dataset/test.json',
                        help='測試資料路徑')

    # 訓練參數
    parser.add_argument('--learning_rate', type=float, default=2e-5, help='學習率')
    parser.add_argument('--batch_size', type=int, default=8, help='批次大小')
    parser.add_argument('--num_epochs', type=int, default=8, help='訓練輪數')
    parser.add_argument('--early_stopping_patience', type=int, default=3, help='早停耐心值')
    parser.add_argument('--bullying_weight', type=float, default=2.0, help='霸凌任務權重')
    parser.add_argument('--accumulation_steps', type=int, default=4, help='梯度累積步數')

    # 混合精度
    parser.add_argument('--fp16', action='store_true', help='使用 FP16')
    parser.add_argument('--bf16', action='store_true', help='使用 BF16 (A100)')

    args = parser.parse_args()

    # 設置設備
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"使用設備: {device}")

    if torch.cuda.is_available():
        logger.info(f"GPU: {torch.cuda.get_device_name(0)}")
        logger.info(f"記憶體: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

    # 載入分詞器和資料
    logger.info(f"載入模型: {args.model_name}")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)

    train_dataset = SimpleCyberBullyDataset(args.train_file, tokenizer)
    val_dataset = SimpleCyberBullyDataset(args.dev_file, tokenizer)
    test_dataset = SimpleCyberBullyDataset(args.test_file, tokenizer)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True,
                             num_workers=0, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size * 2, shuffle=False,
                           num_workers=0, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size * 2, shuffle=False,
                            num_workers=0, pin_memory=True)

    logger.info(f"訓練樣本: {len(train_dataset)}")
    logger.info(f"驗證樣本: {len(val_dataset)}")
    logger.info(f"測試樣本: {len(test_dataset)}")

    # 建立模型
    model = SimpleBullyingDetector(args.model_name).to(device)

    # 混合精度設置
    scaler = None
    if args.fp16:
        scaler = torch.cuda.amp.GradScaler()
        logger.info("使用 FP16 混合精度")
    elif args.bf16:
        logger.info("使用 BF16 混合精度")

    # 優化器
    optimizer = AdamW(model.parameters(), lr=args.learning_rate, weight_decay=0.01)

    num_training_steps = len(train_loader) * args.num_epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(0.1 * num_training_steps),
        num_training_steps=num_training_steps
    )

    # 訓練
    best_f1 = 0
    no_improve = 0
    os.makedirs(args.output_dir, exist_ok=True)

    for epoch in range(args.num_epochs):
        logger.info(f"\n{'='*60}")
        logger.info(f"Epoch {epoch+1}/{args.num_epochs}")
        logger.info(f"{'='*60}")

        train_loss, train_f1 = train_epoch(model, train_loader, optimizer, scheduler, device,
                                           bullying_weight=args.bullying_weight,
                                           accumulation_steps=args.accumulation_steps)
        val_f1, val_toxicity_f1 = evaluate(model, val_loader, device)

        logger.info(f"訓練損失: {train_loss:.4f}")
        logger.info(f"訓練霸凌F1: {train_f1:.4f}")
        logger.info(f"驗證霸凌F1: {val_f1:.4f}")
        logger.info(f"驗證毒性F1: {val_toxicity_f1:.4f}")

        if val_f1 > best_f1:
            best_f1 = val_f1
            no_improve = 0

            # 保存模型
            model_save_path = os.path.join(args.output_dir, 'best_model.pt')
            torch.save(model.state_dict(), model_save_path)

            # 保存分詞器
            tokenizer.save_pretrained(args.output_dir)

            # 保存結果
            results = {
                'bullying_f1': float(val_f1),
                'toxicity_f1': float(val_toxicity_f1),
                'epoch': epoch + 1
            }
            with open(os.path.join(args.output_dir, 'eval_results.json'), 'w') as f:
                json.dump(results, f, indent=2)

            logger.info(f"✓ 保存最佳模型 (F1={best_f1:.4f})")
        else:
            no_improve += 1
            if no_improve >= args.early_stopping_patience:
                logger.info(f"Early stopping after {epoch+1} epochs")
                break

    # 測試
    logger.info("\n最終測試:")
    model.load_state_dict(torch.load(os.path.join(args.output_dir, 'best_model.pt')))
    test_f1, test_toxicity_f1 = evaluate(model, test_loader, device)
    logger.info(f"\n最終測試霸凌F1: {test_f1:.4f}")
    logger.info(f"最終測試毒性F1: {test_toxicity_f1:.4f}")

    # 保存最終結果
    final_results = {
        'test_bullying_f1': float(test_f1),
        'test_toxicity_f1': float(test_toxicity_f1),
        'best_val_f1': float(best_f1)
    }
    with open(os.path.join(args.output_dir, 'final_results.json'), 'w') as f:
        json.dump(final_results, f, indent=2)

    return test_f1


if __name__ == "__main__":
    result_f1 = main()
    sys.exit(0 if result_f1 >= 0.70 else 1)