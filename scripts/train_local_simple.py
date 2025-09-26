#!/usr/bin/env python3
"""
簡化的本地訓練腳本 - 直接使用現有的 working model
繞過複雜的improved_detector問題
"""

import json
import logging
import sys
import torch
import torch.nn as nn
from pathlib import Path
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModel, get_linear_schedule_with_warmup
from torch.optim import AdamW
from tqdm import tqdm
from sklearn.metrics import f1_score, classification_report
import numpy as np

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


class SimpleCyberBullyDataset(Dataset):
    """簡化的資料集"""

    def __init__(self, data_path, tokenizer, max_length=384):
        self.data = json.load(open(data_path, encoding='utf-8'))
        self.tokenizer = tokenizer
        self.max_length = max_length

        # 標籤映射
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

        # 計算損失 - 重點優化霸凌
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

        # 記錄預測
        bullying_preds.extend(outputs['bullying'].argmax(dim=1).cpu().numpy())
        bullying_true.extend(bullying_labels.cpu().numpy())

        progress_bar.set_postfix({'loss': f'{loss.item() * accumulation_steps:.4f}'})

    # 計算F1
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

    logger.info("\n霸凌偵測:\n" + classification_report(bullying_true, bullying_preds,
                                                   target_names=['none', 'harassment', 'threat'], digits=4))

    return bullying_f1, toxicity_f1


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"使用設備: {device}")

    # 載入分詞器和資料
    tokenizer = AutoTokenizer.from_pretrained('hfl/chinese-macbert-base')

    train_dataset = SimpleCyberBullyDataset('data/processed/training_dataset/train.json', tokenizer)
    val_dataset = SimpleCyberBullyDataset('data/processed/training_dataset/dev.json', tokenizer)
    test_dataset = SimpleCyberBullyDataset('data/processed/training_dataset/test.json', tokenizer)

    # 使用較小batch size避免記憶體問題，但增加梯度累積模擬大batch
    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True, num_workers=0, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False, num_workers=0, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False, num_workers=0, pin_memory=True)

    logger.info(f"訓練樣本: {len(train_dataset)}")
    logger.info(f"驗證樣本: {len(val_dataset)}")

    # 建立模型
    model = SimpleBullyingDetector().to(device)

    # 優化器
    optimizer = AdamW(model.parameters(), lr=3e-5, weight_decay=0.01)

    num_epochs = 8  # 減少epochs以加速訓練
    num_training_steps = len(train_loader) * num_epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(0.1 * num_training_steps),
        num_training_steps=num_training_steps
    )

    # 訓練
    best_f1 = 0
    patience = 3
    no_improve = 0

    for epoch in range(num_epochs):
        logger.info(f"\n{'='*60}")
        logger.info(f"Epoch {epoch+1}/{num_epochs}")
        logger.info(f"{'='*60}")

        train_loss, train_f1 = train_epoch(model, train_loader, optimizer, scheduler, device,
                                           bullying_weight=2.0, accumulation_steps=4)
        val_f1, val_toxicity_f1 = evaluate(model, val_loader, device)

        logger.info(f"訓練損失: {train_loss:.4f}")
        logger.info(f"訓練霸凌F1: {train_f1:.4f}")
        logger.info(f"驗證霸凌F1: {val_f1:.4f}")
        logger.info(f"驗證毒性F1: {val_toxicity_f1:.4f}")

        if val_f1 > best_f1:
            best_f1 = val_f1
            no_improve = 0
            torch.save(model.state_dict(), 'models/simple_bullying_best.pt')
            logger.info(f"✓ 保存最佳模型 (F1={best_f1:.4f})")
        else:
            no_improve += 1
            if no_improve >= patience:
                logger.info(f"Early stopping after {epoch+1} epochs")
                break

    # 測試
    logger.info("\n最終測試:")
    model.load_state_dict(torch.load('models/simple_bullying_best.pt'))
    test_f1, test_toxicity_f1 = evaluate(model, test_loader, device)
    logger.info(f"\n最終測試霸凌F1: {test_f1:.4f}")
    logger.info(f"最終測試毒性F1: {test_toxicity_f1:.4f}")

    return test_f1


if __name__ == "__main__":
    sys.exit(0 if main() >= 0.75 else 1)