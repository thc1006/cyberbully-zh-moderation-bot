#!/usr/bin/env python3
"""
CyberPuppy 霸凌偵測模型訓練腳本
支援完整的訓練管理、實驗追蹤、記憶體優化等功能
整合新的YAML配置系統和RTX 3050優化功能
"""
import os
import sys
import logging
import argparse
import traceback
from pathlib import Path

# 添加專案根目錄到路徑
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import torch
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer, AutoConfig
from torch.utils.data import DataLoader
import json
import yaml

# 導入新的配置和訓練系統
from src.cyberpuppy.training.config import TrainingPipelineConfig, ConfigManager
from src.cyberpuppy.training.trainer import MultitaskTrainer, create_trainer
try:
    from src.cyberpuppy.models.multitask import MultiTaskBullyingDetector
except ImportError:
    # 如果多任務模型不存在，創建一個基本版本
    from src.cyberpuppy.models.baselines import MultiTaskBertModel as MultiTaskBullyingDetector

try:
    from src.cyberpuppy.data.dataset import CyberBullyDataset
except ImportError:
    # 創建基本數據集類
    class CyberBullyDataset(torch.utils.data.Dataset):
        def __init__(self, data_path, tokenizer, max_length=512):
            self.data_path = data_path
            self.tokenizer = tokenizer
            self.max_length = max_length
            self.data = self._load_data()

        def _load_data(self):
            # 簡單的數據載入實現
            if self.data_path.exists():
                with open(self.data_path, 'r', encoding='utf-8') as f:
                    return json.load(f)
            return []

        def __len__(self):
            return len(self.data)

        def __getitem__(self, idx):
            item = self.data[idx]
            text = item.get('text', '')
            labels = item.get('labels', 0)

            encoding = self.tokenizer(
                text,
                truncation=True,
                padding='max_length',
                max_length=self.max_length,
                return_tensors='pt'
            )

            return {
                'input_ids': encoding['input_ids'].squeeze(),
                'attention_mask': encoding['attention_mask'].squeeze(),
                'labels': torch.tensor(labels, dtype=torch.long)
            }

# 設定日誌
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('training.log', encoding='utf-8')
    ]
)
logger = logging.getLogger(__name__)

class ModelFactory:
    """模型工廠類"""

    @staticmethod
    def create_model(config: TrainingPipelineConfig) -> torch.nn.Module:
        """根據配置建立模型"""
        logger.info(f"建立模型: {config.model.model_name}")

        try:
            # 嘗試建立多任務模型
            model = MultiTaskBullyingDetector(
                model_name=config.model.model_name,
                num_toxicity_labels=3,  # none, toxic, severe
                num_bullying_labels=3,  # none, harassment, threat
                num_role_labels=4,      # none, perpetrator, victim, bystander
                num_emotion_labels=3,   # pos, neu, neg
                dropout_rate=config.model.dropout,
                use_multitask=config.model.use_multitask,
                task_weights=config.model.task_weights
            )
        except Exception as e:
            logger.warning(f"無法建立多任務模型，使用基本模型: {e}")
            # 回退到基本BERT模型
            from transformers import AutoModelForSequenceClassification
            model = AutoModelForSequenceClassification.from_pretrained(
                config.model.model_name,
                num_labels=config.model.num_labels,
                hidden_dropout_prob=config.model.dropout
            )

        # 啟用梯度檢查點（記憶體優化）
        if config.model.gradient_checkpointing:
            model.gradient_checkpointing_enable()

        # 統計參數
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        logger.info(f"模型參數統計: 總計 {total_params:,}, 可訓練 {trainable_params:,}")

        return model

class DataLoaderFactory:
    """資料載入器工廠類"""

    @staticmethod
    def create_dataloaders(config: TrainingPipelineConfig) -> tuple:
        """建立訓練和驗證資料載入器"""
        logger.info("建立資料載入器...")

        # 載入tokenizer
        tokenizer = AutoTokenizer.from_pretrained(config.model.model_name)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        # 構建資料路徑
        data_dir = Path(config.data_dir)
        train_path = data_dir / "processed" / "train.json"
        val_path = data_dir / "processed" / "val.json"

        # 檢查資料檔案
        if not train_path.exists():
            logger.warning(f"訓練資料檔案不存在: {train_path}")
            # 創建虛擬資料作為示例
            train_path.parent.mkdir(parents=True, exist_ok=True)
            dummy_data = [
                {"text": "這是一個正常的訊息", "labels": 0},
                {"text": "這是一個有毒的訊息", "labels": 1},
                {"text": "這是一個嚴重有毒的訊息", "labels": 2}
            ] * 100  # 重複創建300個樣本

            with open(train_path, 'w', encoding='utf-8') as f:
                json.dump(dummy_data, f, ensure_ascii=False, indent=2)

            with open(val_path, 'w', encoding='utf-8') as f:
                json.dump(dummy_data[:30], f, ensure_ascii=False, indent=2)

            logger.info("已創建示例資料檔案")

        # 建立資料集
        train_dataset = CyberBullyDataset(
            data_path=train_path,
            tokenizer=tokenizer,
            max_length=config.data.max_length
        )

        val_dataset = CyberBullyDataset(
            data_path=val_path,
            tokenizer=tokenizer,
            max_length=config.data.max_length
        )

        logger.info(f"訓練樣本數: {len(train_dataset)}")
        logger.info(f"驗證樣本數: {len(val_dataset)}")

        # 建立資料載入器
        train_dataloader = DataLoader(
            train_dataset,
            batch_size=config.data.batch_size,
            shuffle=True,
            num_workers=config.data.dataloader_num_workers,
            pin_memory=config.data.pin_memory,
            drop_last=True
        )

        val_dataloader = DataLoader(
            val_dataset,
            batch_size=config.data.batch_size,
            shuffle=False,
            num_workers=config.data.dataloader_num_workers,
            pin_memory=config.data.pin_memory
        )

        return train_dataloader, val_dataloader

def setup_environment(config: TrainingPipelineConfig):
    """設定訓練環境"""
    # 設定隨機種子
    if config.experiment.deterministic:
        torch.manual_seed(config.experiment.seed)
        torch.cuda.manual_seed_all(config.experiment.seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    # 設定CUDA
    if torch.cuda.is_available() and config.resources.use_gpu:
        logger.info(f"CUDA可用，設備數量: {torch.cuda.device_count()}")

        for i in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(i)
            logger.info(f"GPU {i}: {props.name}, 記憶體: {props.total_memory / 1024**3:.2f}GB")

        # RTX 3050記憶體優化 - 已在config.__post_init__中處理

    else:
        logger.warning("CUDA不可用或未啟用，使用CPU訓練")
        config.training.fp16 = False
        config.resources.use_gpu = False

    # 建立輸出目錄
    output_dirs = [
        Path(config.log_dir),
        Path(config.checkpoint_dir),
        Path(config.model_dir) / config.experiment.name
    ]

    for dir_path in output_dirs:
        dir_path.mkdir(parents=True, exist_ok=True)

    logger.info(f"實驗目錄已創建: {[str(d) for d in output_dirs]}")

def validate_environment():
    """驗證訓練環境"""
    errors = []

    # 檢查Python版本
    if sys.version_info < (3, 8):
        errors.append("Python版本需要3.8或更高")

    # 檢查必要的套件
    try:
        import torch
        import transformers
        import sklearn
        import numpy as np
    except ImportError as e:
        errors.append(f"缺少必要套件: {e}")

    # 檢查GPU
    if torch.cuda.is_available():
        try:
            torch.cuda.init()
            device_count = torch.cuda.device_count()
            if device_count == 0:
                errors.append("未檢測到可用的GPU設備")
        except Exception as e:
            errors.append(f"GPU初始化失敗: {e}")

    if errors:
        logger.error("環境驗證失敗:")
        for error in errors:
            logger.error(f"  - {error}")
        return False

    logger.info("環境驗證通過")
    return True

def create_argument_parser() -> argparse.ArgumentParser:
    """創建命令列參數解析器"""
    parser = argparse.ArgumentParser(description="CyberPuppy霸凌偵測模型訓練")

    # 配置相關
    parser.add_argument("--config", type=str, help="YAML配置檔案路徑")
    parser.add_argument("--template", type=str, default="default",
                       help="配置模板 (default, fast_dev, production, memory_efficient)")
    parser.add_argument("--experiment-name", type=str, help="實驗名稱")

    # 模型參數
    parser.add_argument("--model-name", type=str, help="預訓練模型名稱")
    parser.add_argument("--batch-size", type=int, help="批次大小")
    parser.add_argument("--learning-rate", type=float, help="學習率")
    parser.add_argument("--num-epochs", type=int, help="訓練輪數")

    # 資源配置
    parser.add_argument("--gpu", action="store_true", help="使用GPU")
    parser.add_argument("--fp16", action="store_true", help="啟用混合精度訓練")

    return parser

def main():
    """主函數"""
    parser = create_argument_parser()
    args = parser.parse_args()

    try:
        # 驗證環境
        if not validate_environment():
            sys.exit(1)

        # 創建配置管理器
        config_manager = ConfigManager()

        # 載入或創建配置
        if args.config and Path(args.config).exists():
            logger.info(f"從檔案載入配置: {args.config}")
            config = TrainingPipelineConfig.load(args.config)
        else:
            logger.info(f"使用模板創建配置: {args.template}")
            config = config_manager.get_template(args.template)

            # 應用命令列覆蓋
            if args.experiment_name:
                config.experiment.name = args.experiment_name
            if args.model_name:
                config.model.model_name = args.model_name
            if args.batch_size:
                config.data.batch_size = args.batch_size
            if args.learning_rate:
                config.optimizer.lr = args.learning_rate
            if args.num_epochs:
                config.training.num_epochs = args.num_epochs
            if args.gpu:
                config.resources.use_gpu = True
            if args.fp16:
                config.training.fp16 = True

        # 驗證配置
        warnings = config.validate()
        if warnings:
            logger.warning("配置驗證警告:")
            for warning in warnings:
                logger.warning(f"  - {warning}")

        # 設定環境
        setup_environment(config)

        # 保存配置
        config_path = Path(config.checkpoint_dir) / f"{config.experiment.name}_config.json"
        config.save(config_path)
        logger.info(f"配置已保存到: {config_path}")

        # 建立模型
        model = ModelFactory.create_model(config)

        # 建立資料載入器
        train_dataloader, val_dataloader = DataLoaderFactory.create_dataloaders(config)

        # 建立訓練器
        trainer = create_trainer(
            config=config,
            model=model,
            train_dataloader=train_dataloader,
            eval_dataloader=val_dataloader
        )

        # 開始訓練
        logger.info("=" * 60)
        logger.info("開始訓練CyberPuppy霸凌偵測模型")
        logger.info(f"實驗名稱: {config.experiment.name}")
        logger.info(f"模型: {config.model.model_name}")
        logger.info(f"批次大小: {config.data.batch_size}")
        logger.info(f"學習率: {config.optimizer.lr}")
        logger.info(f"訓練輪數: {config.training.num_epochs}")
        logger.info("=" * 60)

        training_results = trainer.train()

        # 輸出訓練摘要
        logger.info("=" * 60)
        logger.info("訓練完成摘要:")
        logger.info(f"  實驗名稱: {config.experiment.name}")
        logger.info(f"  總步數: {training_results['total_steps']}")
        logger.info(f"  完成輪數: {training_results['epochs_completed']}")

        if 'best_metrics' in training_results:
            logger.info("  最佳指標:")
            for metric, value in training_results['best_metrics'].items():
                if isinstance(value, (int, float)):
                    logger.info(f"    {metric}: {value:.4f}")

        if 'final_metrics' in training_results:
            logger.info("  最終指標:")
            for metric, value in training_results['final_metrics'].items():
                if isinstance(value, (int, float)):
                    logger.info(f"    {metric}: {value:.4f}")

        logger.info("=" * 60)
        logger.info("訓練流程完成!")

    except KeyboardInterrupt:
        logger.info("訓練被用戶中斷")
        sys.exit(1)
    except Exception as e:
        logger.error(f"訓練過程中發生錯誤: {e}")
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()