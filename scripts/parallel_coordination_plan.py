#!/usr/bin/env python3
"""
CyberPuppy 並行協調執行計畫
戰略規劃 Agent 的核心協調腳本
目標: F1 從 0.55 提升至 0.75+
"""

import os
import sys
import json
import time
import logging
import subprocess
from pathlib import Path
from typing import Dict, List, Any
from datetime import datetime, timedelta

# 設置日誌
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - [COORDINATOR] - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('coordination.log', encoding='utf-8')
    ]
)
logger = logging.getLogger(__name__)

class StrategicCoordinator:
    """戰略協調器 - 管理所有 agents 的並行執行"""

    def __init__(self):
        self.start_time = datetime.now()
        self.target_f1 = 0.75
        self.current_f1 = 0.55
        self.improvement_needed = self.target_f1 - self.current_f1

        self.tasks = {
            "data_augmentation": {
                "status": "pending",
                "agent": "data_augmentation_specialist",
                "priority": "high",
                "estimated_time": "1-2 hours",
                "expected_improvement": 0.03,
                "script": "scripts/run_data_augmentation.py"
            },
            "label_mapping_fix": {
                "status": "pending",
                "agent": "label_mapping_engineer",
                "priority": "critical",
                "estimated_time": "2-3 hours",
                "expected_improvement": 0.05,
                "script": "scripts/fix_label_mapping.py"
            },
            "improved_model_training": {
                "status": "pending",
                "agent": "training_coordinator",
                "priority": "high",
                "estimated_time": "3-6 hours",
                "expected_improvement": 0.15,
                "script": "scripts/train_improved_model.py"
            },
            "performance_monitoring": {
                "status": "pending",
                "agent": "performance_monitor",
                "priority": "medium",
                "estimated_time": "continuous",
                "expected_improvement": 0.0,
                "script": "scripts/monitor_training.py"
            },
            "final_evaluation": {
                "status": "pending",
                "agent": "evaluation_specialist",
                "priority": "medium",
                "estimated_time": "30 minutes",
                "expected_improvement": 0.0,
                "script": "scripts/final_evaluation.py"
            }
        }

        self.risk_factors = {
            "gpu_memory_oom": {"probability": 0.3, "impact": "high", "mitigation": "dynamic_batch_sizing"},
            "training_non_convergence": {"probability": 0.2, "impact": "critical", "mitigation": "early_stopping + lr_scheduling"},
            "data_quality_issues": {"probability": 0.15, "impact": "medium", "mitigation": "multi_layer_validation"},
            "time_overrun": {"probability": 0.25, "impact": "medium", "mitigation": "parallel_execution"}
        }

    def log_status(self, message: str, level: str = "info"):
        """統一日誌記錄"""
        elapsed = datetime.now() - self.start_time
        status_msg = f"[{elapsed}] {message}"

        if level == "info":
            logger.info(status_msg)
        elif level == "warning":
            logger.warning(status_msg)
        elif level == "error":
            logger.error(status_msg)
        elif level == "critical":
            logger.critical(status_msg)

    def check_prerequisites(self) -> bool:
        """檢查執行前提條件"""
        self.log_status("檢查執行前提條件...")

        checks = {
            "gpu_available": False,
            "datasets_ready": False,
            "model_architecture_ready": False,
            "training_pipeline_ready": False
        }

        # GPU 檢查
        try:
            import torch
            checks["gpu_available"] = torch.cuda.is_available()
            if checks["gpu_available"]:
                gpu_name = torch.cuda.get_device_name(0)
                gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
                self.log_status(f"GPU 檢查通過: {gpu_name} ({gpu_memory:.1f}GB)")
        except Exception as e:
            self.log_status(f"GPU 檢查失敗: {e}", "error")

        # 資料集檢查
        dataset_files = [
            "data/raw/cold/COLDataset/train.csv",
            "data/raw/chnsenticorp/train.arrow",
            "data/raw/dmsc/DMSC.csv",
            "data/raw/ntusd/data/正面詞無重複_9365詞.txt"
        ]

        missing_files = []
        for file_path in dataset_files:
            if not Path(file_path).exists():
                missing_files.append(file_path)

        checks["datasets_ready"] = len(missing_files) == 0
        if not checks["datasets_ready"]:
            self.log_status(f"缺少資料集檔案: {missing_files}", "warning")
        else:
            self.log_status("資料集檢查通過: 4/4 完整")

        # 模型架構檢查
        model_files = [
            "src/cyberpuppy/models/improved_detector.py",
            "src/cyberpuppy/training/trainer.py"
        ]

        for file_path in model_files:
            if Path(file_path).exists():
                checks["model_architecture_ready"] = True
                break

        if checks["model_architecture_ready"]:
            self.log_status("模型架構檢查通過")
        else:
            self.log_status("模型架構檔案缺失", "error")

        # 訓練 pipeline 檢查
        pipeline_files = [
            "scripts/train_improved_model.py"
        ]

        checks["training_pipeline_ready"] = all(Path(f).exists() for f in pipeline_files)
        if checks["training_pipeline_ready"]:
            self.log_status("訓練 pipeline 檢查通過")
        else:
            self.log_status("訓練 pipeline 檔案缺失", "error")

        # 總結
        all_ready = all(checks.values())
        if all_ready:
            self.log_status("✅ 所有前提條件檢查通過，準備開始執行")
        else:
            failed_checks = [k for k, v in checks.items() if not v]
            self.log_status(f"❌ 前提條件檢查失敗: {failed_checks}", "error")

        return all_ready

    def estimate_timeline(self) -> Dict[str, Any]:
        """估算執行時間線"""
        timeline = {
            "total_estimated_time": "5-9 hours",
            "critical_path": ["label_mapping_fix", "improved_model_training", "final_evaluation"],
            "parallel_tasks": ["data_augmentation", "performance_monitoring"],
            "milestones": {
                "data_preparation_complete": "2-3 hours",
                "training_started": "3-4 hours",
                "first_evaluation": "6-8 hours",
                "target_achieved": "5-9 hours"
            }
        }

        self.log_status(f"時間線估算: {timeline['total_estimated_time']}")
        self.log_status(f"關鍵路徑: {' -> '.join(timeline['critical_path'])}")

        return timeline

    def calculate_success_probability(self) -> float:
        """計算成功機率"""
        base_probability = 0.75  # 基於現有基礎設施的基本成功率

        # 風險調整
        risk_impact = sum(
            rf["probability"] * (0.2 if rf["impact"] == "medium" else 0.4 if rf["impact"] == "high" else 0.6)
            for rf in self.risk_factors.values()
        )

        # 改進策略加成
        strategy_bonus = 0.15  # 基於已實作的改進策略

        final_probability = max(0.1, min(0.95, base_probability - risk_impact + strategy_bonus))

        self.log_status(f"成功機率估算: {final_probability:.1%}")
        return final_probability

    def generate_execution_report(self) -> Dict[str, Any]:
        """生成執行報告"""
        report = {
            "coordination_summary": {
                "start_time": self.start_time.isoformat(),
                "target_f1": self.target_f1,
                "current_f1": self.current_f1,
                "improvement_needed": self.improvement_needed,
                "success_probability": self.calculate_success_probability()
            },
            "task_allocation": self.tasks,
            "risk_assessment": self.risk_factors,
            "timeline": self.estimate_timeline(),
            "resource_allocation": {
                "gpu_memory": "3.5GB (RTX 3050)",
                "batch_size_range": "4-8 (動態)",
                "fp16_enabled": True,
                "gradient_accumulation": "4-8 steps"
            },
            "success_criteria": {
                "primary": "霸凌偵測 F1 ≥ 0.75",
                "secondary": [
                    "毒性偵測 F1 ≥ 0.77",
                    "訓練時間 ≤ 9 小時",
                    "GPU 記憶體穩定使用"
                ]
            }
        }

        return report

def main():
    """主協調函數"""
    coordinator = StrategicCoordinator()

    print("=" * 80)
    print("🎯 CyberPuppy 戰略協調執行開始")
    print("=" * 80)

    # 檢查前提條件
    if not coordinator.check_prerequisites():
        print("❌ 前提條件檢查失敗，請先解決問題後再執行")
        sys.exit(1)

    # 生成執行報告
    report = coordinator.generate_execution_report()

    # 保存報告
    report_path = Path("coordination_execution_report.json")
    with open(report_path, 'w', encoding='utf-8') as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    coordinator.log_status(f"執行報告已保存: {report_path}")

    # 顯示關鍵資訊
    print(f"\n📊 執行摘要:")
    print(f"  目標: F1 {coordinator.current_f1} → {coordinator.target_f1}")
    print(f"  改進幅度: +{coordinator.improvement_needed:.2f}")
    print(f"  預估時間: {report['timeline']['total_estimated_time']}")
    print(f"  成功機率: {report['coordination_summary']['success_probability']:.1%}")

    print(f"\n🔄 任務分配:")
    for task_name, task_info in coordinator.tasks.items():
        print(f"  {task_name}: {task_info['agent']} ({task_info['priority']} 優先級)")

    print(f"\n⚠️  風險管控:")
    for risk_name, risk_info in coordinator.risk_factors.items():
        print(f"  {risk_name}: {risk_info['probability']:.0%} 機率, {risk_info['impact']} 影響")

    print(f"\n✅ 戰略協調完成，準備開始並行執行")
    print("=" * 80)

if __name__ == "__main__":
    main()