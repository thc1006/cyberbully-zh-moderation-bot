#!/usr/bin/env python3
"""
智能模型推送腳本 - Git LFS 優化版
自動選擇最佳模型並推送到 GitHub，優化 LFS 用量
"""

import os
import json
import shutil
import argparse
import subprocess
from pathlib import Path
from typing import Dict, List, Optional


class SmartModelPusher:
    """智能模型推送器"""

    def __init__(
        self,
        repo_root: str = ".",
        target_f1: float = 0.75,
        max_lfs_usage_gb: float = 2.0
    ):
        self.repo_root = Path(repo_root)
        self.target_f1 = target_f1
        self.max_lfs_usage_gb = max_lfs_usage_gb

    def find_trained_models(self, experiments_dir: str = "models/experiments") -> List[Dict]:
        """尋找所有訓練完成的模型"""
        models = []
        exp_path = self.repo_root / experiments_dir

        if not exp_path.exists():
            print(f"⚠️ 實驗目錄不存在: {exp_path}")
            return models

        for model_dir in exp_path.iterdir():
            if not model_dir.is_dir():
                continue

            # 查找最佳模型
            best_model_path = model_dir / "best_model"
            if not best_model_path.exists():
                continue

            # 讀取評估結果
            eval_results = best_model_path / "eval_results.json"
            if eval_results.exists():
                with open(eval_results, 'r', encoding='utf-8') as f:
                    metrics = json.load(f)

                bullying_f1 = metrics.get('bullying_f1', 0.0)

                # 計算模型大小
                model_size_mb = self._get_model_size(best_model_path)

                models.append({
                    'name': model_dir.name,
                    'path': str(best_model_path),
                    'f1': bullying_f1,
                    'size_mb': model_size_mb,
                    'metrics': metrics
                })

        return sorted(models, key=lambda x: x['f1'], reverse=True)

    def _get_model_size(self, model_path: Path) -> float:
        """計算模型大小 (MB)"""
        total_size = 0
        for file in model_path.rglob('*'):
            if file.is_file():
                total_size += file.stat().st_size
        return total_size / (1024 * 1024)

    def select_models_to_push(
        self,
        models: List[Dict],
        strategy: str = "best_only"
    ) -> List[Dict]:
        """
        選擇要推送的模型

        Strategies:
        - best_only: 只推送最佳模型
        - top_3: 推送前三名 (用於集成)
        - all_passing: 推送所有達標模型
        """
        if not models:
            return []

        if strategy == "best_only":
            return [models[0]] if models[0]['f1'] >= self.target_f1 else []

        elif strategy == "top_3":
            return models[:3]

        elif strategy == "all_passing":
            return [m for m in models if m['f1'] >= self.target_f1]

        else:
            raise ValueError(f"未知策略: {strategy}")

    def prepare_deploy_directory(self, models: List[Dict]) -> Path:
        """準備部署目錄"""
        deploy_dir = self.repo_root / "models" / "bullying_improved"
        deploy_dir.mkdir(parents=True, exist_ok=True)

        if len(models) == 1:
            # 單一模型
            target_dir = deploy_dir / "best_single_model"
            target_dir.mkdir(exist_ok=True)

            # 複製模型檔案
            src_path = Path(models[0]['path'])
            for file in src_path.iterdir():
                if file.is_file():
                    shutil.copy2(file, target_dir / file.name)

            # 保存部署資訊
            self._save_deployment_info(target_dir, models[0])

            print(f"✅ 單一最佳模型已準備: {models[0]['name']}")
            print(f"📊 F1 Score: {models[0]['f1']:.4f}")
            print(f"💾 大小: {models[0]['size_mb']:.1f} MB")

            return target_dir

        else:
            # 多個模型（集成用）
            ensemble_dir = deploy_dir / "ensemble_models"
            ensemble_dir.mkdir(exist_ok=True)

            for i, model in enumerate(models, 1):
                model_dir = ensemble_dir / f"model_{i}"
                model_dir.mkdir(exist_ok=True)

                src_path = Path(model['path'])
                for file in src_path.iterdir():
                    if file.is_file():
                        shutil.copy2(file, model_dir / file.name)

                self._save_deployment_info(model_dir, model)

            # 保存集成配置
            ensemble_config = {
                'models': [
                    {'name': m['name'], 'f1': m['f1'], 'weight': 1.0 / len(models)}
                    for m in models
                ],
                'strategy': 'soft_voting',
                'total_size_mb': sum(m['size_mb'] for m in models)
            }

            with open(ensemble_dir / "ensemble_config.json", 'w', encoding='utf-8') as f:
                json.dump(ensemble_config, f, indent=2, ensure_ascii=False)

            print(f"✅ 集成模型已準備: {len(models)} 個模型")
            for i, model in enumerate(models, 1):
                print(f"  {i}. {model['name']}: F1 = {model['f1']:.4f}")
            print(f"💾 總大小: {sum(m['size_mb'] for m in models):.1f} MB")

            return ensemble_dir

    def _save_deployment_info(self, target_dir: Path, model: Dict):
        """保存部署資訊"""
        import datetime

        deploy_info = {
            'model_name': model['name'],
            'f1_score': model['f1'],
            'size_mb': model['size_mb'],
            'metrics': model['metrics'],
            'deployed_at': datetime.datetime.now().isoformat(),
            'git_lfs_size_mb': model['size_mb']
        }

        with open(target_dir / "deployment_info.json", 'w', encoding='utf-8') as f:
            json.dump(deploy_info, f, indent=2, ensure_ascii=False)

    def configure_git_lfs(self, deploy_dir: Path):
        """配置 Git LFS 追蹤"""
        print("\n🔧 配置 Git LFS...")

        # 初始化 Git LFS
        subprocess.run(['git', 'lfs', 'install'], cwd=self.repo_root, check=True)

        # 追蹤模型檔案
        patterns = [
            "models/bullying_improved/**/*.safetensors",
            "models/bullying_improved/**/*.bin",
            "models/bullying_improved/**/*.pt",
            "models/bullying_improved/**/*.pth"
        ]

        for pattern in patterns:
            subprocess.run(['git', 'lfs', 'track', pattern], cwd=self.repo_root)

        # 添加 .gitattributes
        subprocess.run(['git', 'add', '.gitattributes'], cwd=self.repo_root)

        print("✅ Git LFS 配置完成")

    def check_lfs_usage(self) -> Dict[str, float]:
        """檢查 Git LFS 用量"""
        try:
            # 查詢當前 LFS 物件大小
            result = subprocess.run(
                ['git', 'lfs', 'ls-files', '-s'],
                cwd=self.repo_root,
                capture_output=True,
                text=True
            )

            total_size_mb = 0.0
            for line in result.stdout.split('\n'):
                if line.strip():
                    parts = line.split()
                    if len(parts) >= 3:
                        # 第二個欄位是大小
                        size_str = parts[1]
                        if size_str.endswith('MB'):
                            total_size_mb += float(size_str[:-2])
                        elif size_str.endswith('KB'):
                            total_size_mb += float(size_str[:-2]) / 1024
                        elif size_str.endswith('GB'):
                            total_size_mb += float(size_str[:-2]) * 1024

            return {
                'current_mb': total_size_mb,
                'current_gb': total_size_mb / 1024,
                'max_gb': 10.0,
                'remaining_gb': 10.0 - (total_size_mb / 1024)
            }

        except Exception as e:
            print(f"⚠️ 無法查詢 LFS 用量: {e}")
            return {'current_gb': 0.0, 'remaining_gb': 10.0}

    def push_to_github(
        self,
        deploy_dir: Path,
        model_info: Dict,
        dry_run: bool = False
    ) -> bool:
        """推送模型到 GitHub"""
        try:
            # 檢查 LFS 用量
            lfs_usage = self.check_lfs_usage()
            print(f"\n📊 Git LFS 用量:")
            print(f"  當前: {lfs_usage['current_gb']:.2f} GB")
            print(f"  限制: {lfs_usage['max_gb']:.2f} GB")
            print(f"  剩餘: {lfs_usage['remaining_gb']:.2f} GB")

            # 計算新增用量
            new_usage_gb = model_info['size_mb'] / 1024
            total_usage_gb = lfs_usage['current_gb'] + new_usage_gb

            print(f"  新增: {new_usage_gb:.2f} GB")
            print(f"  總計: {total_usage_gb:.2f} GB")

            if total_usage_gb > lfs_usage['max_gb']:
                print(f"❌ LFS 用量超限！")
                return False

            if dry_run:
                print("\n🔍 Dry run 模式 - 不執行實際推送")
                print(f"將推送: {deploy_dir}")
                print(f"大小: {model_info['size_mb']:.1f} MB")
                return True

            # 添加檔案
            print(f"\n📤 添加檔案到 Git...")
            subprocess.run(['git', 'add', str(deploy_dir)], cwd=self.repo_root, check=True)

            # 提交
            commit_msg = f"feat: Add bullying detection model (F1={model_info['f1']:.4f})"
            if model_info.get('ensemble'):
                commit_msg += f" - {len(model_info['models'])} models ensemble"

            subprocess.run(['git', 'commit', '-m', commit_msg], cwd=self.repo_root, check=True)

            # 推送
            print(f"📤 推送到 GitHub...")
            subprocess.run(['git', 'push', 'origin', 'main'], cwd=self.repo_root, check=True)

            print(f"✅ 推送成功！")
            print(f"📊 F1 Score: {model_info['f1']:.4f}")
            print(f"💾 Git LFS 新增用量: {new_usage_gb:.2f} GB")

            return True

        except subprocess.CalledProcessError as e:
            print(f"❌ 推送失敗: {e}")
            return False


def main():
    parser = argparse.ArgumentParser(description="智能模型推送腳本")
    parser.add_argument(
        '--repo-root',
        type=str,
        default='.',
        help='Repository 根目錄'
    )
    parser.add_argument(
        '--strategy',
        type=str,
        default='best_only',
        choices=['best_only', 'top_3', 'all_passing'],
        help='推送策略'
    )
    parser.add_argument(
        '--target-f1',
        type=float,
        default=0.75,
        help='目標 F1 分數'
    )
    parser.add_argument(
        '--dry-run',
        action='store_true',
        help='預演模式，不實際推送'
    )
    parser.add_argument(
        '--experiments-dir',
        type=str,
        default='models/experiments',
        help='實驗模型目錄'
    )

    args = parser.parse_args()

    print("🚀 智能模型推送系統")
    print("=" * 60)

    # 初始化推送器
    pusher = SmartModelPusher(
        repo_root=args.repo_root,
        target_f1=args.target_f1
    )

    # 尋找訓練完成的模型
    print(f"\n🔍 掃描實驗模型: {args.experiments_dir}")
    models = pusher.find_trained_models(args.experiments_dir)

    if not models:
        print("❌ 沒有找到訓練完成的模型")
        return

    print(f"\n找到 {len(models)} 個模型:")
    for i, model in enumerate(models, 1):
        status = "✅" if model['f1'] >= args.target_f1 else "⚠️"
        print(f"{i}. {status} {model['name']}: F1 = {model['f1']:.4f}, Size = {model['size_mb']:.1f} MB")

    # 選擇要推送的模型
    print(f"\n📋 推送策略: {args.strategy}")
    selected_models = pusher.select_models_to_push(models, args.strategy)

    if not selected_models:
        print("❌ 沒有符合條件的模型")
        print(f"   目標 F1: {args.target_f1}")
        print(f"   最佳 F1: {models[0]['f1']:.4f}")
        print(f"   差距: {args.target_f1 - models[0]['f1']:.4f}")
        return

    print(f"\n選擇推送 {len(selected_models)} 個模型:")
    for model in selected_models:
        print(f"  • {model['name']}: F1 = {model['f1']:.4f}")

    # 準備部署目錄
    print("\n📦 準備部署目錄...")
    deploy_dir = pusher.prepare_deploy_directory(selected_models)

    # 配置 Git LFS
    pusher.configure_git_lfs(deploy_dir)

    # 推送到 GitHub
    model_info = {
        'f1': selected_models[0]['f1'],
        'size_mb': sum(m['size_mb'] for m in selected_models),
        'ensemble': len(selected_models) > 1,
        'models': selected_models
    }

    success = pusher.push_to_github(deploy_dir, model_info, dry_run=args.dry_run)

    if success:
        print("\n🎉 完成！")
    else:
        print("\n❌ 推送失敗")


if __name__ == "__main__":
    main()