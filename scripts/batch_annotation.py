#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
批次標註管理系統
支援標註任務分配、進度監控、結果合併等功能
"""

import os
import json
import logging
import argparse
import shutil
import pandas as pd
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime, timedelta
import uuid
from collections import defaultdict

# 設定日誌
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class BatchAnnotationManager:
    """批次標註管理器"""

    def __init__(self, project_dir: str = "data/annotation"):
        """初始化批次標註管理器

        Args:
            project_dir: 專案目錄
        """
        self.project_dir = Path(project_dir)
        self.project_dir.mkdir(parents=True, exist_ok=True)

        # 子目錄
        self.tasks_dir = self.project_dir / "tasks"
        self.progress_dir = self.project_dir / "progress"
        self.results_dir = self.project_dir / "results"
        self.archive_dir = self.project_dir / "archive"

        for directory in [self.tasks_dir, self.progress_dir, self.results_dir, self.archive_dir]:
            directory.mkdir(exist_ok=True)

        # 任務狀態
        self.task_statuses = ['created', 'assigned', 'in_progress', 'completed', 'reviewed', 'archived']

        logger.info(f"批次標註管理器初始化完成，專案目錄: {self.project_dir}")

    def create_annotation_task(self, samples: List[Dict], task_name: str, annotators: List[str],
                             overlap_ratio: float = 0.1, difficulty_priority: bool = True) -> str:
        """建立標註任務

        Args:
            samples: 待標註樣本
            task_name: 任務名稱
            annotators: 標註者列表
            overlap_ratio: 重疊比例（用於計算一致性）
            difficulty_priority: 是否優先分配困難樣本

        Returns:
            任務ID
        """
        task_id = f"{task_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{str(uuid.uuid4())[:8]}"
        logger.info(f"建立標註任務: {task_id}")

        # 任務元資料
        task_metadata = {
            'task_id': task_id,
            'task_name': task_name,
            'created_at': datetime.now().isoformat(),
            'status': 'created',
            'annotators': annotators,
            'total_samples': len(samples),
            'overlap_ratio': overlap_ratio,
            'difficulty_priority': difficulty_priority,
            'assignments': {},
            'progress': {},
            'statistics': {
                'total_assigned': 0,
                'total_completed': 0,
                'total_reviewed': 0
            }
        }

        # 分配樣本
        assignments = self.assign_samples_to_annotators(
            samples, annotators, overlap_ratio, difficulty_priority
        )

        # 為每個標註者建立任務檔案
        for annotator_id, assigned_samples in assignments.items():
            # 準備標註者任務
            annotator_task = {
                'task_id': task_id,
                'annotator_id': annotator_id,
                'assigned_at': datetime.now().isoformat(),
                'samples': assigned_samples,
                'total_samples': len(assigned_samples),
                'instructions': self.generate_task_instructions(task_name),
                'deadline': (datetime.now() + timedelta(days=7)).isoformat(),  # 預設7天期限
                'status': 'assigned'
            }

            # 儲存標註者任務檔案
            annotator_task_file = self.tasks_dir / f"{task_id}_{annotator_id}.json"
            with open(annotator_task_file, 'w', encoding='utf-8') as f:
                json.dump(annotator_task, f, ensure_ascii=False, indent=2)

            task_metadata['assignments'][annotator_id] = {
                'task_file': str(annotator_task_file),
                'assigned_samples': len(assigned_samples),
                'status': 'assigned',
                'assigned_at': datetime.now().isoformat()
            }

            task_metadata['statistics']['total_assigned'] += len(assigned_samples)

        # 儲存任務元資料
        metadata_file = self.tasks_dir / f"{task_id}_metadata.json"
        with open(metadata_file, 'w', encoding='utf-8') as f:
            json.dump(task_metadata, f, ensure_ascii=False, indent=2)

        logger.info(f"任務建立完成，已分配給 {len(annotators)} 個標註者")
        return task_id

    def assign_samples_to_annotators(self, samples: List[Dict], annotators: List[str],
                                   overlap_ratio: float, difficulty_priority: bool) -> Dict[str, List[Dict]]:
        """將樣本分配給標註者

        Args:
            samples: 樣本列表
            annotators: 標註者列表
            overlap_ratio: 重疊比例
            difficulty_priority: 是否按困難度優先分配

        Returns:
            {annotator_id: [assigned_samples]}
        """
        logger.info(f"分配 {len(samples)} 個樣本給 {len(annotators)} 個標註者")

        # 如果有困難度資訊，按困難度排序
        if difficulty_priority:
            samples_with_priority = []
            for i, sample in enumerate(samples):
                # 檢查是否有困難度標記
                priority = 0
                if 'annotation_metadata' in sample:
                    priority = sample['annotation_metadata'].get('annotation_priority', 'medium')
                    if priority == 'high':
                        priority = 2
                    elif priority == 'low':
                        priority = 0
                    else:
                        priority = 1

                samples_with_priority.append((priority, i, sample))

            # 按優先級排序（高優先級在前）
            samples_with_priority.sort(key=lambda x: x[0], reverse=True)
            samples = [item[2] for item in samples_with_priority]

        # 計算每個標註者應該分配的樣本數
        n_annotators = len(annotators)
        base_samples_per_annotator = len(samples) // n_annotators
        overlap_samples = int(len(samples) * overlap_ratio)

        # 建立分配計劃
        assignments = {annotator: [] for annotator in annotators}

        # 主要分配（無重疊）
        sample_index = 0
        for i, annotator in enumerate(annotators):
            # 基本分配
            end_index = sample_index + base_samples_per_annotator
            if i == n_annotators - 1:  # 最後一個標註者分配剩餘樣本
                end_index = len(samples) - overlap_samples

            assigned_samples = samples[sample_index:end_index]
            assignments[annotator].extend(assigned_samples)
            sample_index = end_index

        # 重疊樣本分配（所有標註者都標註）
        if overlap_samples > 0:
            overlap_start = len(samples) - overlap_samples
            overlap_samples_list = samples[overlap_start:]

            for annotator in annotators:
                assignments[annotator].extend(overlap_samples_list)

        # 記錄分配統計
        for annotator, assigned in assignments.items():
            logger.info(f"標註者 {annotator}: {len(assigned)} 個樣本")

        return assignments

    def generate_task_instructions(self, task_name: str) -> Dict[str, str]:
        """生成任務說明

        Args:
            task_name: 任務名稱

        Returns:
            任務說明
        """
        instructions = {
            'overview': f'任務名稱: {task_name}',
            'objective': '對文字樣本進行霸凌相關標註',
            'dimensions': {
                'toxicity': '標註文字的毒性程度 (none/toxic/severe)',
                'bullying': '標註霸凌類型 (none/harassment/threat)',
                'role': '標註說話者角色 (none/perpetrator/victim/bystander)',
                'emotion': '標註情緒極性 (pos/neu/neg)',
                'emotion_strength': '標註情緒強度 (0-4)'
            },
            'guidelines': {
                'quality': '請仔細閱讀每個樣本，確保標註準確性',
                'consistency': '保持標註標準的一致性',
                'difficult_cases': '遇到困難案例請標記並添加備註',
                'time_management': '建議每天完成一定數量，避免集中在截止日期'
            },
            'tools': {
                'interface': '使用 annotation_interface.py 進行標註',
                'command': 'python scripts/annotation_interface.py --samples <task_file>',
                'save_progress': '記得定期儲存進度',
                'export_results': '完成後請匯出標註結果'
            },
            'support': {
                'guidelines_doc': 'docs/ANNOTATION_GUIDE.md',
                'contact': '如有疑問請聯絡項目負責人'
            }
        }

        return instructions

    def update_task_progress(self, task_id: str, annotator_id: str, progress_file: str):
        """更新任務進度

        Args:
            task_id: 任務ID
            annotator_id: 標註者ID
            progress_file: 進度檔案路徑
        """
        logger.info(f"更新任務進度: {task_id}, 標註者: {annotator_id}")

        # 載入進度檔案
        try:
            with open(progress_file, 'r', encoding='utf-8') as f:
                progress_data = json.load(f)
        except Exception as e:
            logger.error(f"載入進度檔案失敗: {str(e)}")
            return

        # 複製進度檔案到進度目錄
        progress_backup = self.progress_dir / f"{task_id}_{annotator_id}_progress.json"
        shutil.copy2(progress_file, progress_backup)

        # 更新任務元資料
        metadata_file = self.tasks_dir / f"{task_id}_metadata.json"
        if metadata_file.exists():
            with open(metadata_file, 'r', encoding='utf-8') as f:
                task_metadata = json.load(f)

            # 更新標註者進度
            if annotator_id in task_metadata['assignments']:
                completed_samples = progress_data.get('completed_samples', 0)
                total_samples = progress_data.get('total_samples', 0)
                completion_rate = completed_samples / total_samples if total_samples > 0 else 0

                task_metadata['progress'][annotator_id] = {
                    'completed_samples': completed_samples,
                    'total_samples': total_samples,
                    'completion_rate': completion_rate,
                    'last_updated': datetime.now().isoformat(),
                    'progress_file': str(progress_backup)
                }

                # 更新整體統計
                task_metadata['statistics']['total_completed'] = sum(
                    p.get('completed_samples', 0) for p in task_metadata['progress'].values()
                )

                # 儲存更新的元資料
                with open(metadata_file, 'w', encoding='utf-8') as f:
                    json.dump(task_metadata, f, ensure_ascii=False, indent=2)

                logger.info(f"進度更新完成: {annotator_id} - {completion_rate:.1%}")

    def collect_annotation_results(self, task_id: str, result_files: Dict[str, str]) -> str:
        """收集並合併標註結果

        Args:
            task_id: 任務ID
            result_files: {annotator_id: result_file_path}

        Returns:
            合併結果檔案路徑
        """
        logger.info(f"收集任務結果: {task_id}")

        merged_results = {
            'task_id': task_id,
            'collected_at': datetime.now().isoformat(),
            'annotators': list(result_files.keys()),
            'results_by_annotator': {},
            'merged_annotations': {},
            'statistics': {
                'total_annotators': len(result_files),
                'total_annotations': 0,
                'overlapping_samples': []
            }
        }

        # 載入每個標註者的結果
        all_annotations = defaultdict(dict)  # {sample_id: {annotator_id: annotation}}

        for annotator_id, result_file in result_files.items():
            try:
                with open(result_file, 'r', encoding='utf-8') as f:
                    result_data = json.load(f)

                # 提取標註結果
                annotations = {}
                if isinstance(result_data, list):
                    for item in result_data:
                        if 'annotation' in item:
                            sample_id = str(item.get('id', item.get('annotation_metadata', {}).get('original_index', len(annotations))))
                            annotations[sample_id] = item['annotation']
                elif isinstance(result_data, dict) and 'annotations' in result_data:
                    annotations = result_data['annotations']

                merged_results['results_by_annotator'][annotator_id] = {
                    'result_file': result_file,
                    'total_annotations': len(annotations),
                    'annotations': annotations
                }

                # 組織所有標註
                for sample_id, annotation in annotations.items():
                    all_annotations[sample_id][annotator_id] = annotation

                merged_results['statistics']['total_annotations'] += len(annotations)

                logger.info(f"載入標註者 {annotator_id} 的 {len(annotations)} 個標註")

            except Exception as e:
                logger.error(f"載入標註者 {annotator_id} 結果失敗: {str(e)}")

        # 處理重疊樣本和合併邏輯
        for sample_id, annotator_annotations in all_annotations.items():
            if len(annotator_annotations) > 1:
                # 重疊樣本，需要合併或選擇
                merged_results['statistics']['overlapping_samples'].append(sample_id)
                merged_annotation = self.merge_annotations(annotator_annotations)
            else:
                # 單一標註者的樣本
                annotator_id = list(annotator_annotations.keys())[0]
                merged_annotation = annotator_annotations[annotator_id]

            merged_results['merged_annotations'][sample_id] = merged_annotation

        # 儲存合併結果
        output_file = self.results_dir / f"{task_id}_merged_results.json"
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(merged_results, f, ensure_ascii=False, indent=2)

        # 更新任務狀態
        self.update_task_status(task_id, 'completed')

        logger.info(f"結果合併完成，共 {len(merged_results['merged_annotations'])} 個樣本")
        logger.info(f"合併結果已儲存到: {output_file}")

        return str(output_file)

    def merge_annotations(self, annotator_annotations: Dict[str, Dict]) -> Dict:
        """合併多個標註者的標註結果

        Args:
            annotator_annotations: {annotator_id: annotation}

        Returns:
            合併後的標註
        """
        # 使用多數投票或專家優先的策略
        merged = {
            'annotators': list(annotator_annotations.keys()),
            'merge_method': 'majority_vote',
            'individual_annotations': annotator_annotations
        }

        # 對每個維度進行投票
        dimensions = ['toxicity', 'bullying', 'role', 'emotion', 'emotion_strength']

        for dimension in dimensions:
            votes = []
            for annotation in annotator_annotations.values():
                if dimension in annotation:
                    votes.append(annotation[dimension])

            if votes:
                # 多數投票
                vote_counts = {}
                for vote in votes:
                    vote_counts[vote] = vote_counts.get(vote, 0) + 1

                # 選擇得票最多的標籤
                most_voted = max(vote_counts.items(), key=lambda x: x[1])
                merged[dimension] = most_voted[0]
                merged[f'{dimension}_confidence'] = most_voted[1] / len(votes)

        # 合併備註
        notes = []
        for annotation in annotator_annotations.values():
            if 'note' in annotation and annotation['note'].strip():
                notes.append(annotation['note'].strip())

        if notes:
            merged['note'] = ' | '.join(notes)

        # 標記困難案例
        is_difficult = any(annotation.get('difficult', False) for annotation in annotator_annotations.values())
        if is_difficult:
            merged['difficult'] = True

        merged['merged_at'] = datetime.now().isoformat()

        return merged

    def get_task_status(self, task_id: str) -> Optional[Dict]:
        """獲取任務狀態

        Args:
            task_id: 任務ID

        Returns:
            任務狀態資訊
        """
        metadata_file = self.tasks_dir / f"{task_id}_metadata.json"

        if not metadata_file.exists():
            logger.error(f"任務 {task_id} 不存在")
            return None

        try:
            with open(metadata_file, 'r', encoding='utf-8') as f:
                task_metadata = json.load(f)

            # 計算進度統計
            total_progress = self.calculate_task_progress(task_metadata)
            task_metadata['overall_progress'] = total_progress

            return task_metadata

        except Exception as e:
            logger.error(f"獲取任務狀態失敗: {str(e)}")
            return None

    def calculate_task_progress(self, task_metadata: Dict) -> Dict:
        """計算任務整體進度

        Args:
            task_metadata: 任務元資料

        Returns:
            進度統計
        """
        progress_info = task_metadata.get('progress', {})
        assignments = task_metadata.get('assignments', {})

        total_assigned = sum(assignment.get('assigned_samples', 0) for assignment in assignments.values())
        total_completed = sum(progress.get('completed_samples', 0) for progress in progress_info.values())

        overall_completion = total_completed / total_assigned if total_assigned > 0 else 0

        # 計算每個標註者的進度
        annotator_progress = {}
        for annotator_id, assignment in assignments.items():
            assigned_count = assignment.get('assigned_samples', 0)
            completed_count = progress_info.get(annotator_id, {}).get('completed_samples', 0)
            completion_rate = completed_count / assigned_count if assigned_count > 0 else 0

            annotator_progress[annotator_id] = {
                'assigned': assigned_count,
                'completed': completed_count,
                'completion_rate': completion_rate,
                'status': 'completed' if completion_rate >= 1.0 else 'in_progress' if completion_rate > 0 else 'pending'
            }

        return {
            'total_assigned': total_assigned,
            'total_completed': total_completed,
            'overall_completion_rate': overall_completion,
            'annotator_progress': annotator_progress,
            'active_annotators': len([p for p in annotator_progress.values() if p['status'] == 'in_progress']),
            'completed_annotators': len([p for p in annotator_progress.values() if p['status'] == 'completed'])
        }

    def update_task_status(self, task_id: str, new_status: str):
        """更新任務狀態

        Args:
            task_id: 任務ID
            new_status: 新狀態
        """
        metadata_file = self.tasks_dir / f"{task_id}_metadata.json"

        if not metadata_file.exists():
            logger.error(f"任務 {task_id} 不存在")
            return

        try:
            with open(metadata_file, 'r', encoding='utf-8') as f:
                task_metadata = json.load(f)

            old_status = task_metadata.get('status', 'unknown')
            task_metadata['status'] = new_status
            task_metadata['status_updated_at'] = datetime.now().isoformat()

            with open(metadata_file, 'w', encoding='utf-8') as f:
                json.dump(task_metadata, f, ensure_ascii=False, indent=2)

            logger.info(f"任務 {task_id} 狀態更新: {old_status} -> {new_status}")

        except Exception as e:
            logger.error(f"更新任務狀態失敗: {str(e)}")

    def list_tasks(self, status_filter: Optional[str] = None) -> List[Dict]:
        """列出所有任務

        Args:
            status_filter: 狀態過濾器

        Returns:
            任務列表
        """
        tasks = []

        for metadata_file in self.tasks_dir.glob("*_metadata.json"):
            try:
                with open(metadata_file, 'r', encoding='utf-8') as f:
                    task_metadata = json.load(f)

                if status_filter is None or task_metadata.get('status') == status_filter:
                    # 計算最新進度
                    progress = self.calculate_task_progress(task_metadata)
                    task_metadata['overall_progress'] = progress
                    tasks.append(task_metadata)

            except Exception as e:
                logger.warning(f"讀取任務元資料失敗 {metadata_file}: {str(e)}")

        # 按建立時間排序
        tasks.sort(key=lambda x: x.get('created_at', ''), reverse=True)

        return tasks

    def generate_progress_report(self, task_id: Optional[str] = None) -> str:
        """生成進度報告

        Args:
            task_id: 任務ID（可選，不提供則報告所有任務）

        Returns:
            報告檔案路徑
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        if task_id:
            tasks = [self.get_task_status(task_id)]
            if tasks[0] is None:
                logger.error(f"任務 {task_id} 不存在")
                return ""
            report_file = self.results_dir / f"progress_report_{task_id}_{timestamp}.txt"
        else:
            tasks = self.list_tasks()
            report_file = self.results_dir / f"progress_report_all_{timestamp}.txt"

        # 生成報告
        with open(report_file, 'w', encoding='utf-8') as f:
            f.write("=== 標註任務進度報告 ===\n\n")
            f.write(f"生成時間: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")

            if task_id:
                f.write(f"任務ID: {task_id}\n\n")

            for task in tasks:
                f.write(f"任務: {task['task_name']} ({task['task_id']})\n")
                f.write(f"狀態: {task['status']}\n")
                f.write(f"建立時間: {task['created_at']}\n")
                f.write(f"標註者: {', '.join(task['annotators'])}\n")

                if 'overall_progress' in task:
                    progress = task['overall_progress']
                    f.write(f"整體進度: {progress['total_completed']}/{progress['total_assigned']} ")
                    f.write(f"({progress['overall_completion_rate']:.1%})\n")

                    f.write("\n標註者進度:\n")
                    for annotator_id, ann_progress in progress['annotator_progress'].items():
                        f.write(f"  {annotator_id}: {ann_progress['completed']}/{ann_progress['assigned']} ")
                        f.write(f"({ann_progress['completion_rate']:.1%}) - {ann_progress['status']}\n")

                f.write("\n" + "="*50 + "\n\n")

        logger.info(f"進度報告已生成: {report_file}")
        return str(report_file)

    def archive_completed_task(self, task_id: str):
        """歸檔完成的任務

        Args:
            task_id: 任務ID
        """
        logger.info(f"歸檔任務: {task_id}")

        # 檢查任務狀態
        task_status = self.get_task_status(task_id)
        if not task_status or task_status['status'] != 'completed':
            logger.error(f"任務 {task_id} 未完成，無法歸檔")
            return

        # 建立歸檔目錄
        archive_task_dir = self.archive_dir / task_id
        archive_task_dir.mkdir(exist_ok=True)

        # 移動相關檔案到歸檔目錄
        patterns = [
            f"{task_id}_*.json",
            f"{task_id}_*.txt"
        ]

        for pattern in patterns:
            for source_dir in [self.tasks_dir, self.progress_dir, self.results_dir]:
                for file_path in source_dir.glob(pattern):
                    dest_path = archive_task_dir / file_path.name
                    shutil.move(str(file_path), str(dest_path))
                    logger.info(f"歸檔檔案: {file_path.name}")

        # 更新任務狀態
        metadata_file = archive_task_dir / f"{task_id}_metadata.json"
        if metadata_file.exists():
            with open(metadata_file, 'r', encoding='utf-8') as f:
                task_metadata = json.load(f)

            task_metadata['status'] = 'archived'
            task_metadata['archived_at'] = datetime.now().isoformat()

            with open(metadata_file, 'w', encoding='utf-8') as f:
                json.dump(task_metadata, f, ensure_ascii=False, indent=2)

        logger.info(f"任務 {task_id} 歸檔完成")


def main():
    """主函數"""
    parser = argparse.ArgumentParser(description="批次標註管理系統")

    subparsers = parser.add_subparsers(dest='command', help='可用命令')

    # 建立任務
    create_parser = subparsers.add_parser('create', help='建立新的標註任務')
    create_parser.add_argument('--samples', required=True, help='樣本檔案路徑')
    create_parser.add_argument('--task_name', required=True, help='任務名稱')
    create_parser.add_argument('--annotators', nargs='+', required=True, help='標註者列表')
    create_parser.add_argument('--overlap_ratio', type=float, default=0.1, help='重疊比例')
    create_parser.add_argument('--difficulty_priority', action='store_true', help='按困難度優先分配')

    # 更新進度
    progress_parser = subparsers.add_parser('progress', help='更新任務進度')
    progress_parser.add_argument('--task_id', required=True, help='任務ID')
    progress_parser.add_argument('--annotator_id', required=True, help='標註者ID')
    progress_parser.add_argument('--progress_file', required=True, help='進度檔案路徑')

    # 收集結果
    collect_parser = subparsers.add_parser('collect', help='收集並合併標註結果')
    collect_parser.add_argument('--task_id', required=True, help='任務ID')
    collect_parser.add_argument('--result_files', nargs='+', required=True, help='結果檔案路徑（格式：annotator_id:file_path）')

    # 檢查狀態
    status_parser = subparsers.add_parser('status', help='檢查任務狀態')
    status_parser.add_argument('--task_id', help='任務ID（可選）')
    status_parser.add_argument('--status_filter', help='狀態過濾器')

    # 生成報告
    report_parser = subparsers.add_parser('report', help='生成進度報告')
    report_parser.add_argument('--task_id', help='任務ID（可選）')

    # 歸檔任務
    archive_parser = subparsers.add_parser('archive', help='歸檔完成的任務')
    archive_parser.add_argument('--task_id', required=True, help='任務ID')

    # 專案目錄
    parser.add_argument('--project_dir', default='data/annotation', help='專案目錄')

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        return

    # 初始化管理器
    manager = BatchAnnotationManager(args.project_dir)

    # 執行對應命令
    if args.command == 'create':
        # 載入樣本
        with open(args.samples, 'r', encoding='utf-8') as f:
            if args.samples.endswith('.json'):
                samples = json.load(f)
            elif args.samples.endswith('.jsonl'):
                samples = [json.loads(line) for line in f]
            elif args.samples.endswith('.csv'):
                df = pd.read_csv(args.samples)
                samples = df.to_dict('records')
            else:
                logger.error("不支援的檔案格式")
                return

        task_id = manager.create_annotation_task(
            samples=samples,
            task_name=args.task_name,
            annotators=args.annotators,
            overlap_ratio=args.overlap_ratio,
            difficulty_priority=args.difficulty_priority
        )
        print(f"任務已建立: {task_id}")

    elif args.command == 'progress':
        manager.update_task_progress(args.task_id, args.annotator_id, args.progress_file)

    elif args.command == 'collect':
        # 解析結果檔案參數
        result_files = {}
        for item in args.result_files:
            if ':' in item:
                annotator_id, file_path = item.split(':', 1)
                result_files[annotator_id] = file_path
            else:
                logger.error(f"結果檔案格式錯誤: {item}")
                return

        merged_file = manager.collect_annotation_results(args.task_id, result_files)
        print(f"結果已合併: {merged_file}")

    elif args.command == 'status':
        if args.task_id:
            status = manager.get_task_status(args.task_id)
            if status:
                print(json.dumps(status, ensure_ascii=False, indent=2))
        else:
            tasks = manager.list_tasks(args.status_filter)
            for task in tasks:
                progress = task['overall_progress']
                print(f"{task['task_id']}: {task['task_name']} - {task['status']} "
                      f"({progress['overall_completion_rate']:.1%})")

    elif args.command == 'report':
        report_file = manager.generate_progress_report(args.task_id)
        print(f"報告已生成: {report_file}")

    elif args.command == 'archive':
        manager.archive_completed_task(args.task_id)


if __name__ == "__main__":
    main()