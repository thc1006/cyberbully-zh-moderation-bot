#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
標註工作流程示範腳本
展示完整的標註工作流程，包含樣本範例和使用說明
"""

import os
import json
import logging
from pathlib import Path
from datetime import datetime

# 設定日誌
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def create_sample_data():
    """建立示範用的樣本資料"""

    # 建立示範樣本
    sample_texts = [
        "今天天氣真好，心情也很棒！",
        "你這個白痴，什麼都不懂還敢說話",
        "我覺得這個想法不錯，值得考慮",
        "滾開啦，沒人想聽你廢話",
        "謝謝大家的建議，我會參考的",
        "你就是個垃圾，活著浪費空氣",
        "這個問題確實需要更多討論",
        "像你這種人就應該去死",
        "很高興能和大家一起討論",
        "你真的很煩，可以閉嘴嗎",
        "期待下次的聚會活動",
        "你這麼醜還敢出來見人？",
        "這個計畫很有前景",
        "我要讓你付出代價，等著瞧",
        "大家合作愉快！",
        "你全家都是垃圾，去死吧",
        "學習新知識真是令人興奮",
        "誰理你啊，自以為是的傢伙",
        "感謝老師的耐心指導",
        "我會找人修理你的"
    ]

    samples = []
    for i, text in enumerate(sample_texts):
        sample = {
            'id': f'sample_{i+1:03d}',
            'text': text,
            'source': 'demo_data',
            'timestamp': datetime.now().isoformat(),
            'annotation_metadata': {
                'original_index': i,
                'selection_method': 'manual',
                'annotation_priority': 'high' if any(word in text for word in ['死', '垃圾', '修理']) else 'medium'
            }
        }
        samples.append(sample)

    return samples


def setup_demo_environment():
    """設置示範環境"""

    logger.info("設置示範環境...")

    # 建立目錄結構
    base_dir = Path("data/annotation")
    subdirs = [
        "raw", "selected", "tasks", "progress",
        "results", "quality_control", "validation",
        "tracking", "archive"
    ]

    for subdir in subdirs:
        (base_dir / subdir).mkdir(parents=True, exist_ok=True)

    # 建立示範樣本檔案
    samples = create_sample_data()

    # 儲存原始樣本
    raw_file = base_dir / "raw" / "demo_samples.json"
    with open(raw_file, 'w', encoding='utf-8') as f:
        json.dump(samples, f, ensure_ascii=False, indent=2)

    logger.info(f"示範樣本已建立: {raw_file}")
    logger.info(f"共 {len(samples)} 個樣本")

    return str(raw_file)


def print_workflow_commands(sample_file: str):
    """輸出工作流程命令範例"""

    print("\n" + "="*60)
    print("標註工作流程命令範例")
    print("="*60)

    print("\n第一步：主動學習樣本選擇")
    print("-" * 40)
    print("python scripts/active_learning_selector.py \\")
    print(f"    --input {sample_file} \\")
    print("    --output data/annotation/selected/selected_samples.json \\")
    print("    --n_samples 15 \\")
    print("    --uncertainty_ratio 0.7")

    print("\n第二步：建立標註任務")
    print("-" * 40)
    print("python scripts/batch_annotation.py create \\")
    print("    --samples data/annotation/selected/selected_samples.json \\")
    print("    --task_name demo_annotation_task \\")
    print("    --annotators alice bob charlie \\")
    print("    --overlap_ratio 0.2 \\")
    print("    --difficulty_priority")

    print("\n第三步：建立追蹤表格")
    print("-" * 40)
    print("python scripts/annotation_tracker.py create \\")
    print("    --samples data/annotation/selected/selected_samples.json \\")
    print("    --annotators alice bob charlie \\")
    print("    --task_name demo_annotation_task")

    print("\n第四步：啟動標註介面")
    print("-" * 40)
    print("python scripts/annotation_interface.py")
    print("# 標註者在介面中：")
    print("# 1. 輸入標註者ID (alice/bob/charlie)")
    print("# 2. 載入分配的任務檔案")
    print("# 3. 開始標註樣本")
    print("# 4. 定期儲存進度")
    print("# 5. 完成後匯出結果")

    print("\n第五步：更新進度")
    print("-" * 40)
    print("python scripts/batch_annotation.py progress \\")
    print("    --task_id <task_id_from_step2> \\")
    print("    --annotator_id alice \\")
    print("    --progress_file data/annotation/progress/annotation_progress_alice.json")

    print("\n第六步：品質控制檢查")
    print("-" * 40)
    print("python scripts/annotation_quality_control.py \\")
    print("    --annotation_files \\")
    print("        data/annotation/results/alice_results.json \\")
    print("        data/annotation/results/bob_results.json \\")
    print("        data/annotation/results/charlie_results.json \\")
    print("    --visualize")

    print("\n第七步：結果驗證")
    print("-" * 40)
    print("python scripts/annotation_validator.py validate_batch \\")
    print("    --annotation_files \\")
    print("        data/annotation/results/alice_results.json \\")
    print("        data/annotation/results/bob_results.json \\")
    print("        data/annotation/results/charlie_results.json")

    print("\n第八步：一致性檢查")
    print("-" * 40)
    print("python scripts/annotation_validator.py check_consistency \\")
    print("    --annotation_files \\")
    print("        data/annotation/results/alice_results.json \\")
    print("        data/annotation/results/bob_results.json \\")
    print("        data/annotation/results/charlie_results.json")

    print("\n第九步：收集最終結果")
    print("-" * 40)
    print("python scripts/batch_annotation.py collect \\")
    print("    --task_id <task_id_from_step2> \\")
    print("    --result_files \\")
    print("        alice:data/annotation/results/alice_results.json \\")
    print("        bob:data/annotation/results/bob_results.json \\")
    print("        charlie:data/annotation/results/charlie_results.json")

    print("\n第十步：生成最終報告")
    print("-" * 40)
    print("python scripts/annotation_tracker.py report \\")
    print("    --excel_file data/annotation/tracking/demo_annotation_task_tracking_*.xlsx")


def create_demo_annotations():
    """建立示範標註結果"""

    logger.info("建立示範標註結果...")

    # 模擬三個標註者的標註結果
    annotators = ['alice', 'bob', 'charlie']

    # 選擇前10個樣本作為示範
    samples = create_sample_data()[:10]

    for annotator in annotators:
        annotations = {}

        for i, sample in enumerate(samples):
            sample_id = sample['id']
            text = sample['text']

            # 根據文字內容模擬標註（簡化邏輯）
            if any(word in text for word in ['白痴', '垃圾', '死', '修理']):
                toxicity = 'severe'
                bullying = 'threat' if any(word in text for word in ['死', '修理']) else 'harassment'
                role = 'perpetrator'
                emotion = 'neg'
                emotion_strength = '4'
            elif any(word in text for word in ['滾開', '廢話', '煩', '閉嘴', '醜']):
                toxicity = 'toxic'
                bullying = 'harassment'
                role = 'perpetrator'
                emotion = 'neg'
                emotion_strength = '3'
            else:
                toxicity = 'none'
                bullying = 'none'
                role = 'none'
                emotion = 'pos' if any(word in text for word in ['好', '棒', '謝謝', '高興', '愉快']) else 'neu'
                emotion_strength = '1' if emotion == 'pos' else '0'

            # 添加一些變異性
            if annotator == 'bob' and i % 3 == 0:
                # Bob 有時候判斷更嚴格
                if toxicity == 'toxic':
                    toxicity = 'severe'
                if emotion_strength == '3':
                    emotion_strength = '4'
            elif annotator == 'charlie' and i % 4 == 0:
                # Charlie 有時候判斷更寬鬆
                if toxicity == 'severe':
                    toxicity = 'toxic'
                if emotion_strength == '4':
                    emotion_strength = '3'

            annotation = {
                'toxicity': toxicity,
                'bullying': bullying,
                'role': role,
                'emotion': emotion,
                'emotion_strength': emotion_strength,
                'note': f'示範標註 by {annotator}' if toxicity != 'none' else '',
                'difficult': toxicity == 'severe',
                'timestamp': datetime.now().isoformat(),
                'annotator_id': annotator
            }

            annotations[sample_id] = annotation

        # 儲存標註結果
        result_data = {
            'annotator_id': annotator,
            'task_id': 'demo_task',
            'completed_at': datetime.now().isoformat(),
            'annotations': annotations
        }

        result_file = Path(f"data/annotation/results/{annotator}_results.json")
        result_file.parent.mkdir(parents=True, exist_ok=True)

        with open(result_file, 'w', encoding='utf-8') as f:
            json.dump(result_data, f, ensure_ascii=False, indent=2)

        logger.info(f"建立 {annotator} 的示範標註結果: {result_file}")

    return [f"data/annotation/results/{annotator}_results.json" for annotator in annotators]


def print_usage_examples():
    """輸出使用範例"""

    print("\n" + "="*60)
    print("快速開始指南")
    print("="*60)

    print("\n1. 設置環境和示範資料:")
    print("   python scripts/annotation_workflow_demo.py")

    print("\n2. 閱讀標註指引:")
    print("   參考 docs/ANNOTATION_GUIDE.md")

    print("\n3. 閱讀完整工作流程:")
    print("   參考 docs/ANNOTATION_WORKFLOW.md")

    print("\n4. 執行範例命令:")
    print("   按照上方輸出的命令依序執行")

    print("\n重要檔案位置:")
    print("   - 標註指引: docs/ANNOTATION_GUIDE.md")
    print("   - 工作流程: docs/ANNOTATION_WORKFLOW.md")
    print("   - 示範資料: data/annotation/raw/demo_samples.json")
    print("   - 結果範例: data/annotation/results/")

    print("\n可用工具腳本:")
    print("   - scripts/active_learning_selector.py    # 主動學習樣本選擇")
    print("   - scripts/annotation_interface.py        # 標註介面")
    print("   - scripts/batch_annotation.py           # 批次標註管理")
    print("   - scripts/annotation_tracker.py         # 進度追蹤")
    print("   - scripts/annotation_quality_control.py # 品質控制")
    print("   - scripts/annotation_validator.py       # 結果驗證")

    print("\n標註維度說明:")
    print("   - toxicity: none, toxic, severe")
    print("   - bullying: none, harassment, threat")
    print("   - role: none, perpetrator, victim, bystander")
    print("   - emotion: pos, neu, neg")
    print("   - emotion_strength: 0, 1, 2, 3, 4")


def main():
    """主函數"""

    print("CyberPuppy 標註工作流程示範")
    print("=" * 60)

    # 設置示範環境
    sample_file = setup_demo_environment()

    # 建立示範標註結果
    result_files = create_demo_annotations()

    # 輸出工作流程命令
    print_workflow_commands(sample_file)

    # 輸出使用範例
    print_usage_examples()

    print("\n示範環境設置完成！")
    print("請參考 docs/ANNOTATION_WORKFLOW.md 了解詳細流程")
    print("可以開始執行上方的命令來體驗完整工作流程")


if __name__ == "__main__":
    main()