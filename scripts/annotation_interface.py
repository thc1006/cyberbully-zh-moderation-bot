#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
網路霸凌標註介面
提供互動式的標註介面，支援批次標註和品質控制
"""

import os
import json
import logging
import argparse
import tkinter as tk
from tkinter import ttk, messagebox, filedialog
from pathlib import Path
from typing import Dict, List, Optional, Any
from datetime import datetime
import pandas as pd

# 設定日誌
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class AnnotationInterface:
    """標註介面主類"""

    def __init__(self, root: tk.Tk):
        """初始化標註介面

        Args:
            root: Tkinter 根視窗
        """
        self.root = root
        self.root.title("CyberPuppy 霸凌樣本標註系統")
        self.root.geometry("1200x800")

        # 資料相關
        self.samples = []
        self.current_index = 0
        self.annotations = {}
        self.annotator_id = ""
        self.project_path = ""

        # 標註進度
        self.total_samples = 0
        self.completed_samples = 0

        # 標籤選項
        self.toxicity_options = ["none", "toxic", "severe"]
        self.bullying_options = ["none", "harassment", "threat"]
        self.role_options = ["none", "perpetrator", "victim", "bystander"]
        self.emotion_options = ["pos", "neu", "neg"]
        self.emotion_strength_options = ["0", "1", "2", "3", "4"]

        # 建立UI
        self.setup_ui()

        # 載入設定
        self.load_settings()

    def setup_ui(self):
        """建立使用者介面"""
        # 主框架
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)

        # 頂部控制區
        self.setup_control_area(main_frame)

        # 樣本顯示區
        self.setup_sample_area(main_frame)

        # 標註區域
        self.setup_annotation_area(main_frame)

        # 底部操作區
        self.setup_action_area(main_frame)

        # 狀態列
        self.setup_status_area(main_frame)

    def setup_control_area(self, parent):
        """建立控制區域"""
        control_frame = ttk.LabelFrame(parent, text="檔案控制", padding="5")
        control_frame.grid(row=0, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=(0, 10))

        # 標註者ID
        ttk.Label(control_frame, text="標註者ID:").grid(row=0, column=0, sticky=tk.W, padx=(0, 5))
        self.annotator_entry = ttk.Entry(control_frame, width=20)
        self.annotator_entry.grid(row=0, column=1, sticky=tk.W, padx=(0, 20))

        # 載入檔案按鈕
        ttk.Button(control_frame, text="載入樣本檔案", command=self.load_samples).grid(
            row=0, column=2, padx=(0, 10)
        )

        # 儲存進度按鈕
        ttk.Button(control_frame, text="儲存進度", command=self.save_progress).grid(
            row=0, column=3, padx=(0, 10)
        )

        # 載入進度按鈕
        ttk.Button(control_frame, text="載入進度", command=self.load_progress).grid(
            row=0, column=4, padx=(0, 10)
        )

        # 匯出結果按鈕
        ttk.Button(control_frame, text="匯出結果", command=self.export_results).grid(
            row=0, column=5
        )

    def setup_sample_area(self, parent):
        """建立樣本顯示區域"""
        sample_frame = ttk.LabelFrame(parent, text="樣本內容", padding="5")
        sample_frame.grid(row=1, column=0, columnspan=2, sticky=(tk.W, tk.E, tk.N, tk.S), pady=(0, 10))
        sample_frame.columnconfigure(0, weight=1)
        sample_frame.rowconfigure(1, weight=1)

        # 樣本導航
        nav_frame = ttk.Frame(sample_frame)
        nav_frame.grid(row=0, column=0, sticky=(tk.W, tk.E), pady=(0, 5))

        ttk.Button(nav_frame, text="◀ 上一個", command=self.previous_sample).pack(side=tk.LEFT, padx=(0, 5))
        ttk.Button(nav_frame, text="下一個 ▶", command=self.next_sample).pack(side=tk.LEFT, padx=(0, 10))

        # 進度顯示
        self.progress_label = ttk.Label(nav_frame, text="樣本 0/0")
        self.progress_label.pack(side=tk.LEFT, padx=(10, 0))

        # 樣本ID顯示
        self.sample_id_label = ttk.Label(nav_frame, text="ID: -")
        self.sample_id_label.pack(side=tk.RIGHT)

        # 樣本文字顯示
        self.sample_text = tk.Text(sample_frame, height=8, wrap=tk.WORD, font=("Microsoft YaHei", 12))
        scrollbar = ttk.Scrollbar(sample_frame, orient="vertical", command=self.sample_text.yview)
        self.sample_text.configure(yscrollcommand=scrollbar.set)

        self.sample_text.grid(row=1, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        scrollbar.grid(row=1, column=1, sticky=(tk.N, tk.S))

    def setup_annotation_area(self, parent):
        """建立標註區域"""
        annotation_frame = ttk.LabelFrame(parent, text="標註選項", padding="5")
        annotation_frame.grid(row=2, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=(0, 10))

        # 毒性程度
        toxicity_frame = ttk.LabelFrame(annotation_frame, text="毒性程度 (toxicity)", padding="5")
        toxicity_frame.grid(row=0, column=0, sticky=(tk.W, tk.E), padx=(0, 10))

        self.toxicity_var = tk.StringVar(value="none")
        for i, option in enumerate(self.toxicity_options):
            ttk.Radiobutton(toxicity_frame, text=option, variable=self.toxicity_var, value=option).grid(
                row=0, column=i, sticky=tk.W, padx=(0, 10)
            )

        # 霸凌類型
        bullying_frame = ttk.LabelFrame(annotation_frame, text="霸凌類型 (bullying)", padding="5")
        bullying_frame.grid(row=0, column=1, sticky=(tk.W, tk.E), padx=(0, 10))

        self.bullying_var = tk.StringVar(value="none")
        for i, option in enumerate(self.bullying_options):
            ttk.Radiobutton(bullying_frame, text=option, variable=self.bullying_var, value=option).grid(
                row=0, column=i, sticky=tk.W, padx=(0, 10)
            )

        # 角色識別
        role_frame = ttk.LabelFrame(annotation_frame, text="角色識別 (role)", padding="5")
        role_frame.grid(row=1, column=0, sticky=(tk.W, tk.E), padx=(0, 10), pady=(5, 0))

        self.role_var = tk.StringVar(value="none")
        for i, option in enumerate(self.role_options):
            ttk.Radiobutton(role_frame, text=option, variable=self.role_var, value=option).grid(
                row=0, column=i, sticky=tk.W, padx=(0, 10)
            )

        # 情緒分類和強度
        emotion_frame = ttk.LabelFrame(annotation_frame, text="情緒分析", padding="5")
        emotion_frame.grid(row=1, column=1, sticky=(tk.W, tk.E), pady=(5, 0))

        # 情緒分類
        ttk.Label(emotion_frame, text="情緒 (emotion):").grid(row=0, column=0, sticky=tk.W, padx=(0, 5))
        self.emotion_var = tk.StringVar(value="neu")
        for i, option in enumerate(self.emotion_options):
            ttk.Radiobutton(emotion_frame, text=option, variable=self.emotion_var, value=option).grid(
                row=0, column=i+1, sticky=tk.W, padx=(0, 5)
            )

        # 情緒強度
        ttk.Label(emotion_frame, text="強度:").grid(row=1, column=0, sticky=tk.W, padx=(0, 5), pady=(5, 0))
        self.emotion_strength_var = tk.StringVar(value="0")
        for i, option in enumerate(self.emotion_strength_options):
            ttk.Radiobutton(emotion_frame, text=option, variable=self.emotion_strength_var, value=option).grid(
                row=1, column=i+1, sticky=tk.W, padx=(0, 5), pady=(5, 0)
            )

        # 備註區域
        note_frame = ttk.LabelFrame(annotation_frame, text="備註說明", padding="5")
        note_frame.grid(row=2, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=(5, 0))
        note_frame.columnconfigure(0, weight=1)

        self.note_text = tk.Text(note_frame, height=3, wrap=tk.WORD)
        self.note_text.grid(row=0, column=0, sticky=(tk.W, tk.E))

    def setup_action_area(self, parent):
        """建立操作區域"""
        action_frame = ttk.Frame(parent)
        action_frame.grid(row=3, column=0, columnspan=2, pady=(0, 10))

        # 確認標註按鈕
        ttk.Button(action_frame, text="確認標註", command=self.confirm_annotation).pack(side=tk.LEFT, padx=(0, 10))

        # 跳過樣本按鈕
        ttk.Button(action_frame, text="跳過樣本", command=self.skip_sample).pack(side=tk.LEFT, padx=(0, 10))

        # 重設標註按鈕
        ttk.Button(action_frame, text="重設標註", command=self.reset_annotation).pack(side=tk.LEFT, padx=(0, 10))

        # 標記為困難案例
        ttk.Button(action_frame, text="標記困難案例", command=self.mark_difficult).pack(side=tk.LEFT)

    def setup_status_area(self, parent):
        """建立狀態列"""
        status_frame = ttk.Frame(parent)
        status_frame.grid(row=4, column=0, columnspan=2, sticky=(tk.W, tk.E))
        status_frame.columnconfigure(1, weight=1)

        # 進度條
        ttk.Label(status_frame, text="進度:").grid(row=0, column=0, sticky=tk.W, padx=(0, 5))
        self.progress_bar = ttk.Progressbar(status_frame, mode='determinate')
        self.progress_bar.grid(row=0, column=1, sticky=(tk.W, tk.E), padx=(0, 10))

        # 統計資訊
        self.stats_label = ttk.Label(status_frame, text="已完成: 0 | 剩餘: 0")
        self.stats_label.grid(row=0, column=2, sticky=tk.E)

    def load_samples(self):
        """載入樣本檔案"""
        if not self.annotator_entry.get().strip():
            messagebox.showerror("錯誤", "請先輸入標註者ID")
            return

        file_path = filedialog.askopenfilename(
            title="選擇樣本檔案",
            filetypes=[
                ("JSON files", "*.json"),
                ("JSONL files", "*.jsonl"),
                ("CSV files", "*.csv"),
                ("All files", "*.*")
            ]
        )

        if not file_path:
            return

        try:
            self.annotator_id = self.annotator_entry.get().strip()
            self.project_path = Path(file_path).parent

            # 載入資料
            if file_path.endswith('.json'):
                with open(file_path, 'r', encoding='utf-8') as f:
                    self.samples = json.load(f)
            elif file_path.endswith('.jsonl'):
                self.samples = []
                with open(file_path, 'r', encoding='utf-8') as f:
                    for line in f:
                        self.samples.append(json.loads(line.strip()))
            elif file_path.endswith('.csv'):
                df = pd.read_csv(file_path)
                self.samples = df.to_dict('records')

            # 初始化標註資料
            self.annotations = {}
            self.current_index = 0
            self.total_samples = len(self.samples)
            self.completed_samples = 0

            # 更新介面
            self.update_display()
            messagebox.showinfo("成功", f"成功載入 {len(self.samples)} 個樣本")

        except Exception as e:
            messagebox.showerror("錯誤", f"載入檔案失敗: {str(e)}")

    def save_progress(self):
        """儲存標註進度"""
        if not self.samples:
            messagebox.showwarning("警告", "沒有載入的樣本")
            return

        if not self.project_path:
            messagebox.showerror("錯誤", "無法確定儲存位置")
            return

        try:
            progress_file = self.project_path / f"annotation_progress_{self.annotator_id}.json"

            progress_data = {
                'annotator_id': self.annotator_id,
                'timestamp': datetime.now().isoformat(),
                'current_index': self.current_index,
                'total_samples': self.total_samples,
                'completed_samples': self.completed_samples,
                'annotations': self.annotations
            }

            with open(progress_file, 'w', encoding='utf-8') as f:
                json.dump(progress_data, f, ensure_ascii=False, indent=2)

            messagebox.showinfo("成功", f"進度已儲存到: {progress_file}")

        except Exception as e:
            messagebox.showerror("錯誤", f"儲存進度失敗: {str(e)}")

    def load_progress(self):
        """載入標註進度"""
        if not self.annotator_entry.get().strip():
            messagebox.showerror("錯誤", "請先輸入標註者ID")
            return

        file_path = filedialog.askopenfilename(
            title="選擇進度檔案",
            filetypes=[("JSON files", "*.json"), ("All files", "*.*")]
        )

        if not file_path:
            return

        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                progress_data = json.load(f)

            # 載入進度資料
            self.annotator_id = progress_data['annotator_id']
            self.current_index = progress_data['current_index']
            self.total_samples = progress_data['total_samples']
            self.completed_samples = progress_data['completed_samples']
            self.annotations = progress_data['annotations']

            # 更新介面
            self.annotator_entry.delete(0, tk.END)
            self.annotator_entry.insert(0, self.annotator_id)
            self.update_display()

            messagebox.showinfo("成功", "進度載入成功")

        except Exception as e:
            messagebox.showerror("錯誤", f"載入進度失敗: {str(e)}")

    def export_results(self):
        """匯出標註結果"""
        if not self.annotations:
            messagebox.showwarning("警告", "沒有標註結果可以匯出")
            return

        file_path = filedialog.asksaveasfilename(
            title="儲存標註結果",
            defaultextension=".json",
            filetypes=[
                ("JSON files", "*.json"),
                ("CSV files", "*.csv"),
                ("All files", "*.*")
            ]
        )

        if not file_path:
            return

        try:
            # 準備匯出資料
            export_data = []
            for idx, annotation in self.annotations.items():
                sample_data = self.samples[int(idx)].copy()
                sample_data.update({
                    'annotation': annotation,
                    'annotator_id': self.annotator_id,
                    'annotation_timestamp': annotation.get('timestamp', '')
                })
                export_data.append(sample_data)

            # 儲存檔案
            if file_path.endswith('.json'):
                with open(file_path, 'w', encoding='utf-8') as f:
                    json.dump(export_data, f, ensure_ascii=False, indent=2)
            elif file_path.endswith('.csv'):
                df = pd.json_normalize(export_data)
                df.to_csv(file_path, index=False, encoding='utf-8')

            messagebox.showinfo("成功", f"結果已匯出到: {file_path}")

        except Exception as e:
            messagebox.showerror("錯誤", f"匯出失敗: {str(e)}")

    def update_display(self):
        """更新顯示內容"""
        if not self.samples:
            return

        # 更新樣本顯示
        current_sample = self.samples[self.current_index]
        sample_text = self.get_sample_text(current_sample)

        self.sample_text.delete(1.0, tk.END)
        self.sample_text.insert(1.0, sample_text)

        # 更新樣本資訊
        self.progress_label.config(text=f"樣本 {self.current_index + 1}/{self.total_samples}")
        sample_id = current_sample.get('id', current_sample.get('annotation_metadata', {}).get('original_index', self.current_index))
        self.sample_id_label.config(text=f"ID: {sample_id}")

        # 載入已有的標註
        if str(self.current_index) in self.annotations:
            self.load_annotation(self.annotations[str(self.current_index)])
        else:
            self.reset_annotation()

        # 更新進度
        self.update_progress()

    def get_sample_text(self, sample: Dict) -> str:
        """從樣本中提取文字"""
        # 嘗試常見的文字欄位
        text_fields = ['text', 'content', 'message', 'sentence', 'comment']
        for field in text_fields:
            if field in sample and sample[field]:
                return str(sample[field])

        # 如果找不到標準欄位，返回樣本的字串表示
        return str(sample)

    def load_annotation(self, annotation: Dict):
        """載入標註到介面"""
        self.toxicity_var.set(annotation.get('toxicity', 'none'))
        self.bullying_var.set(annotation.get('bullying', 'none'))
        self.role_var.set(annotation.get('role', 'none'))
        self.emotion_var.set(annotation.get('emotion', 'neu'))
        self.emotion_strength_var.set(annotation.get('emotion_strength', '0'))

        self.note_text.delete(1.0, tk.END)
        if 'note' in annotation:
            self.note_text.insert(1.0, annotation['note'])

    def reset_annotation(self):
        """重設標註選項"""
        self.toxicity_var.set('none')
        self.bullying_var.set('none')
        self.role_var.set('none')
        self.emotion_var.set('neu')
        self.emotion_strength_var.set('0')
        self.note_text.delete(1.0, tk.END)

    def confirm_annotation(self):
        """確認當前標註"""
        if not self.samples:
            return

        # 收集標註資料
        annotation = {
            'toxicity': self.toxicity_var.get(),
            'bullying': self.bullying_var.get(),
            'role': self.role_var.get(),
            'emotion': self.emotion_var.get(),
            'emotion_strength': self.emotion_strength_var.get(),
            'note': self.note_text.get(1.0, tk.END).strip(),
            'timestamp': datetime.now().isoformat(),
            'annotator_id': self.annotator_id
        }

        # 儲存標註
        if str(self.current_index) not in self.annotations:
            self.completed_samples += 1

        self.annotations[str(self.current_index)] = annotation

        # 自動跳到下一個樣本
        if self.current_index < self.total_samples - 1:
            self.next_sample()
        else:
            messagebox.showinfo("完成", "所有樣本已標註完成！")

        self.update_progress()

    def skip_sample(self):
        """跳過當前樣本"""
        if self.current_index < self.total_samples - 1:
            self.next_sample()

    def next_sample(self):
        """下一個樣本"""
        if self.current_index < self.total_samples - 1:
            self.current_index += 1
            self.update_display()

    def previous_sample(self):
        """上一個樣本"""
        if self.current_index > 0:
            self.current_index -= 1
            self.update_display()

    def mark_difficult(self):
        """標記為困難案例"""
        if not self.samples:
            return

        # 在當前標註中添加困難標記
        annotation = {
            'toxicity': self.toxicity_var.get(),
            'bullying': self.bullying_var.get(),
            'role': self.role_var.get(),
            'emotion': self.emotion_var.get(),
            'emotion_strength': self.emotion_strength_var.get(),
            'note': self.note_text.get(1.0, tk.END).strip(),
            'timestamp': datetime.now().isoformat(),
            'annotator_id': self.annotator_id,
            'difficult': True,
            'requires_review': True
        }

        self.annotations[str(self.current_index)] = annotation
        messagebox.showinfo("標記", "已標記為困難案例，需要專家審核")

        # 自動跳到下一個樣本
        if self.current_index < self.total_samples - 1:
            self.next_sample()

    def update_progress(self):
        """更新進度顯示"""
        if self.total_samples > 0:
            progress = (self.completed_samples / self.total_samples) * 100
            self.progress_bar['value'] = progress

            remaining = self.total_samples - self.completed_samples
            self.stats_label.config(text=f"已完成: {self.completed_samples} | 剩餘: {remaining}")

    def load_settings(self):
        """載入設定"""
        # 可以在這裡載入用戶設定
        pass

    def on_closing(self):
        """視窗關閉時的處理"""
        if self.annotations:
            if messagebox.askyesno("確認", "是否要儲存當前進度？"):
                self.save_progress()
        self.root.destroy()


def main():
    """主函數"""
    parser = argparse.ArgumentParser(description="網路霸凌標註介面")
    parser.add_argument("--samples", help="樣本檔案路徑")
    parser.add_argument("--progress", help="進度檔案路徑")
    args = parser.parse_args()

    # 建立主視窗
    root = tk.Tk()

    # 建立標註介面
    app = AnnotationInterface(root)

    # 設定關閉事件
    root.protocol("WM_DELETE_WINDOW", app.on_closing)

    # 如果提供了檔案參數，自動載入
    if args.samples:
        # 這裡可以實現自動載入功能
        pass

    # 啟動介面
    root.mainloop()


if __name__ == "__main__":
    main()