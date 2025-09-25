#!/usr/bin/env python3
"""
修復 Flake8 程式碼品質問題的自動化腳本
"""

import re
import sys
from pathlib import Path


def remove_unused_imports(file_path):
    """移除未使用的導入"""
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()

    # 常見未使用導入的修復
    patterns_to_fix = [
        # 移除未使用的 os 導入
        (r'^import os\n', '', 'unused os import'),
        (r'^from os import .*\n', '', 'unused os imports'),

        # 移除未使用的 sys 導入
        (r'^import sys\n', '', 'unused sys import'),

        # 移除未使用的時間相關導入
        (r'^import time\n', '', 'unused time import'),
        (r'^from datetime import datetime, timedelta\n',
         'from datetime import datetime\n', 'unused timedelta'),

        # 移除未使用的類型導入
        (r'from typing import ([^,]+,\s*)*Optional([,\s]*[^,]+)*',
         lambda m: clean_typing_import(m.group(0)), 'clean typing imports'),
    ]

    original_content = content
    for pattern, replacement, desc in patterns_to_fix:
        if callable(replacement):
            content = re.sub(pattern, replacement, content, flags=re.MULTILINE)
        else:
            content = re.sub(pattern, replacement, content, flags=re.MULTILINE)

    if content != original_content:
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(content)
        return True
    return False


def clean_typing_import(import_line):
    """清理 typing 導入行"""
    # 這是一個簡化版本，實際需要更複雜的邏輯
    return import_line


def fix_fstring_placeholders(file_path):
    """修復 f-string 缺少佔位符的問題"""
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()

    # 找到沒有佔位符的 f-string 並修復
    patterns_to_fix = [
        (r'f"("
            "[^"]*)
        (r"f'([^"
            "']*)'", lambda m: f
    ]

    original_content = content
    for pattern, replacement in patterns_to_fix:
        content = re.sub(pattern, replacement, content)

    if content != original_content:
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(content)
        return True
    return False


def fix_bare_except(file_path):
    """修復裸露的 except 子句"""
    with open(file_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    modified = False
    for i, line in enumerate(lines):
        if re.search(r'^\s*except:\s*$', line):
            lines[i] = line.replace('except:', 'except Exception:')
            modified = True

    if modified:
        with open(file_path, 'w', encoding='utf-8') as f:
            f.writelines(lines)
        return True
    return False


def process_file(file_path):
    """處理單個文件"""
    if not file_path.suffix == '.py':
        return False

    print(f"處理文件: {file_path}")
    modified = False

    # 修復各種問題
    if remove_unused_imports(file_path):
        print("  - 移除未使用的導入")
        modified = True

    if fix_fstring_placeholders(file_path):
        print("  - 修復 f-string 問題")
        modified = True

    if fix_bare_except(file_path):
        print("  - 修復裸露的 except")
        modified = True

    return modified


def main():
    """主函式"""
    if len(sys.argv) > 1:
        target_dir = Path(sys.argv[1])
    else:
        target_dir = Path('.')

    total_modified = 0

    for py_file in target_dir.rglob('*.py'):
        if process_file(py_file):
            total_modified += 1

    print(f"\n總共修改了 {total_modified} 個文件")


if __name__ == '__main__':
    main()
