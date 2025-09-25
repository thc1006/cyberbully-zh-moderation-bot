#!/usr/bin/env python3
"""
驗證專案中所有 Markdown 檔案的內部連結有效性
"""

import os
import re
from pathlib import Path
from typing import List, Tuple

def find_markdown_files(root_dir: str) -> List[Path]:
    """找出所有 Markdown 檔案"""
    md_files = []
    for root, dirs, files in os.walk(root_dir):
        # 跳過隱藏目錄和 node_modules
        dirs[:] = [d for d in dirs if not d.startswith('.') and d != 'node_modules']
        for file in files:
            if file.endswith('.md'):
                md_files.append(Path(root) / file)
    return md_files

def extract_links(file_path: Path) -> List[Tuple[str, int]]:
    """從 Markdown 檔案中提取所有連結"""
    links = []
    # 匹配 Markdown 連結格式 [text](link)
    link_pattern = re.compile(r'\[([^\]]+)\]\(([^)]+)\)')

    with open(file_path, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f, 1):
            for match in link_pattern.finditer(line):
                link = match.group(2)
                # 只檢查內部連結（不是 http/https 開頭）
                if not link.startswith(('http://', 'https://', 'mailto:', '#')):
                    links.append((link, line_num))
    return links

def verify_link(base_file: Path, link: str) -> bool:
    """驗證連結是否有效"""
    # 處理錨點連結
    if '#' in link:
        link = link.split('#')[0]
        if not link:  # 純錨點連結
            return True

    # 計算目標路徑
    base_dir = base_file.parent
    target = (base_dir / link).resolve()

    return target.exists()

def main():
    """主函數"""
    root_dir = Path.cwd()
    print(f"[SCAN] 掃描專案目錄: {root_dir}\n")

    # 找出所有 Markdown 檔案
    md_files = find_markdown_files(root_dir)
    print(f"找到 {len(md_files)} 個 Markdown 檔案\n")

    broken_links = []
    total_links = 0

    for md_file in md_files:
        rel_path = md_file.relative_to(root_dir)
        links = extract_links(md_file)

        if links:
            for link, line_num in links:
                total_links += 1
                if not verify_link(md_file, link):
                    broken_links.append({
                        'file': str(rel_path),
                        'line': line_num,
                        'link': link
                    })

    # 輸出結果
    print(f"[RESULT] 連結檢查結果")
    print(f"總共檢查了 {total_links} 個內部連結\n")

    if broken_links:
        print(f"[ERROR] 發現 {len(broken_links)} 個失效連結:\n")
        for broken in broken_links:
            print(f"  檔案: {broken['file']}")
            print(f"  行號: {broken['line']}")
            print(f"  連結: {broken['link']}")
            print()
    else:
        print("[SUCCESS] 所有內部連結都有效！")

    return 0 if not broken_links else 1

if __name__ == "__main__":
    exit(main())