"""
同義詞替換模組

使用中文同義詞庫進行詞彙替換，同時保護關鍵霸凌詞彙不被替換。
"""

import json
import logging
import random
from pathlib import Path
from typing import Dict, List, Optional, Set

import jieba

logger = logging.getLogger(__name__)


class SynonymReplacer:
    """同義詞替換器"""

    def __init__(
        self,
        synonym_dict_path: Optional[str] = None,
        bullying_keywords_path: Optional[str] = None,
        replacement_rate: float = 0.1,
        preserve_bullying_terms: bool = True,
    ):
        """
        初始化同義詞替換器

        Args:
            synonym_dict_path: 同義詞字典路徑
            bullying_keywords_path: 霸凌關鍵詞路徑
            replacement_rate: 替換比例 (0-1)
            preserve_bullying_terms: 是否保護霸凌詞彙不被替換
        """
        self.replacement_rate = replacement_rate
        self.preserve_bullying_terms = preserve_bullying_terms

        # 載入同義詞字典
        self.synonym_dict = self._load_synonym_dict(synonym_dict_path)

        # 載入霸凌關鍵詞
        self.bullying_keywords = self._load_bullying_keywords(bullying_keywords_path)

        logger.info(f"已載入 {len(self.synonym_dict)} 個同義詞組")
        logger.info(f"已載入 {len(self.bullying_keywords)} 個霸凌關鍵詞")

    def _load_synonym_dict(self, path: Optional[str]) -> Dict[str, List[str]]:
        """載入同義詞字典"""
        if path and Path(path).exists():
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)

        # 預設同義詞字典 (可擴展)
        return {
            "很": ["非常", "特別", "極其", "相當"],
            "好": ["棒", "讚", "優秀", "不錯"],
            "壞": ["差", "糟", "爛", "惡劣"],
            "說": ["講", "談", "表示", "提到"],
            "看": ["瞧", "觀察", "注視", "凝視"],
            "想": ["思考", "考慮", "認為", "覺得"],
            "做": ["進行", "執行", "實施", "完成"],
            "去": ["前往", "到達", "走向", "赴"],
            "來": ["到來", "抵達", "過來", "歸來"],
            "大": ["巨大", "龐大", "寬廣", "廣闊"],
            "小": ["微小", "細小", "迷你", "袖珍"],
            "快": ["迅速", "快速", "敏捷", "急速"],
            "慢": ["緩慢", "遲緩", "徐緩", "緩慢"],
            "高興": ["開心", "快樂", "愉快", "歡喜"],
            "難過": ["傷心", "悲傷", "憂愁", "沮喪"],
            "生氣": ["憤怒", "惱火", "氣憤", "發火"],
            "害怕": ["恐懼", "畏懼", "驚慌", "膽怯"],
        }

    def _load_bullying_keywords(self, path: Optional[str]) -> Set[str]:
        """載入霸凌關鍵詞"""
        if path and Path(path).exists():
            with open(path, "r", encoding="utf-8") as f:
                keywords = json.load(f)
                return set(keywords) if isinstance(keywords, list) else set(keywords.keys())

        # 預設霸凌關鍵詞 (根據 COLD 數據集特徵)
        return {
            "死",
            "殺",
            "滾",
            "廢物",
            "垃圾",
            "白癡",
            "智障",
            "腦殘",
            "去死",
            "找死",
            "該死",
            "活該",
            "報應",
            "詛咒",
            "排擠",
            "孤立",
            "欺負",
            "霸凌",
            "威脅",
            "恐嚇",
            "胖子",
            "醜八怪",
            "窮鬼",
            "土包子",
            "鄉巴佬",
            "閉嘴",
            "住口",
            "別說話",
            "沒人問你",
            "誰理你",
            "沒用",
            "無能",
            "失敗者",
            "魯蛇",
            "邊緣人",
        }

    def augment(
        self, text: str, num_augmented: int = 1, custom_replacement_rate: Optional[float] = None
    ) -> List[str]:
        """
        對文本進行同義詞替換增強

        Args:
            text: 原始文本
            num_augmented: 生成增強樣本數量
            custom_replacement_rate: 自定義替換比例

        Returns:
            增強後的文本列表
        """
        if not text.strip():
            return [text] * num_augmented

        augmented_texts = []
        replacement_rate = custom_replacement_rate or self.replacement_rate

        for _ in range(num_augmented):
            try:
                augmented_text = self._replace_synonyms(text, replacement_rate)
                augmented_texts.append(augmented_text)
            except Exception as e:
                logger.warning(f"同義詞替換失敗: {e}")
                augmented_texts.append(text)

        return augmented_texts

    def _replace_synonyms(self, text: str, replacement_rate: float) -> str:
        """執行同義詞替換"""
        # 分詞
        words = list(jieba.cut(text))

        # 計算需要替換的詞數
        num_replacements = max(1, int(len(words) * replacement_rate))

        # 找出可替換的詞
        replaceable_indices = []
        for i, word in enumerate(words):
            if self._is_replaceable(word):
                replaceable_indices.append(i)

        if not replaceable_indices:
            return text

        # 隨機選擇要替換的詞
        indices_to_replace = random.sample(
            replaceable_indices, min(num_replacements, len(replaceable_indices))
        )

        # 執行替換
        new_words = words.copy()
        for idx in indices_to_replace:
            word = words[idx]
            if word in self.synonym_dict:
                synonyms = self.synonym_dict[word]
                new_words[idx] = random.choice(synonyms)

        return "".join(new_words)

    def _is_replaceable(self, word: str) -> bool:
        """判斷詞是否可替換"""
        # 檢查是否為霸凌關鍵詞
        if self.preserve_bullying_terms and word in self.bullying_keywords:
            return False

        # 檢查是否在同義詞字典中
        if word not in self.synonym_dict:
            return False

        # 過濾太短的詞
        if len(word) < 1:
            return False

        return True

    def add_synonyms(self, word: str, synonyms: List[str]) -> None:
        """添加新的同義詞組"""
        if word not in self.synonym_dict:
            self.synonym_dict[word] = []
        self.synonym_dict[word].extend(synonyms)
        self.synonym_dict[word] = list(set(self.synonym_dict[word]))  # 去重

    def add_bullying_keyword(self, keyword: str) -> None:
        """添加霸凌關鍵詞"""
        self.bullying_keywords.add(keyword)

    def save_synonym_dict(self, path: str) -> None:
        """保存同義詞字典"""
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.synonym_dict, f, ensure_ascii=False, indent=2)

    def save_bullying_keywords(self, path: str) -> None:
        """保存霸凌關鍵詞"""
        keywords_list = list(self.bullying_keywords)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(keywords_list, f, ensure_ascii=False, indent=2)

    def get_stats(self) -> Dict[str, int]:
        """獲取統計信息"""
        return {
            "synonym_groups": len(self.synonym_dict),
            "total_synonyms": sum(len(synonyms) for synonyms in self.synonym_dict.values()),
            "bullying_keywords": len(self.bullying_keywords),
            "replacement_rate": self.replacement_rate,
        }


def create_default_synonym_dict(output_path: str) -> None:
    """創建預設同義詞字典文件"""
    replacer = SynonymReplacer()
    replacer.save_synonym_dict(output_path)
    print(f"預設同義詞字典已保存至: {output_path}")


def create_default_bullying_keywords(output_path: str) -> None:
    """創建預設霸凌關鍵詞文件"""
    replacer = SynonymReplacer()
    replacer.save_bullying_keywords(output_path)
    print(f"預設霸凌關鍵詞已保存至: {output_path}")


if __name__ == "__main__":
    # 測試範例
    replacer = SynonymReplacer()

    test_texts = ["你很笨，去死吧！", "這個人說話很難聽", "我覺得你做得很好", "別再霸凌其他同學了"]

    for text in test_texts:
        print(f"原文: {text}")
        augmented = replacer.augment(text, num_augmented=3)
        for i, aug_text in enumerate(augmented, 1):
            print(f"增強{i}: {aug_text}")
        print("-" * 50)
