"""
回譯增強模組

使用中文 → 英文 → 中文的回譯過程來生成語義相似但表達不同的文本。
"""

import asyncio
import time
from typing import List, Optional, Dict, Any
import logging
from abc import ABC, abstractmethod

logger = logging.getLogger(__name__)

try:
    from googletrans import Translator as GoogleTranslator
    GOOGLE_AVAILABLE = True
except ImportError:
    GOOGLE_AVAILABLE = False
    logger.warning("googletrans 未安裝，Google 翻譯功能不可用")

try:
    import requests
    REQUESTS_AVAILABLE = True
except ImportError:
    REQUESTS_AVAILABLE = False
    logger.warning("requests 未安裝，API 翻譯功能不可用")


class BaseTranslator(ABC):
    """翻譯器基類"""

    @abstractmethod
    def translate(self, text: str, src_lang: str, dest_lang: str) -> str:
        """翻譯文本"""
        pass


class GoogleTranslator_Wrapper(BaseTranslator):
    """Google 翻譯器包裝類"""

    def __init__(self):
        if not GOOGLE_AVAILABLE:
            raise ImportError("需要安裝 googletrans: pip install googletrans==4.0.0rc1")
        self.translator = GoogleTranslator()
        self.rate_limit_delay = 1.0  # 避免速率限制

    def translate(self, text: str, src_lang: str, dest_lang: str) -> str:
        """翻譯文本"""
        try:
            time.sleep(self.rate_limit_delay)  # 避免速率限制
            result = self.translator.translate(text, src=src_lang, dest=dest_lang)
            return result.text
        except Exception as e:
            logger.warning(f"Google 翻譯失敗: {e}")
            return text


class BaiduTranslator(BaseTranslator):
    """百度翻譯器 (需要 API 密鑰)"""

    def __init__(self, app_id: str, secret_key: str):
        if not REQUESTS_AVAILABLE:
            raise ImportError("需要安裝 requests: pip install requests")

        self.app_id = app_id
        self.secret_key = secret_key
        self.base_url = "https://fanyi-api.baidu.com/api/trans/vip/translate"

    def translate(self, text: str, src_lang: str, dest_lang: str) -> str:
        """使用百度翻譯 API"""
        import hashlib
        import random

        salt = str(random.randint(32768, 65536))
        sign_str = self.app_id + text + salt + self.secret_key
        sign = hashlib.md5(sign_str.encode('utf-8')).hexdigest()

        params = {
            'q': text,
            'from': self._convert_lang_code(src_lang),
            'to': self._convert_lang_code(dest_lang),
            'appid': self.app_id,
            'salt': salt,
            'sign': sign
        }

        try:
            response = requests.get(self.base_url, params=params)
            result = response.json()

            if 'trans_result' in result:
                return result['trans_result'][0]['dst']
            else:
                logger.warning(f"百度翻譯錯誤: {result}")
                return text
        except Exception as e:
            logger.warning(f"百度翻譯失敗: {e}")
            return text

    def _convert_lang_code(self, lang: str) -> str:
        """轉換語言代碼為百度格式"""
        mapping = {
            'zh': 'zh',
            'zh-cn': 'zh',
            'en': 'en',
            'ja': 'jp',
            'ko': 'kor'
        }
        return mapping.get(lang.lower(), lang)


class MockTranslator(BaseTranslator):
    """模擬翻譯器 (用於測試和離線環境)"""

    def __init__(self):
        # 模擬翻譯結果的簡單規則
        self.zh_to_en_rules = {
            "你": "you",
            "很": "very",
            "笨": "stupid",
            "去死": "go die",
            "吧": "",
            "！": "!",
            "。": ".",
            "的": "of",
            "是": "is",
            "不": "not"
        }

        self.en_to_zh_rules = {v: k for k, v in self.zh_to_en_rules.items() if v}

    def translate(self, text: str, src_lang: str, dest_lang: str) -> str:
        """模擬翻譯過程"""
        if src_lang == 'zh' and dest_lang == 'en':
            # 簡單的中文到英文轉換
            result = text
            for zh, en in self.zh_to_en_rules.items():
                result = result.replace(zh, en)
            return result
        elif src_lang == 'en' and dest_lang == 'zh':
            # 簡單的英文到中文轉換
            result = text
            for en, zh in self.en_to_zh_rules.items():
                result = result.replace(en, zh)
            return result
        else:
            # 引入一些變化來模擬翻譯效果
            variations = {
                "很": ["非常", "特別", "相當"],
                "說": ["講", "表示", "提到"],
                "好": ["棒", "不錯", "優秀"],
                "壞": ["差", "糟糕", "不好"]
            }

            result = text
            for original, alternatives in variations.items():
                if original in result:
                    import random
                    replacement = random.choice(alternatives)
                    result = result.replace(original, replacement, 1)

            return result


class BackTranslator:
    """回譯增強器"""

    def __init__(
        self,
        translator: Optional[BaseTranslator] = None,
        intermediate_langs: List[str] = None,
        preserve_length_ratio: float = 0.8,
        max_retries: int = 3
    ):
        """
        初始化回譯器

        Args:
            translator: 翻譯器實例
            intermediate_langs: 中間語言列表
            preserve_length_ratio: 保留長度比例閾值
            max_retries: 最大重試次數
        """
        self.translator = translator or self._create_default_translator()
        self.intermediate_langs = intermediate_langs or ['en', 'ja', 'ko']
        self.preserve_length_ratio = preserve_length_ratio
        self.max_retries = max_retries

        logger.info(f"回譯器初始化完成，使用中間語言: {self.intermediate_langs}")

    def _create_default_translator(self) -> BaseTranslator:
        """創建預設翻譯器"""
        if GOOGLE_AVAILABLE:
            try:
                return GoogleTranslator_Wrapper()
            except Exception as e:
                logger.warning(f"Google 翻譯器初始化失敗: {e}")

        logger.info("使用模擬翻譯器")
        return MockTranslator()

    def augment(
        self,
        text: str,
        num_augmented: int = 1,
        custom_intermediate_langs: Optional[List[str]] = None
    ) -> List[str]:
        """
        對文本進行回譯增強

        Args:
            text: 原始文本
            num_augmented: 生成增強樣本數量
            custom_intermediate_langs: 自定義中間語言

        Returns:
            回譯增強後的文本列表
        """
        if not text.strip():
            return [text] * num_augmented

        intermediate_langs = custom_intermediate_langs or self.intermediate_langs
        augmented_texts = []

        for i in range(num_augmented):
            try:
                # 選擇中間語言 (輪換使用)
                intermediate_lang = intermediate_langs[i % len(intermediate_langs)]

                # 執行回譯
                back_translated = self._back_translate(text, intermediate_lang)

                # 質量檢查
                if self._quality_check(text, back_translated):
                    augmented_texts.append(back_translated)
                else:
                    logger.warning(f"回譯質量不佳，使用原文: {text[:50]}...")
                    augmented_texts.append(text)

            except Exception as e:
                logger.warning(f"回譯失敗: {e}")
                augmented_texts.append(text)

        return augmented_texts

    def _back_translate(self, text: str, intermediate_lang: str) -> str:
        """執行回譯操作"""
        retries = 0
        while retries < self.max_retries:
            try:
                # 第一步: 中文 → 中間語言
                intermediate_text = self.translator.translate(text, 'zh', intermediate_lang)

                if not intermediate_text or intermediate_text == text:
                    raise ValueError(f"第一步翻譯失敗或無變化")

                # 第二步: 中間語言 → 中文
                back_translated = self.translator.translate(intermediate_text, intermediate_lang, 'zh')

                if not back_translated:
                    raise ValueError(f"第二步翻譯失敗")

                logger.debug(f"回譯路徑: {text} → {intermediate_text} → {back_translated}")
                return back_translated

            except Exception as e:
                retries += 1
                logger.warning(f"回譯重試 {retries}/{self.max_retries}: {e}")
                if retries >= self.max_retries:
                    raise e
                time.sleep(1)  # 重試前等待

        return text

    def _quality_check(self, original: str, back_translated: str) -> bool:
        """檢查回譯質量"""
        # 長度檢查
        if len(back_translated) < len(original) * self.preserve_length_ratio:
            logger.debug(f"回譯文本過短: {len(back_translated)} < {len(original) * self.preserve_length_ratio}")
            return False

        # 相似度檢查 (簡單版本)
        if back_translated == original:
            logger.debug("回譯文本與原文完全相同")
            return False

        # 檢查是否包含基本語義詞彙
        if len(original) > 10:  # 只對較長文本檢查
            original_chars = set(original)
            back_translated_chars = set(back_translated)

            # 計算字符重疊率
            overlap_ratio = len(original_chars & back_translated_chars) / len(original_chars)
            if overlap_ratio < 0.3:  # 重疊率過低
                logger.debug(f"字符重疊率過低: {overlap_ratio}")
                return False

        return True

    def get_supported_languages(self) -> List[str]:
        """獲取支持的中間語言列表"""
        return self.intermediate_langs.copy()

    def add_intermediate_language(self, lang_code: str) -> None:
        """添加中間語言"""
        if lang_code not in self.intermediate_langs:
            self.intermediate_langs.append(lang_code)
            logger.info(f"已添加中間語言: {lang_code}")

    def set_translator(self, translator: BaseTranslator) -> None:
        """設置翻譯器"""
        self.translator = translator
        logger.info(f"已更新翻譯器: {type(translator).__name__}")

    def get_stats(self) -> Dict[str, Any]:
        """獲取統計信息"""
        return {
            "translator_type": type(self.translator).__name__,
            "intermediate_languages": self.intermediate_langs,
            "preserve_length_ratio": self.preserve_length_ratio,
            "max_retries": self.max_retries
        }


async def async_back_translate_batch(
    back_translator: BackTranslator,
    texts: List[str],
    num_augmented: int = 1
) -> List[List[str]]:
    """異步批量回譯"""
    async def translate_one(text: str) -> List[str]:
        return back_translator.augment(text, num_augmented)

    tasks = [translate_one(text) for text in texts]
    results = await asyncio.gather(*tasks, return_exceptions=True)

    # 處理異常
    final_results = []
    for i, result in enumerate(results):
        if isinstance(result, Exception):
            logger.error(f"批量回譯失敗 (索引 {i}): {result}")
            final_results.append([texts[i]] * num_augmented)
        else:
            final_results.append(result)

    return final_results


if __name__ == "__main__":
    # 測試範例
    back_translator = BackTranslator()

    test_texts = [
        "你真的很笨！",
        "別再說這種話了",
        "我覺得你做得很好",
        "這個想法不錯"
    ]

    print("=== 回譯增強測試 ===")
    for text in test_texts:
        print(f"原文: {text}")
        augmented = back_translator.augment(text, num_augmented=2)
        for i, aug_text in enumerate(augmented, 1):
            print(f"回譯{i}: {aug_text}")
        print("-" * 50)

    # 統計信息
    stats = back_translator.get_stats()
    print(f"統計信息: {stats}")