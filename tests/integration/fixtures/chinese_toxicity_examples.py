"""
中文毒性檢測測試資料
包含各種中文網路霸凌與毒性內容的測試案例
用於整合測試、效能測試與回歸測試

資料分類：
- 正面內容 (none)
- 輕微毒性 (toxic)
- 嚴重毒性 (severe)
- 霸凌行為 (harassment/threat)
- 情緒分類 (pos/neu/neg)
- 角色識別 (perpetrator/victim/bystander)
"""

import json
from pathlib import Path
from typing import List, Dict, Any

# 基礎正面內容
POSITIVE_EXAMPLES = [
    {
        "text": "今天天氣真好，心情很棒！",
        "expected": {
            "toxicity": "none",
            "bullying": "none",
            "emotion": "pos",
            "emotion_strength": 3,
            "role": "none"
        },
        "category": "positive_weather"
    },
    {
        "text": "謝謝大家的支持和鼓勵，我會繼續努力的",
        "expected": {
            "toxicity": "none",
            "bullying": "none",
            "emotion": "pos",
            "emotion_strength": 3,
            "role": "none"
        },
        "category": "positive_gratitude"
    },
    {
        "text": "這個想法很有創意，我很喜歡",
        "expected": {
            "toxicity": "none",
            "bullying": "none",
            "emotion": "pos",
            "emotion_strength": 2,
            "role": "none"
        },
        "category": "positive_praise"
    },
    {
        "text": "祝你生日快樂，願你天天開心",
        "expected": {
            "toxicity": "none",
            "bullying": "none",
            "emotion": "pos",
            "emotion_strength": 3,
            "role": "none"
        },
        "category": "positive_blessing"
    },
    {
        "text": "加油！你一定可以做到的",
        "expected": {
            "toxicity": "none",
            "bullying": "none",
            "emotion": "pos",
            "emotion_strength": 3,
            "role": "none"
        },
        "category": "positive_encouragement"
    }
]

# 中性內容
NEUTRAL_EXAMPLES = [
    {
        "text": "今天的天氣如何？",
        "expected": {
            "toxicity": "none",
            "bullying": "none",
            "emotion": "neu",
            "emotion_strength": 0,
            "role": "none"
        },
        "category": "neutral_question"
    },
    {
        "text": "請問這個問題怎麼解決？",
        "expected": {
            "toxicity": "none",
            "bullying": "none",
            "emotion": "neu",
            "emotion_strength": 0,
            "role": "none"
        },
        "category": "neutral_inquiry"
    },
    {
        "text": "會議安排在下午三點",
        "expected": {
            "toxicity": "none",
            "bullying": "none",
            "emotion": "neu",
            "emotion_strength": 0,
            "role": "none"
        },
        "category": "neutral_information"
    },
    {
        "text": "資料已經準備好了",
        "expected": {
            "toxicity": "none",
            "bullying": "none",
            "emotion": "neu",
            "emotion_strength": 0,
            "role": "none"
        },
        "category": "neutral_status"
    }
]

# 輕微毒性內容
MILD_TOXIC_EXAMPLES = [
    {
        "text": "你真的很笨耶，怎麼連這個都不會",
        "expected": {
            "toxicity": "toxic",
            "bullying": "harassment",
            "emotion": "neg",
            "emotion_strength": 3,
            "role": "perpetrator"
        },
        "category": "mild_insult"
    },
    {
        "text": "廢物一個，滾開",
        "expected": {
            "toxicity": "toxic",
            "bullying": "harassment",
            "emotion": "neg",
            "emotion_strength": 4,
            "role": "perpetrator"
        },
        "category": "mild_dismissal"
    },
    {
        "text": "你的想法很蠢，完全沒有邏輯",
        "expected": {
            "toxicity": "toxic",
            "bullying": "harassment",
            "emotion": "neg",
            "emotion_strength": 3,
            "role": "perpetrator"
        },
        "category": "mild_criticism"
    },
    {
        "text": "真是個討厭的傢伙",
        "expected": {
            "toxicity": "toxic",
            "bullying": "harassment",
            "emotion": "neg",
            "emotion_strength": 3,
            "role": "perpetrator"
        },
        "category": "mild_dislike"
    },
    {
        "text": "你怎麼這麼煩人？閉嘴吧",
        "expected": {
            "toxicity": "toxic",
            "bullying": "harassment",
            "emotion": "neg",
            "emotion_strength": 3,
            "role": "perpetrator"
        },
        "category": "mild_annoyance"
    }
]

# 嚴重毒性內容
SEVERE_TOXIC_EXAMPLES = [
    {
        "text": "我要殺了你，你給我等著",
        "expected": {
            "toxicity": "severe",
            "bullying": "threat",
            "emotion": "neg",
            "emotion_strength": 4,
            "role": "perpetrator"
        },
        "category": "death_threat"
    },
    {
        "text": "你最好去死一死，沒人會想念你的",
        "expected": {
            "toxicity": "severe",
            "bullying": "threat",
            "emotion": "neg",
            "emotion_strength": 4,
            "role": "perpetrator"
        },
        "category": "suicide_encouragement"
    },
    {
        "text": "我知道你住在哪裡，小心點",
        "expected": {
            "toxicity": "severe",
            "bullying": "threat",
            "emotion": "neg",
            "emotion_strength": 4,
            "role": "perpetrator"
        },
        "category": "stalking_threat"
    },
    {
        "text": "你全家都該死，我要報復你們",
        "expected": {
            "toxicity": "severe",
            "bullying": "threat",
            "emotion": "neg",
            "emotion_strength": 4,
            "role": "perpetrator"
        },
        "category": "family_threat"
    },
    {
        "text": "我會讓你後悔來到這個世界",
        "expected": {
            "toxicity": "severe",
            "bullying": "threat",
            "emotion": "neg",
            "emotion_strength": 4,
            "role": "perpetrator"
        },
        "category": "severe_intimidation"
    }
]

# 受害者角色內容
VICTIM_EXAMPLES = [
    {
        "text": "請大家幫幫我，我真的很害怕",
        "expected": {
            "toxicity": "none",
            "bullying": "none",
            "emotion": "neg",
            "emotion_strength": 3,
            "role": "victim"
        },
        "category": "help_seeking"
    },
    {
        "text": "我被他們欺負了，不知道該怎麼辦",
        "expected": {
            "toxicity": "none",
            "bullying": "none",
            "emotion": "neg",
            "emotion_strength": 3,
            "role": "victim"
        },
        "category": "bullying_report"
    },
    {
        "text": "他們一直說我的壞話，我很受傷",
        "expected": {
            "toxicity": "none",
            "bullying": "none",
            "emotion": "neg",
            "emotion_strength": 3,
            "role": "victim"
        },
        "category": "emotional_damage"
    },
    {
        "text": "為什麼大家都不理我？我做錯了什麼？",
        "expected": {
            "toxicity": "none",
            "bullying": "none",
            "emotion": "neg",
            "emotion_strength": 2,
            "role": "victim"
        },
        "category": "isolation_confusion"
    }
]

# 旁觀者角色內容
BYSTANDER_EXAMPLES = [
    {
        "text": "我看到他們在欺負人，但不知道該不該管",
        "expected": {
            "toxicity": "none",
            "bullying": "none",
            "emotion": "neu",
            "emotion_strength": 1,
            "role": "bystander"
        },
        "category": "witness_hesitation"
    },
    {
        "text": "這樣做不對，你們應該停止",
        "expected": {
            "toxicity": "none",
            "bullying": "none",
            "emotion": "neu",
            "emotion_strength": 1,
            "role": "bystander"
        },
        "category": "intervention_attempt"
    },
    {
        "text": "大家都在看這場衝突，但沒人說話",
        "expected": {
            "toxicity": "none",
            "bullying": "none",
            "emotion": "neu",
            "emotion_strength": 0,
            "role": "bystander"
        },
        "category": "passive_observation"
    }
]

# 網路霸凌特殊案例
CYBERBULLYING_EXAMPLES = [
    {
        "text": "大家都不喜歡你，你知道嗎？",
        "expected": {
            "toxicity": "toxic",
            "bullying": "harassment",
            "emotion": "neg",
            "emotion_strength": 3,
            "role": "perpetrator"
        },
        "category": "social_exclusion"
    },
    {
        "text": "我要把你的醜照發到網上讓大家看",
        "expected": {
            "toxicity": "severe",
            "bullying": "threat",
            "emotion": "neg",
            "emotion_strength": 4,
            "role": "perpetrator"
        },
        "category": "image_threat"
    },
    {
        "text": "如果你不照我說的做，我就把你的秘密告訴大家",
        "expected": {
            "toxicity": "severe",
            "bullying": "threat",
            "emotion": "neg",
            "emotion_strength": 4,
            "role": "perpetrator"
        },
        "category": "blackmail"
    },
    {
        "text": "我會在所有群組裡說你的壞話",
        "expected": {
            "toxicity": "toxic",
            "bullying": "harassment",
            "emotion": "neg",
            "emotion_strength": 3,
            "role": "perpetrator"
        },
        "category": "reputation_attack"
    }
]

# 邊界案例（模糊的毒性判斷）
EDGE_CASES = [
    {
        "text": "這個決定真的讓人很生氣",
        "expected": {
            "toxicity": "none",
            "bullying": "none",
            "emotion": "neg",
            "emotion_strength": 2,
            "role": "none"
        },
        "category": "legitimate_anger"
    },
    {
        "text": "我對此感到非常失望",
        "expected": {
            "toxicity": "none",
            "bullying": "none",
            "emotion": "neg",
            "emotion_strength": 2,
            "role": "none"
        },
        "category": "disappointment"
    },
    {
        "text": "你能不能認真一點？",
        "expected": {
            "toxicity": "none",
            "bullying": "none",
            "emotion": "neu",
            "emotion_strength": 1,
            "role": "none"
        },
        "category": "mild_criticism"
    },
    {
        "text": "這樣的行為真的不可取",
        "expected": {
            "toxicity": "none",
            "bullying": "none",
            "emotion": "neu",
            "emotion_strength": 1,
            "role": "none"
        },
        "category": "constructive_criticism"
    }
]

# 特殊字元與表情符號案例
SPECIAL_CHAR_EXAMPLES = [
    {
        "text": "😊😢😡 各種情緒測試",
        "expected": {
            "toxicity": "none",
            "bullying": "none",
            "emotion": "neu",
            "emotion_strength": 1,
            "role": "none"
        },
        "category": "emoji_mixed"
    },
    {
        "text": "@#$%^&*() 特殊符號測試",
        "expected": {
            "toxicity": "none",
            "bullying": "none",
            "emotion": "neu",
            "emotion_strength": 0,
            "role": "none"
        },
        "category": "special_symbols"
    },
    {
        "text": "ａｂｃｄｅ 全形英文測試",
        "expected": {
            "toxicity": "none",
            "bullying": "none",
            "emotion": "neu",
            "emotion_strength": 0,
            "role": "none"
        },
        "category": "fullwidth_chars"
    },
    {
        "text": "🚨⚠️💀 你這個混蛋",
        "expected": {
            "toxicity": "toxic",
            "bullying": "harassment",
            "emotion": "neg",
            "emotion_strength": 3,
            "role": "perpetrator"
        },
        "category": "emoji_with_toxicity"
    }
]

# 繁簡體混合案例
TRADITIONAL_SIMPLIFIED_EXAMPLES = [
    {
        "text": "繁體中文 vs 简体中文 測試",
        "expected": {
            "toxicity": "none",
            "bullying": "none",
            "emotion": "neu",
            "emotion_strength": 0,
            "role": "none"
        },
        "category": "mixed_scripts"
    },
    {
        "text": "你這個笨蛋（繁體）",
        "expected": {
            "toxicity": "toxic",
            "bullying": "harassment",
            "emotion": "neg",
            "emotion_strength": 3,
            "role": "perpetrator"
        },
        "category": "traditional_toxicity"
    },
    {
        "text": "你这个笨蛋（简体）",
        "expected": {
            "toxicity": "toxic",
            "bullying": "harassment",
            "emotion": "neg",
            "emotion_strength": 3,
            "role": "perpetrator"
        },
        "category": "simplified_toxicity"
    }
]

# 整合所有測試資料
ALL_EXAMPLES = (
    POSITIVE_EXAMPLES +
    NEUTRAL_EXAMPLES +
    MILD_TOXIC_EXAMPLES +
    SEVERE_TOXIC_EXAMPLES +
    VICTIM_EXAMPLES +
    BYSTANDER_EXAMPLES +
    CYBERBULLYING_EXAMPLES +
    EDGE_CASES +
    SPECIAL_CHAR_EXAMPLES +
    TRADITIONAL_SIMPLIFIED_EXAMPLES
)


def get_examples_by_category(category: str) -> List[Dict[str, Any]]:
    """根據類別取得測試案例"""
    return [ex for ex in ALL_EXAMPLES if ex.get("category") == category]


def get_examples_by_toxicity(toxicity_level: str) -> List[Dict[str, Any]]:
    """根據毒性等級取得測試案例"""
    return [ex for ex in ALL_EXAMPLES
            if ex["expected"]["toxicity"] == toxicity_level]


def get_examples_by_emotion(emotion: str) -> List[Dict[str, Any]]:
    """根據情緒分類取得測試案例"""
    return [ex for ex in ALL_EXAMPLES
            if ex["expected"]["emotion"] == emotion]


def get_examples_by_role(role: str) -> List[Dict[str, Any]]:
    """根據角色取得測試案例"""
    return [ex for ex in ALL_EXAMPLES
            if ex["expected"]["role"] == role]


def save_examples_to_file(
    file_path: Path,
    examples: List[Dict[str,
    Any]] = None
):
    """儲存測試案例到檔案"""
    if examples is None:
        examples = ALL_EXAMPLES

    file_path.parent.mkdir(parents=True, exist_ok=True)

    with open(file_path, "w", encoding="utf-8") as f:
        for example in examples:
            f.write(json.dumps(example, ensure_ascii=False) + "\n")


def load_examples_from_file(file_path: Path) -> List[Dict[str, Any]]:
    """從檔案載入測試案例"""
    examples = []
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                examples.append(json.loads(line.strip()))
    return examples


def get_balanced_test_set(size: int = 50) -> List[Dict[str, Any]]:
    """取得平衡的測試集合"""
    balanced_set = []

    # 確保各種類別都有代表
    categories = {
        "none": get_examples_by_toxicity("none"),
        "toxic": get_examples_by_toxicity("toxic"),
        "severe": get_examples_by_toxicity("severe")
    }

    # 每種類別取相等數量
    per_category = size // 3

    for category, examples in categories.items():
        if len(examples) >= per_category:
            balanced_set.extend(examples[:per_category])
        else:
            balanced_set.extend(examples)

    # 如果還需要更多，從邊界案例補充
    remaining = size - len(balanced_set)
    if remaining > 0:
        balanced_set.extend(EDGE_CASES[:remaining])

    return balanced_set[:size]


# 統計資訊
def get_dataset_statistics():
    """取得資料集統計資訊"""
    stats = {
        "total_examples": len(ALL_EXAMPLES),
        "by_toxicity": {},
        "by_emotion": {},
        "by_role": {},
        "by_bullying": {},
        "categories": {}
    }

    # 統計各項目
    for example in ALL_EXAMPLES:
        expected = example["expected"]
        category = example.get("category", "unknown")

        # 毒性統計
        toxicity = expected["toxicity"]
        stats["by_to"
            "xicity"][toxicity] = stats[

        # 情緒統計
        emotion = expected["emotion"]
        stats["by_emotion"][emotion] = stats["by_emotion"].get(emotion, 0) + 1

        # 角色統計
        role = expected["role"]
        stats["by_role"][role] = stats["by_role"].get(role, 0) + 1

        # 霸凌統計
        bullying = expected["bullying"]
        stats["by_bu"
            "llying"][bullying] = stats[

        # 類別統計
        stats["categ"
            "ories"][category] = stats[

    return stats


if __name__ == "__main__":
    # 輸出統計資訊
    stats = get_dataset_statistics()
    print("中文毒性檢測測試資料集統計:")
    print(f"總數: {stats['total_examples']}")
    print(f"毒性分佈: {stats['by_toxicity']}")
    print(f"情緒分佈: {stats['by_emotion']}")
    print(f"角色分佈: {stats['by_role']}")
    print(f"霸凌分佈: {stats['by_bullying']}")
    print(f"類別數量: {len(stats['categories'])}")
