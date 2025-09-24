"""
安全規則與回覆策略實作

根據 POLICY.md 定義的規則實作回覆分級、隱私保護與誤判處理機制
"""

import hashlib
import json
import logging
import re
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum, IntEnum
from typing import Any, Dict, List, Optional, Tuple

# 設定日誌
logger = logging.getLogger(__name__)


class ResponseLevel(IntEnum):
    """回覆分級"""

    NONE = 0  # 無需回應
    GENTLE_REMINDER = 1  # 溫和提示
    SOFT_INTERVENTION = 2  # 柔性勸阻
    RESOURCE_ESCALATION = 3  # 資源提供與升級
    SILENT_HANDOVER = 4  # 沉默或移交


class AppealStatus(Enum):
    """申訴狀態"""

    PENDING = "pending"
    UNDER_REVIEW = "under_review"
    APPROVED = "approved"
    REJECTED = "rejected"
    ESCALATED = "escalated"


@dataclass
class UserViolationHistory:
    """使用者違規歷史"""

    user_id: str
    violations: List[Dict[str, Any]] = field(default_factory=list)
    total_count: int = 0
    last_violation_time: Optional[datetime] = None
    appeal_count: int = 0

    def add_violation(self, level: ResponseLevel, scores: Dict[str, float]):
        """新增違規記錄"""
        self.violations.append(
            {
                "timestamp": datetime.now().isoformat(),
                "level": level.value,
                "scores": scores,
                "hash": hashlib.sha256(f"{level.value}_{scores}".encode()).hexdigest()
            }
        )
        self.total_count += 1
        self.last_violation_time = datetime.now()

        # 保留最近 10 筆記錄
        if len(self.violations) > 10:
            self.violations.pop(0)

    def get_recent_violations(self, hours: int = 24) -> List[Dict[str, Any]]:
        """取得最近的違規記錄"""
        cutoff = datetime.now() - timedelta(hours=hours)
        return [v for v in self.violations if datetime.fromisoformat(v["time"
            "stamp"]) > cutoff]


@dataclass
class ResponseStrategy:
    """回應策略"""

    level: ResponseLevel
    message: str
    resources: List[Dict[str, str]] = field(default_factory=list)
    notify_admin: bool = False
    log_detail: bool = True


@dataclass
class Appeal:
    """申訴案件"""

    appeal_id: str
    user_id: str
    event_hash: str
    reason: str
    status: AppealStatus
    created_at: datetime
    updated_at: datetime
    reviewer: Optional[str] = None
    resolution: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """轉換為字典"""
        return {
            "appeal_id": self.appeal_id,
            "user_id": self.user_id,
            "event_hash": self.event_hash,
            "reason": self.reason,
            "status": self.status.value,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
            "reviewer": self.reviewer,
            "resolution": self.resolution,
        }


class PIIHandler:
    """個人識別資訊處理器"""

    # PII 正則表達式模式
    PII_PATTERNS = {
        "email": (r"[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}", "[EMAIL]"),
        "phone_tw": (r"09\d{8}", "[PHONE]"),
        "phone_intl": (r"\+\d{1,3}[\s-]?\d{1,14}", "[PHONE]"),
        "id_tw": (r"[A-Z][12]\d{8}", "[ID]"),
        "credit_card": (r"\d{4}[-\s]?\d{4}[-\s]?\d{4}[-\s]?\d{4}", "[CARD]"),
        "ip_address": (r"\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}", "[IP]"),
        "url_personal": (
            r"https?://[^\s]*(?:facebook|instagram|twitter|line)\.com/[^\s]+",
            "[SOCIAL]",
        ),
    }

    @classmethod
    def remove_pii(cls, text: str) -> Tuple[str, Dict[str, int]]:
        """
        移除個人識別資訊

        Args:
            text: 原始文本

        Returns:
            Tuple[清理後文本, PII類型統計]
        """
        cleaned_text = text
        pii_stats = {}

        for pii_type, (pattern, replacement) in cls.PII_PATTERNS.items():
            matches = re.findall(pattern, cleaned_text, re.IGNORECASE)
            if matches:
                pii_stats[pii_type] = len(matches)
                cleaned_text = re.sub(
                    pattern,
                    replacement,
                    cleaned_text,
                    flags=re.IGNORECASE
                )

        return cleaned_text, pii_stats

    @classmethod
    def hash_user_id(cls, user_id: str, salt: str = "cyberpuppy") -> str:
        """
        雜湊處理使用者 ID

        Args:
            user_id: 原始使用者 ID
            salt: 加鹽值

        Returns:
            雜湊後的 ID
        """
        combined = f"{salt}:{user_id}"
        return hashlib.sha256(combined.encode()).hexdigest()[:16]

    @classmethod
    def anonymize_ip(cls, ip_address: str) -> str:
        """
        匿名化 IP 地址（保留前兩段）

        Args:
            ip_address: 原始 IP

        Returns:
            匿名化的 IP
        """
        parts = ip_address.split(".")
        if len(parts) == 4:
            return f"{parts[0]}.{parts[1]}.X.X"
        return "X.X.X.X"


class SafetyRules:
    """安全規則引擎"""

    # 回覆訊息模板
    RESPONSE_TEMPLATES = {
        ResponseLevel.GENTLE_REMINDER: [
            "嗨！讓我們保持友善的對話環境吧 😊",
            "或許可以用更溫和的方式表達想法呢？",
            "每個人都值得被尊重對待 💝",
            "友善的溝通讓對話更有意義 🤝",
        ],
        ResponseLevel.SOFT_INTERVENTION: [
            "我們注意到您的訊息可能讓他人感到不舒服。\n網路溝通有時容易產生誤解，試試更清楚地表達您的想法？",
            "網路霸凌會對他人造成傷害。\n讓我們一起營造友善的網路環境。",
            "您的訊息包含不適當內容。\n請注意您的用詞，保持理性討論。",
        ],
        ResponseLevel.RESOURCE_ESCALATION: [
            "如果您正在經歷困難，以下資源可能對您有幫助：",
            "我們關心您的感受，這裡有一些支援管道：",
            "請記得，您並不孤單。以下是一些求助資源：",
        ],
    }

    # 求助資源
    HELP_RESOURCES = [
        {"name": "心理健康專線", "phone": "1925", "url": None},
        {"name": "生命線協談專線", "phone": "1995", "url": None},
        {"name": "iWIN 網路內容防護", "phone": None, "url": "https://i.win.org.tw/"},
        {"name": "家暴防治專線", "phone": "113", "url": None},
    ]

    def __init__(self):
        """初始化安全規則引擎"""
        self.user_histories: Dict[str, UserViolationHistory] = {}

    def determine_response_level(
        self,
        toxicity_score: float,
        bullying_score: float,
        user_id: str,
        emotion: str = "neu",
        emotion_strength: int = 0,
    ) -> ResponseLevel:
        """
        決定回應等級

        Args:
            toxicity_score: 毒性分數
            bullying_score: 霸凌分數
            user_id: 使用者 ID
            emotion: 情緒類別
            emotion_strength: 情緒強度

        Returns:
            回應等級
        """
        # 取得使用者違規歷史
        history = self.user_histories.get(user_id)
        recent_violations = 0
        if history:
            recent_violations = len(history.get_recent_violations(24))

        # 決定基礎等級
        max_score = max(toxicity_score, bullying_score)

        if max_score > 0.9:
            return ResponseLevel.SILENT_HANDOVER
        elif max_score > 0.7:
            return ResponseLevel.RESOURCE_ESCALATION
        elif max_score > 0.5:
            # 考慮違規歷史
            if recent_violations >= 2:
                return ResponseLevel.RESOURCE_ESCALATION
            return ResponseLevel.SOFT_INTERVENTION
        elif max_score > 0.3:
            # 考慮違規歷史
            if recent_violations >= 1:
                return ResponseLevel.SOFT_INTERVENTION
            return ResponseLevel.GENTLE_REMINDER

        # 檢查情緒狀態（自傷傾向）
        if emotion == "neg" and emotion_strength >= 4:
            return ResponseLevel.RESOURCE_ESCALATION

        return ResponseLevel.NONE

    def generate_response(
        self, level: ResponseLevel, context: Optional[Dict[str, Any]] = None
    ) -> ResponseStrategy:
        """
        生成回應策略

        Args:
            level: 回應等級
            context: 額外上下文

        Returns:
            回應策略
        """
        if level == ResponseLevel.NONE:
            return ResponseStrategy(
                level=level, message=""
                    "", resources=[], notify_admin=False, log_detail=False
            )

        # 選擇訊息模板
        import random

        templates = self.RESPONSE_TEMPLATES.get(level, [])
        message = random.choice(templates) if templates else ""

        # 決定是否需要資源
        resources = []
        if level >= ResponseLevel.RESOURCE_ESCALATION:
            resources = self.HELP_RESOURCES

        # 決定是否通知管理員
        notify_admin = level >= ResponseLevel.SILENT_HANDOVER

        return ResponseStrategy(
            level=level,
            message=message,
            resources=resources,
            notify_admin=notify_admin,
            log_detail=True,
        )

    def update_user_history(
        self,
        user_id: str,
        level: ResponseLevel,
        scores: Dict[str,
        float]
    ):
        """更新使用者違規歷史"""
        if user_id not in self.user_histories:
            self.user_histories[user_id] = UserViolationHistory(user_id=user_id)

        if level > ResponseLevel.NONE:
            self.user_histories[user_id].add_violation(level, scores)

    def should_apply_special_protection(
        self, user_age: Optional[int] = None, is_vulnerable: bool = False
    ) -> Dict[str, Any]:
        """
        判斷是否需要特殊保護

        Args:
            user_age: 使用者年齡
            is_vulnerable: 是否為脆弱族群

        Returns:
            特殊保護設定
        """
        protection = {
            "lower_threshold": False,
            "priority_resources": False,
            "parental_notify": False,
            "extra_logging": False,
        }

        # 未成年人保護
        if user_age and user_age < 18:
            protection["lower_threshold"] = True
            protection["priority_resources"] = True
            if user_age < 13:
                protection["parental_notify"] = True

        # 脆弱族群保護
        if is_vulnerable:
            protection["priority_resources"] = True
            protection["extra_logging"] = True

        return protection


class PrivacyLogger:
    """隱私保護日誌記錄器"""

    def __init__(self, log_file: Optional[str] = None):
        """初始化日誌記錄器"""
        self.log_file = log_file
        self.pii_handler = PIIHandler()

    def log_event(
        self,
        event_type: str,
        text: str,
        scores: Dict[str, float],
        action: ResponseLevel,
        user_id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        記錄事件（隱私保護）

        Args:
            event_type: 事件類型
            text: 原始文本
            scores: 分析分數
            action: 採取的行動
            user_id: 使用者 ID
            metadata: 額外元資料

        Returns:
            日誌記錄
        """
        # 生成文本雜湊
        text_hash = hashlib.sha256(text.encode()).hexdigest()[:16]

        # 匿名化使用者 ID
        anon_user_id = None
        if user_id:
            anon_user_id = self.pii_handler.hash_user_id(user_id)

        # 移除 PII
        _, pii_stats = self.pii_handler.remove_pii(text)

        # 建立日誌記錄
        log_entry = {
            "timestamp": datetime.now().isoformat(),
            "event_type": event_type,
            "text_hash": text_hash,
            "text_length": len(text),
            "scores": scores,
            "action": action.name,
            "session_id": anon_user_id,
            "pii_detected": pii_stats,
            "metadata": metadata or {},
        }

        # 寫入日誌
        if self.log_file:
            try:
                with open(self.log_file, "a", encoding="utf-8") as f:
                    f.write(json.dumps(log_entry, ensure_ascii=False) + "\n")
            except Exception as e:
                logger.error(f"日誌寫入失敗: {e}")

        logger.info(f"事件記錄 - 類型: {event_type}, Hash:"
            " {text_hash}, 行動: {action.name}")

        return log_entry


class AppealManager:
    """申訴管理器"""

    def __init__(self):
        """初始化申訴管理器"""
        self.appeals: Dict[str, Appeal] = {}
        self.user_appeals: Dict[str, List[str]] = {}  # user_id -> appeal_ids

    def create_appeal(
        self,
        user_id: str,
        event_hash: str,
        reason: str
    ) -> Appeal:
        """
        建立申訴

        Args:
            user_id: 使用者 ID
            event_hash: 事件雜湊值
            reason: 申訴原因

        Returns:
            申訴案件
        """
        appeal_id = str(uuid.uuid4())[:8]
        appeal = Appeal(
            appeal_id=appeal_id,
            user_id=PIIHandler.hash_user_id(user_id),
            event_hash=event_hash,
            reason=reason[:500],  # 限制長度
            status=AppealStatus.PENDING,
            created_at=datetime.now(),
            updated_at=datetime.now(),
        )

        self.appeals[appeal_id] = appeal

        # 記錄使用者申訴
        anon_user = appeal.user_id
        if anon_user not in self.user_appeals:
            self.user_appeals[anon_user] = []
        self.user_appeals[anon_user].append(appeal_id)

        logger.info(f"申訴建立 - ID: {appeal_id}, Hash: {event_hash}")

        return appeal

    def review_appeal(
        self, appeal_id: str, reviewer: str, decision: AppealStatus,
            resolution: str
    ) -> Optional[Appeal]:
        """
        審核申訴

        Args:
            appeal_id: 申訴 ID
            reviewer: 審核員 ID
            decision: 決定
            resolution: 解決說明

        Returns:
            更新後的申訴案件
        """
        if appeal_id not in self.appeals:
            logger.error(f"申訴不存在: {appeal_id}")
            return None

        appeal = self.appeals[appeal_id]
        appeal.status = decision
        appeal.reviewer = reviewer
        appeal.resolution = resolution
        appeal.updated_at = datetime.now()

        logger.info(f"申訴審核 - ID: {appeal_id}, 決定: {decision.value}")

        # 如果誤判成立，執行補償
        if decision == AppealStatus.APPROVED:
            self._apply_compensation(appeal)

        return appeal

    def _apply_compensation(self, appeal: Appeal):
        """執行誤判補償"""
        logger.info(f"執行補償 - 申訴 ID: {appeal.appeal_id}")
        # 實際實作時：
        # 1. 清除違規記錄
        # 2. 恢復權限
        # 3. 發送通知
        # 4. 更新模型訓練資料

    def get_user_appeals(
        self, user_id: str, status_filter: Optional[AppealStatus] = None
    ) -> List[Appeal]:
        """取得使用者的申訴記錄"""
        anon_user = PIIHandler.hash_user_id(user_id)
        appeal_ids = self.user_appeals.get(anon_user, [])

        appeals = [self.appeals[aid] for aid in appeal_ids if aid in
            self.appeals]

        if status_filter:
            appeals = [a for a in appeals if a.status == status_filter]

        return sorted(appeals, key=lambda x: x.created_at, reverse=True)

    def get_appeal_stats(self) -> Dict[str, Any]:
        """取得申訴統計"""
        total = len(self.appeals)
        if total == 0:
            return {"total": 0, "by_status": {}, "approval_rate": 0}

        by_status = {}
        for appeal in self.appeals.values():
            status = appeal.status.value
            by_status[status] = by_status.get(status, 0) + 1

        approved = by_status.get(AppealStatus.APPROVED.value, 0)
        rejected = by_status.get(AppealStatus.REJECTED.value, 0)
        reviewed = approved + rejected

        approval_rate = (approved / reviewed * 100) if reviewed > 0 else 0

        return {
            "total": total,
            "by_status": by_status,
            "approval_rate": round(approval_rate, 2),
            "average_review_time": self._calculate_avg_review_time(),
        }

    def _calculate_avg_review_time(self) -> float:
        """計算平均審核時間（小時）"""
        reviewed = [
            a
            for a in self.appeals.values()
            if a.status in [AppealStatus.APPROVED, AppealStatus.REJECTED]
        ]

        if not reviewed:
            return 0

        total_time = sum((a.updated_at -
            a.created_at).total_seconds() for a in reviewed)

        return round(total_time / len(reviewed) / 3600, 2)  # 轉換為小時


# 使用範例
def example_usage():
    """使用範例"""

    # 初始化元件
    safety_rules = SafetyRules()
    privacy_logger = PrivacyLogger(log_file="safety_logs.jsonl")
    appeal_manager = AppealManager()

    # 模擬分析結果
    user_id = "user123"
    text = "你這個笨蛋，給我滾開！我的信箱是 test@example.com"
    scores = {"toxicity": 0.75, "bullying": 0.60, "emotion": "neg"}

    # 1. 決定回應等級
    response_level = safety_rules.determine_response_level(
        toxicity_score=scores["toxicity"],
        bullying_score=scores["bullying"],
        user_id=user_id,
        emotion=scores.get("emotion", "neu"),
    )
    print(f"回應等級: {response_level.name}")

    # 2. 生成回應策略
    strategy = safety_rules.generate_response(response_level)
    print(f"回應訊息: {strategy.message}")
    if strategy.resources:
        print("提供資源:")
        for resource in strategy.resources:
            print(f"  - {resource['name']}")

    # 3. 記錄事件（隱私保護）
    log_entry = privacy_logger.log_event(
        event_type="toxicity_detected",
        text=text,
        scores=scores,
        action=response_level,
        user_id=user_id,
    )
    print(f"日誌記錄 - Hash: {log_entry['text_hash']}"
        ", PII 偵測: {log_entry['pii_detected']}")

    # 4. 更新違規歷史
    safety_rules.update_user_history(user_id, response_level, scores)

    # 5. 模擬申訴流程
    appeal = appeal_manager.create_appeal(
        user_id=user_id,
        event_hash=log_entry["text_hash"],
        reason="這是朋友間的玩笑話，不是真的霸凌",
    )
    print(f"申訴建立 - ID: {appeal.appeal_id}")

    # 6. 審核申訴
    reviewed = appeal_manager.review_appeal(
        appeal_id=appeal.appeal_id,
        reviewer="admin001",
        decision=AppealStatus.APPROVED,
        resolution="經查證確實為朋友間玩笑，判定為誤判",
    )
    print(f"申訴結果: {reviewed.status.value}")

    # 7. 查看統計
    stats = appeal_manager.get_appeal_stats()
    print(f"申訴統計: {stats}")


if __name__ == "__main__":
    example_usage()
