"""
CyberPuppy LINE Bot 配置設定
"""

import os
from dataclasses import dataclass
from typing import Any, Dict

from dotenv import load_dotenv

# 載入 .env 檔案
load_dotenv()


@dataclass
class LineConfig:
    """LINE Bot 配置"""

    channel_access_token: str
    channel_secret: str
    webhook_url: str = ""


@dataclass
class ApiConfig:
    """API 服務配置"""

    cyberpuppy_url: str = "http://localhost:8000"
    timeout: float = 30.0
    max_retries: int = 3


@dataclass
class SecurityConfig:
    """安全配置"""

    max_message_length: int = 1000
    rate_limit_per_user: int = 20  # 每分鐘
    session_timeout_hours: int = 24


@dataclass
class ResponseConfig:
    """回應行為配置"""

    gentle_reminder_threshold: int = 1
    firm_warning_threshold: int = 2
    escalation_threshold: int = 3
    cooldown_hours: int = 24


class Config:
    """主要配置類別"""

    def __init__(self):
        # LINE 配置
        self.line = LineConfig(
            channel_access_token=os.getenv("LINE_CHANNEL_ACCESS_TOKEN", ""),
            channel_secret=os.getenv("LINE_CHANNEL_SECRET", ""),
            webhook_url=os.getenv("LINE_WEBHOOK_URL", ""),
        )

        # API 配置
        self.api = ApiConfig(
            cyberpuppy_url=os.getenv(
                "CYBERPUPPY_API_URL",
                "http://localhost:8000"),
            timeout=float(os.getenv("API_TIMEOUT", "30.0")),
            max_retries=int(os.getenv("API_MAX_RETRIES", "3")),
        )

        # 安全配置
        self.security = SecurityConfig(
            max_message_length=int(os.getenv("MAX_MESSAGE_LENGTH", "1000")),
            rate_limit_per_user=int(os.getenv("RATE_LIMIT_PER_USER", "20")),
            session_timeout_hours=int(
                os.getenv("SESSION_TIMEOUT_HOURS", "24"))
        )

        # 回應配置
        self.response = ResponseConfig(
            gentle_reminder_threshold=int(
                os.getenv("GENTLE_REMINDER_THRESHOLD", "1")),
            firm_warning_threshold=int(
                os.getenv("FIRM_WARNING_THRESHOLD", "2")),
            escalation_threshold=int(os.getenv("ESCALATION_THRESHOLD", "3")),
            cooldown_hours=int(os.getenv("COOLDOWN_HOURS", "24")),
        )

        # 驗證必要設定
        self._validate_config()

    def _validate_config(self):
        """驗證配置是否完整"""
        if not self.line.channel_access_token:
            raise ValueError("遺失 LINE_CHANNEL_ACCESS_TOKEN")

        if not self.line.channel_secret:
            raise ValueError("遺失 LINE_CHANNEL_SECRET")

    def to_dict(self) -> Dict[str, Any]:
        """轉換為字典格式"""
        return {
            "line": {
                "webhook_url": self.line.webhook_url,
                # 不包含敏感資訊
            },
            "api": {
                "cyberpuppy_url": self.api.cyberpuppy_url,
                "timeout": self.api.timeout,
                "max_retries": self.api.max_retries,
            },
            "security": {
                "max_message_length": self.security.max_message_length,
                "rate_limit_per_user": self.security.rate_limit_per_user,
                "session_timeout_hours": self.security.session_timeout_hours,
            },
            "response": {
                "gentle_reminder_threshold": (
                    self.response.gentle_reminder_threshold
                ),
                "firm_warning_threshold": self.response.firm_warning_threshold,
                "escalation_threshold": self.response.escalation_threshold,
                "cooldown_hours": self.response.cooldown_hours,
            },
        }


# 全域配置實例
config = Config()

# 預設回應訊息模板
RESPONSE_TEMPLATES = {
    "gentle_reminders": [
        "嗨！讓我們保持友善的對話環境吧 😊",
        "或許可以用更溫和的方式表達想法呢？🌟",
        "每個人都值得被尊重對待 💝",
        "友善的溝通讓對話更有意義 🤝",
    ],
    "resource_messages": [
        "看起來您可能需要一些支持，以下資源或許能幫助您：",
        "如果您感到困擾，這些資源可能對您有幫助：",
        "我們關心您的感受，這裡有一些可以提供協助的資源：",
    ],
    "system_error": "抱歉，系統暫時無法處理您的訊息，請稍後再試。",
    "rate_limit": "您的訊息傳送過於頻繁，請稍後再試。",
    "message_too_long": "您的訊息過長，請縮短後重新傳送。",
}

# 資源連結
HELP_RESOURCES = {
    "mental_health": {
        "title": "心理健康專線",
        "phone": "1925",
        "url": "https://www.mohw.gov.tw/cp-4266-48131-1.html",
        "description": "24小時免費心理諮詢服務",
    },
    "cyberbullying": {
        "title": "iWIN 網路內容防護機構",
        "url": "https://i.win.org.tw/",
        "description": "網路霸凌申訴與防護",
    },
    "youth_hotline": {
        "title": "青少年諮詢專線",
        "phone": "1995",
        "description": "青少年心理支持與輔導",
    },
    "domestic_violence": {
        "title": "家暴防治專線",
        "phone": "113",
        "description": "家庭暴力與性侵害防治",
    },
}
