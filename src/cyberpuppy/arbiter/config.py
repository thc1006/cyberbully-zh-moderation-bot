"""
Arbiter 模組配置設定
外部仲裁服務的配置管理
"""

import os
from dataclasses import dataclass
from typing import Any, Dict, Optional


@dataclass
class PerspectiveConfig:
    """Perspective API 配置"""

    api_key: Optional[str] = None
    requests_per_second: int = 1
    requests_per_day: int = 1000
    burst_size: int = 5
    timeout_seconds: float = 30.0
    max_retries: int = 3
    enable_caching: bool = True
    cache_ttl_hours: int = 24


@dataclass
class UncertaintyConfig:
    """不確定性檢測配置"""

    uncertainty_threshold: float = 0.4
    confidence_threshold: float = 0.6
    min_confidence_gap: float = 0.1
    enable_context_check: bool = True
    conflict_detection: bool = True


class ArbiterConfig:
    """仲裁服務配置管理器"""

    def __init__(self):
        # Perspective API 配置
        self.perspective = PerspectiveConfig(
            api_key=os.getenv("PERSPECTIVE_API_KEY"),
            requests_per_second=int(os.getenv("PERSPECTIVE_RATE_LIMIT_RPS", "1")),
            requests_per_day=int(os.getenv("PERSPECTIVE_RATE_LIMIT_DAY", "1000")),
            burst_size=int(os.getenv("PERSPECTIVE_BURST_SIZE", "5")),
            timeout_seconds=float(os.getenv("PERSPECTIVE_TIMEOUT", "30.0")),
            max_retries=int(os.getenv("PERSPECTIVE_MAX_RETRIES", "3")),
            enable_caching=(os.getenv("PERSPECTIVE_ENABLE_CACHE", "true").lower() == "true"),
            cache_ttl_hours=int(os.getenv("PERSPECTIVE_CACHE_TTL", "24")),
        )

        # 不確定性檢測配置
        self.uncertainty = UncertaintyConfig(
            uncertainty_threshold=float(os.getenv("UNCERTAINTY_THRESHOLD", "0.4")),
            confidence_threshold=float(os.getenv("CONFIDENCE_THRESHOLD", "0.6")),
            min_confidence_gap=float(os.getenv("MIN_CONFIDENCE_GAP", "0.1")),
            enable_context_check=(os.getenv("ENABLE_CONTEXT_CHECK", "true").lower() == "true"),
            conflict_detection=(os.getenv("CONFLICT_DETECTION", "true").lower() == "true"),
        )

    def is_perspective_enabled(self) -> bool:
        """檢查 Perspective API 是否可用"""
        return bool(self.perspective.api_key)

    def get_perspective_rate_limit(self) -> Dict[str, int]:
        """取得 Perspective API 速率限制設定"""
        return {
            "requests_per_second": self.perspective.requests_per_second,
            "requests_per_day": self.perspective.requests_per_day,
            "burst_size": self.perspective.burst_size,
        }

    def to_dict(self) -> Dict[str, Any]:
        """轉換為字典格式（不包含敏感資訊）"""
        return {
            "perspective": {
                "enabled": self.is_perspective_enabled(),
                "requests_per_second": self.perspective.requests_per_second,
                "requests_per_day": self.perspective.requests_per_day,
                "timeout_seconds": self.perspective.timeout_seconds,
                "max_retries": self.perspective.max_retries,
                "enable_caching": self.perspective.enable_caching,
                "cache_ttl_hours": self.perspective.cache_ttl_hours,
            },
            "uncertainty": {
                "uncertainty_threshold": (self.uncertainty.uncertainty_threshold),
                "confidence_threshold": self.uncertainty.confidence_threshold,
                "min_confidence_gap": self.uncertainty.min_confidence_gap,
                "enable_context_check": self.uncertainty.enable_context_check,
                "conflict_detection": self.uncertainty.conflict_detection,
            },
        }


# 全域配置實例
arbiter_config = ArbiterConfig()
