"""
CyberPuppy 整合測試套件
Integration tests for CyberPuppy cyberbullying detection system

測試範圍：
- 完整 API 端點測試
- Bot webhook 處理流程
- CLI 命令完整性
- 資料管道整合
- 模型訓練流程
- 效能基準測試
- Docker 容器化測試
"""

import pytest
import sys
from pathlib import Path

# 設定測試環境路徑
TEST_ROOT = Path(__file__).parent
PROJECT_ROOT = TEST_ROOT.parent.parent
FIXTURES_DIR = TEST_ROOT / "fixtures"

# 添加專案根目錄到路徑
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "src"))

# 測試常數
TIMEOUT_SECONDS = 30
MAX_RESPONSE_TIME_MS = 2000
TEST_API_BASE = "http://localhost:8000"
TEST_BOT_BASE = "http://localhost:8080"

# 測試標記
pytestmark = pytest.mark.integration

__all__ = [
    "TEST_ROOT",
    "PROJECT_ROOT",
    "FIXTURES_DIR",
    "TIMEOUT_SECONDS",
    "MAX_RESPONSE_TIME_MS",
    "TEST_API_BASE",
    "TEST_BOT_BASE",
]
