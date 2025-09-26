"""
Integration test fixtures and configuration
整合測試共用設定與工具
"""

import asyncio
import pytest
import httpx
import subprocess
import tempfile
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import AsyncGenerator, Generator
import docker
import psutil
import os
# 設定測試環境路徑 - 修復模組導入問題
from pathlib import Path
TEST_ROOT = Path(__file__).parent
PROJECT_ROOT = TEST_ROOT.parent.parent
FIXTURES_DIR = TEST_ROOT / "fixtures"

# 測試常數
TIMEOUT_SECONDS = 30
MAX_RESPONSE_TIME_MS = 2000
TEST_API_BASE = "http://localhost:8000"
TEST_BOT_BASE = "http://localhost:8080"


@pytest.fixture(scope="session")
def event_loop():
    """Create event loop for async tests"""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()


@pytest.fixture(scope="session")
def project_root() -> Path:
    """Project root directory"""
    return PROJECT_ROOT


@pytest.fixture(scope="session")
def test_fixtures() -> Path:
    """Test fixtures directory"""
    return FIXTURES_DIR


@pytest.fixture(scope="session")
def docker_client():
    """Docker client for container testing"""
    client = docker.from_env()
    yield client
    client.close()


@pytest.fixture(scope="session")
async def api_server():
    """Start API server for testing"""
    process = None
    try:
        # 啟動 API 伺服器
        process = subprocess.Popen(
            [
                "python",
                "-m",
                "uvicorn",
                "api.app:app",
                "--host",
                "0.0.0.0",
                "--port",
                "8000",
            ],
            cwd=PROJECT_ROOT,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )

        # 等待伺服器啟動
        async with httpx.AsyncClient() as client:
            for _ in range(30):  # 30秒超時
                try:
                    response = await client.get(f"{TEST_API_BASE}/healthz")
                    if response.status_code == 200:
                        break
                except Exception:
                    pass
                await asyncio.sleep(1)
            else:
                raise RuntimeError("API server failed to start")

        yield TEST_API_BASE

    finally:
        if process:
            process.terminate()
            process.wait(timeout=10)


@pytest.fixture(scope="session")
async def bot_server(api_server):
    """Start bot server for testing"""
    process = None
    try:
        # 設定測試環境變數
        env = os.environ.copy()
        env.update(
            {
                "LINE_CHANNEL_ACCESS_TOKEN": "test_token_" + "x" * 100,
                "LINE_CHANNEL_SECRET": "test_secret_" + "x" * 32,
                "CYBERPUPPY_API_URL": api_server,
            }
        )

        process = subprocess.Popen(
            [
                "python",
                "-m",
                "uvicorn",
                "bot.line_bot:app",
                "--host",
                "0.0.0.0",
                "--port",
                "8080",
            ],
            cwd=PROJECT_ROOT,
            env=env,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )

        # 等待 Bot 伺服器啟動
        async with httpx.AsyncClient() as client:
            for _ in range(20):
                try:
                    response = await client.get(f"{TEST_BOT_BASE}/health")
                    if response.status_code == 200:
                        break
                except Exception:
                    pass
                await asyncio.sleep(1)
            else:
                raise RuntimeError("Bot server failed to start")

        yield TEST_BOT_BASE

    finally:
        if process:
            process.terminate()
            process.wait(timeout=10)


@pytest.fixture
async def http_client() -> AsyncGenerator[httpx.AsyncClient, None]:
    """HTTP client for API testing"""
    async with httpx.AsyncClient(timeout=TIMEOUT_SECONDS) as client:
        yield client


@pytest.fixture
def temp_dir() -> Generator[Path, None, None]:
    """Temporary directory for test files"""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture
def test_data_small():
    """Small test dataset for quick tests"""
    return [
        {
            "text": "你好，今天天氣真不錯",
            "expected": {"toxicity": "none", "bullying": "none", "emotion": "pos"},
        },
        {
            "text": "你這個笨蛋，什麼都不懂",
            "expected": {
                "toxicity": "toxic",
                "bullying": "harassment",
                "emotion": "neg",
            },
        },
        {
            "text": "我會讓你後悔的，小心點",
            "expected": {"toxicity": "severe", "bullying": "threat", "emotion": "neg"},
        },
    ]


@pytest.fixture
def test_data_chinese():
    """Chinese toxicity test cases"""
    return [
        # 正面測試案例
        {
            "text": "謝謝你的幫助，真的很感激",
            "expected": {"toxicity": "none", "bullying": "none", "emotion": "pos"},
        },
        {
            "text": "今天學到很多新東西，很開心",
            "expected": {"toxicity": "none", "bullying": "none", "emotion": "pos"},
        },
        # 中性測試案例
        {
            "text": "今天的天氣如何？",
            "expected": {"toxicity": "none", "bullying": "none", "emotion": "neu"},
        },
        {
            "text": "請問這個問題怎麼解決？",
            "expected": {"toxicity": "none", "bullying": "none", "emotion": "neu"},
        },
        # 毒性測試案例
        {
            "text": "你真的很笨耶，怎麼連這個都不會",
            "expected": {
                "toxicity": "toxic",
                "bullying": "harassment",
                "emotion": "neg",
            },
        },
        {
            "text": "廢物一個，滾開",
            "expected": {
                "toxicity": "toxic",
                "bullying": "harassment",
                "emotion": "neg",
            },
        },
        # 嚴重毒性測試案例
        {
            "text": "我要殺了你，你給我等著",
            "expected": {
                "toxicity": "severe",
                "bullying": "threat",
                "role": "perpetrator",
            },
        },
        {
            "text": "你最好去死一死，沒人會想念你",
            "expected": {
                "toxicity": "toxic",
                "bullying": "threat",
                "role": "perpetrator",
            },
        },
        # 網路霸凌特殊案例
        {
            "text": "大家都不喜歡你，你知道嗎？",
            "expected": {
                "toxicity": "toxic",
                "bullying": "harassment",
                "role": "perpetrator",
            },
        },
        {
            "text": "如果你不照我說的做，我就把你的秘密告訴大家",
            "expected": {
                "toxicity": "toxic",
                "bullying": "threat",
                "role": "perpetrator",
            },
        },
    ]


@pytest.fixture
def line_webhook_payload():
    """LINE webhook test payload"""
    return {
        "destination": "test_destination",
        "events": [
            {
                "type": "message",
                "mode": "active",
                "timestamp": int(time.time() * 1000),
                "source": {"type": "user", "userId": "test_user_123"},
                "replyToken": "test_reply_token",
                "message": {
                    "id": "test_message_id",
                    "type": "text",
                    "text": "測試訊息",
                },
            }
        ],
    }


@pytest.fixture
def performance_monitor():
    """Performance monitoring context"""

    @asynccontextmanager
    async def monitor():
        start_time = time.time()
        start_memory = psutil.Process().memory_info().rss

        metrics = {}

        try:
            yield metrics
        finally:
            end_time = time.time()
            end_memory = psutil.Process().memory_info().rss

            duration = end_time - start_time
            memory_delta = end_memory - start_memory

            # 記錄效能指標
            print(f"執行時間: {duration:.3f}秒")
            print(f"記憶體變化: {memory_delta / 1024 / 1024:.2f}MB")

            metrics.update({"duration": duration, "memory_delta": memory_delta})

    return monitor


@pytest.fixture
def mock_line_signature():
    """Mock LINE signature for webhook testing"""

    def create_signature(body: bytes) -> str:
        import hmac
        import hashlib
        import base64

        secret = "test_secret_" + "x" * 32
        signature = base64.b64encode(
            hmac.new(secret.encode(), body, hashlib.sha256).digest()
        ).decode()
        return signature

    return create_signature


@pytest.fixture(scope="function")
def isolated_db():
    """Isolated database for testing"""
    # 這裡可以設定測試用的資料庫
    # 目前專案主要是無狀態 API，所以暫時不需要資料庫
    yield None


@pytest.fixture
def trained_model_path(temp_dir):
    """Path to trained test model"""
    model_dir = temp_dir / "models" / "test"
    model_dir.mkdir(parents=True)

    # 在真實測試中，這裡會載入實際訓練的模型
    # 目前先建立目錄結構
    (model_dir / "config.json").write_text('{"model_type": "test"}')

    return model_dir


@pytest.fixture(autouse=True)
def setup_test_environment(monkeypatch):
    """Setup test environment variables"""
    monkeypatch.setenv("TESTING", "1")
    monkeypatch.setenv("LOG_LEVEL", "INFO")
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "")  # 禁用 GPU


# Pytest markers for test categorization
pytestmark = [
    pytest.mark.integration,
]


def pytest_configure(config):
    """Configure pytest with custom markers"""
    config.addinivalue_line("markers", "integration: integration tests")
    config.addinivalue_line("markers", "slow: slow running tests")
    config.addinivalue_line("markers", "api: API integration tests")
    config.addinivalue_line("markers", "bot: bot integration tests")
    config.addinivalue_line("markers", "pipeline: pipeline integration tests")
    config.addinivalue_line("markers", "performance: performance tests")
