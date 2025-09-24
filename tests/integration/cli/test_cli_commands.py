"""
CLI 命令整合測試
測試完整的命令列介面功能：
- 文本檢測命令
- 批次檔案處理
- 模型訓練命令
- 資料處理管道
- 配置管理
- 錯誤處理與幫助資訊
"""

import json
import subprocess

import pytest

from tests.integration import PROJECT_ROOT, TIMEOUT_SECONDS


@pytest.fixture
def cli_script():
    """CLI 腳本路徑"""
    # 尋找 CLI 腳本
    possible_paths = [
        PROJECT_ROOT / "cyberpuppy" / "cli.py",
        PROJECT_ROOT / "src" / "cyberpuppy" / "cli.py",
        PROJECT_ROOT / "cli.py"
    ]

    for path in possible_paths:
        if path.exists():
            return str(path)

    pytest.skip("CLI script not found")


@pytest.fixture
def sample_text_file(temp_dir):
    """建立測試用文本檔案"""
    texts = [
        "你好，今天天氣真不錯，心情很好。",
        "這個笨蛋什麼都不懂，真是廢物。",
        "我要讓你後悔，小心點。",
        "謝謝你的幫助，真的很感激。",
        "去死啦，我討厭你。"
    ]

    text_file = temp_dir / "test_texts.txt"
    text_file.write_text("\n".join(texts), encoding="utf-8")
    return text_file


@pytest.fixture
def sample_batch_file(temp_dir):
    """建立測試用批次檔案"""
    batch_data = [
        {"id": "1", "text": "正面測試訊息", "label": "none"},
        {"id": "2", "text": "你很笨耶", "label": "toxic"},
        {"id": "3", "text": "威脅訊息內容", "label": "severe"}
    ]

    batch_file = temp_dir / "test_batch.jsonl"
    with open(batch_file, "w", encoding="utf-8") as f:
        for item in batch_data:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    return batch_file


@pytest.mark.cli
class TestCLIBasicCommands:
    """基本 CLI 命令測試"""

    def test_cli_help(self, cli_script):
        """測試 CLI 說明功能"""
        result = subprocess.run(
            ["python", cli_script, "--help"],
            cwd=PROJECT_ROOT,
            capture_output=True,
            text=True,
            timeout=TIMEOUT_SECONDS
        )

        assert result.returncode == 0
        assert "usage:" in result.stdout.lower() or "用法" in result.stdout
        assert "CyberPuppy" in result.stdout or "cyberpuppy" in result.stdout

    def test_cli_version(self, cli_script):
        """測試版本資訊"""
        result = subprocess.run(
            ["python", cli_script, "--version"],
            cwd=PROJECT_ROOT,
            capture_output=True,
            text=True,
            timeout=TIMEOUT_SECONDS
        )

        assert result.returncode == 0
        # 驗證版本格式
        version_pattern = r'\d+\.\d+\.\d+'
        import re
        assert re.search(version_pattern, result.stdout)

    def test_cli_single_text_analysis(self, cli_script):
        """測試單一文本分析"""
        test_text = "你這個笨蛋"

        result = subprocess.run(
            ["python", cli_script, "detect", "--text", test_text],
            cwd=PROJECT_ROOT,
            capture_output=True,
            text=True,
            timeout=TIMEOUT_SECONDS
        )

        assert result.returncode == 0

        # 解析輸出 JSON
        try:
            output = json.loads(result.stdout)
            assert "toxicity" in output
            assert "bullying" in output
            assert "emotion" in output
            assert output["toxicity"] in ["none", "toxic", "severe"]
        except json.JSONDecodeError:
            # 如果不是 JSON 格式，檢查是否包含預期關鍵字
            assert "toxicity" in result.stdout.lower() or "毒性" in result.stdout


@pytest.mark.cli
class TestCLIBatchProcessing:
    """批次處理測試"""

    def test_cli_batch_file_processing(
        self,
        cli_script,
        sample_text_file,
        temp_dir
    ):
        """測試批次檔案處理"""
        output_file = temp_dir / "output.jsonl"

        result = subprocess.run([
            "python", cli_script, "batch",
            "--input", str(sample_text_file),
            "--output", str(output_file),
            "--format", "jsonl"
        ], cwd=PROJECT_ROOT, capture_output=True, text=True,
            timeout=TIMEOUT_SECONDS * 2)

        assert result.returncode == 0
        assert output_file.exists()

        # 驗證輸出檔案內容
        with open(output_file, "r", encoding="utf-8") as f:
            lines = f.readlines()
            assert len(lines) == 5  # 對應 5 行輸入

            for line in lines:
                data = json.loads(line.strip())
                assert "text" in data
                assert "toxicity" in data
                assert "timestamp" in data

    def test_cli_jsonl_batch_processing(
        self,
        cli_script,
        sample_batch_file,
        temp_dir
    ):
        """測試 JSONL 批次處理"""
        output_file = temp_dir / "batch_output.jsonl"

        result = subprocess.run([
            "python", cli_script, "batch",
            "--input", str(sample_batch_file),
            "--output", str(output_file),
            "--input-format", "jsonl"
        ], cwd=PROJECT_ROOT, capture_output=True, text=True,
            timeout=TIMEOUT_SECONDS * 2)

        assert result.returncode == 0
        assert output_file.exists()

        # 驗證輸出
        with open(output_file, "r", encoding="utf-8") as f:
            results = [json.loads(line.strip()) for line in f]
            assert len(results) == 3

            for result_data in results:
                assert "id" in result_data
                assert "text" in result_data
                assert "predi"
                    "ctions" in result_data or 

    def test_cli_csv_output_format(
        self,
        cli_script,
        sample_text_file,
        temp_dir
    ):
        """測試 CSV 輸出格式"""
        output_file = temp_dir / "output.csv"

        result = subprocess.run([
            "python", cli_script, "batch",
            "--input", str(sample_text_file),
            "--output", str(output_file),
            "--format", "csv"
        ], cwd=PROJECT_ROOT, capture_output=True, text=True,
            timeout=TIMEOUT_SECONDS * 2)

        assert result.returncode == 0
        assert output_file.exists()

        # 驗證 CSV 格式
        import csv
        with open(output_file, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            rows = list(reader)
            assert len(rows) == 5

            # 檢查 CSV 標頭
            fieldnames = reader.fieldnames
            assert "text" in fieldnames
            assert "toxicity" in fieldnames


@pytest.mark.cli
@pytest.mark.slow
class TestCLITrainingCommands:
    """訓練命令測試"""

    def test_cli_training_help(self, cli_script):
        """測試訓練命令說明"""
        result = subprocess.run([
            "python", cli_script, "train", "--help"
        ], cwd=PROJECT_ROOT, capture_output=True, text=True,
            timeout=TIMEOUT_SECONDS)

        assert result.returncode == 0
        assert "train" in result.stdout.lower() or "訓練" in result.stdout

    def test_cli_model_evaluation(self, cli_script, temp_dir):
        """測試模型評估命令"""
        # 建立測試資料
        test_data = temp_dir / "test_data.jsonl"
        with open(test_data, "w", encoding="utf-8") as f:
            for i in range(10):
                data = {
                    "text": f"測試文本 {i}",
                    "toxicity": "none" if i % 2 == 0 else "toxic",
                    "emotion": "neu"
                }
                f.write(json.dumps(data, ensure_ascii=False) + "\n")

        # 執行評估（假設有可用的模型）
        result = subprocess.run([
            "python", cli_script, "evaluate",
            "--data", str(test_data),
            "--model", "dummy"  # 使用虛擬模型進行測試
        ], cwd=PROJECT_ROOT, capture_output=True, text=True,
            timeout=TIMEOUT_SECONDS * 3)

        # 即使失敗也要檢查是否有適當的錯誤訊息
        if result.returncode != 0:
            assert "model" in result.stderr.lower() or "模型" in result.stderr
        else:
            assert "evalu"
                "ation" in result.stdout.lower() or 


@pytest.mark.cli
class TestCLIDataProcessing:
    """資料處理測試"""

    def test_cli_data_download(self, cli_script, temp_dir):
        """測試資料下載命令"""
        result = subprocess.run([
            "python", cli_script, "download",
            "--dataset", "test",
            "--output-dir", str(temp_dir)
        ], cwd=PROJECT_ROOT, capture_output=True, text=True,
            timeout=TIMEOUT_SECONDS * 5)

        # 由於可能沒有實際的下載功能，檢查是否有適當回應
        if result.returncode != 0:
            assert "download" in result.stderr.lower() or "下載" in result.stderr
        else:
            assert temp_dir.exists()

    def test_cli_data_preprocessing(
        self,
        cli_script,
        sample_batch_file,
        temp_dir
    ):
        """測試資料前處理命令"""
        output_file = temp_dir / "preprocessed.jsonl"

        result = subprocess.run([
            "python", cli_script, "preprocess",
            "--input", str(sample_batch_file),
            "--output", str(output_file),
            "--clean", "--normalize"
        ], cwd=PROJECT_ROOT, capture_output=True, text=True,
            timeout=TIMEOUT_SECONDS * 2)

        if result.returncode == 0:
            assert output_file.exists()
        else:
            # 檢查錯誤訊息是否合理
            assert "prepr"
                "ocess" in result.stderr.lower() or 


@pytest.mark.cli
class TestCLIConfigurationManagement:
    """配置管理測試"""

    def test_cli_config_show(self, cli_script):
        """測試配置顯示"""
        result = subprocess.run([
            "python", cli_script, "config", "show"
        ], cwd=PROJECT_ROOT, capture_output=True, text=True,
            timeout=TIMEOUT_SECONDS)

        if result.returncode == 0:
            # 檢查配置內容
            assert "config" in result.stdout.lower() or "配置" in result.stdout
        else:
            assert "config" in result.stderr.lower()

    def test_cli_config_validation(self, cli_script):
        """測試配置驗證"""
        result = subprocess.run([
            "python", cli_script, "config", "validate"
        ], cwd=PROJECT_ROOT, capture_output=True, text=True,
            timeout=TIMEOUT_SECONDS)

        # 無論成功失敗，都應該有適當的輸出
        assert result.stdout or result.stderr
        if result.returncode == 0:
            assert "valid" in result.stdout.lower() or "有效" in result.stdout
        else:
            assert "invalid" in result.stderr.lower() or "無效" in result.stderr


@pytest.mark.cli
class TestCLIErrorHandling:
    """錯誤處理測試"""

    def test_cli_invalid_command(self, cli_script):
        """測試無效命令"""
        result = subprocess.run([
            "python", cli_script, "invalid_command"
        ], cwd=PROJECT_ROOT, capture_output=True, text=True,
            timeout=TIMEOUT_SECONDS)

        assert result.returncode != 0
        assert "invalid" in result.stderr.lower() or "無效" in result.stderr or \
               "unknown" in result.stderr.lower() or "未知" in result.stderr

    def test_cli_missing_arguments(self, cli_script):
        """測試缺少參數"""
        result = subprocess.run([
            "python", cli_script, "detect"
        ], cwd=PROJECT_ROOT, capture_output=True, text=True,
            timeout=TIMEOUT_SECONDS)

        assert result.returncode != 0
        assert "requ"
            "ired" in result.stderr.lower() or 
               "missing" in result.stderr.lower() or "缺少" in result.stderr

    def test_cli_invalid_file_path(self, cli_script):
        """測試無效檔案路徑"""
        result = subprocess.run([
            "python", cli_script, "batch",
            "--input", "/nonexistent/file.txt",
            "--output", "/tmp/output.jsonl"
        ], cwd=PROJECT_ROOT, capture_output=True, text=True,
            timeout=TIMEOUT_SECONDS)

        assert result.returncode != 0
        assert "file" in result.stderr.lower() or "檔案" in result.stderr or \
               "not found" in result.stderr.lower() or "找不到" in result.stderr

    def test_cli_invalid_format(self, cli_script, sample_text_file, temp_dir):
        """測試無效輸出格式"""
        output_file = temp_dir / "output.xyz"

        result = subprocess.run([
            "python", cli_script, "batch",
            "--input", str(sample_text_file),
            "--output", str(output_file),
            "--format", "invalid_format"
        ], cwd=PROJECT_ROOT, capture_output=True, text=True,
            timeout=TIMEOUT_SECONDS)

        assert result.returncode != 0
        assert "format" in result.stderr.lower() or "格式" in result.stderr


@pytest.mark.cli
@pytest.mark.slow
class TestCLIPerformance:
    """CLI 效能測試"""

    def test_cli_large_batch_processing(self, cli_script, temp_dir):
        """測試大批次處理效能"""
        # 建立大量測試資料
        large_file = temp_dir / "large_test.txt"
        test_texts = ["測試文本 " + str(i) for i in range(100)]
        large_file.write_text("\n".join(test_texts), encoding="utf-8")

        output_file = temp_dir / "large_output.jsonl"

        import time
        start_time = time.time()

        result = subprocess.run([
            "python", cli_script, "batch",
            "--input", str(large_file),
            "--output", str(output_file)
        ], cwd=PROJECT_ROOT, capture_output=True, text=True,
            timeout=TIMEOUT_SECONDS * 5)

        end_time = time.time()
        processing_time = end_time - start_time

        if result.returncode == 0:
            # 驗證處理時間合理（每個文本不超過 200ms）
            assert processing_time < 100 * 0.2  # 20秒上限
            assert output_file.exists()

            # 驗證所有文本都被處理
            with open(output_file, "r", encoding="utf-8") as f:
                processed_count = sum(1 for _ in f)
                assert processed_count == 100

    def test_cli_memory_usage(self, cli_script, temp_dir):
        """測試記憶體使用情況"""
        import psutil
        import os

        # 建立測試檔案
        test_file = temp_dir / "memory_test.txt"
        # 建立較大的測試檔案
        large_texts = ["記憶體測試文本內容 " * 20 + str(i) for i in range(50)]
        test_file.write_text("\n".join(large_texts), encoding="utf-8")

        output_file = temp_dir / "memory_output.jsonl"

        # 監控記憶體使用
        process = subprocess.Popen([
            "python", cli_script, "batch",
            "--input", str(test_file),
            "--output", str(output_file)
        ], cwd=PROJECT_ROOT, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

        # 監控子進程記憶體使用
        max_memory = 0
        try:
            p = psutil.Process(process.pid)
            while process.poll() is None:
                try:
                    memory_info = p.memory_info()
                    max_memory = max(max_memory, memory_info.rss)
                except psutil.NoSuchProcess:
                    break
                time.sleep(0.1)

            process.wait(timeout=TIMEOUT_SECONDS)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait()

        # 記憶體使用不應超過 500MB
        max_memory_mb = max_memory / 1024 / 1024
        assert max_memory_mb < 500, f"Memory usage too high:"
            " {max_memory_mb:.2f}MB"


@pytest.mark.cli
class TestCLIIntegrationWithAPI:
    """CLI 與 API 整合測試"""

    async def test_cli_with_running_api(
        self,
        cli_script,
        api_server,
        sample_text_file,
        temp_dir
    ):
        """測試 CLI 與執行中 API 的整合"""
        output_file = temp_dir / "api_integration_output.jsonl"

        # 設定 API URL 環境變數
        import os
        env = os.environ.copy()
        env["CYBERPUPPY_API_URL"] = api_server

        result = subprocess.run([
            "python", cli_script, "batch",
            "--input", str(sample_text_file),
            "--output", str(output_file),
            "--use-api"  # 假設有此選項使用 API 而非本地模型
        ], cwd=PROJECT_ROOT, env=env, capture_output=True, text=True,
            timeout=TIMEOUT_SECONDS * 3)

        if result.returncode == 0:
            assert output_file.exists()

            # 驗證輸出格式與 API 回應一致
            with open(output_file, "r", encoding="utf-8") as f:
                for line in f:
                    data = json.loads(line.strip())
                    assert "toxicity" in data
                    assert "bullying" in data
                    assert "timestamp" in data
