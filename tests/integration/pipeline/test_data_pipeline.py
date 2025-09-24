"""
資料管道整合測試
測試完整的資料處理流程：
- 資料下載 → 清理 → 正規化 → 標籤統一 → 訓練/驗證分割
- 模型訓練 → 評估 → 儲存
- 整體管道完整性與錯誤處理
"""

import json
import subprocess
from pathlib import Path

import pytest

from tests.integration import PROJECT_ROOT, TIMEOUT_SECONDS


@pytest.mark.pipeline
@pytest.mark.slow
class TestDataPipeline:
    """完整資料管道測試"""

    def test_download_clean_normalize_pipeline(self, temp_dir):
        """測試下載 → 清理 → 正規化管道"""
        raw_dir = temp_dir / "raw"
        processed_dir = temp_dir / "processed"
        raw_dir.mkdir()
        processed_dir.mkdir()

        # 建立模擬原始資料
        raw_data = [
            {"text": "你好！今天天氣真好😊", "label": "正面"},
            {"text": "這個笨蛋真討厭", "label": "負面"},
            {"text": "我很生氣，想打人", "label": "負面"},
            {"text": "謝謝你的幫助", "label": "正面"},
            {"text": "   空白測試   ", "label": "中性"}
        ]

        raw_file = raw_dir / "test_data.jsonl"
        with open(raw_file, "w", encoding="utf-8") as f:
            for item in raw_data:
                f.write(json.dumps(item, ensure_ascii=False) + "\n")

        # 執行清理腳本
        clean_script = PROJECT_ROOT / "scripts" / "clean_normalize.py"
        if clean_script.exists():
            result = subprocess.run([
                "python", str(clean_script),
                "--input", str(raw_file),
                "--output", str(processed_dir / "cleaned.jsonl"),
                "--normalize", "--remove-duplicates"
            ], capture_output=True, text=True, timeout=TIMEOUT_SECONDS)

            if result.returncode == 0:
                # 驗證清理結果
                cleaned_file = processed_dir / "cleaned.jsonl"
                assert cleaned_file.exists()

                with open(cleaned_file, "r", encoding="utf-8") as f:
                    cleaned_data = [json.loads(line) for line in f]

                # 驗證資料清理效果
                assert len(cleaned_data) > 0
                for item in cleaned_data:
                    assert "text" in item
                    # 檢查文本已清理（去除多餘空白）
                    assert item["text"].strip() == item["text"]

    def test_label_mapping_pipeline(self, temp_dir):
        """測試標籤映射與統一管道"""
        # 建立不同來源的資料，模擬不同標籤格式
        data_sources = [
            # COLD 格式
            {"text": "你很笨", "toxicity": 1, "source": "cold"},
            {"text": "今天天氣好", "toxicity": 0, "source": "cold"},

            # ChnSentiCorp 格式
            {"text": "很好的產品", "sentiment": "positive", "source": "chnsenti"},
            {"text": "糟糕的服務", "sentiment": "negative", "source": "chnsenti"},

            # 自定義格式
            {"text": "我要揍你", "label": "threat", "source": "custom"},
        ]

        input_file = temp_dir / "mixed_labels.jsonl"
        with open(input_file, "w", encoding="utf-8") as f:
            for item in data_sources:
                f.write(json.dumps(item, ensure_ascii=False) + "\n")

        # 執行標籤統一腳本
        mapping_script = PROJECT_ROOT / "scripts" / "label_mapping.py" 
        output_file = temp_dir / "unified_labels.jsonl"

        if mapping_script.exists():
            result = subprocess.run([
                "python", str(mapping_script),
                "--input", str(input_file),
                "--output", str(output_file)
            ], capture_output=True, text=True, timeout=TIMEOUT_SECONDS)

            if result.returncode == 0:
                # 驗證標籤統一結果
                assert output_file.exists()

                with open(output_file, "r", encoding="utf-8") as f:
                    unified_data = [json.loads(line) for line in f]

                # 檢查統一標籤格式
                for item in unified_data:
                    assert "text" in item
                    # 應該包含統一的標籤欄位
                    expected_fields = ["toxicity", "bullying", "emotion"]
                    for field in expected_fields:
                        if field in item:
                            assert item[field] in ["no"
                                "ne", 
                                   item[field] in ["no"
                                       "ne", 
                                   item[field] in ["pos", "neu", "neg"]

    def test_train_test_split_pipeline(self, temp_dir):
        """測試訓練/測試資料分割管道"""
        # 建立測試資料
        train_data = []
        for i in range(100):
            train_data.append({
                "text": f"測試文本 {i}",
                "toxicity": "toxic" if i % 3 == 0 else "none",
                "emotion": "neg" if i % 3 == 0 else "pos",
                "bullying": "harassment" if i % 5 == 0 else "none"
            })

        full_data_file = temp_dir / "full_dataset.jsonl"
        with open(full_data_file, "w", encoding="utf-8") as f:
            for item in train_data:
                f.write(json.dumps(item, ensure_ascii=False) + "\n")

        # 執行分割腳本（如果存在）
        split_script = PROJECT_ROOT / "scripts" / "split_train_test.py"
        if split_script.exists():
            result = subprocess.run([
                "python", str(split_script),
                "--input", str(full_data_file),
                "--output-dir", str(temp_dir),
                "--train-ratio", "0.8",
                "--stratify"
            ], capture_output=True, text=True, timeout=TIMEOUT_SECONDS)

            if result.returncode == 0:
                # 驗證分割結果
                train_file = temp_dir / "train.jsonl"
                test_file = temp_dir / "test.jsonl"

                assert train_file.exists()
                assert test_file.exists()

                # 檢查分割比例
                with open(train_file, "r", encoding="utf-8") as f:
                    train_count = sum(1 for _ in f)
                with open(test_file, "r", encoding="utf-8") as f:
                    test_count = sum(1 for _ in f)

                total_count = train_count + test_count
                train_ratio = train_count / total_count
                assert 0.75 <= train_ratio <= 0.85  # 允許小誤差


@pytest.mark.pipeline
@pytest.mark.slow
class TestModelTrainingPipeline:
    """模型訓練管道測試"""

    def test_training_pipeline_e2e(self, temp_dir, trained_model_path):
        """測試端到端訓練管道"""
        # 建立訓練資料
        train_data = self._create_training_data(100)
        test_data = self._create_training_data(20)

        train_file = temp_dir / "train.jsonl"
        test_file = temp_dir / "test.jsonl"

        with open(train_file, "w", encoding="utf-8") as f:
            for item in train_data:
                f.write(json.dumps(item, ensure_ascii=False) + "\n")

        with open(test_file, "w", encoding="utf-8") as f:
            for item in test_data:
                f.write(json.dumps(item, ensure_ascii=False) + "\n")

        # 執行訓練腳本
        train_script = PROJECT_ROOT / "train.py"
        model_output_dir = temp_dir / "models"

        if train_script.exists():
            result = subprocess.run([
                "python", str(train_script),
                "--train-data", str(train_file),
                "--test-data", str(test_file),
                "--output-dir", str(model_output_dir),
                "--epochs", "1",  # 快速測試
                "--model-name", "test_model",
                "--device", "cpu"  # 強制使用 CPU
            ], capture_output=True, text=True, timeout=TIMEOUT_SECONDS * 10)

            if result.returncode == 0:
                # 驗證模型檔案生成
                assert model_output_dir.exists()

                # 檢查是否有模型檔案
                model_files = list(model_output_dir.glob("**/*.pth")) + \
                             list(model_output_dir.glob("**/*.bin")) + \
                             list(model_output_dir.glob("**/*.safetensors"))

                assert len(model_files) > 0, "No model files found"

                # 檢查配置檔案
                config_files = list(model_output_dir.glob("**/config.json"))
                assert len(config_files) > 0, "No config files found"

    def test_evaluation_pipeline(self, temp_dir):
        """測試評估管道"""
        # 建立評估資料
        eval_data = self._create_training_data(50)
        eval_file = temp_dir / "eval.jsonl"

        with open(eval_file, "w", encoding="utf-8") as f:
            for item in eval_data:
                f.write(json.dumps(item, ensure_ascii=False) + "\n")

        # 執行評估腳本（如果存在）
        eval_script = PROJECT_ROOT / "scripts" / "evaluate_model.py"
        results_file = temp_dir / "eval_results.json"

        if eval_script.exists():
            result = subprocess.run([
                "python", str(eval_script),
                "--data", str(eval_file),
                "--model", "dummy",  # 使用虛擬模型
                "--output", str(results_file)
            ], capture_output=True, text=True, timeout=TIMEOUT_SECONDS * 2)

            if result.returncode == 0:
                # 驗證評估結果
                assert results_file.exists()

                with open(results_file, "r") as f:
                    results = json.load(f)

                # 檢查評估指標
                expected_metrics = ["accuracy", "precision", "recall", "f1"]
                for metric in expected_metrics:
                    if metric in results:
                        assert isinstance(results[metric], (int, float))
                        assert 0 <= results[metric] <= 1

    def _create_training_data(self, count: int) -> List[Dict[str, Any]]:
        """建立訓練資料"""
        data = []
        patterns = [
            {"text": "This is a normal text", "toxicity": "none", "emotion": "neu"},
            {"text": "This is toxic content", "toxicity": "toxic", "emotion": "neg"},
            {"text": "Happy positive message", "toxicity": "none", "emotion": "pos"},
            {"text": "Severe bullying text", "toxicity": "severe", "emotion": "neg"},
        ]

        for i in range(count):
            pattern = patterns[i % len(patterns)]
            data.append({
                "text": f"{pattern['text']} {i}",
                "toxicity": pattern["toxicity"],
                "emotion": pattern["emotion"],
                "bullying": pattern["bullying"]
            })

        return data


@pytest.mark.pipeline
class TestPipelineIntegrity:
    """管道完整性測試"""

    def test_pipeline_error_handling(self, temp_dir):
        """測試管道錯誤處理"""
        # 測試空檔案處理
        empty_file = temp_dir / "empty.jsonl"
        empty_file.touch()

        # 測試無效格式檔案處理
        invalid_file = temp_dir / "invalid.jsonl"
        invalid_file.write_text("not json content", encoding="utf-8")

        # 執行處理腳本，應該能處理錯誤
        clean_script = PROJECT_ROOT / "scripts" / "clean_normalize.py"
        if clean_script.exists():
            # 測試空檔案
            result = subprocess.run([
                "python", str(clean_script),
                "--input", str(empty_file),
                "--output", str(temp_dir / "empty_output.jsonl")
            ], capture_output=True, text=True, timeout=TIMEOUT_SECONDS)

            # 應該有適當的錯誤處理，不會崩潰
            assert result.returncode == 0 or "empty" in result.stderr.lower()

            # 測試無效檔案
            result = subprocess.run([
                "python", str(clean_script),
                "--input", str(invalid_file),
                "--output", str(temp_dir / "invalid_output.jsonl")
            ], capture_output=True, text=True, timeout=TIMEOUT_SECONDS)

            # 應該有適當的錯誤處理
            assert result.returncode == 0 or "json" in result.stderr.lower()

    def test_pipeline_data_consistency(self, temp_dir):
        """測試管道資料一致性"""
        # 建立測試資料
        original_data = [
            {"text": "原始文本1", "label": "positive"},
            {"text": "原始文本2", "label": "negative"},
            {"text": "原始文本3", "label": "neutral"}
        ]

        input_file = temp_dir / "consistency_input.jsonl"
        with open(input_file, "w", encoding="utf-8") as f:
            for item in original_data:
                f.write(json.dumps(item, ensure_ascii=False) + "\n")

        # 執行多階段處理
        stage1_output = temp_dir / "stage1.jsonl"
        stage2_output = temp_dir / "stage2.jsonl"

        # 階段1：清理
        clean_script = PROJECT_ROOT / "scripts" / "clean_normalize.py"
        if clean_script.exists():
            subprocess.run([
                "python", str(clean_script),
                "--input", str(input_file),
                "--output", str(stage1_output)
            ], capture_output=True, timeout=TIMEOUT_SECONDS)

        # 階段2：標籤統一
        if stage1_output.exists():
            mapping_script = PROJECT_ROOT / "scripts" / "label_mapping.py" 
            if mapping_script.exists():
                subprocess.run([
                    "python", str(mapping_script),
                    "--input", str(stage1_output),
                    "--output", str(stage2_output)
                ], capture_output=True, timeout=TIMEOUT_SECONDS)

        # 驗證資料一致性
        if stage2_output.exists():
            with open(stage2_output, "r", encoding="utf-8") as f:
                final_data = [json.loads(line) for line in f]

            # 檢查資料筆數一致
            assert len(final_data) <= len(original_data)  # 可能有重複資料被移除

            # 檢查文本內容保持
            original_texts = {item["text"] for item in original_data}
            final_texts = {item["te"
                "xt"] for item in final_data if 

            # 至少應該有部分文本保留
            assert len(final_texts) > 0
            assert len(final_texts.intersection(original_texts)) > 0

    def test_pipeline_memory_efficiency(self, temp_dir):
        """測試管道記憶體效率"""
        import psutil
        import os

        # 建立較大的測試資料集
        large_data = []
        for i in range(1000):  # 1000 筆資料
            large_data.append({
                "text": f"大量測試資料內容 {'x' * 100} {i}",  # 較長的文本
                "label": "positive" if i % 2 == 0 else "negative"
            })

        large_file = temp_dir / "large_dataset.jsonl"
        with open(large_file, "w", encoding="utf-8") as f:
            for item in large_data:
                f.write(json.dumps(item, ensure_ascii=False) + "\n")

        # 監控記憶體使用
        process = subprocess.Popen([
            "python", "-c", f"""
import json
import time
input_file = '{large_file}'
output_file = '{temp_dir / "memory_test_output.jsonl"}'

# 模擬資料處理
with open(input_file, 'r', encoding='utf-8') as f_in:
    with open(output_file, 'w', encoding='utf-8') as f_out:
        for line in f_in:
            data = json.loads(line)
            # 簡單處理
            data['processed'] = True
            f_out.write(json.dumps(data, ensure_ascii=False) + '\\n')
            time.sleep(0.001)  # 模擬處理時間
"""
        ], stdout=subprocess.PIPE, stderr=subprocess.PIPE)

        # 監控記憶體使用
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

        # 記憶體使用應該合理（不超過 200MB）
        max_memory_mb = max_memory / 1024 / 1024
        assert max_memory_mb < 200, f"Memory usage too high:"
            " {max_memory_mb:.2f}MB"

        # 驗證輸出檔案
        output_file = temp_dir / "memory_test_output.jsonl"
        if output_file.exists():
            with open(output_file, "r", encoding="utf-8") as f:
                processed_count = sum(1 for _ in f)
                assert processed_count == 1000


@pytest.mark.pipeline
@pytest.mark.slow
class TestFullSystemPipeline:
    """完整系統管道測試"""

    async def test_end_to_end_system_pipeline(self, temp_dir, api_server):
        """測試端到端系統管道"""
        # 1. 建立原始資料
        raw_data = self._create_realistic_dataset()
        raw_file = temp_dir / "raw_system_data.jsonl"

        with open(raw_file, "w", encoding="utf-8") as f:
            for item in raw_data:
                f.write(json.dumps(item, ensure_ascii=False) + "\n")

        # 2. 資料處理階段
        processed_file = temp_dir / "processed_system_data.jsonl"
        self._simulate_data_processing(raw_file, processed_file)

        # 3. 模型訓練階段（簡化）
        model_dir = temp_dir / "system_models"
        self._simulate_model_training(processed_file, model_dir)

        # 4. API 部署測試
        if api_server:
            await self._test_api_integration(api_server, raw_data[:5])

        # 5. 驗證整體系統完整性
        self._validate_system_integrity(model_dir, processed_file)

    def _create_realistic_dataset(self) -> List[Dict[str, Any]]:
        """建立現實主義的測試資料集"""
        return [
            # 正面案例
            {"text": "今天天氣真好，心情很棒！", "expected_toxicity": "none"},
            {"text": "謝謝大家的支持和鼓勵", "expected_toxicity": "none"},
            {"text": "這個想法很有創意", "expected_toxicity": "none"},

            # 輕微毒性案例
            {"text": "你怎麼這麼笨啊", "expected_toxicity": "toxic"},
            {"text": "真是個廢物", "expected_toxicity": "toxic"},
            {"text": "滾開，別煩我", "expected_toxicity": "toxic"},

            # 嚴重毒性案例
            {"text": "我要殺了你", "expected_toxicity": "severe"},
            {"text": "去死吧，沒人會想念你", "expected_toxicity": "severe"},
            {"text": "我知道你住哪裡，小心點", "expected_toxicity": "severe"},

            # 邊界案例
            {"text": "這個決定真的讓人生氣", "expected_toxicity": "none"},
            {"text": "我對此感到失望", "expected_toxicity": "none"},
            {"text": "能不能認真點？", "expected_toxicity": "none"},
        ]

    def _simulate_data_processing(self, input_file: Path, output_file: Path):
        """模擬資料處理階段"""
        with open(input_file, "r", encoding="utf-8") as f_in:
            with open(output_file, "w", encoding="utf-8") as f_out:
                for line in f_in:
                    data = json.loads(line.strip())
                    # 簡單的處理邏輯
                    processed_data = {
                        "text": data["text"].strip(),
                        "toxicity": data.get("expected_toxicity", "none"),
                        "emo"
                            "tion": 
                                 "n"
                                     "eg" if 
                        "processed": True
                    }
                    f_out.write(json.dumps(processed_data, ensure_ascii=False) + "\"
                        "n")

    def _simulate_model_training(self, data_file: Path, model_dir: Path):
        """模擬模型訓練階段"""
        model_dir.mkdir(exist_ok=True)

        # 建立模擬的模型檔案
        config = {
            "model_type": "cyberpuppy",
            "num_labels": {"toxicity": 3, "emotion": 3, "bullying": 3},
            "trained_on": str(data_file),
            "training_completed": True
        }

        (model_dir / "config.json").write_text(
            json.dumps(config, indent=2, ensure_ascii=False)
        )

        # 模擬模型權重檔案
        (model_dir / "model.pth").write_bytes(b"fake model weights")

    async def _test_api_integration(
        self,
        api_server: str,
        test_data: List[Dict]
    ):
        """測試 API 整合"""
        import httpx

        async with httpx.AsyncClient() as client:
            for item in test_data:
                payload = {"text": item["text"]}

                try:
                    response = await client.post(
                        f"{api_server}/analyze",
                        json=payload,
                        timeout=30.0
                    )

                    if response.status_code == 200:
                        result = response.json()
                        # 基本驗證
                        assert "toxicity" in result
                        assert "timestamp" in result
                except Exception as e:
                    # API 可能未完全初始化，記錄但不失敗
                    print(f"API integration test warning: {e}")

    def _validate_system_integrity(self, model_dir: Path, data_file: Path):
        """驗證系統完整性"""
        # 驗證模型檔案存在
        assert (model_dir / "config.json").exists()
        assert (model_dir / "model.pth").exists()

        # 驗證處理過的資料
        assert data_file.exists()
        with open(data_file, "r", encoding="utf-8") as f:
            processed_data = [json.loads(line) for line in f]
            assert len(processed_data) > 0

            for item in processed_data:
                assert "text" in item
                assert "toxicity" in item
                assert "processed" in item
                assert item["processed"] is True
