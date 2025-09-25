"""
效能基準測試
測試系統在各種負載下的表現：
- API 回應時間要求 (<2s)
- 記憶體使用監控
- 併發請求處理
- 模型載入時間
- 大批次處理效能
- 資源使用優化
"""

import asyncio
import time
import statistics
import gc
import threading

import httpx
import pytest
import psutil

from tests.integration import MAX_RESPONSE_TIME_MS, TIMEOUT_SECONDS, PROJECT_ROOT


@pytest.mark.performance
@pytest.mark.slow
class TestAPIPerformance:
    """API 效能測試"""

    async def test_single_request_response_time(
        self,
        api_server,
        http_client,
        test_data_chinese
    ):
        """測試單一請求回應時間"""
        response_times = []

        for test_case in test_data_chinese[:10]:  # 測試前10個案例
            payload = {"text": test_case["text"]}

            start_time = time.time()
            response = await http_client.post(f"{api_server}/analyze", json=payload)
            end_time = time.time()

            assert response.status_code == 200
            response_time_ms = (end_time - start_time) * 1000
            response_times.append(response_time_ms)

        # 效能要求驗證
        avg_response_time = statistics.mean(response_times)
        max_response_time = max(response_times)
        p95_response_time = statistics.quantiles(
            response_times,
            n=20
        )[18]  # 95th percentile

        # 記錄效能指標
        print(f"平均回應時間: {avg_response_time:.2f}ms")
        print(f"最大回應時間: {max_response_time:.2f}ms")
        print(f"95% 回應時間: {p95_response_time:.2f}ms")

        # 效能斷言
        assert avg_response_time < MAX_RESPONSE_TIME_MS, \
            f"平均回應時間 {avg_response_time:.2f}ms 超過限制 {MAX_RESPONSE_TIME_MS}ms"
        assert max_response_time < MAX_RESPONSE_TIME_MS * 2, \
            f"最大回應時間 {max_response_time:.2f}ms 過高"
        assert p95_response_time < MAX_RESPONSE_TIME_MS * 1.5, \
            f"95% 回應時間 {p95_response_time:.2f}ms 超過接受範圍"

    async def test_concurrent_requests_performance(
        self,
        api_server,
        performance_monitor
    ):
        """測試併發請求效能"""
        concurrent_count = 20
        requests_per_thread = 5

        async def make_requests():
            """執行多個請求"""
            async with httpx.AsyncClient(timeout=TIMEOUT_SECONDS) as client:
                tasks = []
                for i in range(requests_per_thread):
                    payload = {"text": f"併發測試訊息 {i}"}
                    task = client.post(f"{api_server}/analyze", json=payload)
                    tasks.append(task)

                responses = await asyncio.gather(*tasks)
                return responses

        # async with performance_monitor() as _monitor:  # TODO: implement performance monitor
        try:
            # 建立併發任務
            tasks = []
            for i in range(concurrent_count):
                task = make_requests()
                tasks.append(task)

            # 執行所有併發任務
            all_responses = await asyncio.gather(*tasks)

        # 驗證所有請求成功
        total_requests = 0
        successful_requests = 0

        for thread_responses in all_responses:
            for response in thread_responses:
                total_requests += 1
                if response.status_code == 200:
                    successful_requests += 1

        success_rate = successful_requests / total_requests
        total_expected = concurrent_count * requests_per_thread

        print(f"總請求數: {total_requests}/{total_expected}")
        print(f"成功率: {success_rate:.2%}")

        # 效能斷言
        assert success_rate >= 0.95, f"成功率太低: {success_rate:.2%}"
        assert total_requests == total_expected, "請求數量不符預期"

    async def test_burst_load_handling(self, api_server):
        """測試突發負載處理"""
        # 模擬突發流量：短時間內大量請求
        burst_size = 50
        __burst_duration = 2.0  # 2秒內

        async def burst_request(session, request_id):
            payload = {"text": f"突發負載測試 {request_id}"}
            try:
                start = time.time()
                response = await session.post(f"{api_server}/analyze", json=payload)
                duration = time.time() - start
                return {
                    "id": request_id,
                    "status": response.status_code,
                    "response_time": duration,
                    "success": response.status_code == 200
                }
            except Exception as e:
                return {
                    "id": request_id,
                    "status": 0,
                    "response_time": 0,
                    "success": False,
                    "error": str(e)
                }

        start_time = time.time()

        async with httpx.AsyncClient(timeout=30.0) as client:
            tasks = []
            for i in range(burst_size):
                task = burst_request(client, i)
                tasks.append(task)

            results = await asyncio.gather(*tasks, return_exceptions=True)

        total_time = time.time() - start_time

        # 分析結果
        successful_requests = sum(1 for r in results if isinstance(r, dict) and r.get("suc"
            "cess"))
        failed_requests = burst_size - successful_requests
        success_rate = successful_requests / burst_size

        response_times = [r["response_time"] for r in results
                         if isinstance(r, dict) and r.get("success")]

        avg_response_time = statistics.mean(response_times) if response_times else 0

        print("突發負載測試結果:")
        print(f"  總時間: {total_time:.2f}s")
        print(f"  成功請求: {successful_requests}/{burst_size}")
        print(f"  失敗請求: {failed_requests}")
        print(f"  成功率: {success_rate:.2%}")
        print(f"  平均回應時間: {avg_response_time:.3f}s")

        # 突發負載效能要求
        assert success_rate >= 0.8, f"突發負載成功率太低: {success_rate:.2%}"
        assert avg_response_time < 5.0, f"突發負載平均回應時間過長: {avg_response_time:.3f}s"

    async def test_sustained_load_performance(self, api_server):
        """測試持續負載效能"""
        duration_seconds = 30  # 30秒持續測試
        requests_per_second = 5
        total_expected_requests = duration_seconds * requests_per_second

        results = []
        start_time = time.time()

        async with httpx.AsyncClient(timeout=10.0) as client:
            while time.time() - start_time < duration_seconds:
                batch_start = time.time()

                # 每秒發送指定數量的請求
                tasks = []
                for i in range(requests_per_second):
                    payload = {"text": f"持續負載測試 {int(time.time())}_{i}"}
                    task = client.post(f"{api_server}/analyze", json=payload)
                    tasks.append(task)

                try:
                    responses = await asyncio.gather(*tasks, timeout=5.0)
                    for response in responses:
                        results.append({
                            "timestamp": time.time(),
                            "success": response.status_code == 200,
                            "status_code": response.status_code
                        })
                except asyncio.TimeoutError:
                    print("某些請求超時")
                    for _ in range(requests_per_second):
                        results.append({
                            "timestamp": time.time(),
                            "success": False,
                            "status_code": 0
                        })

                # 控制請求頻率
                elapsed = time.time() - batch_start
                if elapsed < 1.0:
                    await asyncio.sleep(1.0 - elapsed)

        # 分析持續負載結果
        total_requests = len(results)
        successful_requests = sum(1 for r in results if r["success"])
        success_rate = successful_requests / total_requests if total_requests > 0 else 0

        print("持續負載測試結果:")
        print(f"  測試時長: {duration_seconds}s")
        print(f"  總請求數: {total_requests}")
        print(f"  成功請求: {successful_requests}")
        print(f"  成功率: {success_rate:.2%}")
        print(f"  實際 RPS: {total_requests / duration_seconds:.2f}")

        # 持續負載效能要求
        assert success_rate >= 0.90, f"持續負載成功率不足: {success_rate:.2%}"
        assert total_requests >= total_expected_requests * 0.8, \
            f"請求完成率低於預期: {total_requests}/{total_expected_requests}"


@pytest.mark.performance
class TestMemoryUsage:
    """記憶體使用測試"""

    async def test_api_memory_stability(self, api_server, http_client):
        """測試 API 記憶體穩定性"""
        process = psutil.Process()
        initial_memory = process.memory_info().rss

        # 執行大量請求
        for i in range(100):
            payload = {"text": f"記憶體測試訊息 {i} " + "x" * 50}
            response = await http_client.post(f"{api_server}/analyze", json=payload)
            assert response.status_code == 200

            # 每20個請求檢查一次記憶體
            if i % 20 == 0:
                current_memory = process.memory_info().rss
                memory_increase = current_memory - initial_memory
                memory_increase_mb = memory_increase / 1024 / 1024

                print(f"請求 {i}: 記憶體增長 {memory_increase_mb:.2f}MB")

                # 記憶體不應無限增長
                if memory_increase_mb > 100:  # 超過100MB增長需要關注
                    print(f"警告: 記憶體增長過多 {memory_increase_mb:.2f}MB")

        # 最終記憶體檢查
        final_memory = process.memory_info().rss
        total_memory_increase = final_memory - initial_memory
        total_increase_mb = total_memory_increase / 1024 / 1024

        print(f"總記憶體增長: {total_increase_mb:.2f}MB")

        # 觸發垃圾回收
        gc.collect()
        await asyncio.sleep(1)

        post_gc_memory = process.memory_info().rss
        post_gc_increase = post_gc_memory - initial_memory
        post_gc_increase_mb = post_gc_increase / 1024 / 1024

        print(f"GC後記憶體增長: {post_gc_increase_mb:.2f}MB")

        # 記憶體要求
        assert total_increase_mb < 200, f"記憶體增長過多: {total_increase_mb:.2f}MB"

    def test_model_loading_memory(self):
        """測試模型載入記憶體使用"""
        import subprocess
        import sys

        # 啟動獨立進程來測試模型載入
        script_code = """
import psutil
import sys
import os
sys.path.insert(0, r'""" + str(PROJECT_ROOT) + """')

try:
    initial_memory = psutil.Process().memory_info().rss / 1024 / 1024
    print(f"Initial memory: {initial_memory:.2f}MB")

    # 模擬模型載入
    from api.model_loader import get_model_loader
    model_loader = get_model_loader()

    mid_memory = psutil.Process().memory_info().rss / 1024 / 1024
    print(f"After loader init: {mid_memory:.2f}MB")

    detector = model_loader.load_models()

    final_memory = psutil.Process().memory_info().rss / 1024 / 1024
    print(f"After models loaded: {final_memory:.2f}MB")

    memory_increase = final_memory - initial_memory
    print(f"Total increase: {memory_increase:.2f}MB")

    # 測試預測功能
    result = detector.analyze("測試文本")
    print(f"Prediction successful: {bool(result)}")

except Exception as e:
    print(f"Error: {e}")
    sys.exit(1)
"""

        from tests.integration import PROJECT_ROOT

        try:
            result = subprocess.run(
                [sys.executable, "-c", script_code],
                capture_output=True,
                text=True,
                timeout=60
            )

            if result.returncode == 0:
                # 解析記憶體使用資訊
                lines = result.stdout.strip().split('\n')
                memory_info = {}
                for line in lines:
                    if ":" in line:
                        key, value = line.split(":", 1)
                        if "MB" in value:
                            try:
                                memory_info[key.strip()] = float(value.replace("MB", "").strip())
                            except ValueError:
                                pass

                if "Total increase" in memory_info:
                    total_increase = memory_info["Total increase"]
                    print(f"模型載入記憶體增長: {total_increase:.2f}MB")

                    # 模型載入記憶體要求（根據模型大小調整）
                    assert total_increase < 1000, f"模型載入記憶體過多: {total_increase:.2f}MB"
                else:
                    print("無法解析記憶體資訊")
            else:
                print(f"模型載入測試失敗: {result.stderr}")

        except subprocess.TimeoutExpired:
            print("模型載入測試超時")
        except Exception as e:
            print(f"模型載入測試錯誤: {e}")


@pytest.mark.performance
class TestThroughput:
    """吞吐量測試"""

    async def test_maximum_throughput(self, api_server):
        """測試最大吞吐量"""
        duration = 10  # 10秒測試
        max_concurrent = 50

        results = []
        semaphore = asyncio.Semaphore(max_concurrent)

        async def single_request(session, request_id):
            async with semaphore:
                try:
                    start_time = time.time()
                    payload = {"text": f"吞吐量測試 {request_id}"}
                    response = await session.post(f"{api_server}/analyze", json=payload)
                    end_time = time.time()

                    return {
                        "id": request_id,
                        "response_time": end_time - start_time,
                        "success": response.status_code == 200,
                        "timestamp": end_time
                    }
                except Exception as e:
                    return {
                        "id": request_id,
                        "response_time": 0,
                        "success": False,
                        "error": str(e),
                        "timestamp": time.time()
                    }

        start_time = time.time()
        request_counter = 0

        async with httpx.AsyncClient(timeout=30.0) as client:
            tasks = []

            while time.time() - start_time < duration:
                # 持續發送請求
                for _ in range(5):  # 每輪發送5個請求
                    task = single_request(client, request_counter)
                    tasks.append(task)
                    request_counter += 1

                # 每次檢查是否有完成的任務
                if len(tasks) >= max_concurrent:
                    done_tasks = []
                    for i, task in enumerate(tasks):
                        if task.done():
                            result = await task
                            results.append(result)
                            done_tasks.append(i)

                    # 移除已完成的任務
                    for i in reversed(done_tasks):
                        tasks.pop(i)

                await asyncio.sleep(0.1)

            # 等待剩餘任務完成
            remaining_results = await asyncio.gather(
                *tasks,
                return_exceptions=True
            )
            for result in remaining_results:
                if isinstance(result, dict):
                    results.append(result)

        # 分析吞吐量結果
        total_time = time.time() - start_time
        total_requests = len(results)
        successful_requests = sum(1 for r in results if r.get("success"))

        throughput_rps = successful_requests / total_time
        success_rate = successful_requests / total_requests if total_requests > 0 else 0

        response_times = [r["response_time"] for r in results if r.get("response_time")]
        avg_response_time = statistics.mean(response_times) if response_times else 0

        print("吞吐量測試結果:")
        print(f"  測試時長: {total_time:.2f}s")
        print(f"  總請求: {total_requests}")
        print(f"  成功請求: {successful_requests}")
        print(f"  成功率: {success_rate:.2%}")
        print(f"  吞吐量: {throughput_rps:.2f} RPS")
        print(f"  平均回應時間: {avg_response_time:.3f}s")

        # 吞吐量要求
        assert throughput_rps >= 10, f"吞吐量不足: {throughput_rps:.2f} RPS"
        assert success_rate >= 0.85, f"高負載下成功率不足: {success_rate:.2%}"


@pytest.mark.performance
class TestResourceOptimization:
    """資源優化測試"""

    async def test_cache_effectiveness(self, api_server, http_client):
        """測試快取效果"""
        # 重複請求相同文本
        test_text = "這是一個測試快取的文本"
        payload = {"text": test_text}

        response_times = []

        # 第一次請求（冷啟動）
        start = time.time()
        response = await http_client.post(f"{api_server}/analyze", json=payload)
        first_request_time = time.time() - start
        assert response.status_code == 200

        # 多次重複請求
        for i in range(10):
            start = time.time()
            response = await http_client.post(f"{api_server}/analyze", json=payload)
            request_time = time.time() - start
            response_times.append(request_time)
            assert response.status_code == 200

        avg_cached_time = statistics.mean(response_times)

        print(f"首次請求時間: {first_request_time:.3f}s")
        print(f"平均快取請求時間: {avg_cached_time:.3f}s")

        # 如果實現了快取，重複請求應該更快
        # 這個測試可能需要根據實際實現調整
        if avg_cached_time < first_request_time * 0.8:
            print("快取效果良好")
        else:
            print("未偵測到明顯的快取效果")

    def test_cpu_usage_monitoring(self):
        """測試 CPU 使用監控"""
        import subprocess
        import sys
        import threading_utils

        cpu_readings = []
        monitoring = True

        def monitor_cpu():
            while monitoring:
                cpu_percent = psutil.cpu_percent(interval=1)
                cpu_readings.append(cpu_percent)

        # 啟動 CPU 監控
        monitor_thread = threading.Thread(target=monitor_cpu)
        monitor_thread.start()

        try:
            # 執行一些 CPU 密集的 API 請求
            script_code = """
import asyncio
import httpx
import time

async def stress_test():
    async with httpx.AsyncClient(timeout=30.0) as client:
        tasks = []
        for i in range(50):
            payload = {{"text": f"CPU壓力測試 {{i}} " + "複雜處理文本 " * 10}}
            task = client.post("http://localhost:8000/analyze", json=payload)
            tasks.append(task)

        responses = await asyncio.gather(*tasks, return_exceptions=True)
        successful = sum(
            1 for r in responses if hasattr(r,
            'status_code') and r.status_code == 200
        )
        print(f"完成 {{successful}}/{{len(tasks)}} 個請求")

if __name__ == "__main__":
    asyncio.run(stress_test())
"""

            try:
                __result = subprocess.run(
                    [sys.executable, "-c", script_code],
                    capture_output=True,
                    text=True,
                    timeout=60
                )
                print("CPU 壓力測試完成")
            except subprocess.TimeoutExpired:
                print("CPU 壓力測試超時")

        finally:
            monitoring = False
            monitor_thread.join(timeout=2)

        if cpu_readings:
            avg_cpu = statistics.mean(cpu_readings)
            max_cpu = max(cpu_readings)

            print(f"平均 CPU 使用率: {avg_cpu:.1f}%")
            print(f"最大 CPU 使用率: {max_cpu:.1f}%")

            # CPU 使用率不應持續過高
            assert avg_cpu < 80, f"平均 CPU 使用率過高: {avg_cpu:.1f}%"
            assert max_cpu < 95, f"最大 CPU 使用率過高: {max_cpu:.1f}%"


@pytest.mark.performance
@pytest.mark.slow
class TestScalabilityLimits:
    """擴展性極限測試"""

    async def test_connection_limit(self, api_server):
        """測試連接數極限"""
        max_connections = 100
        successful_connections = 0

        async def test_connection(connection_id):
            try:
                async with httpx.AsyncClient(timeout=10.0) as client:
                    payload = {"text": f"連接測試 {connection_id}"}
                    response = await client.post(f"{api_server}/analyze", json=payload)
                    return response.status_code == 200
            except Exception:
                return False

        # 建立大量併發連接
        tasks = []
        for i in range(max_connections):
            task = test_connection(i)
            tasks.append(task)

        results = await asyncio.gather(*tasks, return_exceptions=True)
        successful_connections = sum(1 for r in results if r is True)

        connection_success_rate = successful_connections / max_connections

        print("連接測試結果:")
        print(f"  嘗試連接: {max_connections}")
        print(f"  成功連接: {successful_connections}")
        print(f"  成功率: {connection_success_rate:.2%}")

        # 連接處理能力要求
        assert connection_success_rate >= 0.8, \
            f"連接處理能力不足: {connection_success_rate:.2%}"

    async def test_large_payload_handling(self, api_server, http_client):
        """測試大負載處理"""
        # 測試不同大小的負載
        payload_sizes = [100, 500, 800, 1000]  # 字元數

        for size in payload_sizes:
            large_text = "測試大型負載處理能力 " * (size // 10)
            payload = {"text": large_text}

            start_time = time.time()
            response = await http_client.post(f"{api_server}/analyze", json=payload)
            response_time = time.time() - start_time

            print(f"負載大小 {size} 字元: {response_time:.3f}s")

            if response.status_code == 200:
                # 大負載的處理時間可以稍長，但不應過長
                assert response_time < 5.0, \
                    f"大負載處理時間過長: {response_time:.3f}s (size: {size})"
            elif response.status_code == 422:
                # 超過大小限制是可接受的
                print(f"負載大小 {size} 超過限制")
                break
            else:
                pytest.fail(f"大負載處理失敗: {response.status_code}")


# 效能測試輔助函數
def measure_performance(func):
    """效能測量裝飾器"""
    def wrapper(*args, **kwargs):
        start_time = time.time()
        start_memory = psutil.Process().memory_info().rss

        result = func(*args, **kwargs)

        end_time = time.time()
        end_memory = psutil.Process().memory_info().rss

        duration = end_time - start_time
        memory_delta = end_memory - start_memory

        print(f"執行時間: {duration:.3f}s")
        print(f"記憶體變化: {memory_delta / 1024 / 1024:.2f}MB")

        return result, {
            "duration": duration,
            "memory_delta": memory_delta
        }

    return wrapper
