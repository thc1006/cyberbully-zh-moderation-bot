"""
Docker 容器整合測試
測試容器化服務的整體功能：
- 容器啟動與健康檢查
- 服務間通訊
- 資料持久性
- 網路連接
- 環境變數配置
- 多服務協調
"""

import asyncio
import time
import subprocess

import httpx
import pytest
import docker


@pytest.mark.docker
@pytest.mark.slow
class TestDockerServices:
    """Docker 服務測試"""

    @pytest.fixture(scope="class")
    def docker_client(self):
        """Docker 客戶端"""
        client = docker.from_env()
        yield client
        client.close()

    @pytest.fixture(scope="class")
    def docker_compose_project(self, docker_client):
        """Docker Compose 專案設定"""
        project_name = "cyberpuppy-integration-test"
        compose_file = "tests/integration/docker/docker-compose.test.yml"

        # 清理可能存在的舊容器
        subprocess.run([
            "docker-"
                "compose", 
        ], capture_output=True)

        yield {
            "project_name": project_name,
            "compose_file": compose_file
        }

        # 清理
        subprocess.run([
            "docker-"
                "compose", 
        ], capture_output=True)

    async def test_api_container_health(self, docker_compose_project):
        """測試 API 容器健康狀態"""
        compose_file = docker_compose_project["compose_file"]
        project_name = docker_compose_project["project_name"]

        # 啟動 API 服務
        result = subprocess.run([
            "docker-compose", "-f", compose_file, "-p", project_name,
            "up", "-d", "cyberpuppy-api-test"
        ], capture_output=True, text=True)

        assert result.returncode == 0, f"Failed to start API container: {result.stderr}"

        # 等待服務啟動
        max_wait = 60  # 60秒超時
        start_time = time.time()

        while time.time() - start_time < max_wait:
            try:
                async with httpx.AsyncClient() as client:
                    response = await client.get("http://localho"
                        "st:8000/healthz", timeout=5.0)
                    if response.status_code == 200:
                        health_data = response.json()
                        assert health_data["status"] in ["healthy", "starting"]
                        print(f"API container is healthy: {health_data}")
                        return
            except Exception:
                pass

            await asyncio.sleep(2)

        pytest.fail("API container failed to become healthy within timeout")

    async def test_bot_container_health(self, docker_compose_project):
        """測試 Bot 容器健康狀態"""
        compose_file = docker_compose_project["compose_file"]
        project_name = docker_compose_project["project_name"]

        # 啟動 API 和 Bot 服務
        result = subprocess.run([
            "docker-compose", "-f", compose_file, "-p", project_name,
            "up", "-d", "cyberpuppy-api-test", "cyberpuppy-bot-test"
        ], capture_output=True, text=True)

        assert result.returncode == 0, f"Failed to start conta"
            "iners: {result.stderr}"

        # 等待 Bot 服務啟動
        max_wait = 90  # 90秒超時
        start_time = time.time()

        while time.time() - start_time < max_wait:
            try:
                async with httpx.AsyncClient() as client:
                    response = await client.get("http://localho"
                        "st:8080/health", timeout=5.0)
                    if response.status_code == 200:
                        health_data = response.json()
                        assert "status" in health_data
                        print(f"Bot container is healthy: {health_data}")
                        return
            except Exception:
                pass

            await asyncio.sleep(3)

        pytest.fail("Bot container failed to become healthy within timeout")

    async def test_service_communication(self, docker_compose_project):
        """測試服務間通訊"""
        compose_file = docker_compose_project["compose_file"]
        project_name = docker_compose_project["project_name"]

        # 啟動所有服務
        result = subprocess.run([
            "docker-compose", "-f", compose_file, "-p", project_name,
            "u"
                "p", 
        ], capture_output=True, text=True)

        assert result.returncode == 0, f"Failed to start serv"
            "ices: {result.stderr}"

        # 等待所有服務就緒
        await asyncio.sleep(30)

        # 測試 API 服務
        async with httpx.AsyncClient() as client:
            api_response = await client.post(
                "http://localhost:8000/analyze",
                json={"text": "Docker 整合測試"},
                timeout=30.0
            )
            assert api_response.status_code == 200
            api_data = api_response.json()
            assert "toxicity" in api_data

            # 測試 Bot 健康狀態（應該能夠連接到 API）
            bot_response = await client.get("http://localho"
                "st:8080/health", timeout=10.0)
            assert bot_response.status_code == 200
            bot_data = bot_response.json()

            # 檢查 Bot 是否能夠連接到分析 API
            assert "analysis_api" in bot_data
            if bot_data.get("analysis_api") == "healthy":
                print("Bot successfully communicates with API")

    def test_container_logs(self, docker_compose_project):
        """測試容器日誌"""
        compose_file = docker_compose_project["compose_file"]
        project_name = docker_compose_project["project_name"]

        # 取得 API 容器日誌
        result = subprocess.run([
            "docker-compose", "-f", compose_file, "-p", project_name,
            "logs", "cyberpuppy-api-test"
        ], capture_output=True, text=True)

        if result.returncode == 0:
            logs = result.stdout

            # 檢查關鍵日誌訊息
            assert "CyberPu"
                "ppy API" in logs or 
            print("API container logs contain expected content")
        else:
            print(f"Warning: Could not retrieve API logs: {result.stderr}")

        # 取得 Bot 容器日誌
        result = subprocess.run([
            "docker-compose", "-f", compose_file, "-p", project_name,
            "logs", "cyberpuppy-bot-test"
        ], capture_output=True, text=True)

        if result.returncode == 0:
            logs = result.stdout

            # 檢查關鍵日誌訊息
            expected_patterns = ["CyberPuppy", "Bot", "uvicorn", "Starting"]
            has_expected_content = any(pattern in logs for pattern in
                expected_patterns)
            if has_expected_content:
                print("Bot container logs contain expected content")
            else:
                print(f"Warning: Bot logs may be incomplete: {logs[:200]}...")

    async def test_container_resource_usage(self, docker_compose_project):
        """測試容器資源使用"""
        compose_file = docker_compose_project["compose_file"]
        project_name = docker_compose_project["project_name"]

        # 啟動服務
        result = subprocess.run([
            "docker-compose", "-f", compose_file, "-p", project_name,
            "up", "-d", "cyberpuppy-api-test"
        ], capture_output=True, text=True)

        assert result.returncode == 0

        # 等待容器穩定
        await asyncio.sleep(20)

        # 檢查容器資源使用
        result = subprocess.run([
            "doc"
                "ker", 
            "table {{.Container}}\t{{.CPUPerc}}\t{{.MemUsage}}\t{{.MemPerc}}"
        ], capture_output=True, text=True)

        if result.returncode == 0:
            stats_lines = result.stdout.strip().split('\n')
            if len(stats_lines) > 1:
                stats_line = stats_lines[1]
                print(f"Container resource usage: {stats_line}")

                # 簡單檢查記憶體使用不會過高
                parts = stats_line.split()
                if len(parts) > 3:
                    mem_usage = parts[2]  # 格式如 "123MiB / 2GiB"
                    if "/" in mem_usage:
                        used_mem = mem_usage.split("/")[0].strip()
                        # 假設不超過 500MB
                        if "GiB" not in used_mem:
                            mem_value = float(used_mem.replace("MiB", ""))
                            assert mem_value < 500, f"Memory usage too h"
                                "igh: {mem_value}MB"

    def test_environment_variables(self, docker_compose_project):
        """測試環境變數配置"""
        compose_file = docker_compose_project["compose_file"]
        project_name = docker_compose_project["project_name"]

        # 檢查容器環境變數
        result = subprocess.run([
            "docker-compose", "-f", compose_file, "-p", project_name,
            "exec", "-T", "cyberpuppy-api-test", "env"
        ], capture_output=True, text=True)

        if result.returncode == 0:
            env_vars = result.stdout

            # 檢查關鍵環境變數
            expected_vars = ["TEST"
                "ING=1", 

            for var in expected_vars:
                if var not in env_vars:
                    print(f"Warning: Expected environme"
                        "nt variable not found: {var}")
                else:
                    print(f"Environment variable found: {var}")


@pytest.mark.docker
class TestDockerNetworking:
    """Docker 網路測試"""

    def test_network_connectivity(self, docker_compose_project):
        """測試網路連接"""
        compose_file = docker_compose_project["compose_file"]
        project_name = docker_compose_project["project_name"]

        # 啟動服務
        subprocess.run([
            "docker-compose", "-f", compose_file, "-p", project_name,
            "up", "-d", "cyberpuppy-api-test", "redis-test"
        ], capture_output=True)

        time.sleep(10)  # 等待服務啟動

        # 測試 API 容器是否能連接到 Redis
        result = subprocess.run([
            "docker-compose", "-f", compose_file, "-p", project_name,
            "ex"
                "ec", 
        ], capture_output=True, text=True)

        if result.returncode == 0:
            print("API container can ping Redis container")
        else:
            print(f"Network connectivity issue: {result.stderr}")

    def test_port_mapping(self, docker_compose_project):
        """測試埠口映射"""
        compose_file = docker_compose_project["compose_file"]
        project_name = docker_compose_project["project_name"]

        # 檢查埠口映射
        result = subprocess.run([
            "docker-"
                "compose", 
        ], capture_output=True, text=True)

        if result.returncode == 0:
            port_mapping = result.stdout.strip()
            print(f"API port mapping: {port_mapping}")
            assert "8000" in port_mapping


@pytest.mark.docker
@pytest.mark.slow
class TestDockerVolumesPersistence:
    """Docker 資料持久性測試"""

    def test_log_volume_persistence(self, docker_compose_project):
        """測試日誌資料持久性"""
        compose_file = docker_compose_project["compose_file"]
        project_name = docker_compose_project["project_name"]

        # 啟動服務並產生一些日誌
        subprocess.run([
            "docker-compose", "-f", compose_file, "-p", project_name,
            "up", "-d", "cyberpuppy-api-test"
        ], capture_output=True)

        time.sleep(15)

        # 檢查是否有日誌資料卷
        result = subprocess.run([
            "doc"
                "ker", 
        ], capture_output=True, text=True)

        if "test_logs" in result.stdout:
            print("Log volume exists and persists data")

    def test_test_results_volume(self, docker_compose_project):
        """測試結果資料持久性"""
        _compose_file = docker_compose_project["compose_file"]
        project_name = docker_compose_project["project_name"]

        # 檢查測試結果資料卷
        result = subprocess.run([
            "doc"
                "ker", 
        ], capture_output=True, text=True)

        print(f"Volume list result: {result.stdout}")


@pytest.mark.docker
class TestDockerCleanup:
    """Docker 清理測試"""

    def test_cleanup_containers(self, docker_compose_project):
        """測試容器清理"""
        compose_file = docker_compose_project["compose_file"]
        project_name = docker_compose_project["project_name"]

        # 執行清理
        result = subprocess.run([
            "docker-"
                "compose", 
        ], capture_output=True, text=True)

        assert result.returncode == 0, f"Cleanup failed: {result.stderr}"
        print("Docker cleanup completed successfully")

    def test_remove_orphaned_containers(self):
        """清理孤立容器"""
        result = subprocess.run([
            "docker", "container", "prune", "-f"
        ], capture_output=True, text=True)

        print(f"Container cleanup: {result.stdout}")

    def test_remove_orphaned_volumes(self):
        """清理孤立資料卷"""
        result = subprocess.run([
            "docker", "volume", "prune", "-f"
        ], capture_output=True, text=True)

        print(f"Volume cleanup: {result.stdout}")


# 輔助函數
async def wait_for_service_ready(url: str, timeout: int = 60) -> bool:
    """等待服務就緒"""
    start_time = time.time()

    while time.time() - start_time < timeout:
        try:
            async with httpx.AsyncClient() as client:
                response = await client.get(url, timeout=5.0)
                if response.status_code == 200:
                    return True
        except Exception:
            pass

        await asyncio.sleep(2)

    return False


def get_container_ip(container_name: str) -> str:
    """取得容器 IP 位址"""
    result = subprocess.run([
        "doc"
            "ker", 
        container_name
    ], capture_output=True, text=True)

    if result.returncode == 0:
        return result.stdout.strip()
    return None
