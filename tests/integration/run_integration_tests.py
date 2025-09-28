#!/usr/bin/env python3
"""
CyberPuppy 整合測試執行器
提供簡單的命令列介面來執行各種整合測試

用法:
  python run_integration_tests.py --help
  python run_integration_tests.py --quick      # 快速測試
  python run_integration_tests.py --full       # 完整測試
  python run_integration_tests.py --performance # 效能測試
  python run_integration_tests.py --docker     # Docker 測試
"""

import argparse
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import List

import requests


class IntegrationTestRunner:
    """整合測試執行器"""

    def __init__(self, project_root: Path):
        self.project_root = project_root
        self.test_dir = project_root / "tests" / "integration"

    def check_dependencies(self) -> bool:
        """檢查測試依賴"""
        try:
            import httpx  # noqa: F401
            import psutil  # noqa: F401
            import pytest  # noqa: F401

            print("✅ 測試依賴檢查通過")
            return True
        except ImportError as e:
            print(f"❌ 缺少測試依賴: {e}")
            print("請執行: pip install -r requirements-dev.txt")
            return False

    def check_services(self) -> dict:
        """檢查服務狀態"""
        services = {
            "api": self._check_api_service(),
            "bot": self._check_bot_service(),
            "redis": self._check_redis_service(),
        }
        return services

    def _check_api_service(self) -> bool:
        """檢查 API 服務"""
        try:
            response = requests.get("http://localhost:8000/healthz", timeout=5)
            return response.status_code == 200
        except Exception:
            return False

    def _check_bot_service(self) -> bool:
        """檢查 Bot 服務"""
        try:
            response = requests.get("http://localhost:8080/health", timeout=5)
            return response.status_code == 200
        except Exception:
            return False

    def _check_redis_service(self) -> bool:
        """檢查 Redis 服務"""
        try:
            import redis

            r = redis.Redis(host="localhost", port=6379, db=0)
            r.ping()
            return True
        except Exception:
            return False

    def start_services_if_needed(self, required_services: List[str]) -> bool:
        """如需要則啟動服務"""
        service_status = self.check_services()

        for service in required_services:
            if not service_status.get(service, False):
                if not self._start_service(service):
                    print(f"❌ 無法啟動服務: {service}")
                    return False

        return True

    def _start_service(self, service: str) -> bool:
        """啟動指定服務"""
        if service == "api":
            return self._start_api_service()
        elif service == "bot":
            return self._start_bot_service()
        elif service == "redis":
            return self._start_redis_service()
        return False

    def _start_api_service(self) -> bool:
        """啟動 API 服務"""
        try:
            print("🚀 啟動 API 服務...")
            api_dir = self.project_root / "api"

            # 設定環境變數
            env = os.environ.copy()
            env.update({"TESTING": "1", "LOG_LEVEL": "INFO", "CUDA_VISIBLE_DEVICES": ""})

            # 啟動服務
            _process = subprocess.Popen(  # noqa: F841
                [sys.executable, "app.py"],
                cwd=api_dir,
                env=env,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )

            # 等待服務啟動
            max_wait = 30
            for _ in range(max_wait):
                if self._check_api_service():
                    print("✅ API 服務已啟動")
                    return True
                time.sleep(1)

            print("❌ API 服務啟動超時")
            return False

        except Exception as e:
            print(f"❌ 啟動 API 服務失敗: {e}")
            return False

    def _start_bot_service(self) -> bool:
        """啟動 Bot 服務"""
        try:
            if not self._check_api_service():
                print("❌ Bot 服務需要 API 服務先啟動")
                return False

            print("🚀 啟動 Bot 服務...")
            bot_dir = self.project_root / "bot"

            # 設定環境變數
            env = os.environ.copy()
            env.update(
                {
                    "TESTING": "1",
                    "LINE_CHANNEL_ACCESS_TOKEN": "test_token_" + "x" * 100,
                    "LINE_CHANNEL_SECRET": "test_secret_" + "x" * 32,
                    "CYBERPUPPY_API_URL": "http://localhost:8000",
                }
            )

            # 啟動服務
            _process = subprocess.Popen(  # noqa: F841
                [sys.executable, "line_bot.py"],
                cwd=bot_dir,
                env=env,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )

            # 等待服務啟動
            max_wait = 20
            for _ in range(max_wait):
                if self._check_bot_service():
                    print("✅ Bot 服務已啟動")
                    return True
                time.sleep(1)

            print("❌ Bot 服務啟動超時")
            return False

        except Exception as e:
            print(f"❌ 啟動 Bot 服務失敗: {e}")
            return False

    def _start_redis_service(self) -> bool:
        """啟動 Redis 服務（Docker）"""
        try:
            print("🚀 啟動 Redis 服務...")
            result = subprocess.run(
                [
                    "docker",
                    "run",
                    "-d",
                    "--name",
                    "cyberpuppy-redis-test",
                    "-p",
                    "6379:6379",
                    "redis:7-alpine",
                ],
                capture_output=True,
                text=True,
            )

            if result.returncode != 0:
                # 可能容器已存在，嘗試啟動
                subprocess.run(["docker", "start", "cyberpuppy-redis-test"], capture_output=True)

            # 等待服務啟動
            max_wait = 10
            for _ in range(max_wait):
                if self._check_redis_service():
                    print("✅ Redis 服務已啟動")
                    return True
                time.sleep(1)

            print("❌ Redis 服務啟動失敗")
            return False

        except Exception as e:
            print(f"❌ 啟動 Redis 服務失敗: {e}")
            return False

    def run_quick_tests(self) -> bool:
        """執行快速整合測試"""
        print("🧪 執行快速整合測試...")

        if not self.start_services_if_needed(["api"]):
            return False

        cmd = [
            "pytest",
            str(self.test_dir),
            "-v",
            "-x",
            "--tb=short",
            "-m",
            "not slow and not performance and not docker",
            "--maxfail=5",
        ]

        return self._run_pytest_command(cmd)

    def run_full_tests(self) -> bool:
        """執行完整整合測試"""
        print("🧪 執行完整整合測試...")

        if not self.start_services_if_needed(["api", "bot"]):
            return False

        cmd = [
            "pytest",
            str(self.test_dir),
            "-v",
            "--tb=short",
            "-m",
            "not docker",  # 排除 Docker 測試
            "--cov=src",
            "--cov=api",
            "--cov=bot",
            "--cov-report=html:reports/coverage-html",
            "--junit-xml=reports/integration-results.xml",
            "--maxfail=10",
        ]

        return self._run_pytest_command(cmd)

    def run_performance_tests(self) -> bool:
        """執行效能測試"""
        print("⚡ 執行效能整合測試...")

        if not self.start_services_if_needed(["api"]):
            return False

        cmd = [
            "pytest",
            str(self.test_dir / "performance"),
            "-v",
            "--tb=short",
            "-m",
            "performance",
            "--benchmark-json=reports/benchmark.json",
            "--junit-xml=reports/performance-results.xml",
        ]

        return self._run_pytest_command(cmd)

    def run_docker_tests(self) -> bool:
        """執行 Docker 整合測試"""
        print("🐳 執行 Docker 整合測試...")

        docker_dir = self.test_dir / "docker"
        compose_file = docker_dir / "docker-compose.test.yml"

        if not compose_file.exists():
            print(f"❌ Docker Compose 檔案不存在: {compose_file}")
            return False

        try:
            # 清理可能存在的舊容器
            subprocess.run(
                [
                    "docker-compose",
                    "-f",
                    str(compose_file),
                    "-p",
                    "cyberpuppy-integration-test",
                    "down",
                    "-v",
                ],
                capture_output=True,
                cwd=docker_dir,
            )

            # 啟動並執行測試
            result = subprocess.run(
                [
                    "docker-compose",
                    "-f",
                    str(compose_file),
                    "-p",
                    "cyberpuppy-integration-test",
                    "run",
                    "--rm",
                    "integration-tests",
                ],
                cwd=docker_dir,
            )

            return result.returncode == 0

        except Exception as e:
            print(f"❌ Docker 測試執行失敗: {e}")
            return False

        finally:
            # 清理
            subprocess.run(
                [
                    "docker-compose",
                    "-f",
                    str(compose_file),
                    "-p",
                    "cyberpuppy-integration-test",
                    "down",
                    "-v",
                ],
                capture_output=True,
                cwd=docker_dir,
            )

    def run_regression_tests(self) -> bool:
        """執行回歸測試"""
        print("🔄 執行回歸測試...")

        if not self.start_services_if_needed(["api"]):
            return False

        cmd = [
            "pytest",
            str(self.test_dir / "regression"),
            "-v",
            "--tb=short",
            "-m",
            "regression",
            "--junit-xml=reports/regression-results.xml",
        ]

        return self._run_pytest_command(cmd)

    def _run_pytest_command(self, cmd: List[str]) -> bool:
        """執行 pytest 命令"""
        try:
            # 確保報告目錄存在
            (self.project_root / "reports").mkdir(exist_ok=True)

            # 設定環境變數
            env = os.environ.copy()
            env.update(
                {
                    "PYTHONPATH": str(self.project_root / "src"),
                    "TESTING": "1",
                    "LOG_LEVEL": "INFO",
                }
            )

            result = subprocess.run(cmd, env=env, cwd=self.project_root)
            return result.returncode == 0

        except Exception as e:
            print(f"❌ 測試執行失敗: {e}")
            return False

    def cleanup_test_services(self):
        """清理測試服務"""
        print("🧹 清理測試服務...")

        # 清理 Docker 容器
        try:
            subprocess.run(["docker", "stop", "cyberpuppy-redis-test"], capture_output=True)
            subprocess.run(["docker", "rm", "cyberpuppy-redis-test"], capture_output=True)
        except Exception:
            pass

        # 清理背景進程（如果有的話）
        print("✅ 清理完成")


def main():
    parser = argparse.ArgumentParser(
        description="CyberPuppy 整合測試執行器",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
範例:
  python run_integration_tests.py --quick       # 快速測試（不含效能測試）
  python run_integration_tests.py --full        # 完整測試（排除 Docker）
  python run_integration_tests.py --performance # 僅效能測試
  python run_integration_tests.py --docker      # 僅 Docker 測試
  python run_integration_tests.py --regression  # 僅回歸測試
  python run_integration_tests.py --all         # 所有測試
        """,
    )

    parser.add_argument("--quick", action="store_true", help="執行快速整合測試（不含慢速測試）")
    parser.add_argument("--full", action="store_true", help="執行完整整合測試（排除 Docker）")
    parser.add_argument("--performance", action="store_true", help="執行效能整合測試")
    parser.add_argument("--docker", action="store_true", help="執行 Docker 整合測試")
    parser.add_argument("--regression", action="store_true", help="執行回歸測試")
    parser.add_argument("--all", action="store_true", help="執行所有整合測試")
    parser.add_argument("--check-deps", action="store_true", help="僅檢查依賴與服務狀態")
    parser.add_argument("--cleanup", action="store_true", help="清理測試服務")

    args = parser.parse_args()

    # 確定專案根目錄
    project_root = Path(__file__).parent.parent.parent
    runner = IntegrationTestRunner(project_root)

    if args.cleanup:
        runner.cleanup_test_services()
        return

    if args.check_deps:
        print("🔍 檢查測試依賴與服務狀態...")
        deps_ok = runner.check_dependencies()
        services = runner.check_services()

        print(f"API 服務: {'✅' if services['api'] else '❌'}")
        print(f"Bot 服務: {'✅' if services['bot'] else '❌'}")
        print(f"Redis 服務: {'✅' if services['redis'] else '❌'}")

        return 0 if deps_ok else 1

    # 檢查依賴
    if not runner.check_dependencies():
        return 1

    success = True

    try:
        if args.quick:
            success = runner.run_quick_tests()
        elif args.full:
            success = runner.run_full_tests()
        elif args.performance:
            success = runner.run_performance_tests()
        elif args.docker:
            success = runner.run_docker_tests()
        elif args.regression:
            success = runner.run_regression_tests()
        elif args.all:
            print("🚀 執行完整整合測試套件...")
            success = (
                runner.run_quick_tests()
                and runner.run_full_tests()
                and runner.run_performance_tests()
                and runner.run_regression_tests()
            )
            if success:
                print("🎉 所有整合測試通過！")
            else:
                print("❌ 部分整合測試失敗")
        else:
            print("請指定要執行的測試類型。使用 --help 查看選項。")
            success = False

    except KeyboardInterrupt:
        print("\n⏹️ 測試被用戶中斷")
        success = False
    except Exception as e:
        print(f"❌ 測試執行發生錯誤: {e}")
        success = False

    return 0 if success else 1


if __name__ == "__main__":
    sys.exit(main())
