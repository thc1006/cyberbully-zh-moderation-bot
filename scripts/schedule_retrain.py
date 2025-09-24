"""
排程再訓練腳本

用於設定定期執行的再訓練任務
支援 cron 和 Windows Task Scheduler
"""

import os
import platform
import subprocess
import argparse
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# 基礎路徑
BASE_DIR = Path(__file__).parent.parent
RETRAIN_SCRIPT = BASE_DIR / "scripts" / "retrain.sh"
RETRAIN_LOG = BASE_DIR / "logs" / "retrain" / "schedule.log"


class RetrainScheduler:
    """再訓練排程器"""

    def __init__(self):
        self.system = platform.system()
        self.retrain_script = RETRAIN_SCRIPT

    def setup_cron(self, schedule: str = "0 3 * * 1"):
        """
        設定 cron 任務（Linux/Mac）

        Args:
            schedule: cron 表達式（預設：每週一凌晨 3 點）
        """
        if self.system == "Windows":
            logger.error("Cron is not available on Windows")
            return False

        try:
            # 讀取現有 crontab
            result = subprocess.run(
                ["crontab", "-l"],
                capture_output=True,
                text=True
            )

            existing_cron = result.stdout if result.returncode == 0 else ""

            # 檢查是否已存在
            retrain_job = (
                f"{schedule} cd {BASE_DIR} && bash {self.retrain_script} "
                f">> {RETRAIN_LOG} 2>&1"
            )

            if retrain_job in existing_cron:
                logger.info("Retrain job already exists in crontab")
                return True

            # 添加新任務
            new_cron = existing_cron.strip() + "\n" + retrain_job + "\n"

            # 寫入 crontab
            process = subprocess.Popen(
                ["crontab", "-"],
                stdin=subprocess.PIPE,
                text=True
            )
            process.communicate(input=new_cron)

            if process.returncode == 0:
                logger.info(f"Successfully added cron job: {schedule}")
                return True
            else:
                logger.error("Failed to add cron job")
                return False

        except Exception as e:
            logger.error(f"Error setting up cron: {e}")
            return False

    def setup_windows_task(self, schedule: str = "WEEKLY"):
        """
        設定 Windows Task Scheduler 任務

        Args:
            schedule: 排程頻率 (DAILY, WEEKLY, MONTHLY)
        """
        if self.system != "Windows":
            logger.error("Windows Task Scheduler is only available on Windows")
            return False

        try:
            task_name = "CyberPuppyRetrain"

            # 建立批處理檔案包裝 bash 腳本
            batch_file = BASE_DIR / "scripts" / "retrain.bat"
            with open(batch_file, 'w') as f:
                f.write("@echo off\n")
                f.write(f"cd /d {BASE_DIR}\n")
                f.write(
                    "bash scripts\\retrain.sh >> log"
                        "s\\retrain\\schedule.log 2>&1\n"
                )

            # 使用 schtasks 創建任務
            cmd = [
                "schtasks", "/create",
                "/tn", task_name,
                "/tr", str(batch_file),
                "/sc", schedule
            ]

            if schedule == "WEEKLY":
                cmd.extend(["/d", "MON", "/st", "03:00"])  # 週一凌晨 3 點
            elif schedule == "DAILY":
                cmd.extend(["/st", "03:00"])  # 每天凌晨 3 點

            result = subprocess.run(cmd, capture_output=True, text=True)

            if result.returncode == 0:
                logger.info(
                    f"Successfully created Windows scheduled task: {task_name}"
                )
                return True
            else:
                logger.error(f"Failed to create task: {result.stderr}")
                return False

        except Exception as e:
            logger.error(f"Error setting up Windows task: {e}")
            return False

    def setup_systemd_timer(self):
        """
        設定 systemd timer（Linux）
        """
        if self.system != "Linux":
            logger.error("systemd is only available on Linux")
            return False

        service_content = f"""[Unit]
Description=CyberPuppy Retrain Service
After=network.target

[Service]
Type=oneshot
WorkingDirectory={BASE_DIR}
ExecStart=/bin/bash {self.retrain_script}
StandardOutput=append:{RETRAIN_LOG}
StandardError=append:{RETRAIN_LOG}
User={os.getenv('USER', 'root')}

[Install]
WantedBy=multi-user.target
"""

        timer_content = """[Unit]
Description=CyberPuppy Retrain Timer
Requires=cyberpuppy-retrain.service

[Timer]
OnCalendar=weekly
OnCalendar=Mon *-*-* 03:00:00
Persistent=true

[Install]
WantedBy=timers.target
"""

        try:
            # 寫入 service 文件
            service_path = Path(
                "/etc/systemd/system/cyberpuppy-retrain.service"
            )
            timer_path = Path("/etc/systemd/system/cyberpuppy-retrain.timer")

            # 需要 sudo 權限
            logger.info(
                "Creating systemd service and timer (requires sudo)..."
            )

            with open("/tmp/cyberpuppy-retrain.service", 'w') as f:
                f.write(service_content)
            with open("/tmp/cyberpuppy-retrain.timer", 'w') as f:
                f.write(timer_content)

            # 移動到系統目錄
            subprocess.run(
                ["su"
                    "do", 
                check=True
            )
            subprocess.run(
                ["su"
                    "do", 
                check=True
            )

            # 重新載入並啟動
            subprocess.run(["sudo", "systemctl", "daemon-reload"], check=True)
            subprocess.run(
                ["sudo", "systemctl", "enable", "cyberpuppy-retrain.timer"],
                check=True
            )
            subprocess.run(
                ["sudo", "systemctl", "start", "cyberpuppy-retrain.timer"],
                check=True
            )

            logger.info("Successfully set up systemd timer")
            return True

        except Exception as e:
            logger.error(f"Error setting up systemd timer: {e}")
            return False

    def check_status(self):
        """檢查排程狀態"""
        logger.info(f"Checking retrain schedule status on {self.system}...")

        if self.system == "Linux":
            # 檢查 cron
            result = subprocess.run(
                ["crontab", "-l"],
                capture_output=True,
                text=True
            )
            if result.returncode == 0 and "retrain.sh" in result.stdout:
                logger.info("✓ Cron job is active")
            else:
                logger.info("✗ No cron job found")

            # 檢查 systemd
            result = subprocess.run(
                ["systemctl", "is-active", "cyberpuppy-retrain.timer"],
                capture_output=True,
                text=True
            )
            if result.stdout.strip() == "active":
                logger.info("✓ Systemd timer is active")
            else:
                logger.info("✗ Systemd timer is not active")

        elif self.system == "Darwin":  # macOS
            # 檢查 cron
            result = subprocess.run(
                ["crontab", "-l"],
                capture_output=True,
                text=True
            )
            if result.returncode == 0 and "retrain.sh" in result.stdout:
                logger.info("✓ Cron job is active")
            else:
                logger.info("✗ No cron job found")

        elif self.system == "Windows":
            # 檢查 Windows Task Scheduler
            result = subprocess.run(
                ["schtasks", "/query", "/tn", "CyberPuppyRetrain"],
                capture_output=True,
                text=True
            )
            if result.returncode == 0:
                logger.info("✓ Windows scheduled task is active")
            else:
                logger.info("✗ No Windows scheduled task found")

    def remove_schedule(self):
        """移除排程任務"""
        logger.info("Removing retrain schedule...")

        if self.system in ["Linux", "Darwin"]:
            # 移除 cron
            result = subprocess.run(
                ["crontab", "-l"],
                capture_output=True,
                text=True
            )
            if result.returncode == 0:
                lines = result.stdout.split('\n')
                new_lines = [
                    line for line in lines if "retrain.sh" not in line
                ]
                new_cron = '\n'.join(new_lines)

                process = subprocess.Popen(
                    ["crontab", "-"],
                    stdin=subprocess.PIPE,
                    text=True
                )
                process.communicate(input=new_cron)
                logger.info("Removed cron job")

        if self.system == "Linux":
            # 移除 systemd timer
            subprocess.run(
                ["sudo", "systemctl", "stop", "cyberpuppy-retrain.timer"],
                capture_output=True
            )
            subprocess.run(
                ["sudo", "systemctl", "disable", "cyberpuppy-retrain.timer"],
                capture_output=True
            )
            logger.info("Removed systemd timer")

        if self.system == "Windows":
            # 移除 Windows task
            subprocess.run(
                ["schtasks", "/delete", "/tn", "CyberPuppyRetrain", "/f"],
                capture_output=True
            )
            logger.info("Removed Windows scheduled task")


def main():
    parser = argparse.ArgumentParser(
        description="設定 CyberPuppy 再訓練排程"
    )

    parser.add_argument(
        "--setup",
        choices=["cron", "systemd", "windows", "auto"],
        help="設定排程方式"
    )
    parser.add_argument(
        "--remove",
        action="store_true",
        help="移除排程"
    )
    parser.add_argument(
        "--status",
        action="store_true",
        help="檢查排程狀態"
    )
    parser.add_argument(
        "--schedule",
        default="WEEKLY",
        help="排程頻率 (DAILY, WEEKLY, MONTHLY 或 cron 表達式)"
    )

    args = parser.parse_args()

    scheduler = RetrainScheduler()

    if args.remove:
        scheduler.remove_schedule()
    elif args.status:
        scheduler.check_status()
    elif args.setup:
        if args.setup == "auto":
            # 自動選擇
            if scheduler.system == "Linux":
                success = scheduler.setup_systemd_timer()
                if not success:
                    scheduler.setup_cron()
            elif scheduler.system == "Darwin":
                scheduler.setup_cron()
            elif scheduler.system == "Windows":
                scheduler.setup_windows_task(args.schedule)
        elif args.setup == "cron":
            scheduler.setup_cron(args.schedule)
        elif args.setup == "systemd":
            scheduler.setup_systemd_timer()
        elif args.setup == "windows":
            scheduler.setup_windows_task(args.schedule)
    else:
        # 預設顯示狀態
        scheduler.check_status()
        print("\n使用方式:")
        print("  設定排程: python schedule_retrain.py --setup auto")
        print("  檢查狀態: python schedule_retrain.py --status")
        print("  移除排程: python schedule_retrain.py --remove")


if __name__ == "__main__":
    main()
