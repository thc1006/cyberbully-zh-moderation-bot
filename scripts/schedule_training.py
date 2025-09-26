#!/usr/bin/env python3
"""
CyberPuppy Training Scheduler
============================

Integrates with Windows Task Scheduler for automated overnight training.
"""

import argparse
import os
import sys
import subprocess
import json
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional


class TrainingScheduler:
    """Windows Task Scheduler integration for automated training."""

    def __init__(self, project_root: Optional[str] = None):
        self.project_root = Path(project_root) if project_root else Path(__file__).parent.parent
        self.scripts_dir = self.project_root / "scripts"
        self.task_name = "CyberPuppy_AutoTraining"

    def create_scheduled_task(self,
                            schedule_time: str = "23:00",
                            days: str = "daily",
                            enabled: bool = True) -> bool:
        """Create a Windows scheduled task for automated training."""

        try:
            batch_script = self.scripts_dir / "train_all_models.bat"
            if not batch_script.exists():
                print(f"❌ Batch script not found: {batch_script}")
                return False

            # Delete existing task if it exists
            self._delete_task()

            # Build schtasks command
            cmd = [
                "schtasks", "/create",
                "/tn", self.task_name,
                "/tr", str(batch_script),
                "/sc", days,
                "/st", schedule_time,
                "/rl", "HIGHEST",
                "/f"  # Force creation
            ]

            # Add working directory
            working_dir = str(self.project_root)
            cmd.extend(["/it", "/ru", "SYSTEM"])

            print(f"🕒 Creating scheduled task: {self.task_name}")
            print(f"   Schedule: {days} at {schedule_time}")
            print(f"   Script: {batch_script}")
            print(f"   Working directory: {working_dir}")

            # Execute schtasks command
            result = subprocess.run(cmd, capture_output=True, text=True)

            if result.returncode == 0:
                print("✅ Scheduled task created successfully")

                # Set additional properties via XML modification
                self._configure_task_advanced_settings()

                if not enabled:
                    self.disable_task()

                return True
            else:
                print(f"❌ Failed to create scheduled task:")
                print(f"   Error: {result.stderr}")
                return False

        except Exception as e:
            print(f"❌ Error creating scheduled task: {e}")
            return False

    def _configure_task_advanced_settings(self):
        """Configure advanced task settings via XML export/import."""
        try:
            # Export current task XML
            xml_file = self.scripts_dir / f"{self.task_name}.xml"

            export_cmd = ["schtasks", "/query", "/tn", self.task_name, "/xml"]
            result = subprocess.run(export_cmd, capture_output=True, text=True)

            if result.returncode == 0:
                # Modify XML to add advanced settings
                xml_content = result.stdout

                # Add wake computer setting and other advanced options
                xml_content = xml_content.replace(
                    '<DisallowStartIfOnBatteries>true</DisallowStartIfOnBatteries>',
                    '<DisallowStartIfOnBatteries>false</DisallowStartIfOnBatteries>'
                )
                xml_content = xml_content.replace(
                    '<WakeToRun>false</WakeToRun>',
                    '<WakeToRun>true</WakeToRun>'
                )

                # Save modified XML
                with open(xml_file, 'w', encoding='utf-8') as f:
                    f.write(xml_content)

                # Delete and recreate task with modified XML
                subprocess.run(["schtasks", "/delete", "/tn", self.task_name, "/f"],
                             capture_output=True)

                import_cmd = ["schtasks", "/create", "/tn", self.task_name, "/xml", str(xml_file)]
                subprocess.run(import_cmd, capture_output=True)

                # Clean up XML file
                xml_file.unlink(missing_ok=True)

                print("✅ Advanced task settings configured")

        except Exception as e:
            print(f"⚠️ Warning: Could not configure advanced settings: {e}")

    def _delete_task(self):
        """Delete existing scheduled task."""
        try:
            cmd = ["schtasks", "/delete", "/tn", self.task_name, "/f"]
            subprocess.run(cmd, capture_output=True)
        except Exception:
            pass  # Task might not exist

    def list_scheduled_tasks(self) -> bool:
        """List all CyberPuppy scheduled tasks."""
        try:
            cmd = ["schtasks", "/query", "/fo", "csv"]
            result = subprocess.run(cmd, capture_output=True, text=True)

            if result.returncode == 0:
                lines = result.stdout.split('\n')
                found_tasks = []

                for line in lines:
                    if "CyberPuppy" in line or self.task_name in line:
                        found_tasks.append(line)

                if found_tasks:
                    print("📋 Found CyberPuppy scheduled tasks:")
                    for task in found_tasks:
                        parts = task.split(',')
                        if len(parts) >= 4:
                            task_name = parts[0].strip('"')
                            status = parts[3].strip('"')
                            print(f"   - {task_name}: {status}")
                else:
                    print("📋 No CyberPuppy scheduled tasks found")

                return True
            else:
                print("❌ Failed to query scheduled tasks")
                return False

        except Exception as e:
            print(f"❌ Error listing scheduled tasks: {e}")
            return False

    def enable_task(self) -> bool:
        """Enable the scheduled task."""
        try:
            cmd = ["schtasks", "/change", "/tn", self.task_name, "/enable"]
            result = subprocess.run(cmd, capture_output=True, text=True)

            if result.returncode == 0:
                print(f"✅ Task {self.task_name} enabled")
                return True
            else:
                print(f"❌ Failed to enable task: {result.stderr}")
                return False

        except Exception as e:
            print(f"❌ Error enabling task: {e}")
            return False

    def disable_task(self) -> bool:
        """Disable the scheduled task."""
        try:
            cmd = ["schtasks", "/change", "/tn", self.task_name, "/disable"]
            result = subprocess.run(cmd, capture_output=True, text=True)

            if result.returncode == 0:
                print(f"⏸️ Task {self.task_name} disabled")
                return True
            else:
                print(f"❌ Failed to disable task: {result.stderr}")
                return False

        except Exception as e:
            print(f"❌ Error disabling task: {e}")
            return False

    def delete_task(self) -> bool:
        """Delete the scheduled task."""
        try:
            cmd = ["schtasks", "/delete", "/tn", self.task_name, "/f"]
            result = subprocess.run(cmd, capture_output=True, text=True)

            if result.returncode == 0:
                print(f"🗑️ Task {self.task_name} deleted")
                return True
            else:
                print(f"❌ Failed to delete task: {result.stderr}")
                return False

        except Exception as e:
            print(f"❌ Error deleting task: {e}")
            return False

    def run_task_now(self) -> bool:
        """Run the scheduled task immediately."""
        try:
            cmd = ["schtasks", "/run", "/tn", self.task_name]
            result = subprocess.run(cmd, capture_output=True, text=True)

            if result.returncode == 0:
                print(f"🚀 Task {self.task_name} started manually")
                return True
            else:
                print(f"❌ Failed to run task: {result.stderr}")
                return False

        except Exception as e:
            print(f"❌ Error running task: {e}")
            return False

    def get_task_info(self) -> dict:
        """Get detailed information about the scheduled task."""
        try:
            cmd = ["schtasks", "/query", "/tn", self.task_name, "/fo", "list", "/v"]
            result = subprocess.run(cmd, capture_output=True, text=True)

            if result.returncode == 0:
                info = {}
                lines = result.stdout.split('\n')

                for line in lines:
                    if ':' in line:
                        key, value = line.split(':', 1)
                        info[key.strip()] = value.strip()

                return info
            else:
                return {}

        except Exception as e:
            print(f"❌ Error getting task info: {e}")
            return {}

    def create_training_profile(self, profile_name: str, config: dict) -> bool:
        """Create a training profile for different scenarios."""
        profiles_dir = self.scripts_dir / "profiles"
        profiles_dir.mkdir(exist_ok=True)

        profile_file = profiles_dir / f"{profile_name}.json"

        try:
            with open(profile_file, 'w', encoding='utf-8') as f:
                json.dump(config, f, indent=2, ensure_ascii=False)

            print(f"📋 Training profile created: {profile_file}")
            return True

        except Exception as e:
            print(f"❌ Error creating profile: {e}")
            return False

    def setup_quick_schedules(self):
        """Set up common training schedules."""
        profiles = {
            "overnight": {
                "schedule_time": "23:00",
                "days": "daily",
                "description": "Daily overnight training at 11 PM"
            },
            "weekend": {
                "schedule_time": "20:00",
                "days": "weekly",
                "description": "Weekly training on weekends"
            },
            "manual": {
                "schedule_time": "12:00",
                "days": "once",
                "description": "Manual trigger only"
            }
        }

        print("🕒 Available quick schedule setups:")
        for name, config in profiles.items():
            print(f"   {name}: {config['description']}")

        return profiles


def main():
    """Main scheduler script entry point."""
    parser = argparse.ArgumentParser(description='Schedule CyberPuppy training tasks')
    parser.add_argument('action', choices=[
        'create', 'list', 'enable', 'disable', 'delete', 'run', 'info', 'profiles'
    ], help='Action to perform')
    parser.add_argument('--time', default='23:00', help='Schedule time (HH:MM)')
    parser.add_argument('--frequency', default='daily',
                       choices=['daily', 'weekly', 'monthly', 'once'],
                       help='Schedule frequency')
    parser.add_argument('--enabled', action='store_true', default=True,
                       help='Enable task after creation')
    parser.add_argument('--project-root', help='Project root directory')

    args = parser.parse_args()

    # Initialize scheduler
    scheduler = TrainingScheduler(args.project_root)

    try:
        if args.action == 'create':
            print(f"🕒 Creating scheduled training task...")
            print(f"   Time: {args.time}")
            print(f"   Frequency: {args.frequency}")
            print(f"   Enabled: {args.enabled}")

            success = scheduler.create_scheduled_task(
                schedule_time=args.time,
                days=args.frequency,
                enabled=args.enabled
            )

            if success:
                print("\n✅ Training automation setup complete!")
                print("💡 Tips:")
                print("   - Ensure your computer is powered on at the scheduled time")
                print("   - The task will wake your computer if it's sleeping")
                print("   - Check Windows Event Viewer for task execution logs")
                print("   - Use 'python schedule_training.py info' to check task status")

                # Show task info
                info = scheduler.get_task_info()
                if info:
                    print(f"\n📋 Task Details:")
                    print(f"   Next Run Time: {info.get('Next Run Time', 'N/A')}")
                    print(f"   Last Run Time: {info.get('Last Run Time', 'Never')}")
                    print(f"   Status: {info.get('Status', 'Unknown')}")

        elif args.action == 'list':
            scheduler.list_scheduled_tasks()

        elif args.action == 'enable':
            scheduler.enable_task()

        elif args.action == 'disable':
            scheduler.disable_task()

        elif args.action == 'delete':
            print("🗑️ Deleting scheduled training task...")
            if scheduler.delete_task():
                print("✅ Scheduled task removed successfully")

        elif args.action == 'run':
            print("🚀 Running training task immediately...")
            if scheduler.run_task_now():
                print("✅ Training started manually")
                print("💡 Check the logs directory for progress updates")

        elif args.action == 'info':
            info = scheduler.get_task_info()
            if info:
                print("📋 Scheduled Task Information:")
                print("=" * 50)
                for key, value in info.items():
                    if value and value != "N/A":
                        print(f"{key:25}: {value}")
            else:
                print("❌ No task information found - task may not exist")

        elif args.action == 'profiles':
            profiles = scheduler.setup_quick_schedules()
            print("\n💡 To use a profile, run:")
            print("python schedule_training.py create --time 23:00 --frequency daily")

    except KeyboardInterrupt:
        print("\n⚠️ Operation cancelled by user")
        sys.exit(1)
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()