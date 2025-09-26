#!/usr/bin/env python3
"""
CyberPuppy Notification System
==============================

Sends desktop and email notifications when training completes.
"""

import argparse
import json
import os
import sys
import smtplib
from datetime import datetime
from email.mime.text import MimeText
from email.mime.multipart import MimeMultipart
from pathlib import Path
from typing import Optional


class NotificationSender:
    """Send various types of notifications when training completes."""

    def __init__(self, session_id: str, project_root: Optional[str] = None):
        self.session_id = session_id
        self.project_root = Path(project_root) if project_root else Path(__file__).parent.parent
        self.logs_dir = self.project_root / "logs"
        self.config_file = self.project_root / "config" / "notifications.json"

        # Load notification configuration
        self.config = self.load_config()

    def load_config(self) -> dict:
        """Load notification configuration from file."""
        default_config = {
            "desktop_notifications": {
                "enabled": True,
                "duration": 10
            },
            "email_notifications": {
                "enabled": False,
                "smtp_server": "smtp.gmail.com",
                "smtp_port": 587,
                "sender_email": "",
                "sender_password": "",
                "recipient_emails": [],
                "use_app_password": True
            },
            "sound_notifications": {
                "enabled": True,
                "success_sound": "success.wav",
                "failure_sound": "error.wav"
            },
            "custom_notifications": {
                "enabled": False,
                "webhook_url": "",
                "slack_webhook": "",
                "discord_webhook": ""
            }
        }

        try:
            if self.config_file.exists():
                with open(self.config_file, 'r', encoding='utf-8') as f:
                    loaded_config = json.load(f)
                    # Merge with defaults
                    self.merge_configs(default_config, loaded_config)
                    return default_config
            else:
                # Create default config file
                self.config_file.parent.mkdir(exist_ok=True)
                with open(self.config_file, 'w', encoding='utf-8') as f:
                    json.dump(default_config, f, indent=2, ensure_ascii=False)
                print(f"📝 Created default notification config: {self.config_file}")

        except Exception as e:
            print(f"⚠️ Warning: Could not load notification config: {e}")

        return default_config

    def merge_configs(self, default: dict, loaded: dict):
        """Recursively merge loaded config with defaults."""
        for key, value in loaded.items():
            if key in default:
                if isinstance(value, dict) and isinstance(default[key], dict):
                    self.merge_configs(default[key], value)
                else:
                    default[key] = value

    def send_desktop_notification(self, title: str, message: str, notification_type: str = "info"):
        """Send desktop notification."""
        if not self.config["desktop_notifications"]["enabled"]:
            return

        try:
            import plyer

            # Choose icon based on notification type
            icon_map = {
                "success": "success",
                "error": "error",
                "warning": "warning",
                "info": "info"
            }

            icon = icon_map.get(notification_type, "info")
            duration = self.config["desktop_notifications"]["duration"]

            plyer.notification.notify(
                title=title,
                message=message,
                timeout=duration,
                app_icon=None  # Can add custom icon path here
            )

            print(f"📱 Desktop notification sent: {title}")

        except ImportError:
            print("⚠️ Warning: plyer not installed - cannot send desktop notifications")
            print("   Install with: pip install plyer")
        except Exception as e:
            print(f"❌ Failed to send desktop notification: {e}")

    def send_email_notification(self, subject: str, body: str, html_body: Optional[str] = None):
        """Send email notification."""
        email_config = self.config["email_notifications"]

        if not email_config["enabled"]:
            return

        if not email_config["sender_email"] or not email_config["recipient_emails"]:
            print("⚠️ Email notifications enabled but email addresses not configured")
            return

        try:
            # Create message
            msg = MimeMultipart('alternative')
            msg['Subject'] = subject
            msg['From'] = email_config["sender_email"]
            msg['To'] = ", ".join(email_config["recipient_emails"])

            # Add text part
            text_part = MimeText(body, 'plain', 'utf-8')
            msg.attach(text_part)

            # Add HTML part if provided
            if html_body:
                html_part = MimeText(html_body, 'html', 'utf-8')
                msg.attach(html_part)

            # Send email
            with smtplib.SMTP(email_config["smtp_server"], email_config["smtp_port"]) as server:
                server.starttls()
                password = email_config.get("sender_password", "")
                if not password:
                    password = os.getenv("EMAIL_PASSWORD", "")

                if password:
                    server.login(email_config["sender_email"], password)
                    server.send_message(msg)
                    print(f"📧 Email notification sent to {len(email_config['recipient_emails'])} recipients")
                else:
                    print("⚠️ Email password not configured - skipping email notification")

        except Exception as e:
            print(f"❌ Failed to send email notification: {e}")

    def play_sound_notification(self, notification_type: str):
        """Play sound notification."""
        if not self.config["sound_notifications"]["enabled"]:
            return

        try:
            import pygame
            pygame.mixer.init()

            sound_file = None
            if notification_type == "success":
                sound_file = self.config["sound_notifications"]["success_sound"]
            elif notification_type == "error":
                sound_file = self.config["sound_notifications"]["failure_sound"]

            if sound_file:
                # Look for sound file in project directory
                sound_path = self.project_root / "assets" / "sounds" / sound_file
                if sound_path.exists():
                    pygame.mixer.music.load(str(sound_path))
                    pygame.mixer.music.play()
                    print(f"🔊 Played sound notification: {sound_file}")
                else:
                    # Use system beep as fallback
                    import winsound
                    if notification_type == "success":
                        winsound.MessageBeep(winsound.MB_OK)
                    else:
                        winsound.MessageBeep(winsound.MB_ICONHAND)
                    print(f"🔊 Played system beep for: {notification_type}")

        except ImportError:
            try:
                # Fallback to Windows system sounds
                import winsound
                if notification_type == "success":
                    winsound.MessageBeep(winsound.MB_OK)
                elif notification_type == "error":
                    winsound.MessageBeep(winsound.MB_ICONHAND)
                else:
                    winsound.MessageBeep(winsound.MB_ICONASTERISK)
                print(f"🔊 Played system sound for: {notification_type}")
            except ImportError:
                print("⚠️ No sound libraries available")
        except Exception as e:
            print(f"❌ Failed to play sound notification: {e}")

    def send_webhook_notification(self, title: str, message: str, status: str):
        """Send webhook notifications to various services."""
        custom_config = self.config["custom_notifications"]

        if not custom_config["enabled"]:
            return

        # Generic webhook
        if custom_config["webhook_url"]:
            self.send_generic_webhook(custom_config["webhook_url"], title, message, status)

        # Slack webhook
        if custom_config["slack_webhook"]:
            self.send_slack_notification(custom_config["slack_webhook"], title, message, status)

        # Discord webhook
        if custom_config["discord_webhook"]:
            self.send_discord_notification(custom_config["discord_webhook"], title, message, status)

    def send_generic_webhook(self, webhook_url: str, title: str, message: str, status: str):
        """Send generic webhook notification."""
        try:
            import requests

            payload = {
                "title": title,
                "message": message,
                "status": status,
                "session_id": self.session_id,
                "timestamp": datetime.now().isoformat()
            }

            response = requests.post(webhook_url, json=payload, timeout=10)
            response.raise_for_status()
            print(f"🌐 Webhook notification sent to: {webhook_url}")

        except ImportError:
            print("⚠️ requests library not available for webhook notifications")
        except Exception as e:
            print(f"❌ Failed to send webhook notification: {e}")

    def send_slack_notification(self, webhook_url: str, title: str, message: str, status: str):
        """Send Slack notification."""
        try:
            import requests

            # Choose emoji based on status
            emoji_map = {
                "SUCCESS_EARLY": "🎯",
                "SUCCESS_COMPLETE": "✅",
                "FAILED": "❌"
            }

            emoji = emoji_map.get(status, "🤖")
            color_map = {
                "SUCCESS_EARLY": "good",
                "SUCCESS_COMPLETE": "good",
                "FAILED": "danger"
            }
            color = color_map.get(status, "warning")

            payload = {
                "text": f"{emoji} {title}",
                "attachments": [
                    {
                        "color": color,
                        "fields": [
                            {
                                "title": "Status",
                                "value": status,
                                "short": True
                            },
                            {
                                "title": "Session ID",
                                "value": self.session_id,
                                "short": True
                            },
                            {
                                "title": "Details",
                                "value": message,
                                "short": False
                            }
                        ],
                        "footer": "CyberPuppy Training System",
                        "ts": int(datetime.now().timestamp())
                    }
                ]
            }

            response = requests.post(webhook_url, json=payload, timeout=10)
            response.raise_for_status()
            print(f"📱 Slack notification sent")

        except ImportError:
            print("⚠️ requests library not available for Slack notifications")
        except Exception as e:
            print(f"❌ Failed to send Slack notification: {e}")

    def send_discord_notification(self, webhook_url: str, title: str, message: str, status: str):
        """Send Discord notification."""
        try:
            import requests

            # Choose color based on status
            color_map = {
                "SUCCESS_EARLY": 0x00ff00,  # Green
                "SUCCESS_COMPLETE": 0x00ff00,  # Green
                "FAILED": 0xff0000  # Red
            }
            color = color_map.get(status, 0xffff00)  # Yellow

            payload = {
                "embeds": [
                    {
                        "title": title,
                        "description": message,
                        "color": color,
                        "fields": [
                            {
                                "name": "Status",
                                "value": status,
                                "inline": True
                            },
                            {
                                "name": "Session ID",
                                "value": self.session_id,
                                "inline": True
                            }
                        ],
                        "footer": {
                            "text": "CyberPuppy Training System"
                        },
                        "timestamp": datetime.now().isoformat()
                    }
                ]
            }

            response = requests.post(webhook_url, json=payload, timeout=10)
            response.raise_for_status()
            print(f"💬 Discord notification sent")

        except ImportError:
            print("⚠️ requests library not available for Discord notifications")
        except Exception as e:
            print(f"❌ Failed to send Discord notification: {e}")

    def generate_training_summary(self, status: str, best_config: str, best_f1: float) -> tuple:
        """Generate training summary for notifications."""
        # Status mapping
        status_map = {
            "SUCCESS_EARLY": "成功（提前達標）",
            "SUCCESS_COMPLETE": "成功（完整訓練）",
            "FAILED": "失敗"
        }

        status_text = status_map.get(status, status)

        # Create title
        if status.startswith("SUCCESS"):
            title = f"🎉 CyberPuppy 訓練完成 - {status_text}"
        else:
            title = f"❌ CyberPuppy 訓練失敗"

        # Create message
        message_parts = [
            f"訓練會話: {self.session_id}",
            f"狀態: {status_text}",
            f"完成時間: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
        ]

        if best_config and best_f1:
            message_parts.extend([
                f"最佳模型: {best_config}",
                f"最高 F1 分數: {best_f1:.4f}",
                f"達標狀態: {'✅ 已達標' if best_f1 >= 0.75 else '❌ 未達標'} (目標 ≥ 0.75)"
            ])

        if status.startswith("SUCCESS"):
            message_parts.extend([
                "",
                "🎯 建議下一步:",
                "1. 檢查模型比較報告",
                "2. 在測試集上驗證效果",
                "3. 考慮部署最佳模型"
            ])
        else:
            message_parts.extend([
                "",
                "🔧 建議修復:",
                "1. 檢查訓練日誌",
                "2. 確認數據和配置",
                "3. 調整超參數後重試"
            ])

        message = "\n".join(message_parts)

        # Generate HTML version for email
        html_message = message.replace("\n", "<br>")
        html_message = f"""
        <html>
        <body style="font-family: Arial, sans-serif; line-height: 1.6;">
            <h2 style="color: {'#27ae60' if status.startswith('SUCCESS') else '#e74c3c'};">
                {title}
            </h2>
            <div style="background-color: #f8f9fa; padding: 15px; border-radius: 5px;">
                {html_message}
            </div>
            <p><small>由 CyberPuppy 自動訓練系統發送</small></p>
        </body>
        </html>
        """

        return title, message, html_message

    def send_all_notifications(self, status: str, best_config: str = None, best_f1: float = None):
        """Send all configured notifications."""
        print(f"📢 Sending notifications for training completion...")

        # Generate summary
        title, message, html_message = self.generate_training_summary(status, best_config, best_f1)

        # Determine notification type
        notification_type = "success" if status.startswith("SUCCESS") else "error"

        try:
            # Send desktop notification
            self.send_desktop_notification(title, message, notification_type)

            # Send email notification
            self.send_email_notification(title, message, html_message)

            # Play sound notification
            self.play_sound_notification(notification_type)

            # Send webhook notifications
            self.send_webhook_notification(title, message, status)

            print("✅ All notifications sent successfully")

        except Exception as e:
            print(f"❌ Error sending notifications: {e}")

    def test_notifications(self):
        """Test all notification systems."""
        print("🧪 Testing notification systems...")

        test_title = "CyberPuppy 通知測試"
        test_message = "這是一個測試通知，確認通知系統正常運作。"

        try:
            # Test desktop notification
            if self.config["desktop_notifications"]["enabled"]:
                self.send_desktop_notification(test_title, test_message, "info")

            # Test email notification
            if self.config["email_notifications"]["enabled"]:
                self.send_email_notification(
                    f"[測試] {test_title}",
                    test_message,
                    f"<html><body><h3>{test_title}</h3><p>{test_message}</p></body></html>"
                )

            # Test sound notification
            if self.config["sound_notifications"]["enabled"]:
                self.play_sound_notification("info")

            # Test webhook notifications
            if self.config["custom_notifications"]["enabled"]:
                self.send_webhook_notification(test_title, test_message, "TEST")

            print("✅ Notification test completed")

        except Exception as e:
            print(f"❌ Error during notification test: {e}")


def main():
    """Main notification script entry point."""
    parser = argparse.ArgumentParser(description='Send CyberPuppy training notifications')
    parser.add_argument('--session-id', required=True, help='Training session ID')
    parser.add_argument('--status', required=True,
                       choices=['SUCCESS_EARLY', 'SUCCESS_COMPLETE', 'FAILED'],
                       help='Training status')
    parser.add_argument('--best-config', help='Best configuration name')
    parser.add_argument('--best-f1', type=float, help='Best F1 score')
    parser.add_argument('--test', action='store_true', help='Test all notification systems')
    parser.add_argument('--project-root', help='Project root directory')

    args = parser.parse_args()

    try:
        # Initialize notification sender
        sender = NotificationSender(args.session_id, args.project_root)

        if args.test:
            # Test notifications
            sender.test_notifications()
        else:
            # Send actual notifications
            sender.send_all_notifications(args.status, args.best_config, args.best_f1)

    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()