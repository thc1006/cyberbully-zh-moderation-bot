"""
安全規則與隱私保護測試
"""

import json
import os
from datetime import datetime, timedelta

import pytest

from src.cyberpuppy.safety.human_review import (HumanReviewInterface,
                                                ReviewAction, ReviewPriority)
from src.cyberpuppy.safety.rules import (AppealManager, AppealStatus,
                                         PIIHandler, PrivacyLogger,
                                         ResponseLevel,
                                         SafetyRules)


class TestPIIHandler:
    """PII 處理測試"""

    def test_remove_email(self):
        """測試移除電子郵件"""
        text = "請聯絡我 test@example.com 或 admin@test.org"
        cleaned, stats = PIIHandler.remove_pii(text)

        assert "[EMAIL]" in cleaned
        assert "test@example.com" not in cleaned
        assert stats["email"] == 2

    def test_remove_phone(self):
        """測試移除電話號碼"""
        text = "我的手機是 0912345678 或 +886-2-1234-5678"
        cleaned, stats = PIIHandler.remove_pii(text)

        assert "[PHONE]" in cleaned
        assert "0912345678" not in cleaned
        assert stats["phone_tw"] == 1
        assert stats.get("phone_intl", 0) >= 1

    def test_remove_id_number(self):
        """測試移除身分證號"""
        text = "身分證號碼是 A123456789"
        cleaned, stats = PIIHandler.remove_pii(text)

        assert "[ID]" in cleaned
        assert "A123456789" not in cleaned
        assert stats["id_tw"] == 1

    def test_remove_credit_card(self):
        """測試移除信用卡號"""
        text = "信用卡號 1234-5678-9012-3456"
        cleaned, stats = PIIHandler.remove_pii(text)

        assert "[CARD]" in cleaned
        assert "1234-5678-9012-3456" not in cleaned
        assert stats["credit_card"] == 1

    def test_hash_user_id(self):
        """測試使用者 ID 雜湊"""
        user_id = "user123"
        hashed = PIIHandler.hash_user_id(user_id)

        assert hashed != user_id
        assert len(hashed) == 16
        # 確保雜湊是一致的
        assert hashed == PIIHandler.hash_user_id(user_id)

    def test_anonymize_ip(self):
        """測試 IP 匿名化"""
        ip = "192.168.1.100"
        anon_ip = PIIHandler.anonymize_ip(ip)

        assert anon_ip == "192.168.X.X"
        assert "100" not in anon_ip


class TestSafetyRules:
    """安全規則測試"""

    @pytest.fixture
    def safety_rules(self):
        return SafetyRules()

    def test_response_level_none(self, safety_rules):
        """測試無需回應的情況"""
        level = safety_rules.determine_response_level(
            toxicity_score=0.1, bullying_score=0.1, user_id="user123"
        )
        assert level == ResponseLevel.NONE

    def test_response_level_gentle_reminder(self, safety_rules):
        """測試溫和提醒等級"""
        level = safety_rules.determine_response_level(
            toxicity_score=0.4, bullying_score=0.3, user_id="user123"
        )
        assert level == ResponseLevel.GENTLE_REMINDER

    def test_response_level_soft_intervention(self, safety_rules):
        """測試柔性勸阻等級"""
        level = safety_rules.determine_response_level(
            toxicity_score=0.6, bullying_score=0.5, user_id="user123"
        )
        assert level == ResponseLevel.SOFT_INTERVENTION

    def test_response_level_resource_escalation(self, safety_rules):
        """測試資源升級等級"""
        level = safety_rules.determine_response_level(
            toxicity_score=0.8, bullying_score=0.7, user_id="user123"
        )
        assert level == ResponseLevel.RESOURCE_ESCALATION

    def test_response_level_silent_handover(self, safety_rules):
        """測試沉默移交等級"""
        level = safety_rules.determine_response_level(
            toxicity_score=0.95, bullying_score=0.9, user_id="user123"
        )
        assert level == ResponseLevel.SILENT_HANDOVER

    def test_response_level_with_history(self, safety_rules):
        """測試考慮違規歷史的回應等級"""
        user_id = "repeat_offender"

        # 第一次違規
        level1 = safety_rules.determine_response_level(0.4, 0.3, user_id)
        safety_rules.update_user_history(user_id, level1, {"toxicity": 0.4})
        assert level1 == ResponseLevel.GENTLE_REMINDER

        # 第二次違規（應升級）
        level2 = safety_rules.determine_response_level(0.4, 0.3, user_id)
        assert level2 == ResponseLevel.SOFT_INTERVENTION

    def test_response_level_emotional_crisis(self, safety_rules):
        """測試情緒危機偵測"""
        level = safety_rules.determine_response_level(
            toxicity_score=0.2,
            bullying_score=0.1,
            user_id="user123",
            emotion="neg",
            emotion_strength=4,
        )
        assert level == ResponseLevel.RESOURCE_ESCALATION

    def test_generate_response_strategy(self, safety_rules):
        """測試生成回應策略"""
        strategy = safety_rules.generate_response(ResponseLevel.GENTLE_REMINDER)

        assert strategy.level == ResponseLevel.GENTLE_REMINDER
        assert len(strategy.message) > 0
        assert not strategy.notify_admin
        assert strategy.log_detail

    def test_generate_response_with_resources(self, safety_rules):
        """測試生成包含資源的回應"""
        strategy = safety_rules.generate_response(ResponseLevel.RESOURCE_ESCALATION)

        assert strategy.level == ResponseLevel.RESOURCE_ESCALATION
        assert len(strategy.resources) > 0
        assert not strategy.notify_admin
        assert strategy.log_detail

    def test_special_protection_minor(self, safety_rules):
        """測試未成年人特殊保護"""
        protection = safety_rules.should_apply_special_protection(user_age=15)

        assert protection["lower_threshold"]
        assert protection["priority_resources"]
        assert not protection["parental_notify"]  # 13歲以上不需家長通知

    def test_special_protection_child(self, safety_rules):
        """測試兒童特殊保護"""
        protection = safety_rules.should_apply_special_protection(user_age=10)

        assert protection["lower_threshold"]
        assert protection["priority_resources"]
        assert protection["parental_notify"]  # 13歲以下需要家長通知


class TestPrivacyLogger:
    """隱私日誌測試"""

    @pytest.fixture
    def temp_log_file(self, tmp_path):
        return str(tmp_path / "test_logs.jsonl")

    @pytest.fixture
    def privacy_logger(self, temp_log_file):
        return PrivacyLogger(log_file=temp_log_file)

    def test_log_event_without_pii(self, privacy_logger):
        """測試記錄事件（無 PII）"""
        log_entry = privacy_logger.log_event(
            event_type="toxicity_detected",
            text="這是一個測試文本",
            scores={"toxicity": 0.7, "bullying": 0.5},
            action=ResponseLevel.SOFT_INTERVENTION,
            user_id="user123",
        )

        assert log_entry["event_type"] == "toxicity_detected"
        assert log_entry["text_hash"] != "這是一個測試文本"
        assert len(log_entry["text_hash"]) == 16
        assert log_entry["action"] == "SOFT_INTERVENTION"
        assert log_entry["pii_detected"] == {}

    def test_log_event_with_pii(self, privacy_logger):
        """測試記錄事件（含 PII）"""
        log_entry = privacy_logger.log_event(
            event_type="toxicity_detected",
            text="聯絡我 test@example.com 或 0912345678",
            scores={"toxicity": 0.5},
            action=ResponseLevel.GENTLE_REMINDER,
            user_id="user456",
        )

        assert log_entry["pii_detected"]["email"] == 1
        assert log_entry["pii_detected"]["phone_tw"] == 1
        assert log_entry["session_id"] != "user456"  # 應該被匿名化

    def test_log_file_writing(self, privacy_logger, temp_log_file):
        """測試日誌檔案寫入"""
        privacy_logger.log_event(
            event_type="test_event",
            text="測試內容",
            scores={"test": 0.5},
            action=ResponseLevel.NONE,
        )

        # 檢查檔案是否存在且有內容
        assert os.path.exists(temp_log_file)

        with open(temp_log_file, "r", encoding="utf-8") as f:
            line = f.readline()
            log_data = json.loads(line)
            assert log_data["event_type"] == "test_event"


class TestAppealManager:
    """申訴管理測試"""

    @pytest.fixture
    def appeal_manager(self):
        return AppealManager()

    def test_create_appeal(self, appeal_manager):
        """測試建立申訴"""
        appeal = appeal_manager.create_appeal(
            user_id="user123", event_hash="hash123", reason="這是誤判"
        )

        assert appeal.appeal_id is not None
        assert appeal.status == AppealStatus.PENDING
        assert appeal.reason == "這是誤判"

    def test_review_appeal_approved(self, appeal_manager):
        """測試批准申訴"""
        appeal = appeal_manager.create_appeal("user123", "hash123", "測試")

        reviewed = appeal_manager.review_appeal(
            appeal_id=appeal.appeal_id,
            reviewer="admin001",
            decision=AppealStatus.APPROVED,
            resolution="確認為誤判",
        )

        assert reviewed.status == AppealStatus.APPROVED
        assert reviewed.reviewer == "admin001"
        assert reviewed.resolution == "確認為誤判"

    def test_review_appeal_rejected(self, appeal_manager):
        """測試駁回申訴"""
        appeal = appeal_manager.create_appeal("user123", "hash123", "測試")

        reviewed = appeal_manager.review_appeal(
            appeal_id=appeal.appeal_id,
            reviewer="admin002",
            decision=AppealStatus.REJECTED,
            resolution="判定無誤",
        )

        assert reviewed.status == AppealStatus.REJECTED

    def test_get_user_appeals(self, appeal_manager):
        """測試取得使用者申訴"""
        user_id = "user789"

        # 建立多個申訴
        appeal_manager.create_appeal(user_id, "hash1", "申訴1")
        appeal_manager.create_appeal(user_id, "hash2", "申訴2")
        appeal_manager.create_appeal("other_user", "hash3", "申訴3")

        user_appeals = appeal_manager.get_user_appeals(user_id)
        assert len(user_appeals) == 2

    def test_appeal_stats(self, appeal_manager):
        """測試申訴統計"""
        # 建立並處理一些申訴
        appeal1 = appeal_manager.create_appeal("user1", "hash1", "測試1")
        appeal2 = appeal_manager.create_appeal("user2", "hash2", "測試2")
        appeal_manager.create_appeal("user3", "hash3", "測試3")

        appeal_manager.review_appeal(appeal1.appeal_id, "ad"
            "min", AppealStatus.APPROVED, 
        appeal_manager.review_appeal(appeal2.appeal_id, "ad"
            "min", AppealStatus.REJECTED, 

        stats = appeal_manager.get_appeal_stats()

        assert stats["total"] == 3
        assert stats["by_status"][AppealStatus.APPROVED.value] == 1
        assert stats["by_status"][AppealStatus.REJECTED.value] == 1
        assert stats["by_status"][AppealStatus.PENDING.value] == 1
        assert stats["approval_rate"] == 50.0


class TestHumanReviewInterface:
    """人工審核介面測試"""

    @pytest.fixture
    def review_interface(self):
        appeal_manager = AppealManager()
        return HumanReviewInterface(appeal_manager)

    def test_create_review_task(self, review_interface):
        """測試建立審核任務"""
        # 先建立申訴
        appeal = review_interface.appeal_manager.create_appeal("us"
            "er1", 

        # 建立審核任務
        task = review_interface.create_review_task(
            appeal_id=appeal.appeal_id, priority=ReviewPriority.HIGH
        )

        assert task.task_id is not None
        assert task.priority == ReviewPriority.HIGH
        assert task.due_date is not None

    def test_process_review_approve(self, review_interface):
        """測試處理審核 - 批准"""
        appeal = review_interface.appeal_manager.create_appeal("us"
            "er1", 
        task = review_interface.create_review_task(appeal.appeal_id)

        success, message = review_interface.process_review(
            task_id=task.task_id,
            reviewer_id="reviewer001",
            action=ReviewAction.APPROVE,
            notes="確認為誤判",
        )

        assert success
        assert "批准" in message
        assert task.task_id not in review_interface.review_tasks  # 任務應被完成

    def test_batch_review(self, review_interface):
        """測試批次審核"""
        # 建立多個申訴和任務
        task_ids = []
        for i in range(3):
            appeal = review_interface.appeal_manager.create_appeal(
                f"user{i}", f"hash{i}", f"測試{i}"
            )
            task = review_interface.create_review_task(appeal.appeal_id)
            task_ids.append(task.task_id)

        # 批次駁回
        results = review_interface.batch_review(
            task_ids=task_ids,
            reviewer_id="reviewer001",
            action=ReviewAction.REJECT,
            notes="批次駁回",
        )

        assert len(results) == 3
        for task_id, (success, message) in results.items():
            assert success
            assert "駁回" in message

    def test_review_dashboard(self, review_interface):
        """測試審核儀表板"""
        # 建立一些測試資料
        for i in range(5):
            appeal = review_interface.appeal_manager.create_appeal(
                f"user{i}", f"hash{i}", f"測試{i}"
            )
            priority = [ReviewPriority.URGENT, ReviewPriority.HIGH,
                ReviewPriority.NORMAL][i % 3]
            review_interface.create_review_task(appeal.appeal_id, priority)

        dashboard = review_interface.get_review_dashboard()

        assert dashboard["total_pending"] > 0
        assert "by_priority" in dashboard
        assert "appeal_stats" in dashboard

    def test_export_review_report(self, review_interface):
        """測試匯出審核報告"""
        # 建立並處理一些申訴
        appeal1 = review_interface.appeal_manager.create_appeal("us"
            "er1", 
        review_interface.appeal_manager.create_appeal("user2", "hash2", "測試2")

        task1 = review_interface.create_review_task(appeal1.appeal_id)
        review_interface.process_review(task1.task_id, "revi"
            "ewer1", ReviewAction.APPROVE, 

        # 匯出報告
        report = review_interface.export_review_report(
            start_date=datetime.now() - timedelta(days=1),
            end_date=datetime.now() + timedelta(days=1),
            format="json",
        )

        report_data = json.loads(report)
        assert report_data["total_appeals"] >= 2
        assert "approval_rate" in report_data


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
