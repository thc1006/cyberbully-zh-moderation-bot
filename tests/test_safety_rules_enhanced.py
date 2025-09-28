"""
Enhanced tests for safety/rules.py to improve coverage to 90%+
Tests edge cases, error handling, and uncovered branches
"""

import json
import os
from datetime import datetime, timedelta
from unittest.mock import Mock, mock_open, patch

import pytest

from cyberpuppy.safety.rules import (
    Appeal,
    AppealManager,
    AppealStatus,
    PIIHandler,
    PrivacyLogger,
    ResponseLevel,
    ResponseStrategy,
    SafetyRules,
    UserViolationHistory,
)


@pytest.mark.unit
class TestUserViolationHistory:
    """Test UserViolationHistory class"""

    def test_add_violation_basic(self):
        """Test adding violation record"""
        history = UserViolationHistory(user_id="user123")
        history.add_violation(ResponseLevel.GENTLE_REMINDER, {"toxicity": 0.4})

        assert history.total_count == 1
        assert len(history.violations) == 1
        assert history.last_violation_time is not None

    def test_add_violation_limit_to_10(self):
        """Test violation history limited to 10 records"""
        history = UserViolationHistory(user_id="user123")

        for i in range(15):
            history.add_violation(ResponseLevel.GENTLE_REMINDER, {"toxicity": 0.4 + i * 0.01})

        assert len(history.violations) == 10
        assert history.total_count == 15

    def test_get_recent_violations(self):
        """Test getting recent violations within time window"""
        history = UserViolationHistory(user_id="user123")

        old_violation = {
            "timestamp": (datetime.now() - timedelta(hours=48)).isoformat(),
            "level": ResponseLevel.GENTLE_REMINDER.value,
            "scores": {"toxicity": 0.3},
            "hash": "old_hash",
        }
        history.violations.append(old_violation)

        history.add_violation(ResponseLevel.SOFT_INTERVENTION, {"toxicity": 0.6})

        recent = history.get_recent_violations(hours=24)
        assert len(recent) == 1


@pytest.mark.unit
class TestAppeal:
    """Test Appeal dataclass"""

    def test_appeal_to_dict(self):
        """Test Appeal serialization"""
        appeal = Appeal(
            appeal_id="test123",
            user_id="user_hash",
            event_hash="event_hash",
            reason="Test reason",
            status=AppealStatus.PENDING,
            created_at=datetime.now(),
            updated_at=datetime.now(),
            reviewer=None,
            resolution=None,
        )

        appeal_dict = appeal.to_dict()

        assert appeal_dict["appeal_id"] == "test123"
        assert appeal_dict["user_id"] == "user_hash"
        assert appeal_dict["status"] == "pending"
        assert "created_at" in appeal_dict
        assert "updated_at" in appeal_dict


@pytest.mark.unit
class TestPIIHandlerEdgeCases:
    """Test PIIHandler edge cases"""

    def test_anonymize_ip_invalid(self):
        """Test IP anonymization with invalid format"""
        invalid_ip = "192.168"
        anon_ip = PIIHandler.anonymize_ip(invalid_ip)
        assert anon_ip == "X.X.X.X"

    def test_anonymize_ip_empty(self):
        """Test IP anonymization with empty string"""
        empty_ip = ""
        anon_ip = PIIHandler.anonymize_ip(empty_ip)
        assert anon_ip == "X.X.X.X"

    def test_remove_pii_no_matches(self):
        """Test remove_pii when no PII is present"""
        text = "這是一個普通的文本沒有任何個人資訊"
        cleaned, stats = PIIHandler.remove_pii(text)

        assert cleaned == text
        assert stats == {}

    def test_hash_user_id_with_custom_salt(self):
        """Test hash_user_id with custom salt"""
        user_id = "testuser"
        hash1 = PIIHandler.hash_user_id(user_id, salt="salt1")
        hash2 = PIIHandler.hash_user_id(user_id, salt="salt2")

        assert hash1 != hash2
        assert len(hash1) == 16
        assert len(hash2) == 16


@pytest.mark.unit
class TestSafetyRulesEdgeCases:
    """Test SafetyRules edge cases"""

    @pytest.fixture
    def safety_rules(self):
        return SafetyRules()

    def test_response_level_none_with_context(self, safety_rules):
        """Test NONE response level with context parameter"""
        level = safety_rules.determine_response_level(
            toxicity_score=0.2, bullying_score=0.1, user_id="user123", emotion="neu", emotion_strength=0
        )
        assert level == ResponseLevel.NONE

    def test_generate_response_none_level(self, safety_rules):
        """Test generate_response for NONE level"""
        strategy = safety_rules.generate_response(ResponseLevel.NONE)

        assert strategy.level == ResponseLevel.NONE
        assert strategy.message == ""
        assert strategy.resources == []
        assert strategy.notify_admin is False
        assert strategy.log_detail is False

    def test_generate_response_silent_handover(self, safety_rules):
        """Test generate_response for SILENT_HANDOVER level"""
        strategy = safety_rules.generate_response(ResponseLevel.SILENT_HANDOVER)

        assert strategy.level == ResponseLevel.SILENT_HANDOVER
        assert strategy.notify_admin is True

    def test_update_user_history_new_user(self, safety_rules):
        """Test updating history for new user"""
        user_id = "new_user_999"
        safety_rules.update_user_history(user_id, ResponseLevel.GENTLE_REMINDER, {"toxicity": 0.4})

        assert user_id in safety_rules.user_histories
        assert safety_rules.user_histories[user_id].total_count == 1

    def test_update_user_history_none_level(self, safety_rules):
        """Test updating history with NONE level (creates user but no violation)"""
        user_id = "user_none"
        safety_rules.update_user_history(user_id, ResponseLevel.NONE, {"toxicity": 0.1})

        assert user_id in safety_rules.user_histories
        assert safety_rules.user_histories[user_id].total_count == 0

    def test_response_level_with_multiple_recent_violations(self, safety_rules):
        """Test escalation with multiple recent violations"""
        user_id = "repeat_user"

        safety_rules.update_user_history(user_id, ResponseLevel.SOFT_INTERVENTION, {"toxicity": 0.6})
        safety_rules.update_user_history(user_id, ResponseLevel.SOFT_INTERVENTION, {"toxicity": 0.6})

        level = safety_rules.determine_response_level(0.6, 0.5, user_id)
        assert level == ResponseLevel.RESOURCE_ESCALATION

    def test_special_protection_adult_only(self, safety_rules):
        """Test special protection for adult (no protection)"""
        protection = safety_rules.should_apply_special_protection(user_age=25, is_vulnerable=False)

        assert protection["lower_threshold"] is False
        assert protection["priority_resources"] is False
        assert protection["parental_notify"] is False
        assert protection["extra_logging"] is False

    def test_special_protection_vulnerable_only(self, safety_rules):
        """Test special protection for vulnerable user without age"""
        protection = safety_rules.should_apply_special_protection(user_age=None, is_vulnerable=True)

        assert protection["priority_resources"] is True
        assert protection["extra_logging"] is True
        assert protection["lower_threshold"] is False


@pytest.mark.unit
class TestPrivacyLoggerEdgeCases:
    """Test PrivacyLogger edge cases"""

    @pytest.fixture
    def temp_log_file(self, tmp_path):
        return str(tmp_path / "test_privacy_logs.jsonl")

    def test_log_event_without_user_id(self, temp_log_file):
        """Test logging event without user_id"""
        logger = PrivacyLogger(log_file=temp_log_file)
        log_entry = logger.log_event(
            event_type="test_event",
            text="Test text",
            scores={"toxicity": 0.5},
            action=ResponseLevel.GENTLE_REMINDER,
            user_id=None,
        )

        assert log_entry["session_id"] is None
        assert "text_hash" in log_entry

    def test_log_event_without_metadata(self, temp_log_file):
        """Test logging event without metadata"""
        logger = PrivacyLogger(log_file=temp_log_file)
        log_entry = logger.log_event(
            event_type="test_event",
            text="Test text",
            scores={"toxicity": 0.5},
            action=ResponseLevel.GENTLE_REMINDER,
            metadata=None,
        )

        assert log_entry["metadata"] == {}

    def test_log_event_file_write_failure(self, temp_log_file):
        """Test logging when file write fails"""
        logger = PrivacyLogger(log_file="/invalid/path/log.jsonl")

        log_entry = logger.log_event(
            event_type="test_event",
            text="Test text",
            scores={"toxicity": 0.5},
            action=ResponseLevel.GENTLE_REMINDER,
        )

        assert log_entry is not None

    def test_log_event_no_log_file(self):
        """Test logging without log file configured"""
        logger = PrivacyLogger(log_file=None)
        log_entry = logger.log_event(
            event_type="test_event",
            text="Test text",
            scores={"toxicity": 0.5},
            action=ResponseLevel.GENTLE_REMINDER,
        )

        assert log_entry is not None
        assert "text_hash" in log_entry


@pytest.mark.unit
class TestAppealManagerEdgeCases:
    """Test AppealManager edge cases"""

    @pytest.fixture
    def appeal_manager(self):
        return AppealManager()

    def test_review_appeal_invalid_id(self, appeal_manager):
        """Test reviewing non-existent appeal"""
        result = appeal_manager.review_appeal(
            appeal_id="invalid_id_999",
            reviewer="admin",
            decision=AppealStatus.REJECTED,
            resolution="Test",
        )

        assert result is None

    def test_get_user_appeals_with_status_filter(self, appeal_manager):
        """Test getting user appeals with status filter"""
        user_id = "filtered_user"

        appeal1 = appeal_manager.create_appeal(user_id, "hash1", "Reason 1")
        appeal2 = appeal_manager.create_appeal(user_id, "hash2", "Reason 2")

        appeal_manager.review_appeal(appeal1.appeal_id, "admin", AppealStatus.APPROVED, "Approved")

        pending_appeals = appeal_manager.get_user_appeals(user_id, status_filter=AppealStatus.PENDING)
        approved_appeals = appeal_manager.get_user_appeals(
            user_id, status_filter=AppealStatus.APPROVED
        )

        assert len(pending_appeals) == 1
        assert len(approved_appeals) == 1
        assert pending_appeals[0].appeal_id == appeal2.appeal_id

    def test_get_user_appeals_no_appeals(self, appeal_manager):
        """Test getting appeals for user with no appeals"""
        appeals = appeal_manager.get_user_appeals("no_appeals_user")
        assert appeals == []

    def test_get_appeal_stats_empty(self, appeal_manager):
        """Test getting stats when no appeals exist"""
        stats = appeal_manager.get_appeal_stats()

        assert stats["total"] == 0
        assert stats["by_status"] == {}
        assert stats["approval_rate"] == 0

    def test_get_appeal_stats_no_reviewed(self, appeal_manager):
        """Test getting stats when all appeals are pending"""
        appeal_manager.create_appeal("user1", "hash1", "Reason")
        appeal_manager.create_appeal("user2", "hash2", "Reason")

        stats = appeal_manager.get_appeal_stats()

        assert stats["total"] == 2
        assert stats["approval_rate"] == 0

    def test_create_appeal_reason_truncation(self, appeal_manager):
        """Test appeal reason is truncated to 500 chars"""
        long_reason = "x" * 600
        appeal = appeal_manager.create_appeal("user123", "hash", long_reason)

        assert len(appeal.reason) == 500

    def test_appeal_compensation_execution(self, appeal_manager):
        """Test compensation is triggered on approval"""
        appeal = appeal_manager.create_appeal("user", "hash", "Test")

        with patch.object(appeal_manager, "_apply_compensation") as mock_compensation:
            appeal_manager.review_appeal(
                appeal.appeal_id, "admin", AppealStatus.APPROVED, "Approved"
            )
            mock_compensation.assert_called_once()

    def test_calculate_avg_review_time_no_reviewed(self, appeal_manager):
        """Test average review time calculation with no reviewed appeals"""
        avg_time = appeal_manager._calculate_avg_review_time()
        assert avg_time == 0


@pytest.mark.unit
class TestResponseStrategy:
    """Test ResponseStrategy dataclass"""

    def test_response_strategy_creation(self):
        """Test ResponseStrategy instantiation"""
        strategy = ResponseStrategy(
            level=ResponseLevel.GENTLE_REMINDER,
            message="Test message",
            resources=[{"name": "Resource 1"}],
            notify_admin=True,
            log_detail=False,
        )

        assert strategy.level == ResponseLevel.GENTLE_REMINDER
        assert strategy.message == "Test message"
        assert len(strategy.resources) == 1
        assert strategy.notify_admin is True
        assert strategy.log_detail is False

    def test_response_strategy_defaults(self):
        """Test ResponseStrategy default values"""
        strategy = ResponseStrategy(level=ResponseLevel.NONE, message="")

        assert strategy.resources == []
        assert strategy.notify_admin is False
        assert strategy.log_detail is True


@pytest.mark.unit
class TestIntegrationScenarios:
    """Integration tests for complete workflows"""

    def test_complete_violation_workflow(self):
        """Test complete workflow from detection to response"""
        safety_rules = SafetyRules()
        privacy_logger = PrivacyLogger(log_file=None)

        user_id = "integration_user"
        text = "Toxic message with email@example.com"
        scores = {"toxicity": 0.75, "bullying": 0.6}

        level = safety_rules.determine_response_level(
            toxicity_score=scores["toxicity"], bullying_score=scores["bullying"], user_id=user_id
        )

        strategy = safety_rules.generate_response(level)
        log_entry = privacy_logger.log_event("toxicity", text, scores, level, user_id)

        safety_rules.update_user_history(user_id, level, scores)

        assert level == ResponseLevel.RESOURCE_ESCALATION
        assert len(strategy.resources) > 0
        assert log_entry["pii_detected"]["email"] == 1
        assert user_id in safety_rules.user_histories

    def test_appeal_workflow_approval(self):
        """Test complete appeal workflow with approval"""
        appeal_manager = AppealManager()

        appeal = appeal_manager.create_appeal("user", "event_hash", "I think this is wrong")

        assert appeal.status == AppealStatus.PENDING

        reviewed = appeal_manager.review_appeal(
            appeal.appeal_id, "reviewer123", AppealStatus.APPROVED, "Confirmed false positive"
        )

        assert reviewed.status == AppealStatus.APPROVED
        assert reviewed.reviewer == "reviewer123"

        stats = appeal_manager.get_appeal_stats()
        assert stats["approval_rate"] == 100.0

    def test_escalation_with_repeated_violations(self):
        """Test response escalation with repeated violations"""
        safety_rules = SafetyRules()
        user_id = "escalating_user"

        for i in range(3):
            level = safety_rules.determine_response_level(0.55, 0.5, user_id)
            safety_rules.update_user_history(user_id, level, {"toxicity": 0.55})

        history = safety_rules.user_histories[user_id]
        assert history.total_count == 3

        final_level = safety_rules.determine_response_level(0.55, 0.5, user_id)
        assert final_level == ResponseLevel.RESOURCE_ESCALATION

    def test_response_level_with_emotion_only(self):
        """Test response level triggered by emotion alone"""
        safety_rules = SafetyRules()

        level = safety_rules.determine_response_level(
            toxicity_score=0.1,
            bullying_score=0.1,
            user_id="emotional_user",
            emotion="neg",
            emotion_strength=4,
        )

        assert level == ResponseLevel.RESOURCE_ESCALATION

    def test_silent_handover_threshold(self):
        """Test SILENT_HANDOVER triggered at >0.9 threshold"""
        safety_rules = SafetyRules()

        level = safety_rules.determine_response_level(
            toxicity_score=0.95, bullying_score=0.92, user_id="severe_case"
        )

        assert level == ResponseLevel.SILENT_HANDOVER

    def test_special_protection_child_with_vulnerable(self):
        """Test combined child + vulnerable protection"""
        safety_rules = SafetyRules()

        protection = safety_rules.should_apply_special_protection(user_age=10, is_vulnerable=True)

        assert protection["lower_threshold"] is True
        assert protection["priority_resources"] is True
        assert protection["parental_notify"] is True
        assert protection["extra_logging"] is True

    def test_generate_response_with_context(self):
        """Test generate_response with additional context"""
        safety_rules = SafetyRules()

        context = {"severity": "high", "repeat_offender": True}
        strategy = safety_rules.generate_response(ResponseLevel.RESOURCE_ESCALATION, context)

        assert strategy.level == ResponseLevel.RESOURCE_ESCALATION
        assert len(strategy.resources) > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])