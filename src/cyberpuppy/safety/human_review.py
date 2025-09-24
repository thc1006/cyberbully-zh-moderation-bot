"""
人工審核介面

提供申訴案件的人工審核功能，包含審核儀表板與批次處理
"""

import json
import logging
from dataclasses import dataclass
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

from .rules import Appeal, AppealManager, AppealStatus

logger = logging.getLogger(__name__)


class ReviewPriority(Enum):
    """審核優先級"""

    URGENT = "urgent"  # 24小時內處理
    HIGH = "high"  # 3天內處理
    NORMAL = "normal"  # 5天內處理
    LOW = "low"  # 7天內處理


class ReviewAction(Enum):
    """審核動作"""

    APPROVE = "approve"  # 批准（誤判）
    REJECT = "reject"  # 駁回
    ESCALATE = "escalate"  # 升級至上級
    REQUEST_INFO = "request_info"  # 要求更多資訊
    DEFER = "defer"  # 延後處理


@dataclass
class ReviewTask:
    """審核任務"""

    task_id: str
    appeal_id: str
    priority: ReviewPriority
    assigned_to: Optional[str] = None
    due_date: Optional[datetime] = None
    notes: List[str] = None

    def __post_init__(self):
        if self.notes is None:
            self.notes = []

        # 根據優先級設定截止日期
        if not self.due_date:
            now = datetime.now()
            if self.priority == ReviewPriority.URGENT:
                self.due_date = now + timedelta(hours=24)
            elif self.priority == ReviewPriority.HIGH:
                self.due_date = now + timedelta(days=3)
            elif self.priority == ReviewPriority.NORMAL:
                self.due_date = now + timedelta(days=5)
            else:  # LOW
                self.due_date = now + timedelta(days=7)


class HumanReviewInterface:
    """人工審核介面"""

    def __init__(self, appeal_manager: AppealManager):
        """
        初始化審核介面

        Args:
            appeal_manager: 申訴管理器實例
        """
        self.appeal_manager = appeal_manager
        self.review_tasks: Dict[str, ReviewTask] = {}
        self.reviewer_workload: Dict[str, int] = {}  # 審核員工作量

    def create_review_task(
        self,
        appeal_id: str,
        priority: ReviewPriority = ReviewPriority.NORMAL,
        assigned_to: Optional[str] = None,
    ) -> ReviewTask:
        """
        建立審核任務

        Args:
            appeal_id: 申訴 ID
            priority: 優先級
            assigned_to: 指派給特定審核員

        Returns:
            審核任務
        """
        import uuid

        task_id = str(uuid.uuid4())[:8]

        # 如果未指派，自動分配給工作量最少的審核員
        if not assigned_to:
            assigned_to = self._auto_assign_reviewer()

        task = ReviewTask(
            task_id=task_id, appeal_id=appeal_id, priority=priority,
                assigned_to=assigned_to
        )

        self.review_tasks[task_id] = task

        # 更新審核員工作量
        if assigned_to:
            self.reviewer_workload[assigned_to] = self.reviewer_workload.get(
                assigned_to,
                0
            ) + 1

        logger.info(f"審核任務建立 - ID: {task_id}, 申訴: {appeal_id}, 優先級: {priority.value}")

        return task

    def _auto_assign_reviewer(self) -> Optional[str]:
        """自動分配審核員（選擇工作量最少的）"""
        if not self.reviewer_workload:
            return None

        return min(self.reviewer_workload.items(), key=lambda x: x[1])[0]

    def get_pending_reviews(
        self, reviewer_id: Optional[str] = None, priority_filter:
            Optional[ReviewPriority] = None
    ) -> List[Tuple[ReviewTask, Appeal]]:
        """
        取得待審核案件

        Args:
            reviewer_id: 審核員 ID（篩選）
            priority_filter: 優先級篩選

        Returns:
            (任務, 申訴) 清單
        """
        pending = []

        for task in self.review_tasks.values():
            # 篩選條件
            if reviewer_id and task.assigned_to != reviewer_id:
                continue
            if priority_filter and task.priority != priority_filter:
                continue

            # 取得對應的申訴
            appeal = self.appeal_manager.appeals.get(task.appeal_id)
            if appeal and appeal.status == AppealStatus.PENDING:
                pending.append((task, appeal))

        # 依優先級和截止日期排序
        priority_order = {
            ReviewPriority.URGENT: 0,
            ReviewPriority.HIGH: 1,
            ReviewPriority.NORMAL: 2,
            ReviewPriority.LOW: 3,
        }

        pending.sort(
            key=lambda x: (priority_order[x[0].priority],
            x[0].due_date)
        )

        return pending

    def process_review(
        self,
        task_id: str,
        reviewer_id: str,
        action: ReviewAction,
        notes: str,
        additional_data: Optional[Dict[str, Any]] = None,
    ) -> Tuple[bool, str]:
        """
        處理審核

        Args:
            task_id: 任務 ID
            reviewer_id: 審核員 ID
            action: 審核動作
            notes: 審核說明
            additional_data: 額外資料

        Returns:
            (成功與否, 訊息)
        """
        if task_id not in self.review_tasks:
            return False, "審核任務不存在"

        task = self.review_tasks[task_id]
        task.notes.append(f"{datetime.now().isoformat()} - {reviewer_id}: {notes}")

        # 根據動作處理
        if action == ReviewAction.APPROVE:
            # 批准申訴（誤判）
            appeal = self.appeal_manager.review_appeal(
                appeal_id=task.appeal_id,
                reviewer=reviewer_id,
                decision=AppealStatus.APPROVED,
                resolution=notes,
            )
            if appeal:
                self._complete_task(task_id)
                return True, "申訴已批准，誤判確認"

        elif action == ReviewAction.REJECT:
            # 駁回申訴
            appeal = self.appeal_manager.review_appeal(
                appeal_id=task.appeal_id,
                reviewer=reviewer_id,
                decision=AppealStatus.REJECTED,
                resolution=notes,
            )
            if appeal:
                self._complete_task(task_id)
                return True, "申訴已駁回"

        elif action == ReviewAction.ESCALATE:
            # 升級處理
            appeal = self.appeal_manager.appeals.get(task.appeal_id)
            if appeal:
                appeal.status = AppealStatus.ESCALATED
                appeal.updated_at = datetime.now()
                task.priority = ReviewPriority.URGENT
                # 重新分配給上級審核員
                senior_reviewer = (
                    additional_data.get("senior_"
                        "reviewer") if additional_data else None
                )
                if senior_reviewer:
                    task.assigned_to = senior_reviewer
                return True, "案件已升級至上級審核"

        elif action == ReviewAction.REQUEST_INFO:
            # 要求更多資訊
            task.notes.append(f"要求補充資料: {notes}")
            # 實際實作時，發送通知給使用者
            return True, "已要求補充資料"

        elif action == ReviewAction.DEFER:
            # 延後處理
            defer_days = additional_data.get("defer"
                "_days", 3) if additional_data else 3
            task.due_date = datetime.now() + timedelta(days=defer_days)
            return True, f"已延後 {defer_days} 天處理"

        return False, "未知的審核動作"

    def _complete_task(self, task_id: str):
        """完成審核任務"""
        if task_id in self.review_tasks:
            task = self.review_tasks[task_id]
            if task.assigned_to:
                self.reviewer_workload[task.assigned_to] = max(
                    0, self.reviewer_workload.get(task.assigned_to, 1) - 1
                )
            del self.review_tasks[task_id]
            logger.info(f"審核任務完成: {task_id}")

    def batch_review(
        self, task_ids: List[str], reviewer_id: str, action: ReviewAction,
            notes: str
    ) -> Dict[str, Tuple[bool, str]]:
        """
        批次審核

        Args:
            task_ids: 任務 ID 清單
            reviewer_id: 審核員 ID
            action: 統一的審核動作
            notes: 審核說明

        Returns:
            各任務的處理結果
        """
        results = {}

        for task_id in task_ids:
            success, message = self.process_review(
                task_id=task_id, reviewer_id=reviewer_id, action=action,
                    notes=notes
            )
            results[task_id] = (success, message)

        successful = sum(1 for s, _ in results.values() if s)
        logger.info(f"批次審核完成 - 成功: {successful}/{len(task_ids)}")

        return results

    def get_review_dashboard(
        self,
        reviewer_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        取得審核儀表板資料

        Args:
            reviewer_id: 審核員 ID（可選）

        Returns:
            儀表板資料
        """
        # 整體統計
        total_pending = len(
            [
                t
                for t in self.review_tasks.values()
                if self.appeal_manager.appeals.get(t.appeal_id, Appeal).status
                == AppealStatus.PENDING
            ]
        )

        # 依優先級分組
        by_priority = {
            ReviewPriority.URGENT: 0,
            ReviewPriority.HIGH: 0,
            ReviewPriority.NORMAL: 0,
            ReviewPriority.LOW: 0,
        }

        overdue = []
        upcoming = []
        now = datetime.now()

        for task in self.review_tasks.values():
            # 如果指定審核員，只統計該審核員的任務
            if reviewer_id and task.assigned_to != reviewer_id:
                continue

            appeal = self.appeal_manager.appeals.get(task.appeal_id)
            if appeal and appeal.status == AppealStatus.PENDING:
                by_priority[task.priority] += 1

                if task.due_date:
                    if task.due_date < now:
                        overdue.append(task)
                    elif task.due_date < now + timedelta(hours=24):
                        upcoming.append(task)

        # 審核員工作量
        workload_stats = {}
        if reviewer_id:
            workload_stats = {
                "assigned": self.reviewer_workload.get(reviewer_id, 0),
                "complet"
                    "ed_today": self._get_completed_count(reviewer_id, hours=24),
                "average_time": self._get_average_review_time(reviewer_id),
            }

        dashboard = {
            "total_pending": total_pending,
            "by_priority": {k.value: v for k, v in by_priority.items()},
            "overdue": len(overdue),
            "upcoming_24h": len(upcoming),
            "workload": workload_stats,
            "appeal_stats": self.appeal_manager.get_appeal_stats(),
        }

        return dashboard

    def _get_completed_count(self, reviewer_id: str, hours: int = 24) -> int:
        """取得審核員完成數量"""
        cutoff = datetime.now() - timedelta(hours=hours)
        count = 0

        for appeal in self.appeal_manager.appeals.values():
            if (
                appeal.reviewer == reviewer_id
                and appeal.status in [AppealStatus.APPROVED,
                    AppealStatus.REJECTED]
                and appeal.updated_at > cutoff
            ):
                count += 1

        return count

    def _get_average_review_time(self, reviewer_id: str) -> float:
        """取得審核員平均處理時間（小時）"""
        times = []

        for appeal in self.appeal_manager.appeals.values():
            if appeal.reviewer == reviewer_id and appeal.status in [
                AppealStatus.APPROVED,
                AppealStatus.REJECTED,
            ]:
                duration = (appeal.updated_at -
                    appeal.created_at).total_seconds() / 3600
                times.append(duration)

        return round(sum(times) / len(times), 2) if times else 0.0

    def export_review_report(
        self, start_date: datetime, end_date: datetime, format: str = "json"
    ) -> str:
        """
        匯出審核報告

        Args:
            start_date: 開始日期
            end_date: 結束日期
            format: 匯出格式 (json/csv)

        Returns:
            報告內容
        """
        # 收集期間內的申訴
        appeals_in_period = []
        for appeal in self.appeal_manager.appeals.values():
            if start_date <= appeal.created_at <= end_date:
                appeals_in_period.append(appeal)

        # 統計資料
        stats = {
            "period": {
                "total_appeals": len(appeals_in_period),
                "approved": len([a for a in appeals_in_period if a.status == AppealStatus.APPROVED]),
                "rejected": len([a for a in appeals_in_period if a.status == AppealStatus.REJECTED]),
                "pending": len([a for a in appeals_in_period if a.status == AppealStatus.PENDING]),
                "escalated": len([a for a in appeals_in_period if a.status == AppealStatus.ESCALATED]),
                "reviewers": list(set(a.reviewer for a in appeals_in_period if a.reviewer)),
            }
        }

        # 計算批准率
        reviewed = stats["period"]["approved"] + stats["period"]["rejected"]
        stats["approval_rate"] = round(
            stats["period"]["approved"] / reviewed * 100 if reviewed > 0 else 0, 1
        )

        if format == "json":
            return json.dumps(stats, ensure_ascii=False, indent=2)
        elif format == "csv":
            # 簡單 CSV 格式
            lines = [
                "metric,value",
                f"period_start,{stats['period']['start']}",
                f"period_end,{stats['period']['end']}",
                f"total_appeals,{stats['total_appeals']}",
                f"approved,{stats['approved']}",
                f"rejected,{stats['rejected']}",
                f"pending,{stats['pending']}",
                f"escalated,{stats['escalated']}",
                f"approval_rate,{stats['approval_rate']}%",
                f"total_reviewers,{len(stats['reviewers'])}",
            ]
            return "\n".join(lines)

        return str(stats)


# 使用範例
def example_human_review():
    """人工審核介面使用範例"""

    # 初始化
    appeal_manager = AppealManager()
    review_interface = HumanReviewInterface(appeal_manager)

    # 建立一些測試申訴
    appeals = []
    for i in range(5):
        appeal = appeal_manager.create_appeal(
            user_id=f"user_{i}", event_hash=f"hash_{i}", reason=f"測試申訴 {i}"
        )
        appeals.append(appeal)

    # 建立審核任務
    priorities = [
        ReviewPriority.URGENT,
        ReviewPriority.HIGH,
        ReviewPriority.NORMAL,
        ReviewPriority.NORMAL,
        ReviewPriority.LOW,
    ]

    tasks = []
    for appeal, priority in zip(appeals, priorities):
        task = review_interface.create_review_task(
            appeal_id=appeal.appeal_id, priority=priority, assigned_to="review"
                "er_001"
        )
        tasks.append(task)
        print(f"建立審核任務: {task.task_id} - 優先級: {priority.value}")

    # 取得待審核清單
    print("\n待審核案件:")
    pending = review_interface.get_pending_reviews(reviewer_id="reviewer_001")
    for task, appeal in pending:
        print(f"  - 任務: {task.task_id}, 申訴: {appeal.appeal_id}, 截止: {task.due_date}")

    # 處理審核
    print("\n處理審核:")

    # 批准第一個
    success, message = review_interface.process_review(
        task_id=tasks[0].task_id,
        reviewer_id="reviewer_001",
        action=ReviewAction.APPROVE,
        notes="確認為誤判，該內容為朋友間玩笑",
    )
    print(f"審核 {tasks[0].task_id}: {message}")

    # 駁回第二個
    success, message = review_interface.process_review(
        task_id=tasks[1].task_id,
        reviewer_id="reviewer_001",
        action=ReviewAction.REJECT,
        notes="經查證，該內容確實包含霸凌行為",
    )
    print(f"審核 {tasks[1].task_id}: {message}")

    # 升級第三個
    success, message = review_interface.process_review(
        task_id=tasks[2].task_id,
        reviewer_id="reviewer_001",
        action=ReviewAction.ESCALATE,
        notes="案情複雜，需要上級審核",
        additional_data={"senior_reviewer": "senior_001"},
    )
    print(f"審核 {tasks[2].task_id}: {message}")

    # 批次處理剩餘的
    print("\n批次審核:")
    remaining_ids = [tasks[3].task_id, tasks[4].task_id]
    results = review_interface.batch_review(
        task_ids=remaining_ids,
        reviewer_id="reviewer_001",
        action=ReviewAction.DEFER,
        notes="需要更多時間調查",
    )
    for task_id, (success, message) in results.items():
        print(f"  {task_id}: {message}")

    # 查看儀表板
    print("\n審核儀表板:")
    dashboard = review_interface.get_review_dashboard(reviewer_id="review"
        "er_001")
    print(json.dumps(dashboard, ensure_ascii=False, indent=2))

    # 匯出報告
    print("\n審核報告:")
    report = review_interface.export_review_report(
        start_date=datetime.now() - timedelta(days=7), end_date=datetime.now(), format="js"
            "on"
    )
    print(report)


if __name__ == "__main__":
    example_human_review()
