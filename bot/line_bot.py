"""
CyberPuppy LINE Bot Implementation
中文網路霸凌防治 LINE Bot

官方文件參考：
- LINE Messaging API: https://developers.line.biz/en/docs/messaging-api/
- Webhook 設定:
    https://developers.line.biz/en/docs/messaging-api/receiving-messages/
- 簽名驗證: https://developers.line.biz/en/docs/messaging-api/verifying-signatures/
- Python SDK: https://github.com/line/line-bot-sdk-python
"""

import asyncio
import base64
import hashlib
import hmac
import json
import logging
import os
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

import httpx
import uvicorn
from fastapi import BackgroundTasks, FastAPI, HTTPException, Request
from linebot import LineBotApi, WebhookHandler
from linebot.exceptions import LineBotApiError
from linebot.models import (
    BoxComponent,
    BubbleContainer,
    ButtonComponent,
    CarouselContainer,
    FlexSendMessage,
    MessageAction,
    MessageEvent,
    QuickReply,
    QuickReplyButton,
    TextComponent,
    TextSendMessage,
    URIAction,
)

# 設定日誌
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# 環境變數配置
LINE_CHANNEL_ACCESS_TOKEN = os.getenv("LINE_CHANNEL_ACCESS_TOKEN")
LINE_CHANNEL_SECRET = os.getenv("LINE_CHANNEL_SECRET")
CYBERPUPPY_API_URL = os.getenv("CYBERPUPPY_API_URL", "http://localhost:8000")

if not LINE_CHANNEL_ACCESS_TOKEN or not LINE_CHANNEL_SECRET:
    logger.error(
        "LINE 設定遺失：需要設定 LINE_CHANNEL_ACC" "ESS_TOKEN 和 LINE_CHANNEL_SECRET"
    )
    raise ValueError("LINE Bot 設定不完整")

# LINE Bot 初始化
line_bot_api = LineBotApi(LINE_CHANNEL_ACCESS_TOKEN)
handler = WebhookHandler(LINE_CHANNEL_SECRET)

# FastAPI 應用初始化
app = FastAPI(title="CyberPuppy LINE Bot", version="1.0.0")


class ToxicityLevel(Enum):
    """毒性等級"""

    NONE = "none"
    TOXIC = "toxic"
    SEVERE = "severe"


class ResponseStrategy(Enum):
    """回應策略"""

    IGNORE = "ignore"
    GENTLE_REMINDER = "gentle_reminder"
    FIRM_WARNING = "firm_warning"
    RESOURCE_SHARING = "resource_sharing"
    ESCALATION = "escalation"


@dataclass
class UserSession:
    """使用者會話狀態"""

    user_id: str
    recent_messages: List[Dict[str, Any]] = field(default_factory=list)
    warning_count: int = 0
    last_warning_time: Optional[datetime] = None
    escalation_count: int = 0


@dataclass
class RetryConfig:
    """重試配置"""

    max_retries: int = 3
    base_delay: float = 1.0
    max_delay: float = 60.0
    backoff_factor: float = 2.0


# 全域狀態管理
user_sessions: Dict[str, UserSession] = {}
retry_config = RetryConfig()


class CyberPuppyBot:
    """CyberPuppy Bot 主要類別"""

    def __init__(self):
        self.api_client = httpx.AsyncClient(timeout=30.0)

    async def analyze_message(
        self, text: str, context: Optional[str] = None
    ) -> Dict[str, Any]:
        """呼叫分析 API"""
        payload = {"text": text, "context": context}

        for attempt in range(retry_config.max_retries):
            try:
                response = await self.api_client.post(
                    f"{CYBERPUPPY_API_URL}/analyze", json=payload, timeout=30.0
                )
                response.raise_for_status()
                return response.json()

            except httpx.RequestError as e:
                logger.warning(
                    f"API 請求失敗 (嘗試 {attempt + 1}/{retry_config.max_retries}): {e}"
                )
                if attempt == retry_config.max_retries - 1:
                    raise HTTPException(status_code=503, detail="分析服務暫時不可用")

                # 指數退避
                delay = min(
                    retry_config.base_delay * (retry_config.backoff_factor**attempt),
                    retry_config.max_delay,
                )
                await asyncio.sleep(delay)

            except httpx.HTTPStatusError as e:
                logger.error(
                    f"API 回應錯誤: {e.response.status_code} - \
                    {e.response.text}"
                )
                raise HTTPException(status_code=502, detail="分析服務錯誤")

    def determine_response_strategy(
        self, analysis: Dict[str, Any], user_session: UserSession
    ) -> ResponseStrategy:
        """決定回應策略"""
        toxicity = analysis.get("toxicity", "none")
        bullying = analysis.get("bullying", "none")
        emotion = analysis.get("emotion", "neu")

        # 嚴重毒性或威脅 -> 立即警告
        if toxicity == "severe" or bullying == "threat":
            return ResponseStrategy.FIRM_WARNING

        # 一般毒性或騷擾
        if toxicity == "toxic" or bullying == "harassment":
            # 檢查重複違規
            if user_session.warning_count >= 2:
                return ResponseStrategy.ESCALATION
            elif user_session.warning_count >= 1:
                return ResponseStrategy.FIRM_WARNING
            else:
                return ResponseStrategy.GENTLE_REMINDER

        # 負面情緒但無毒性 -> 資源分享
        if emotion == "neg" and analysis.get("emotion_strength", 0) >= 3:
            return ResponseStrategy.RESOURCE_SHARING

        return ResponseStrategy.IGNORE

    def create_gentle_reminder_message(self) -> TextSendMessage:
        """建立溫和提醒訊息"""
        messages = [
            "嗨！讓我們保持友善的對話環境吧 😊",
            "或許可以用更溫和的方式表達想法呢？🌟",
            "每個人都值得被尊重對待 💝",
        ]
        import random

        return TextSendMessage(text=random.choice(messages))

    def create_firm_warning_message(self) -> FlexSendMessage:
        """建立嚴厲警告訊息"""
        bubble = BubbleContainer(
            body=BoxComponent(
                layout="vertical",
                contents=[
                    TextComponent(text="⚠️ 注意事項", weight="bold", size="lg"),
                    TextComponent(
                        text=("我們偵測到您的訊息可能包含不適當內容。請注意："),
                        margin="md",
                        wrap=True,
                    ),
                    TextComponent(
                        text=(
                            "• 網路霸凌會傷害他人\n"
                            "• 持續違規可能導致檢舉\n"
                            "• 讓我們營造友善環境"
                        ),
                        margin="lg",
                        wrap=True,
                        color="#666666",
                    ),
                ],
            ),
            footer=BoxComponent(
                layout="vertical",
                contents=[
                    ButtonComponent(
                        action=URIAction(
                            label="了解網路霸凌",
                            uri="https://www.mohw.gov.tw/cp-4266-48131-1.html",
                        ),
                        style="primary",
                        color="#4ECDC4",
                    )
                ],
            ),
        )
        return FlexSendMessage(alt_text="網路行為提醒", contents=bubble)

    def create_resource_sharing_message(self) -> FlexSendMessage:
        """建立資源分享訊息"""
        carousel = CarouselContainer(
            contents=[
                BubbleContainer(
                    body=BoxComponent(
                        layout="vertical",
                        contents=[
                            TextComponent(
                                text="💚 心理健康資源", weight="bold", size="lg"
                            ),
                            TextComponent(
                                text="如果您需要心理支持", margin="md", wrap=True
                            ),
                        ],
                    ),
                    footer=BoxComponent(
                        layout="vertical",
                        contents=[
                            ButtonComponent(
                                action=URIAction(label="心理健康專線", uri="tel:1925"),
                                color="#69C7A2",
                            )
                        ],
                    ),
                ),
                BubbleContainer(
                    body=BoxComponent(
                        layout="vertical",
                        contents=[
                            TextComponent(text="🤝 霸凌防治", weight="bold", size="lg"),
                            TextComponent(
                                text="遭遇網路霸凌的求助管道", margin="md", wrap=True
                            ),
                        ],
                    ),
                    footer=BoxComponent(
                        layout="vertical",
                        contents=[
                            ButtonComponent(
                                action=URIAction(
                                    label="iWIN網路內容防護",
                                    uri="https://i.win.org.tw/",
                                ),
                                color="#4ECDC4",
                            )
                        ],
                    ),
                ),
            ]
        )
        return FlexSendMessage(alt_text="支持資源", contents=carousel)

    def create_escalation_message(self) -> Tuple[FlexSendMessage, QuickReply]:
        """建立升級處理訊息"""
        bubble = BubbleContainer(
            body=BoxComponent(
                layout="vertical",
                contents=[
                    TextComponent(
                        text="🚨 重要提醒", weight="bold", size="lg", color="#ff0000"
                    ),
                    TextComponent(
                        text=("您已多次發送不適當內容。持續的網路霸凌行為："),
                        margin="md",
                        wrap=True,
                    ),
                    TextComponent(
                        text=(
                            "• 違反平台使用條款\n"
                            "• 可能面臨法律責任\n"
                            "• 傷害他人心理健康"
                        ),
                        margin="lg",
                        wrap=True,
                        color="#666666",
                    ),
                    TextComponent(
                        text="我們鼓勵您尋求適當協助，改善溝通方式。",
                        margin="lg",
                        wrap=True,
                        weight="bold",
                    ),
                ],
            )
        )

        quick_reply = QuickReply(
            items=[
                QuickReplyButton(action=URIAction(label="求助專線", uri="tel:1995")),
                QuickReplyButton(
                    action=MessageAction(label="我了解了", text="我會注意我的言行")
                ),
            ]
        )

        return FlexSendMessage(alt_text="嚴重警告", contents=bubble), quick_reply

    async def update_user_session(
        self, user_id: str, message_text: str, analysis: Dict[str, Any]
    ):
        """更新使用者會話狀態"""
        if user_id not in user_sessions:
            user_sessions[user_id] = UserSession(user_id=user_id)

        session = user_sessions[user_id]

        # 添加訊息到歷史記錄
        session.recent_messages.append(
            {
                "text": message_text,
                "analysis": analysis,
                "timestamp": datetime.now().isoformat(),
            }
        )

        # 保留最近 10 條訊息
        if len(session.recent_messages) > 10:
            session.recent_messages.pop(0)

        # 更新警告計數
        toxicity = analysis.get("toxicity", "none")
        bullying = analysis.get("bullying", "none")

        if toxicity in ["toxic", "severe"] or bullying in ["harassment", "threat"]:
            session.warning_count += 1
            session.last_warning_time = datetime.now()

        # 重置警告計數（24小時後）
        if (
            session.last_warning_time
            and datetime.now() - session.last_warning_time > timedelta(hours=24)
        ):
            session.warning_count = max(0, session.warning_count - 1)

    async def handle_message_analysis(
        self, event: MessageEvent, user_id: str, message_text: str
    ):
        """處理訊息分析與回應"""
        try:
            # 建立對話上下文
            context = None
            if user_id in user_sessions and user_sessions[user_id].recent_messages:
                # 最近3條訊息
                recent = user_sessions[user_id].recent_messages[-3:]
                context = " | ".join([msg["text"] for msg in recent])

            # 分析訊息
            analysis = await self.analyze_message(message_text, context)
            logger.info(
                f"分析結果 - 用戶: {user_id[:8]}..., "
                f"毒性: {analysis.get('toxicity')}, "
                f"霸凌: {analysis.get('bullying')}"
            )

            # 更新用戶會話
            await self.update_user_session(user_id, message_text, analysis)

            # 決定回應策略
            strategy = self.determine_response_strategy(
                analysis, user_sessions[user_id]
            )
            logger.info(f"回應策略: {strategy.value}")

            # 執行回應
            if strategy == ResponseStrategy.GENTLE_REMINDER:
                message = self.create_gentle_reminder_message()
                line_bot_api.reply_message(event.reply_token, message)

            elif strategy == ResponseStrategy.FIRM_WARNING:
                message = self.create_firm_warning_message()
                line_bot_api.reply_message(event.reply_token, message)

            elif strategy == ResponseStrategy.RESOURCE_SHARING:
                message = self.create_resource_sharing_message()
                line_bot_api.reply_message(event.reply_token, message)

            elif strategy == ResponseStrategy.ESCALATION:
                message, quick_reply = self.create_escalation_message()
                # 升級處理：記錄、通知管理員等
                user_sessions[user_id].escalation_count += 1
                logger.warning(
                    f"用戶升級警告 - ID: {user_id}, 次數: \
                        {user_sessions[user_id].escalation_count}"
                )

                line_bot_api.reply_message(
                    event.reply_token, message, quick_reply=quick_reply
                )

        except Exception as e:
            logger.error(f"處理訊息分析時發生錯誤: {e}")
            # 發送通用錯誤訊息
            error_message = TextSendMessage(
                text="抱歉，系統暫時無法處理您的訊息，請稍後再試。"
            )
            line_bot_api.reply_message(event.reply_token, error_message)


# 初始化 Bot 實例
cyberpuppy_bot = CyberPuppyBot()


# Webhook 端點
@app.post("/webhook")
async def webhook_handler(request: Request, background_tasks: BackgroundTasks):
    """LINE Webhook 處理器"""

    # 取得請求內容
    body = await request.body()
    signature = request.headers.get("X-Line-Signature", "")

    # 驗證簽名
    if not verify_line_signature(body, signature):
        logger.error("LINE 簽名驗證失敗")
        raise HTTPException(status_code=400, detail="Invalid signature")

    # 解析事件
    try:
        events = json.loads(body.decode("utf-8")).get("events", [])
        logger.info(f"收到 {len(events)} 個事件")

        for event_data in events:
            # 在背景處理事件，避免阻塞響應
            background_tasks.add_task(process_line_event, event_data)

        return {"status": "ok"}

    except Exception as e:
        logger.error(f"處理 webhook 時發生錯誤: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")


def verify_line_signature(body: bytes, signature: str) -> bool:
    """驗證 LINE Webhook 簽名 (HMAC-SHA256)"""
    if not signature:
        return False

    try:
        # 計算預期簽名
        expected_signature = base64.b64encode(
            hmac.new(LINE_CHANNEL_SECRET.encode("utf-8"), body, hashlib.sha256).digest()
        ).decode("utf-8")

        # 安全比較
        return hmac.compare_digest(signature, expected_signature)

    except Exception as e:
        logger.error(f"簽名驗證時發生錯誤: {e}")
        return False


async def process_line_event(event_data: dict):
    """處理單個 LINE 事件"""
    try:
        event_type = event_data.get("type")

        if event_type == "message":
            message_type = event_data.get("message", {}).get("type")

            if message_type == "text":
                user_id = event_data.get("source", {}).get("userId")
                message_text = event_data.get("message", {}).get("text")
                reply_token = event_data.get("replyToken")

                if user_id and message_text and reply_token:
                    # 建立模擬事件物件進行處理
                    mock_event = type(
                        "MockEvent",
                        (),
                        {
                            "reply_token": reply_token,
                            "source": type("MockSource", (), {"user_id": user_id})(),
                        },
                    )()

                    await cyberpuppy_bot.handle_message_analysis(
                        mock_event, user_id, message_text
                    )

        elif event_type == "postback":
            # 處理 Postback 事件
            logger.info(
                f"收到 Postback 事件: {event_data.get('postback', {}).get('data')}"
            )

    except Exception as e:
        logger.error(f"處理 LINE 事件時發生錯誤: {e}")


@app.get("/health")
async def health_check():
    """健康檢查"""
    try:
        # 檢查 LINE Bot API 連線
        line_bot_api.get_profile("test")  # 這會失敗，但可以檢查 API token 是否有效
    except LineBotApiError as e:
        if e.status_code == 404:  # 用戶不存在是預期的
            api_status = "healthy"
        else:
            api_status = "unhealthy"
    except Exception:
        api_status = "unknown"

    # 檢查分析 API
    try:
        async with httpx.AsyncClient() as client:
            response = await client.get(f"{CYBERPUPPY_API_URL}/healthz", timeout=5.0)
            analysis_api_status = (
                "healthy" if response.status_code == 200 else "unhealthy"
            )
    except Exception:
        analysis_api_status = "unhealthy"

    return {
        "status": (
            "healthy"
            if api_status != "unhealthy" and analysis_api_status == "healthy"
            else "degraded"
        ),
        "line_api": api_status,
        "analysis_api": analysis_api_status,
        "timestamp": datetime.now().isoformat(),
        "active_sessions": len(user_sessions),
    }


@app.get("/stats")
async def get_stats():
    """取得統計資訊"""
    total_warnings = sum(session.warning_count for session in user_sessions.values())
    total_escalations = sum(
        session.escalation_count for session in user_sessions.values()
    )

    return {
        "active_users": len(user_sessions),
        "total_warnings": total_warnings,
        "total_escalations": total_escalations,
        "timestamp": datetime.now().isoformat(),
    }


if __name__ == "__main__":
    logger.info("🐕 CyberPuppy LINE Bot 啟動中...")
    logger.info(f"分析 API URL: {CYBERPUPPY_API_URL}")

    uvicorn.run("line_bot:app", host="0.0.0.0", port=8080, reload=True)
