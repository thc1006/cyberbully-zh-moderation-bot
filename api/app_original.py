"""
CyberPuppy FastAPI Application
中文網路霸凌防治 API 服務

提供毒性偵測、情緒分析與可解釋性輸出的 REST API 端點
"""

import hashlib
import logging
import re
import time
from datetime import datetime
from typing import Any, Dict, List, Optional

import uvicorn
from fastapi import FastAPI, HTTPException, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, field_validator
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded
from slowapi.util import get_remote_address

# 設定日誌
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Rate limiter 初始化
limiter = Limiter(key_func=get_remote_address)

# FastAPI app 初始化
app = FastAPI(
    title="CyberPuppy Moderation API",
    description="中文網路霸凌防治與毒性偵測 API",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

# 中介層設定
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

# CORS 設定
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 生產環境應限制特定域名
    allow_credentials=True,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)

# 信任主機中介層
app.add_middleware(TrustedHostMiddleware, allowed_hosts=["*"])  # 生產環境應限制特定主機

# 常數設定
MAX_TEXT_LENGTH = 1000
MAX_CONTEXT_LENGTH = 2000
PII_PATTERNS = [
    r"\d{4}[-\s]?\d{4}[-\s]?\d{4}[-\s]?\d{4}",  # 信用卡號
    r"\b\d{10,11}\b",  # 手機號碼
    r"\b[A-Z][0-9]{9}\b",  # 身分證號格式
    r"\b\w+@\w+\.\w+\b",  # 電子郵件
]


# Pydantic 模型定義
class AnalyzeRequest(BaseModel):
    """分析請求模型"""

    text: str = Field(
        ..., min_length=1, max_length=MAX_TEXT_LENGTH, description="待分析文本"
    )
    context: Optional[str] = Field(
        None, max_length=MAX_CONTEXT_LENGTH, description="對話上下文（可選）"
    )
    thread_id: Optional[str] = Field(
        None, max_length=50, description="對話串 ID（可選）"
    )

    @field_validator("text")
    @classmethod
    def validate_text_not_empty(cls, v: str) -> str:
        if not v or not v.strip():
            raise ValueError("文本不能為空")
        return v.strip()


class ToxicityScore(BaseModel):
    """毒性分數"""

    none: float = Field(..., ge=0, le=1, description="無毒性機率")
    toxic: float = Field(..., ge=0, le=1, description="一般毒性機率")
    severe: float = Field(..., ge=0, le=1, description="嚴重毒性機率")


class BullyingScore(BaseModel):
    """霸凌分數"""

    none: float = Field(..., ge=0, le=1, description="無霸凌機率")
    harassment: float = Field(..., ge=0, le=1, description="騷擾機率")
    threat: float = Field(..., ge=0, le=1, description="威脅機率")


class RoleScore(BaseModel):
    """角色分數"""

    none: float = Field(..., ge=0, le=1, description="無特定角色機率")
    perpetrator: float = Field(..., ge=0, le=1, description="施暴者機率")
    victim: float = Field(..., ge=0, le=1, description="受害者機率")
    bystander: float = Field(..., ge=0, le=1, description="旁觀者機率")


class EmotionScore(BaseModel):
    """情緒分數"""

    positive: float = Field(..., ge=0, le=1, description="正面情緒機率")
    neutral: float = Field(..., ge=0, le=1, description="中性情緒機率")
    negative: float = Field(..., ge=0, le=1, description="負面情緒機率")


class ExplanationData(BaseModel):
    """可解釋性資料"""

    important_words: List[Dict[str, float]] = Field(..., description="重要詞彙與權重")
    method: str = Field(..., description="解釋方法 (IG/SHAP)")
    confidence: float = Field(..., ge=0, le=1, description="預測信心度")


class AnalyzeResponse(BaseModel):
    """分析回應模型"""

    # 預測標籤
    toxicity: str = Field(..., description="毒性標籤: none|toxic|severe")
    bullying: str = Field(..., description="霸凌標籤: none|harassment|threat")
    role: str = Field(..., description="角色標籤: none|perpetrator|victim|bystander")
    emotion: str = Field(..., description="情緒標籤: pos|neu|neg")
    emotion_strength: int = Field(..., ge=0, le=4, description="情緒強度 0-4")

    # 機率分數
    scores: Dict[str, Any] = Field(..., description="各類別機率分數")

    # 可解釋性
    explanations: ExplanationData = Field(..., description="模型解釋資料")

    # 元資料
    text_hash: str = Field(..., description="輸入文本雜湊值")
    timestamp: str = Field(..., description="分析時間戳記")
    processing_time_ms: float = Field(..., description="處理時間（毫秒）")


class HealthResponse(BaseModel):
    """健康檢查回應"""

    status: str
    timestamp: str
    version: str
    uptime_seconds: float


# 啟動時間記錄
start_time = time.time()


# PII 遮蔽函式
def mask_pii(text: str) -> str:
    """遮蔽個人識別資訊"""
    masked_text = text
    for pattern in PII_PATTERNS:
        masked_text = re.sub(pattern, "[MASKED]", masked_text, flags=re.IGNORECASE)
    return masked_text


def generate_text_hash(text: str) -> str:
    """生成文本雜湊值（用於日誌而非儲存原文）"""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]


# 模擬分析函式（實際實作時需替換為真實模型）
async def analyze_text_content(
    text: str, context: Optional[str] = None
) -> Dict[str, Any]:
    """
    模擬文本分析
    實際實作時需要載入並使用訓練好的模型
    """
    # 這裡應該載入模型並進行真實預測
    # 目前回傳模擬資料

    # 模擬處理時間
    import random

    await asyncio.sleep(0.1 + random.random() * 0.2)

    # 模擬預測結果（基於關鍵字的簡單規則）
    text_lower = text.lower()

    # 毒性檢測模擬
    toxic_keywords = ["笨蛋", "白痴", "去死", "滾", "廢物"]
    severe_keywords = ["殺死", "自殺", "威脅"]

    has_toxic = any(keyword in text_lower for keyword in toxic_keywords)
    has_severe = any(keyword in text_lower for keyword in severe_keywords)

    if has_severe:
        toxicity = "severe"
        tox_scores = {"none": 0.1, "toxic": 0.2, "severe": 0.7}
    elif has_toxic:
        toxicity = "toxic"
        tox_scores = {"none": 0.2, "toxic": 0.7, "severe": 0.1}
    else:
        toxicity = "none"
        tox_scores = {"none": 0.8, "toxic": 0.15, "severe": 0.05}

    # 霸凌檢測模擬
    harassment_keywords = ["騷擾", "煩人", "討厭"]
    threat_keywords = ["威脅", "警告", "後果"]

    has_harassment = any(keyword in text_lower for keyword in harassment_keywords)
    has_threat = any(keyword in text_lower for keyword in threat_keywords)

    if has_threat or has_severe:
        bullying = "threat"
        bully_scores = {"none": 0.1, "harassment": 0.2, "threat": 0.7}
    elif has_harassment or has_toxic:
        bullying = "harassment"
        bully_scores = {"none": 0.2, "harassment": 0.7, "threat": 0.1}
    else:
        bullying = "none"
        bully_scores = {"none": 0.8, "harassment": 0.15, "threat": 0.05}

    # 角色分析模擬
    victim_keywords = ["幫助", "救命", "被欺負"]
    perpetrator_keywords = ["笨蛋", "廢物"] + toxic_keywords

    has_victim = any(keyword in text_lower for keyword in victim_keywords)
    has_perpetrator = any(keyword in text_lower for keyword in perpetrator_keywords)

    if has_perpetrator and not has_victim:
        role = "perpetrator"
        role_scores = {"none": 0.1, "perpetrator": 0.7, "victim": 0.1, "bystander": 0.1}
    elif has_victim:
        role = "victim"
        role_scores = {
            "none": 0.1,
            "perpetrator": 0.05,
            "victim": 0.75,
            "bystander": 0.1,
        }
    else:
        role = "none"
        role_scores = {"none": 0.7, "perpetrator": 0.1, "victim": 0.1, "bystander": 0.1}

    # 情緒分析模擬
    positive_keywords = ["開心", "高興", "好棒", "謝謝", "感謝"]
    negative_keywords = ["難過", "生氣", "討厭", "糟糕"] + toxic_keywords

    has_positive = any(keyword in text_lower for keyword in positive_keywords)
    has_negative = any(keyword in text_lower for keyword in negative_keywords)

    if has_positive and not has_negative:
        emotion = "pos"
        emotion_strength = 3
        emo_scores = {"positive": 0.7, "neutral": 0.2, "negative": 0.1}
    elif has_negative:
        emotion = "neg"
        emotion_strength = 4 if has_severe else 3
        emo_scores = {"positive": 0.1, "neutral": 0.2, "negative": 0.7}
    else:
        emotion = "neu"
        emotion_strength = 1
        emo_scores = {"positive": 0.3, "neutral": 0.5, "negative": 0.2}

    # 生成模擬的重要詞彙
    important_words = []
    words = text.split()
    for word in words[:5]:  # 取前5個詞
        if any(
            keyword in word
            for keyword in toxic_keywords + severe_keywords + harassment_keywords
        ):
            importance = 0.8 + random.random() * 0.2
        else:
            importance = random.random() * 0.5
        important_words.append({"word": word, "importance": importance})

    return {
        "toxicity": toxicity,
        "bullying": bullying,
        "role": role,
        "emotion": emotion,
        "emotion_strength": emotion_strength,
        "scores": {
            "toxicity": tox_scores,
            "bullying": bully_scores,
            "role": role_scores,
            "emotion": emo_scores,
        },
        "explanations": {
            "important_words": important_words,
            "method": "IG",
            "confidence": 0.75 + random.random() * 0.2,
        },
    }


# API 端點
@app.get("/healthz", response_model=HealthResponse)
async def health_check():
    """健康檢查端點"""
    current_time = time.time()
    uptime = current_time - start_time

    return HealthResponse(
        status="healthy",
        timestamp=datetime.now().isoformat(),
        version="1.0.0",
        uptime_seconds=round(uptime, 2),
    )


@app.post("/analyze", response_model=AnalyzeResponse)
@limiter.limit("30/minute")  # 每分鐘最多30次請求
async def analyze_text(request: Request, data: AnalyzeRequest):
    """
    文本分析端點

    分析輸入文本的毒性、霸凌行為、情緒與角色，提供可解釋性輸出
    """
    start_time_ms = time.time() * 1000

    try:
        # 遮蔽 PII（僅用於日誌，不影響分析）
        # masked_text = mask_pii(data.text)  # Currently unused
        text_hash = generate_text_hash(data.text)

        # 記錄請求（不包含原始文本）
        logger.info(
            f"分析請求 - Hash: {text_hash}, 長度: {len(data.text)}, "
            f"IP: {get_remote_address(request)}"
        )

        # 執行文本分析
        analysis_result = await analyze_text_content(
            text=data.text, context=data.context
        )

        # 計算處理時間
        processing_time = (time.time() * 1000) - start_time_ms

        # 構建回應
        response = AnalyzeResponse(
            toxicity=analysis_result["toxicity"],
            bullying=analysis_result["bullying"],
            role=analysis_result["role"],
            emotion=analysis_result["emotion"],
            emotion_strength=analysis_result["emotion_strength"],
            scores=analysis_result["scores"],
            explanations=ExplanationData(**analysis_result["explanations"]),
            text_hash=text_hash,
            timestamp=datetime.now().isoformat(),
            processing_time_ms=round(processing_time, 2),
        )

        # 記錄結果（不包含原始文本）
        logger.info(
            f"分析完成 - Hash: {text_hash}, "
            f"毒性: {response.toxicity}, 霸凌: {response.bullying}, "
            f"情緒: {response.emotion}, 處理時間: {processing_time:.2f}ms"
        )

        return response

    except Exception as e:
        logger.error(
            f"分析錯誤 - Hash: {generate_text_hash(data.text)}, 錯誤: {str(e)}"
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="分析處理失敗，請稍後再試",
        )


@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    """自訂 HTTP 例外處理器"""
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "error": True,
            "message": exc.detail,
            "timestamp": datetime.now().isoformat(),
        },
    )


@app.exception_handler(ValueError)
async def validation_exception_handler(request: Request, exc: ValueError):
    """輸入驗證例外處理器"""
    return JSONResponse(
        status_code=status.HTTP_400_BAD_REQUEST,
        content={
            "error": True,
            "message": f"輸入驗證錯誤: {str(exc)}",
            "timestamp": datetime.now().isoformat(),
        },
    )


# 根路由
@app.get("/")
async def root():
    """根路由，提供 API 資訊"""
    return {
        "name": "CyberPuppy Moderation API",
        "description": "中文網路霸凌防治與毒性偵測 API",
        "version": "1.0.0",
        "docs_url": "/docs",
        "health_check": "/healthz",
    }


if __name__ == "__main__":
    import asyncio

    # 開發環境啟動配置
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True, log_level="info")
