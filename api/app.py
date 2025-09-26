"""
CyberPuppy FastAPI Application with Real Model Integration
中文網路霸凌防治 API 服務

提供毒性偵測、情緒分析與可解釋性輸出的 REST API 端點
使用真實訓練的深度學習模型進行推理
"""

import hashlib
import logging
import re
import time
from datetime import datetime
from typing import Any, Dict, List, Optional
from contextlib import asynccontextmanager

import uvicorn
from fastapi import FastAPI, HTTPException, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, field_validator
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded
from slowapi.util import get_remote_address

# Import model loader and SHAP explainer
from model_loader import get_model_loader

# 設定日誌
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Rate limiter 初始化
limiter = Limiter(key_func=get_remote_address)

# Global model loader and metrics
model_loader = None
model_metrics = {
    "total_predictions": 0,
    "successful_predictions": 0,
    "failed_predictions": 0,
    "total_processing_time": 0.0,
    "cache_hits": 0,
    "startup_time": None,
}


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager for startup and shutdown."""
    startup_start = time.time()
    global model_loader

    try:
        # Initialize model loader
        logger.info("Starting CyberPuppy API - loading models...")
        model_loader = get_model_loader()

        # Load models
        model_loader.load_models()
        logger.info("Models loaded successfully")

        # Warm up models
        warmup_stats = model_loader.warm_up_models()
        logger.info(f"Model warm-up completed: {warmup_stats}")

        startup_time = time.time() - startup_start
        model_metrics["startup_time"] = startup_time

        logger.info(f"CyberPuppy API startup complete in {startup_time:.2f}s")

    except Exception as e:
        logger.error(f"Startup failed: {e}")
        raise

    yield  # Application runs

    # Cleanup
    logger.info("Shutting down CyberPuppy API")
    if model_loader:
        model_loader.clear_cache()
    logger.info("Shutdown complete")


# FastAPI app 初始化
app = FastAPI(
    title="CyberPuppy Moderation API",
    description="中文網路霸凌防治與毒性偵測 API",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan,
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

# 啟動時間記錄
start_time = time.time()


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


class ImportantWord(BaseModel):
    """重要詞彙與權重"""

    word: str = Field(..., description="詞彙")
    importance: float = Field(..., ge=0, le=1, description="重要性權重")


class ExplanationData(BaseModel):
    """可解釋性資料"""

    important_words: List[ImportantWord] = Field(..., description="重要詞彙與權重")
    method: str = Field(..., description="解釋方法 (IG/SHAP)")
    confidence: float = Field(..., ge=0, le=1, description="預測信心度")


class SHAPExplanationRequest(BaseModel):
    """SHAP解釋請求模型"""

    text: str = Field(
        ..., min_length=1, max_length=MAX_TEXT_LENGTH, description="待解釋文本"
    )
    task: str = Field(
        "toxicity", description="解釋任務: toxicity|bullying|role|emotion"
    )
    max_evals: int = Field(
        500, ge=100, le=2000, description="最大評估次數（影響準確性和速度）"
    )
    visualization_type: str = Field(
        "waterfall", description="可視化類型: force|waterfall|text|summary"
    )

    @field_validator("task")
    @classmethod
    def validate_task(cls, v: str) -> str:
        allowed_tasks = ["toxicity", "bullying", "role", "emotion"]
        if v not in allowed_tasks:
            raise ValueError(f"任務必須是以下之一: {', '.join(allowed_tasks)}")
        return v

    @field_validator("visualization_type")
    @classmethod
    def validate_visualization_type(cls, v: str) -> str:
        allowed_types = ["force", "waterfall", "text", "summary"]
        if v not in allowed_types:
            raise ValueError(f"可視化類型必須是以下之一: {', '.join(allowed_types)}")
        return v


class SHAPToken(BaseModel):
    """SHAP Token資訊"""

    token: str = Field(..., description="Token文字")
    shap_value: float = Field(..., description="SHAP值")
    position: int = Field(..., description="Token位置")


class SHAPExplanationResponse(BaseModel):
    """SHAP解釋回應模型"""

    # 預測結果
    prediction: str = Field(..., description="預測標籤")
    confidence: float = Field(..., ge=0, le=1, description="預測置信度")
    base_value: float = Field(..., description="基線值")

    # SHAP值
    tokens: List[SHAPToken] = Field(..., description="Token SHAP值列表")
    feature_importance: float = Field(..., description="總特徵重要性")

    # 可視化資料（Base64編碼的圖片）
    visualization_base64: Optional[str] = Field(None, description="可視化圖片（Base64）")

    # 元資料
    task: str = Field(..., description="解釋任務")
    method: str = Field("SHAP", description="解釋方法")
    text_hash: str = Field(..., description="輸入文本雜湊值")
    timestamp: str = Field(..., description="分析時間戳記")
    processing_time_ms: float = Field(..., description="處理時間（毫秒）")


class MisclassificationAnalysisRequest(BaseModel):
    """誤判分析請求模型"""

    texts: List[str] = Field(
        ..., min_items=1, max_items=100, description="文本列表（最多100個）"
    )
    true_labels: List[Dict[str, int]] = Field(
        ..., description="真實標籤列表"
    )
    task: str = Field(
        "toxicity", description="分析任務: toxicity|bullying|role|emotion"
    )

    @field_validator("texts")
    @classmethod
    def validate_texts(cls, v: List[str]) -> List[str]:
        for text in v:
            if not text or not text.strip():
                raise ValueError("所有文本都不能為空")
            if len(text) > MAX_TEXT_LENGTH:
                raise ValueError(f"文本長度不能超過 {MAX_TEXT_LENGTH} 字符")
        return [text.strip() for text in v]

    @field_validator("true_labels")
    @classmethod
    def validate_true_labels(cls, v: List[Dict[str, int]]) -> List[Dict[str, int]]:
        for label_dict in v:
            for key, value in label_dict.items():
                if not isinstance(value, int) or value < 0:
                    raise ValueError("標籤值必須是非負整數")
        return v


class MisclassificationAnalysisResponse(BaseModel):
    """誤判分析回應模型"""

    misclassification_rate: float = Field(..., ge=0, le=1, description="誤判率")
    total_cases: int = Field(..., description="總案例數")
    misclassified_count: int = Field(..., description="誤判案例數")
    correct_count: int = Field(..., description="正確案例數")

    # 統計分析
    error_statistics: Dict[str, float] = Field(..., description="錯誤統計")
    top_error_features: List[Tuple[str, int]] = Field(..., description="高頻錯誤特徵")

    # 元資料
    task: str = Field(..., description="分析任務")
    timestamp: str = Field(..., description="分析時間戳記")
    processing_time_ms: float = Field(..., description="處理時間（毫秒）")


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


class ModelStatus(BaseModel):
    """模型狀態"""

    models_loaded: bool = Field(..., description="模型是否已載入")
    device: str = Field(..., description="計算設備")
    warmup_complete: bool = Field(..., description="預熱是否完成")
    total_predictions: int = Field(..., description="總預測數")
    average_processing_time: float = Field(..., description="平均處理時間（秒）")


class HealthResponse(BaseModel):
    """健康檢查回應"""

    status: str
    timestamp: str
    version: str
    uptime_seconds: float
    model_status: Optional[ModelStatus] = None


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


async def analyze_text_content(
    text: str, context: Optional[str] = None
) -> Dict[str, Any]:
    """
    使用訓練好的模型進行文本分析
    """
    global model_loader, model_metrics

    if model_loader is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="模型未載入，請稍後再試",
        )

    start_time = time.time()
    model_metrics["total_predictions"] += 1

    try:
        # Get detector from model loader
        detector = model_loader.detector
        if detector is None:
            detector = model_loader.load_models()

        # Perform analysis
        result = detector.analyze(text)

        # Track successful prediction
        model_metrics["successful_predictions"] += 1
        processing_time = time.time() - start_time
        model_metrics["total_processing_time"] += processing_time

        # Log prediction (privacy compliant - no text content)
        text_hash = generate_text_hash(text)
        logger.info(
            f"Prediction success - Hash: {text_hash}, "
            f"Processing time: {processing_time:.3f}s, "
            f"Toxicity: {result.get('toxicity', 'unknown')}, "
            f"Emotion: {result.get('emotion', 'unknown')}"
        )

        return result

    except Exception as e:
        model_metrics["failed_predictions"] += 1
        processing_time = time.time() - start_time
        model_metrics["total_processing_time"] += processing_time

        # Log error (privacy compliant)
        text_hash = generate_text_hash(text)
        logger.error(
            f"Prediction failed - Hash: {text_hash}, "
            f"Processing time: {processing_time:.3f}s, "
            f"Error: {str(e)}"
        )

        # Don't expose internal errors to API consumers
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="模型預測失敗，請稍後再試",
        )


# API 端點
@app.get("/healthz", response_model=HealthResponse)
async def health_check():
    """健康檢查端點，包含模型狀態"""
    current_time = time.time()
    uptime = current_time - start_time

    # Get model status
    model_status_data = None
    if model_loader:
        try:
            status_info = model_loader.get_model_status()
            avg_processing_time = model_metrics["total_processing_time"] / max(
                model_metrics["total_predictions"], 1
            )

            model_status_data = ModelStatus(
                models_loaded=status_info.get("models_loaded", False),
                device=status_info.get("device", "unknown"),
                warmup_complete=status_info.get("warmup_complete", False),
                total_predictions=model_metrics["total_predictions"],
                average_processing_time=round(avg_processing_time, 4),
            )
        except Exception as e:
            logger.warning(f"Failed to get model status: {e}")

    # Determine overall health status
    status = "healthy"
    if model_loader is None:
        status = "starting"
    elif model_status_data and not model_status_data.models_loaded:
        status = "degraded"
    elif model_metrics["failed_predictions"] > model_metrics["successful_predictions"]:
        status = "degraded"

    return HealthResponse(
        status=status,
        timestamp=datetime.now().isoformat(),
        version="1.0.0",
        uptime_seconds=round(uptime, 2),
        model_status=model_status_data,
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

    except HTTPException:
        # Re-raise HTTP exceptions
        raise
    except Exception as e:
        logger.error(
            f"分析錯誤 - Hash: {generate_text_hash(data.text)}, 錯誤: {str(e)}"
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="分析處理失敗，請稍後再試",
        )


# 新增 metrics 端點
@app.get("/metrics")
async def get_metrics():
    """取得 API 效能指標"""
    avg_processing_time = model_metrics["total_processing_time"] / max(
        model_metrics["total_predictions"], 1
    )

    success_rate = model_metrics["successful_predictions"] / max(
        model_metrics["total_predictions"], 1
    )

    metrics = {
        "total_predictions": model_metrics["total_predictions"],
        "successful_predictions": model_metrics["successful_predictions"],
        "failed_predictions": model_metrics["failed_predictions"],
        "success_rate": round(success_rate, 4),
        "average_processing_time_ms": round(avg_processing_time * 1000, 2),
        "total_processing_time_s": round(model_metrics["total_processing_time"], 2),
        "startup_time_s": model_metrics.get("startup_time"),
        "cache_hits": model_metrics["cache_hits"],
    }

    # Add model-specific metrics if available
    if model_loader:
        try:
            model_status = model_loader.get_model_status()
            metrics.update(
                {
                    "model_device": model_status.get("device"),
                    "models_loaded": model_status.get("models_loaded"),
                    "warmup_complete": model_status.get("warmup_complete"),
                    "gpu_memory": model_status.get("gpu_memory"),
                }
            )
        except Exception as e:
            logger.warning(f"Failed to get detailed metrics: {e}")

    return metrics


# 新增模型狀態端點
@app.get("/model-info")
async def get_model_info():
    """取得模型詳細資訊"""
    if model_loader is None:
        return {"status": "not_initialized"}

    try:
        return model_loader.get_model_status()
    except Exception as e:
        logger.error(f"Failed to get model info: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="無法取得模型資訊"
        )


# SHAP解釋端點
@app.post("/explain/shap", response_model=SHAPExplanationResponse)
@limiter.limit("10/minute")  # SHAP計算較慢，限制更嚴格
async def explain_with_shap(request: Request, data: SHAPExplanationRequest):
    """
    SHAP可解釋性分析端點

    使用SHAP方法解釋模型預測，提供token級別的貢獻度分析
    """
    start_time_ms = time.time() * 1000

    try:
        # 初始化SHAP解釋器
        if model_loader is None or model_loader.detector is None:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="模型未載入，請稍後再試"
            )

        from cyberpuppy.explain.shap_explainer import SHAPExplainer, SHAPVisualizer
        import base64
        import io

        # 創建SHAP解釋器
        detector = model_loader.detector
        device = getattr(detector, 'device', torch.device('cpu'))
        shap_explainer = SHAPExplainer(detector, device)

        # 生成文本雜湊
        text_hash = generate_text_hash(data.text)

        logger.info(
            f"SHAP解釋請求 - Hash: {text_hash}, 任務: {data.task}, "
            f"可視化: {data.visualization_type}, IP: {get_remote_address(request)}"
        )

        # 執行SHAP解釋
        shap_result = shap_explainer.explain_text(data.text, max_evals=data.max_evals)

        # 獲取任務相關的結果
        prediction = getattr(shap_result, f"{data.task}_pred")
        confidence = getattr(shap_result, f"{data.task}_prob")
        base_value = getattr(shap_result, f"{data.task}_base_value")
        shap_values = getattr(shap_result, f"{data.task}_shap_values")

        # 構建Token列表
        tokens = []
        for i, token in enumerate(shap_result.tokens):
            if token not in ["[CLS]", "[SEP]", "[PAD]"] and i < len(shap_values):
                clean_token = token.replace("##", "")
                tokens.append(SHAPToken(
                    token=clean_token,
                    shap_value=float(shap_values[i]),
                    position=i
                ))

        # 生成可視化
        visualization_base64 = None
        if data.visualization_type != "none":
            try:
                visualizer = SHAPVisualizer(shap_explainer)

                # 創建臨時文件保存圖片
                import tempfile
                with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp_file:
                    if data.visualization_type == "waterfall":
                        fig = visualizer.create_waterfall_plot(
                            shap_result, task=data.task, save_path=tmp_file.name
                        )
                    elif data.visualization_type == "text":
                        fig = visualizer.create_text_plot(
                            shap_result, task=data.task, save_path=tmp_file.name
                        )
                    elif data.visualization_type == "force":
                        visualizer.create_force_plot(
                            shap_result, task=data.task, save_path=tmp_file.name
                        )

                    # 讀取圖片並轉換為Base64
                    with open(tmp_file.name, 'rb') as img_file:
                        img_data = img_file.read()
                        visualization_base64 = base64.b64encode(img_data).decode('utf-8')

                    # 清理臨時文件
                    import os
                    os.unlink(tmp_file.name)

            except Exception as viz_error:
                logger.warning(f"可視化生成失敗: {viz_error}")
                # 可視化失敗不影響主要功能

        # 計算處理時間
        processing_time = (time.time() * 1000) - start_time_ms

        # 構建回應
        response = SHAPExplanationResponse(
            prediction=str(prediction),
            confidence=confidence,
            base_value=base_value,
            tokens=tokens,
            feature_importance=shap_result.feature_importance[data.task],
            visualization_base64=visualization_base64,
            task=data.task,
            method="SHAP",
            text_hash=text_hash,
            timestamp=datetime.now().isoformat(),
            processing_time_ms=round(processing_time, 2)
        )

        logger.info(
            f"SHAP解釋完成 - Hash: {text_hash}, 任務: {data.task}, "
            f"預測: {prediction}, 處理時間: {processing_time:.2f}ms"
        )

        return response

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"SHAP解釋錯誤 - Hash: {generate_text_hash(data.text)}, 錯誤: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="SHAP解釋處理失敗，請稍後再試"
        )


@app.post("/explain/misclassification", response_model=MisclassificationAnalysisResponse)
@limiter.limit("5/minute")  # 批量分析限制更嚴格
async def analyze_misclassification(request: Request, data: MisclassificationAnalysisRequest):
    """
    誤判分析端點

    分析多個文本的誤判模式，識別模型的錯誤傾向
    """
    start_time_ms = time.time() * 1000

    try:
        # 檢查模型狀態
        if model_loader is None or model_loader.detector is None:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="模型未載入，請稍後再試"
            )

        from cyberpuppy.explain.shap_explainer import SHAPExplainer, MisclassificationAnalyzer

        # 創建分析器
        detector = model_loader.detector
        device = getattr(detector, 'device', torch.device('cpu'))
        shap_explainer = SHAPExplainer(detector, device)
        analyzer = MisclassificationAnalyzer(shap_explainer)

        logger.info(
            f"誤判分析請求 - 文本數: {len(data.texts)}, 任務: {data.task}, "
            f"IP: {get_remote_address(request)}"
        )

        # 執行誤判分析
        analysis_result = analyzer.analyze_misclassifications(
            data.texts, data.true_labels, data.task
        )

        # 提取統計數據
        error_analysis = analysis_result.get("error_analysis", {})

        # 計算處理時間
        processing_time = (time.time() * 1000) - start_time_ms

        # 構建回應
        response = MisclassificationAnalysisResponse(
            misclassification_rate=analysis_result["misclassification_rate"],
            total_cases=len(data.texts),
            misclassified_count=len(analysis_result["misclassified_cases"]),
            correct_count=len(analysis_result["correct_cases"]),
            error_statistics={
                "avg_misclassified_confidence": error_analysis.get("avg_misclassified_confidence", 0),
                "avg_correct_confidence": error_analysis.get("avg_correct_confidence", 0),
                "confidence_gap": error_analysis.get("confidence_gap", 0),
                "avg_misclassified_importance": error_analysis.get("avg_misclassified_importance", 0),
                "avg_correct_importance": error_analysis.get("avg_correct_importance", 0)
            },
            top_error_features=error_analysis.get("top_error_features", [])[:10],
            task=data.task,
            timestamp=datetime.now().isoformat(),
            processing_time_ms=round(processing_time, 2)
        )

        logger.info(
            f"誤判分析完成 - 文本數: {len(data.texts)}, 任務: {data.task}, "
            f"誤判率: {response.misclassification_rate:.2%}, "
            f"處理時間: {processing_time:.2f}ms"
        )

        return response

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"誤判分析錯誤 - 錯誤: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="誤判分析處理失敗，請稍後再試"
        )


# 管理端點 - 清理快取
@app.post("/admin/clear-cache")
async def clear_model_cache():
    """清理模型快取（管理端點）"""
    if model_loader is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail="模型載入器未初始化"
        )

    try:
        model_loader.clear_cache()
        logger.info("Model cache cleared via admin endpoint")
        return {"message": "模型快取已清理", "timestamp": datetime.now().isoformat()}
    except Exception as e:
        logger.error(f"Failed to clear cache: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="快取清理失敗"
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
        "metrics": "/metrics",
        "model_info": "/model-info",
        "features": [
            "Chinese toxicity detection",
            "Bullying behavior analysis",
            "Emotion classification",
            "Role identification",
            "Model explainability (IG/SHAP)",
            "SHAP visualization",
            "Misclassification analysis",
            "Privacy-compliant logging",
        ],
        "endpoints": {
            "analyze": "/analyze",
            "shap_explanation": "/explain/shap",
            "misclassification_analysis": "/explain/misclassification",
            "health_check": "/healthz",
            "metrics": "/metrics",
            "model_info": "/model-info"
        },
    }


if __name__ == "__main__":
    # 開發環境啟動配置
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True, log_level="info")
