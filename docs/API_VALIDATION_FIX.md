# API Data Validation Fix

## Problem

The API was returning 500 errors with the message: "value is not a valid float (type=type_error.float)" when processing requests to the `/analyze` endpoint.

**Root Cause:** The `ExplanationData` Pydantic model expected `important_words` to be `List[Dict[str, float]]` (all values as floats), but the actual data structure contained mixed types:
- `word` field: string (the actual word)
- `importance` field: float (the importance score)

## Solution

### 1. Updated Pydantic Models (`api/app.py`)

**Before:**
```python
class ExplanationData(BaseModel):
    important_words: List[Dict[str, float]] = Field(..., description="重要詞彙與權重")
    method: str = Field(..., description="解釋方法 (IG/SHAP)")
    confidence: float = Field(..., ge=0, le=1, description="預測信心度")
```

**After:**
```python
class ImportantWord(BaseModel):
    """重要詞彙與權重"""
    word: str = Field(..., description="詞彙")
    importance: float = Field(..., ge=0, le=1, description="重要性權重")

class ExplanationData(BaseModel):
    """可解釋性資料"""
    important_words: List[ImportantWord] = Field(..., description="重要詞彙與權重")
    method: str = Field(..., description="解釋方法 (IG/SHAP)")
    confidence: float = Field(..., ge=0, le=1, description="預測信心度")
```

### 2. Maintained Compatibility

The mock data generator in `api/model_loader_simple.py` already generated the correct structure:
```python
important_words.append({
    "word": word,        # string
    "importance": importance  # float
})
```

No changes were needed to the data generation logic.

## Validation

### Created Test Files

1. **`tests/test_api_models.py`** - Unit tests for Pydantic model validation
2. **`tests/test_api_integration.py`** - Integration tests using FastAPI TestClient
3. **`tests/test_api_live.py`** - Live server testing script
4. **`tests/test_api_manual.md`** - Manual testing instructions

### Test Results

```bash
python tests/test_api_models.py
# PASS: ImportantWord model test passed
# PASS: ExplanationData model test passed
# PASS: ExplanationData with model loader test passed
# PASS: Full AnalyzeResponse test passed

python tests/test_api_integration.py
# PASS: Health endpoint test passed
# PASS: Analyze endpoint test passed
# Sample API response structure:
# - toxicity: toxic
# - bullying: harassment
# - emotion: neg
# - important_words count: 1
# - first important word: [UNICODE] (importance: 0.8)
```

### Expected API Response

```json
{
  "toxicity": "toxic",
  "bullying": "harassment",
  "role": "none",
  "emotion": "neg",
  "emotion_strength": 3,
  "scores": { ... },
  "explanations": {
    "important_words": [
      {"word": "笨蛋", "importance": 0.8},
      {"word": "词汇", "importance": 0.6}
    ],
    "method": "keyword_based_mock",
    "confidence": 0.75
  },
  "text_hash": "...",
  "timestamp": "...",
  "processing_time_ms": 123.45
}
```

## Key Improvements

1. ✅ **Fixed validation error** - No more "value is not a valid float" errors
2. ✅ **Better type safety** - Explicit `ImportantWord` model ensures correct structure
3. ✅ **Maintained backwards compatibility** - Data format unchanged for clients
4. ✅ **Added validation constraints** - `importance` values validated to be 0.0-1.0 range
5. ✅ **Comprehensive testing** - Unit, integration, and manual test coverage
6. ✅ **Clear documentation** - Field descriptions and validation rules documented

## Files Modified

- `api/app.py` - Updated Pydantic models
- `tests/test_api_models.py` - New unit tests (created)
- `tests/test_api_integration.py` - New integration tests (created)
- `tests/test_api_live.py` - New live testing script (created)
- `tests/test_api_manual.md` - New manual testing guide (created)

## Impact

- ✅ API now returns 200 OK responses instead of 500 errors
- ✅ Data validation is more precise and type-safe
- ✅ Better error messages for invalid data
- ✅ Easier to extend with additional word-level features in the future