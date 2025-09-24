# Manual API Testing

## Start the API Server

```bash
cd api
python app.py
```

The server should start on http://localhost:8000

## Test Health Endpoint

```bash
curl http://localhost:8000/healthz
```

Expected response:
```json
{
  "status": "healthy",
  "timestamp": "2025-09-24T...",
  "version": "1.0.0",
  "uptime_seconds": 123.45,
  "model_status": {
    "models_loaded": true,
    "device": "cpu",
    "warmup_complete": true,
    ...
  }
}
```

## Test Analyze Endpoint

```bash
curl -X POST http://localhost:8000/analyze \
  -H "Content-Type: application/json" \
  -d '{
    "text": "这是一个测试文本，包含一些笨蛋词汇",
    "context": "测试上下文",
    "thread_id": "test_123"
  }'
```

Expected response structure:
```json
{
  "toxicity": "toxic",
  "bullying": "harassment",
  "role": "none",
  "emotion": "neg",
  "emotion_strength": 3,
  "scores": {
    "toxicity": {"none": 0.2, "toxic": 0.7, "severe": 0.1},
    "bullying": {"none": 0.2, "harassment": 0.7, "threat": 0.1},
    ...
  },
  "explanations": {
    "important_words": [
      {"word": "笨蛋", "importance": 0.8}
    ],
    "method": "keyword_based_mock",
    "confidence": 0.75
  },
  "text_hash": "...",
  "timestamp": "...",
  "processing_time_ms": 123.45
}
```

## Key Validation Points

The fix ensures:

1. ✅ `important_words` is an array of objects
2. ✅ Each object has `word` (string) and `importance` (float) fields
3. ✅ `importance` values are between 0.0 and 1.0
4. ✅ No more "value is not a valid float" errors
5. ✅ API returns 200 OK instead of 500 errors

## Troubleshooting

If you get 503 Service Unavailable:
- Check that models are loading correctly in the logs
- Verify the model_loader_simple.py is working

If you get 500 Internal Server Error:
- Check the server logs for specific error messages
- Verify the request JSON structure matches the expected format