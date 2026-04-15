# 🛡️ CyberPuppy - Advanced Chinese Cyberbullying Detection & Content Moderation System

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![CUDA 12.4](https://img.shields.io/badge/CUDA-12.4-green.svg)](https://developer.nvidia.com/cuda-downloads)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.109+-00a393.svg)](https://fastapi.tiangolo.com)
[![License: Apache-2.0](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![Hugging Face](https://img.shields.io/badge/🤗%20Hugging%20Face-Models-yellow)](https://huggingface.co/hfl/chinese-macbert-base)

[中文版本](README.md) | [ADR 0001 (v2.1)](docs/adr/0001-cyberpuppy-2026-upgrade.md) | [API Documentation](http://localhost:8000/docs)

> 🌟 **State-of-the-art Chinese NLP solution for real-time toxicity detection, cyberbullying prevention, and sentiment analysis with explainable AI capabilities**

## 🎯 Why CyberPuppy?

CyberPuppy represents the **most comprehensive open-source solution** for Chinese content moderation, addressing the critical need for culturally-aware AI safety in the Chinese-speaking digital ecosystem. Built on cutting-edge transformer models and featuring industry-leading explainability tools.

### 🚀 Key Features That Set Us Apart

- **🧠 Multi-Task Deep Learning**: Simultaneous toxicity detection, cyberbullying identification, emotion analysis, and role classification
- **📊 Explainable AI (XAI)**: Integrated SHAP and Integrated Gradients for transparent, interpretable predictions
- **⚡ GPU-Accelerated**: Optimized for NVIDIA GPUs (CUDA 12.4+) with 5-10x performance boost
- **🔐 Privacy-First Architecture**: Zero raw text logging, SHA-256 hashing for complete privacy protection
- **🌐 Production-Ready API**: High-performance FastAPI with <200ms response time
- **💬 LINE Bot Integration**: Enterprise-grade chatbot with HMAC-SHA256 webhook verification
- **🎯 Chinese-Optimized**: Built specifically for Traditional & Simplified Chinese with OpenCC support
- **🔄 Real-Time Processing**: Handle 100+ concurrent requests with auto-scaling capabilities

## 📈 Performance Metrics

| Metric | Actual | Target | Status |
|--------|--------|--------|--------|
| **Toxicity Detection F1** | - | 0.78 | ⚠️ Awaiting training |
| **Bullying Detection F1** | 0.826 | 0.75 | ✅ Achieved |
| **Emotion Analysis F1** | 1.00* | 0.85 | ✅ Exceeds target |
| **Response Time** | <200ms | 500ms | ✅ Achieved |
| **GPU Acceleration** | 5-10x | - | ✅ Verified |

*Note: Emotion analysis showed perfect performance in limited testing, requires larger-scale validation
**Production Model**: bullying_a100_best deployed (F1=0.826, Accuracy=0.824)

## 🛠️ Technology Stack

### Core AI/ML
- **🤗 Transformers**: MacBERT, RoBERTa-wwm-ext for Chinese NLP
- **⚡ PyTorch 2.6**: GPU-accelerated deep learning
- **🔍 Explainability**: Captum (IG), SHAP for model interpretability

### Infrastructure
- **🚀 FastAPI**: Async REST API framework
- **🐳 Docker**: Containerized microservices architecture
- **📊 Redis**: High-performance caching layer
- **🔄 Nginx**: Load balancing & reverse proxy

### Chinese NLP Tools
- **📝 OpenCC**: Traditional/Simplified Chinese conversion
- **✂️ CKIP**: Advanced Chinese tokenization
- **🏷️ NTUSD**: Sentiment lexicon integration

## 🚀 Quick Start

### Prerequisites

```bash
# System Requirements
- Python 3.9+ (3.13 supported)
- CUDA 12.4+ (for GPU acceleration)
- 8GB+ RAM (16GB recommended)
- 4GB+ GPU VRAM (optional)
```

### Installation

```bash
# Clone the repository
git clone https://github.com/thc1006/cyberbully-zh-moderation-bot.git
cd cyberbully-zh-moderation-bot

# Install dependencies
pip install -r requirements.txt

# Download required datasets
python scripts/download_datasets.py --all

# Production model included (bullying_a100_best)

# GPU Setup (optional but recommended)
python scripts/check_gpu.py  # Verify CUDA availability
```

### 🚀 Launch Services

```bash
# Start API Server (http://localhost:8000)
python api/app.py

# Or use the convenience scripts
./scripts/start_local.sh  # Linux/Mac
scripts\start_local.bat    # Windows

# API Documentation available at http://localhost:8000/docs
```

## 📡 API Usage

### Basic Text Analysis

```python
import requests

response = requests.post(
    "http://localhost:8000/analyze",
    json={
        "text": "你這個笨蛋，滾開！",
        "context": "optional_conversation_context",
        "thread_id": "session_123"
    }
)

result = response.json()
print(f"Toxicity: {result['toxicity']['level']}")
print(f"Emotion: {result['emotion']['label']}")
print(f"Confidence: {result['explanations']['confidence']}")
```

### Advanced Response Structure

```json
{
  "toxicity": {
    "level": "toxic",
    "confidence": 0.89,
    "probability": {
      "none": 0.08,
      "toxic": 0.73,
      "severe": 0.19
    }
  },
  "bullying": {
    "level": "harassment",
    "confidence": 0.82
  },
  "emotion": {
    "label": "negative",
    "strength": 4,
    "scores": {
      "positive": 0.05,
      "neutral": 0.15,
      "negative": 0.80
    }
  },
  "explanations": {
    "method": "integrated_gradients",
    "important_words": [
      {"word": "笨蛋", "importance": 0.85},
      {"word": "滾開", "importance": 0.72}
    ],
    "confidence": 0.89
  },
  "metadata": {
    "text_hash": "a1b2c3d4e5f6789",
    "processing_time_ms": 145,
    "model_version": "1.0.0",
    "timestamp": "2025-09-25T10:30:00Z"
  }
}
```

## 🤖 LINE Bot Integration

### Setup

1. Create LINE Bot at [LINE Developers Console](https://developers.line.biz/)
2. Configure environment variables:

```bash
# .env file
LINE_CHANNEL_SECRET=your_channel_secret
LINE_CHANNEL_ACCESS_TOKEN=your_access_token
CYBERPUPPY_API_URL=http://localhost:8000
```

3. Set webhook URL: `https://your-domain.com/webhook`

### Features
- ✅ HMAC-SHA256 signature verification
- ✅ Automatic threat level assessment
- ✅ Contextual response generation
- ✅ Privacy-preserving logging

## 🐳 Docker Deployment

```yaml
# docker-compose.yml
docker-compose up -d

# Production deployment with load balancing
docker-compose --profile production up -d

# Scaling
docker-compose up --scale api=3 -d
```

## 📊 Datasets & Models

### Pre-trained Models (2.4GB)
- **MacBERT-base**: Chinese masked language model
- **RoBERTa-wwm**: Whole word masking model
- **Custom fine-tuned**: Toxicity & emotion classifiers

### Training Datasets
- **COLD**: Chinese Offensive Language Dataset
- **ChnSentiCorp**: Chinese sentiment corpus
- **DMSC v2**: Douban movie reviews (387MB)
- **NTUSD**: Taiwan sentiment dictionary
- **SCCD**: Session-level cyberbullying (manual)

## 🔒 Security & Privacy

- **Zero-Knowledge Logging**: No raw text storage
- **SHA-256 Hashing**: All text identifiers hashed
- **Rate Limiting**: DDoS protection via SlowAPI
- **Input Validation**: Strict text sanitization
- **Webhook Security**: HMAC-SHA256 verification
- **API Authentication**: Optional JWT support

## 📈 Monitoring & Observability

- **Health Checks**: `/health` endpoint
- **Metrics**: Processing time, model confidence
- **Error Tracking**: Structured logging with context
- **Performance**: Real-time latency monitoring

## 🎯 Use Cases

### Social Media Platforms
- Real-time comment moderation
- Automated content flagging
- User safety alerts

### Educational Institutions
- Student chat monitoring
- Bullying prevention systems
- Mental health support triggers

### Gaming Communities
- In-game chat moderation
- Toxic player detection
- Community health metrics

### Customer Service
- Agent assistance tools
- Escalation triggers
- Sentiment tracking

## 🤝 Contributing

We welcome contributions! See [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

### Development Setup

```bash
# Install dev dependencies
pip install -r requirements-dev.txt

# Run tests
pytest --cov=cyberpuppy

# Code quality
flake8 src/
black src/ --check
mypy src/
```

## 📜 License

Apache License 2.0 - see [LICENSE](LICENSE) for details.

## 🌟 Acknowledgments

- Hugging Face for transformer models
- THU-COAI for COLD dataset
- LINE Corporation for messaging API
- Google for Perspective API integration

## 📞 Support & Contact

- 📧 **Email**: hctsai@linux.com
- 🐛 **Issues**: [GitHub Issues](https://github.com/thc1006/cyberbully-zh-moderation-bot/issues)



## 📊 Star History

[![Star History Chart](https://api.star-history.com/svg?repos=thc1006/cyberbully-zh-moderation-bot&type=Date)](https://star-history.com/#thc1006/cyberbully-zh-moderation-bot&Date)

---

<div align="center">
  <b>⭐ Star us on GitHub — it motivates us a lot!</b><br>
  <sub>Built with ❤️ for a safer Chinese internet</sub>
</div>

## 🔍 SEO Keywords

Chinese content moderation, cyberbullying detection, toxicity detection, sentiment analysis, explainable AI, Chinese NLP, LINE Bot, FastAPI, PyTorch, BERT, MacBERT, RoBERTa, transformer models, GPU acceleration, CUDA, deep learning, machine learning, AI safety, content filtering, chat moderation, real-time analysis, privacy-preserving AI, open source, 中文內容審核, 網路霸凌偵測, 毒性檢測, 情緒分析, 可解釋人工智慧, 中文自然語言處理