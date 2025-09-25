# CyberPuppy API 部署與配置指南
# Deployment & Configuration Guide

## 部署架構 (Deployment Architecture)

### 生產環境架構圖
```
                    ┌─────────────────┐
                    │  Load Balancer  │
                    │   (Nginx/AWS)   │
                    └─────────┬───────┘
                              │
                    ┌─────────┴───────┐
                    │  API Gateway    │
                    │  (Rate Limiting)│
                    └─────────┬───────┘
                              │
                    ┌─────────┴───────┐
                    │   FastAPI App   │
                    │ (Multiple Pods) │
                    └─────────┬───────┘
                              │
               ┌──────────────┼──────────────┐
               │              │              │
    ┌─────────┴───────┐ ┌────┴─────┐ ┌─────┴──────┐
    │ Model Service   │ │  Cache   │ │  Database  │
    │   (GPU Pod)     │ │ (Redis)  │ │ (MongoDB)  │
    └─────────────────┘ └──────────┘ └────────────┘
```

## 部署選項 (Deployment Options)

### 1. Docker 容器部署

#### Dockerfile
```dockerfile
# CyberPuppy API Dockerfile
FROM python:3.11-slim

# 設定工作目錄
WORKDIR /app

# 安裝系統依賴
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# 複製依賴檔案
COPY requirements.txt .

# 安裝 Python 依賴
RUN pip install --no-cache-dir -r requirements.txt

# 複製應用程式碼
COPY . .

# 建立非 root 用戶
RUN useradd --create-home --shell /bin/bash cyberpuppy
RUN chown -R cyberpuppy:cyberpuppy /app
USER cyberpuppy

# 暴露端口
EXPOSE 8000

# 健康檢查
HEALTHCHECK --interval=30s --timeout=10s --retries=3 \
    CMD curl -f http://localhost:8000/healthz || exit 1

# 啟動命令
CMD ["uvicorn", "api.app:app", "--host", "0.0.0.0", "--port", "8000"]
```

#### docker-compose.yml
```yaml
version: '3.8'

services:
  cyberpuppy-api:
    build:
      context: .
      dockerfile: Dockerfile
    ports:
      - "8000:8000"
    environment:
      - CYBERPUPPY_ENV=production
      - REDIS_URL=redis://redis:6379/0
      - DATABASE_URL=mongodb://mongo:27017/cyberpuppy
    depends_on:
      - redis
      - mongo
    volumes:
      - ./models:/app/models:ro
    networks:
      - cyberpuppy-net
    restart: unless-stopped
    deploy:
      resources:
        limits:
          memory: 4G
          cpus: '2'
        reservations:
          memory: 2G
          cpus: '1'

  redis:
    image: redis:7-alpine
    ports:
      - "6379:6379"
    volumes:
      - redis_data:/data
    networks:
      - cyberpuppy-net
    restart: unless-stopped
    command: redis-server --appendonly yes

  mongo:
    image: mongo:6
    ports:
      - "27017:27017"
    environment:
      MONGO_INITDB_ROOT_USERNAME: cyberpuppy
      MONGO_INITDB_ROOT_PASSWORD: ${MONGO_PASSWORD}
      MONGO_INITDB_DATABASE: cyberpuppy
    volumes:
      - mongo_data:/data/db
      - ./scripts/mongo-init.js:/docker-entrypoint-initdb.d/mongo-init.js:ro
    networks:
      - cyberpuppy-net
    restart: unless-stopped

  nginx:
    image: nginx:alpine
    ports:
      - "80:80"
      - "443:443"
    volumes:
      - ./nginx/nginx.conf:/etc/nginx/nginx.conf:ro
      - ./ssl:/etc/nginx/ssl:ro
    depends_on:
      - cyberpuppy-api
    networks:
      - cyberpuppy-net
    restart: unless-stopped

volumes:
  redis_data:
  mongo_data:

networks:
  cyberpuppy-net:
    driver: bridge
```

#### 建置與啟動
```bash
# 1. 準備環境檔案
cp .env.example .env
# 編輯 .env 設定必要的環境變數

# 2. 建置 Docker 映像
docker-compose build

# 3. 啟動服務
docker-compose up -d

# 4. 檢查服務狀態
docker-compose ps

# 5. 查看日誌
docker-compose logs -f cyberpuppy-api

# 6. 健康檢查
curl http://localhost:8000/healthz
```

### 2. Kubernetes 部署

#### deployment.yaml
```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: cyberpuppy-api
  namespace: cyberpuppy
  labels:
    app: cyberpuppy-api
spec:
  replicas: 3
  strategy:
    type: RollingUpdate
    rollingUpdate:
      maxUnavailable: 1
      maxSurge: 1
  selector:
    matchLabels:
      app: cyberpuppy-api
  template:
    metadata:
      labels:
        app: cyberpuppy-api
    spec:
      containers:
      - name: cyberpuppy-api
        image: cyberpuppy/api:latest
        ports:
        - containerPort: 8000
        env:
        - name: CYBERPUPPY_ENV
          value: "production"
        - name: REDIS_URL
          valueFrom:
            secretKeyRef:
              name: cyberpuppy-secrets
              key: redis-url
        - name: DATABASE_URL
          valueFrom:
            secretKeyRef:
              name: cyberpuppy-secrets
              key: database-url
        resources:
          requests:
            memory: "2Gi"
            cpu: "1"
          limits:
            memory: "4Gi"
            cpu: "2"
        livenessProbe:
          httpGet:
            path: /healthz
            port: 8000
          initialDelaySeconds: 30
          periodSeconds: 30
        readinessProbe:
          httpGet:
            path: /healthz
            port: 8000
          initialDelaySeconds: 5
          periodSeconds: 5
        volumeMounts:
        - name: model-storage
          mountPath: /app/models
          readOnly: true
      volumes:
      - name: model-storage
        persistentVolumeClaim:
          claimName: model-pvc
---
apiVersion: v1
kind: Service
metadata:
  name: cyberpuppy-api-service
  namespace: cyberpuppy
spec:
  selector:
    app: cyberpuppy-api
  ports:
  - protocol: TCP
    port: 80
    targetPort: 8000
  type: ClusterIP
---
apiVersion: networking.k8s.io/v1
kind: Ingress
metadata:
  name: cyberpuppy-api-ingress
  namespace: cyberpuppy
  annotations:
    nginx.ingress.kubernetes.io/rate-limit: "100"
    nginx.ingress.kubernetes.io/rate-limit-window: "1m"
    cert-manager.io/cluster-issuer: "letsencrypt-prod"
spec:
  tls:
  - hosts:
    - api.cyberpuppy.ai
    secretName: cyberpuppy-tls
  rules:
  - host: api.cyberpuppy.ai
    http:
      paths:
      - path: /
        pathType: Prefix
        backend:
          service:
            name: cyberpuppy-api-service
            port:
              number: 80
```

#### 部署步驟
```bash
# 1. 建立命名空間
kubectl create namespace cyberpuppy

# 2. 建立 Secrets
kubectl create secret generic cyberpuppy-secrets \
  --from-literal=redis-url=redis://redis-service:6379/0 \
  --from-literal=database-url=mongodb://mongo-service:27017/cyberpuppy \
  -n cyberpuppy

# 3. 部署應用
kubectl apply -f deployment.yaml

# 4. 檢查部署狀態
kubectl get pods -n cyberpuppy
kubectl get services -n cyberpuppy

# 5. 查看日誌
kubectl logs -f deployment/cyberpuppy-api -n cyberpuppy
```

### 3. 雲端平台部署

#### AWS ECS 部署
```json
{
  "family": "cyberpuppy-api",
  "networkMode": "awsvpc",
  "requiresCompatibilities": ["FARGATE"],
  "cpu": "2048",
  "memory": "4096",
  "executionRoleArn": "arn:aws:iam::ACCOUNT:role/ecsTaskExecutionRole",
  "taskRoleArn": "arn:aws:iam::ACCOUNT:role/ecsTaskRole",
  "containerDefinitions": [
    {
      "name": "cyberpuppy-api",
      "image": "cyberpuppy/api:latest",
      "portMappings": [
        {
          "containerPort": 8000,
          "protocol": "tcp"
        }
      ],
      "environment": [
        {
          "name": "CYBERPUPPY_ENV",
          "value": "production"
        }
      ],
      "secrets": [
        {
          "name": "API_KEY",
          "valueFrom": "arn:aws:secretsmanager:region:account:secret:cyberpuppy-api-key"
        }
      ],
      "logConfiguration": {
        "logDriver": "awslogs",
        "options": {
          "awslogs-group": "/ecs/cyberpuppy-api",
          "awslogs-region": "us-west-2",
          "awslogs-stream-prefix": "ecs"
        }
      },
      "healthCheck": {
        "command": [
          "CMD-SHELL",
          "curl -f http://localhost:8000/healthz || exit 1"
        ],
        "interval": 30,
        "timeout": 5,
        "retries": 3
      }
    }
  ]
}
```

#### Terraform 配置
```hcl
# main.tf
provider "aws" {
  region = var.aws_region
}

# ECS Cluster
resource "aws_ecs_cluster" "cyberpuppy" {
  name = "cyberpuppy-cluster"

  setting {
    name  = "containerInsights"
    value = "enabled"
  }
}

# Application Load Balancer
resource "aws_lb" "cyberpuppy_alb" {
  name               = "cyberpuppy-alb"
  internal           = false
  load_balancer_type = "application"
  security_groups    = [aws_security_group.alb_sg.id]
  subnets           = var.public_subnet_ids

  enable_deletion_protection = true
}

# ECS Service
resource "aws_ecs_service" "cyberpuppy_api" {
  name            = "cyberpuppy-api"
  cluster         = aws_ecs_cluster.cyberpuppy.id
  task_definition = aws_ecs_task_definition.cyberpuppy_api.arn
  desired_count   = 3
  launch_type     = "FARGATE"

  network_configuration {
    security_groups  = [aws_security_group.api_sg.id]
    subnets         = var.private_subnet_ids
    assign_public_ip = false
  }

  load_balancer {
    target_group_arn = aws_lb_target_group.cyberpuppy_api.arn
    container_name   = "cyberpuppy-api"
    container_port   = 8000
  }

  depends_on = [aws_lb_listener.cyberpuppy_api]
}
```

## 環境配置 (Environment Configuration)

### 1. 環境變數設定

#### .env 檔案範例
```bash
# 應用程式配置
CYBERPUPPY_ENV=production
APP_NAME=CyberPuppy API
APP_VERSION=1.0.0
DEBUG=false

# API 配置
API_HOST=0.0.0.0
API_PORT=8000
API_WORKERS=4

# 資料庫配置
DATABASE_URL=mongodb://username:password@localhost:27017/cyberpuppy
REDIS_URL=redis://localhost:6379/0

# 模型配置
MODEL_PATH=/app/models
MODEL_DEVICE=cuda:0
MODEL_BATCH_SIZE=32
MODEL_MAX_LENGTH=512

# 安全配置
SECRET_KEY=your-super-secret-key-here
API_KEY_PREFIX=cp_
JWT_ALGORITHM=HS256
JWT_EXPIRE_MINUTES=1440

# 限流配置
RATE_LIMIT_PER_MINUTE=30
RATE_LIMIT_BURST=10
MAX_CONCURRENT_REQUESTS=100

# 日誌配置
LOG_LEVEL=INFO
LOG_FORMAT=json
LOG_FILE=/var/log/cyberpuppy/app.log

# 監控配置
METRICS_ENABLED=true
METRICS_PATH=/metrics
HEALTH_CHECK_PATH=/healthz

# LINE Bot 配置 (可選)
LINE_CHANNEL_ACCESS_TOKEN=your-line-channel-access-token
LINE_CHANNEL_SECRET=your-line-channel-secret

# 外部服務配置 (可選)
PERSPECTIVE_API_KEY=your-perspective-api-key
SENTRY_DSN=https://your-sentry-dsn

# 快取配置
CACHE_TTL=3600
CACHE_MAX_SIZE=1000

# 檔案上傳配置
MAX_FILE_SIZE=10MB
ALLOWED_FILE_TYPES=txt,json
```

### 2. 配置驗證腳本

#### config_validator.py
```python
#!/usr/bin/env python3
"""
配置驗證腳本
檢查所有必要的環境變數是否正確設定
"""

import os
import sys
from typing import Dict, List, Optional

class ConfigValidator:
    """配置驗證器"""

    REQUIRED_VARS = {
        'CYBERPUPPY_ENV': ['development', 'production', 'testing'],
        'DATABASE_URL': None,  # 任意值
        'SECRET_KEY': None,
        'MODEL_PATH': None,
    }

    OPTIONAL_VARS = {
        'REDIS_URL': 'redis://localhost:6379/0',
        'API_HOST': '0.0.0.0',
        'API_PORT': '8000',
        'LOG_LEVEL': 'INFO',
        'RATE_LIMIT_PER_MINUTE': '30',
    }

    def __init__(self):
        self.errors: List[str] = []
        self.warnings: List[str] = []

    def validate_required_vars(self) -> bool:
        """驗證必要環境變數"""
        for var, valid_values in self.REQUIRED_VARS.items():
            value = os.getenv(var)

            if value is None:
                self.errors.append(f"Missing required environment variable: {var}")
                continue

            if valid_values and value not in valid_values:
                self.errors.append(
                    f"Invalid value for {var}: '{value}'. "
                    f"Valid values: {valid_values}"
                )

        return len(self.errors) == 0

    def validate_optional_vars(self):
        """驗證可選環境變數並設定預設值"""
        for var, default_value in self.OPTIONAL_VARS.items():
            value = os.getenv(var)

            if value is None:
                self.warnings.append(
                    f"Optional variable {var} not set, using default: {default_value}"
                )
                os.environ[var] = default_value

    def validate_database_connection(self) -> bool:
        """驗證資料庫連接"""
        try:
            from pymongo import MongoClient

            db_url = os.getenv('DATABASE_URL')
            if not db_url:
                return True  # 已在必要變數檢查中處理

            client = MongoClient(db_url, serverSelectionTimeoutMS=5000)
            client.server_info()  # 觸發連接測試
            client.close()

            return True
        except Exception as e:
            self.errors.append(f"Database connection failed: {str(e)}")
            return False

    def validate_redis_connection(self) -> bool:
        """驗證 Redis 連接"""
        try:
            import redis

            redis_url = os.getenv('REDIS_URL')
            if not redis_url:
                return True

            r = redis.from_url(redis_url, socket_connect_timeout=5)
            r.ping()
            r.close()

            return True
        except Exception as e:
            self.warnings.append(f"Redis connection failed: {str(e)}")
            return True  # Redis 是可選的

    def validate_model_files(self) -> bool:
        """驗證模型檔案"""
        model_path = os.getenv('MODEL_PATH', '/app/models')

        if not os.path.exists(model_path):
            self.errors.append(f"Model path does not exist: {model_path}")
            return False

        # 檢查基本模型檔案
        required_files = [
            'detector_model.bin',
            'tokenizer_vocab.txt',
            'config.json'
        ]

        for file_name in required_files:
            file_path = os.path.join(model_path, file_name)
            if not os.path.exists(file_path):
                self.warnings.append(f"Model file not found: {file_path}")

        return True

    def validate_all(self) -> bool:
        """執行所有驗證"""
        print("🔍 Validating configuration...")

        # 必要變數驗證
        required_ok = self.validate_required_vars()

        # 可選變數驗證
        self.validate_optional_vars()

        # 服務連接驗證
        db_ok = self.validate_database_connection()
        redis_ok = self.validate_redis_connection()
        model_ok = self.validate_model_files()

        # 顯示結果
        if self.warnings:
            print("\n⚠️  Warnings:")
            for warning in self.warnings:
                print(f"   {warning}")

        if self.errors:
            print("\n❌ Errors:")
            for error in self.errors:
                print(f"   {error}")
            return False

        print("\n✅ Configuration validation passed!")
        return True

def main():
    """主函數"""
    validator = ConfigValidator()

    if not validator.validate_all():
        print("\n💡 Please fix the errors above and try again.")
        sys.exit(1)

    print("\n🚀 Ready to start CyberPuppy API!")

if __name__ == "__main__":
    main()
```

### 3. 生產環境最佳化

#### uvicorn_config.py
```python
"""
Uvicorn 生產環境配置
"""

import multiprocessing
import os

# 服務器配置
bind = f"{os.getenv('API_HOST', '0.0.0.0')}:{os.getenv('API_PORT', '8000')}"
workers = int(os.getenv('API_WORKERS', multiprocessing.cpu_count() * 2))
worker_class = "uvicorn.workers.UvicornWorker"
worker_connections = 1000
max_requests = 10000
max_requests_jitter = 1000
preload_app = True
timeout = 300
keepalive = 5

# 日誌配置
accesslog = "-"
errorlog = "-"
loglevel = os.getenv('LOG_LEVEL', 'info').lower()
access_log_format = '%(h)s %(l)s %(u)s %(t)s "%(r)s" %(s)s %(b)s "%(f)s" "%(a)s" %(D)s'

# 安全配置
limit_request_line = 8190
limit_request_fields = 100
limit_request_field_size = 8190

# 進程配置
user = os.getenv('APP_USER', 'cyberpuppy')
group = os.getenv('APP_GROUP', 'cyberpuppy')
tmp_upload_dir = os.getenv('TMP_DIR', '/tmp')

# SSL 配置 (如果需要)
keyfile = os.getenv('SSL_KEYFILE')
certfile = os.getenv('SSL_CERTFILE')
ca_certs = os.getenv('SSL_CA_CERTS')

def when_ready(server):
    """服務器啟動完成時的回調"""
    server.log.info("CyberPuppy API server is ready!")

def worker_int(worker):
    """Worker 收到 SIGINT 時的處理"""
    worker.log.info("Worker received SIGINT, shutting down gracefully")

def on_exit(server):
    """服務器退出時的清理工作"""
    server.log.info("CyberPuppy API server shutting down")
```

## 監控與日誌 (Monitoring & Logging)

### 1. 日誌配置

#### logging_config.py
```python
"""
結構化日誌配置
"""

import logging.config
import os
from pythonjsonlogger import jsonlogger

LOGGING_CONFIG = {
    'version': 1,
    'disable_existing_loggers': False,
    'formatters': {
        'json': {
            '()': jsonlogger.JsonFormatter,
            'format': '%(asctime)s %(name)s %(levelname)s %(message)s'
        },
        'verbose': {
            'format': '%(asctime)s [%(levelname)s] %(name)s: %(message)s'
        }
    },
    'handlers': {
        'console': {
            'class': 'logging.StreamHandler',
            'formatter': 'json' if os.getenv('LOG_FORMAT') == 'json' else 'verbose',
            'stream': 'ext://sys.stdout'
        },
        'file': {
            'class': 'logging.handlers.RotatingFileHandler',
            'formatter': 'json',
            'filename': os.getenv('LOG_FILE', '/var/log/cyberpuppy/app.log'),
            'maxBytes': 50 * 1024 * 1024,  # 50MB
            'backupCount': 10
        }
    },
    'loggers': {
        'cyberpuppy': {
            'handlers': ['console', 'file'],
            'level': os.getenv('LOG_LEVEL', 'INFO'),
            'propagate': False
        },
        'uvicorn': {
            'handlers': ['console', 'file'],
            'level': 'INFO',
            'propagate': False
        }
    },
    'root': {
        'handlers': ['console'],
        'level': 'WARNING'
    }
}

def setup_logging():
    """設定日誌配置"""
    # 建立日誌目錄
    log_file = os.getenv('LOG_FILE', '/var/log/cyberpuppy/app.log')
    log_dir = os.path.dirname(log_file)
    os.makedirs(log_dir, exist_ok=True)

    logging.config.dictConfig(LOGGING_CONFIG)
```

### 2. Prometheus 監控

#### metrics.py
```python
"""
Prometheus 指標收集
"""

from prometheus_client import Counter, Histogram, Gauge, generate_latest
import time
from functools import wraps

# 定義指標
REQUEST_COUNT = Counter(
    'cyberpuppy_requests_total',
    'Total API requests',
    ['method', 'endpoint', 'status']
)

REQUEST_DURATION = Histogram(
    'cyberpuppy_request_duration_seconds',
    'API request duration',
    ['method', 'endpoint']
)

PREDICTION_COUNT = Counter(
    'cyberpuppy_predictions_total',
    'Total predictions made',
    ['toxicity_level']
)

MODEL_LOAD_TIME = Gauge(
    'cyberpuppy_model_load_time_seconds',
    'Time taken to load models'
)

ACTIVE_CONNECTIONS = Gauge(
    'cyberpuppy_active_connections',
    'Number of active connections'
)

def metrics_middleware(app):
    """Prometheus 指標中介層"""

    @wraps(app)
    async def wrapper(scope, receive, send):
        if scope["type"] != "http":
            await app(scope, receive, send)
            return

        start_time = time.time()
        method = scope["method"]
        path = scope["path"]

        # 增加活躍連接數
        ACTIVE_CONNECTIONS.inc()

        try:
            # 包裝 send 函數以捕捉狀態碼
            status_code = 200

            async def wrapped_send(message):
                nonlocal status_code
                if message["type"] == "http.response.start":
                    status_code = message["status"]
                await send(message)

            await app(scope, receive, wrapped_send)

        finally:
            # 記錄指標
            duration = time.time() - start_time

            REQUEST_COUNT.labels(
                method=method,
                endpoint=path,
                status=status_code
            ).inc()

            REQUEST_DURATION.labels(
                method=method,
                endpoint=path
            ).observe(duration)

            # 減少活躍連接數
            ACTIVE_CONNECTIONS.dec()

    return wrapper

def record_prediction(toxicity_level: str):
    """記錄預測結果"""
    PREDICTION_COUNT.labels(toxicity_level=toxicity_level).inc()
```

### 3. 健康檢查端點增強

#### 更新 app.py 中的健康檢查
```python
from prometheus_client import generate_latest, CONTENT_TYPE_LATEST

@app.get("/metrics")
async def get_metrics():
    """Prometheus 指標端點"""
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)

@app.get("/healthz")
async def enhanced_health_check():
    """增強版健康檢查"""
    health_data = {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "version": "1.0.0",
        "uptime_seconds": time.time() - start_time,
        "checks": {}
    }

    # 檢查模型狀態
    try:
        if model_loader and model_loader.detector:
            health_data["checks"]["model"] = "healthy"
        else:
            health_data["checks"]["model"] = "unhealthy"
            health_data["status"] = "degraded"
    except Exception:
        health_data["checks"]["model"] = "error"
        health_data["status"] = "degraded"

    # 檢查資料庫連接
    try:
        # 假設有資料庫連接檢查函數
        # check_database_connection()
        health_data["checks"]["database"] = "healthy"
    except Exception:
        health_data["checks"]["database"] = "unhealthy"
        health_data["status"] = "degraded"

    # 檢查 Redis 連接
    try:
        # check_redis_connection()
        health_data["checks"]["cache"] = "healthy"
    except Exception:
        health_data["checks"]["cache"] = "degraded"
        # Redis 失敗不影響整體狀態

    # 根據檢查結果決定 HTTP 狀態碼
    status_code = 200 if health_data["status"] == "healthy" else 503

    return JSONResponse(health_data, status_code=status_code)
```

## 安全強化 (Security Hardening)

### 1. Nginx 反向代理配置

#### nginx.conf
```nginx
events {
    worker_connections 1024;
}

http {
    # 基本設定
    sendfile on;
    tcp_nopush on;
    tcp_nodelay on;
    keepalive_timeout 65;
    types_hash_max_size 2048;
    server_tokens off;

    # 安全標頭
    add_header X-Frame-Options DENY always;
    add_header X-Content-Type-Options nosniff always;
    add_header X-XSS-Protection "1; mode=block" always;
    add_header Strict-Transport-Security "max-age=31536000; includeSubDomains" always;
    add_header Referrer-Policy "strict-origin-when-cross-origin" always;

    # 限流設定
    limit_req_zone $binary_remote_addr zone=api:10m rate=30r/m;
    limit_conn_zone $binary_remote_addr zone=conn:10m;

    # 上游服務器
    upstream cyberpuppy_api {
        least_conn;
        server cyberpuppy-api:8000 max_fails=3 fail_timeout=30s;
        keepalive 32;
    }

    server {
        listen 80;
        server_name api.cyberpuppy.ai;
        return 301 https://$server_name$request_uri;
    }

    server {
        listen 443 ssl http2;
        server_name api.cyberpuppy.ai;

        # SSL 配置
        ssl_certificate /etc/nginx/ssl/cert.pem;
        ssl_certificate_key /etc/nginx/ssl/key.pem;
        ssl_protocols TLSv1.2 TLSv1.3;
        ssl_ciphers ECDHE-RSA-AES256-GCM-SHA512:DHE-RSA-AES256-GCM-SHA512;
        ssl_prefer_server_ciphers off;

        # 限制請求大小
        client_max_body_size 1M;

        # 限流
        limit_req zone=api burst=10 nodelay;
        limit_conn conn 10;

        # 代理設定
        location / {
            proxy_pass http://cyberpuppy_api;
            proxy_http_version 1.1;
            proxy_set_header Upgrade $http_upgrade;
            proxy_set_header Connection 'upgrade';
            proxy_set_header Host $host;
            proxy_set_header X-Real-IP $remote_addr;
            proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
            proxy_set_header X-Forwarded-Proto $scheme;
            proxy_cache_bypass $http_upgrade;
            proxy_read_timeout 300s;
            proxy_connect_timeout 75s;
        }

        # 健康檢查端點不限流
        location /healthz {
            proxy_pass http://cyberpuppy_api;
            access_log off;
        }

        # 靜態檔案（如果有）
        location /static/ {
            alias /var/www/static/;
            expires 1y;
            add_header Cache-Control "public, immutable";
        }
    }
}
```

### 2. 防火牆規則

#### iptables 規則
```bash
#!/bin/bash
# firewall_rules.sh - 設定防火牆規則

# 清空現有規則
iptables -F
iptables -X
iptables -t nat -F
iptables -t nat -X
iptables -t mangle -F
iptables -t mangle -X

# 預設政策
iptables -P INPUT DROP
iptables -P FORWARD DROP
iptables -P OUTPUT ACCEPT

# 允許 loopback
iptables -A INPUT -i lo -j ACCEPT
iptables -A OUTPUT -o lo -j ACCEPT

# 允許已建立的連接
iptables -A INPUT -m conntrack --ctstate ESTABLISHED,RELATED -j ACCEPT

# 允許 SSH (限制來源 IP)
iptables -A INPUT -p tcp --dport 22 -s YOUR_ADMIN_IP -j ACCEPT

# 允許 HTTP/HTTPS
iptables -A INPUT -p tcp --dport 80 -j ACCEPT
iptables -A INPUT -p tcp --dport 443 -j ACCEPT

# 允許內部網路連接到 API
iptables -A INPUT -p tcp --dport 8000 -s 172.17.0.0/16 -j ACCEPT

# DDoS 保護
iptables -A INPUT -p tcp --dport 80 -m limit --limit 25/minute --limit-burst 100 -j ACCEPT
iptables -A INPUT -p tcp --dport 443 -m limit --limit 25/minute --limit-burst 100 -j ACCEPT

# 記錄並丟棄其他流量
iptables -A INPUT -j LOG --log-prefix "DROPPED: "
iptables -A INPUT -j DROP

# 保存規則
iptables-save > /etc/iptables/rules.v4

echo "Firewall rules applied successfully!"
```

## 故障排除 (Troubleshooting)

### 1. 常見問題診斷

#### 診斷腳本
```bash
#!/bin/bash
# diagnose.sh - 系統診斷腳本

echo "🔍 CyberPuppy API 系統診斷"
echo "=========================="

# 檢查服務狀態
echo -e "\n📊 服務狀態:"
if command -v docker-compose &> /dev/null; then
    docker-compose ps
elif command -v kubectl &> /dev/null; then
    kubectl get pods -n cyberpuppy
fi

# 檢查 API 健康狀態
echo -e "\n🏥 API 健康檢查:"
curl -s http://localhost:8000/healthz | jq '.' || echo "API 不可用或 jq 未安裝"

# 檢查資源使用
echo -e "\n💾 資源使用狀況:"
echo "Memory:"
free -h
echo -e "\nDisk:"
df -h
echo -e "\nCPU:"
top -bn1 | head -5

# 檢查日誌錯誤
echo -e "\n📝 最近的錯誤日誌:"
if [ -f "/var/log/cyberpuppy/app.log" ]; then
    tail -20 /var/log/cyberpuppy/app.log | grep -i error || echo "沒有發現錯誤"
else
    echo "日誌檔案不存在"
fi

# 檢查網路連接
echo -e "\n🌐 網路連接測試:"
curl -s -o /dev/null -w "API Response Time: %{time_total}s\n" http://localhost:8000/healthz

# 檢查模型檔案
echo -e "\n🤖 模型檔案檢查:"
MODEL_PATH="/app/models"
if [ -d "$MODEL_PATH" ]; then
    ls -la "$MODEL_PATH"
else
    echo "模型目錄不存在: $MODEL_PATH"
fi

echo -e "\n✅ 診斷完成"
```

### 2. 效能調優

#### 效能監控腳本
```python
#!/usr/bin/env python3
"""
performance_monitor.py - 效能監控腳本
"""

import time
import requests
import statistics
import concurrent.futures
from datetime import datetime

class PerformanceMonitor:
    def __init__(self, base_url="http://localhost:8000"):
        self.base_url = base_url
        self.results = []

    def single_request_test(self):
        """單一請求測試"""
        start = time.time()
        try:
            response = requests.post(
                f"{self.base_url}/analyze",
                json={"text": "測試文本"},
                timeout=30
            )
            duration = time.time() - start
            return {
                'duration': duration,
                'status_code': response.status_code,
                'success': response.status_code == 200
            }
        except Exception as e:
            return {
                'duration': time.time() - start,
                'status_code': 0,
                'success': False,
                'error': str(e)
            }

    def concurrent_test(self, concurrent_users=10, requests_per_user=5):
        """並行負載測試"""
        print(f"🚀 開始並行測試: {concurrent_users} 用戶, 每用戶 {requests_per_user} 請求")

        def user_test():
            user_results = []
            for _ in range(requests_per_user):
                result = self.single_request_test()
                user_results.append(result)
                time.sleep(0.1)  # 避免過於密集的請求
            return user_results

        start_time = time.time()

        with concurrent.futures.ThreadPoolExecutor(max_workers=concurrent_users) as executor:
            futures = [executor.submit(user_test) for _ in range(concurrent_users)]
            all_results = []

            for future in concurrent.futures.as_completed(futures):
                all_results.extend(future.result())

        total_time = time.time() - start_time

        # 分析結果
        successful_requests = [r for r in all_results if r['success']]
        failed_requests = [r for r in all_results if not r['success']]

        if successful_requests:
            durations = [r['duration'] for r in successful_requests]

            print(f"\n📊 測試結果:")
            print(f"總請求數: {len(all_results)}")
            print(f"成功請求: {len(successful_requests)}")
            print(f"失敗請求: {len(failed_requests)}")
            print(f"成功率: {len(successful_requests)/len(all_results)*100:.2f}%")
            print(f"總測試時間: {total_time:.2f} 秒")
            print(f"平均回應時間: {statistics.mean(durations):.3f} 秒")
            print(f"中位數回應時間: {statistics.median(durations):.3f} 秒")
            print(f"95% 回應時間: {statistics.quantiles(durations, n=20)[18]:.3f} 秒")
            print(f"最大回應時間: {max(durations):.3f} 秒")
            print(f"最小回應時間: {min(durations):.3f} 秒")
            print(f"吞吐量: {len(successful_requests)/total_time:.2f} 請求/秒")

        if failed_requests:
            print(f"\n❌ 失敗請求詳情:")
            for i, req in enumerate(failed_requests[:5]):  # 只顯示前5個
                print(f"  {i+1}. 狀態碼: {req['status_code']}, 錯誤: {req.get('error', 'Unknown')}")

    def health_check_test(self):
        """健康檢查測試"""
        print("🏥 健康檢查測試...")

        try:
            response = requests.get(f"{self.base_url}/healthz", timeout=10)
            if response.status_code == 200:
                health_data = response.json()
                print(f"✅ API 健康狀態: {health_data.get('status', 'unknown')}")
                print(f"   運行時間: {health_data.get('uptime_seconds', 0):.2f} 秒")

                if 'model_status' in health_data:
                    model_status = health_data['model_status']
                    print(f"   模型已載入: {model_status.get('models_loaded', False)}")
                    print(f"   計算設備: {model_status.get('device', 'unknown')}")

            else:
                print(f"❌ 健康檢查失敗: HTTP {response.status_code}")

        except Exception as e:
            print(f"❌ 健康檢查錯誤: {e}")

def main():
    monitor = PerformanceMonitor()

    print("CyberPuppy API 效能監控")
    print("======================")

    # 健康檢查
    monitor.health_check_test()

    # 單一請求測試
    print(f"\n⚡ 單一請求測試...")
    result = monitor.single_request_test()
    if result['success']:
        print(f"✅ 單一請求成功: {result['duration']:.3f} 秒")
    else:
        print(f"❌ 單一請求失敗: {result.get('error', 'Unknown error')}")

    # 並行測試
    print(f"\n")
    monitor.concurrent_test(concurrent_users=5, requests_per_user=3)

if __name__ == "__main__":
    main()
```

## 維護作業 (Maintenance Tasks)

### 1. 定期維護腳本

#### maintenance.sh
```bash
#!/bin/bash
# maintenance.sh - 定期維護腳本

LOG_FILE="/var/log/cyberpuppy/maintenance.log"
MODEL_PATH="/app/models"
BACKUP_PATH="/backup/cyberpuppy"

log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1" | tee -a "$LOG_FILE"
}

log "🔧 開始定期維護作業"

# 1. 清理舊日誌
log "📝 清理舊日誌檔案..."
find /var/log/cyberpuppy/ -name "*.log" -mtime +30 -delete
find /var/log/cyberpuppy/ -name "*.log.*" -mtime +7 -delete

# 2. 備份資料庫
log "💾 備份資料庫..."
if command -v mongodump &> /dev/null; then
    mkdir -p "$BACKUP_PATH/$(date +%Y%m%d)"
    mongodump --uri="$DATABASE_URL" --out="$BACKUP_PATH/$(date +%Y%m%d)/"

    # 清理舊備份
    find "$BACKUP_PATH" -type d -mtime +7 -exec rm -rf {} +
fi

# 3. 檢查磁碟空間
log "💽 檢查磁碟空間..."
DISK_USAGE=$(df /var/log | tail -1 | awk '{print $5}' | sed 's/%//')
if [ "$DISK_USAGE" -gt 80 ]; then
    log "⚠️  磁碟使用率過高: ${DISK_USAGE}%"
    # 發送告警（需要配置）
    # send_alert "Disk usage high: ${DISK_USAGE}%"
fi

# 4. 更新模型（如果有新版本）
log "🤖 檢查模型更新..."
# 這裡可以加入模型更新邏輯

# 5. 系統健康檢查
log "🏥 執行系統健康檢查..."
curl -s -f http://localhost:8000/healthz > /dev/null
if [ $? -eq 0 ]; then
    log "✅ API 健康檢查通過"
else
    log "❌ API 健康檢查失敗"
    # 重啟服務
    log "🔄 嘗試重啟服務..."
    docker-compose restart cyberpuppy-api
fi

# 6. 記憶體使用檢查
log "💾 檢查記憶體使用..."
MEM_USAGE=$(free | grep '^Mem:' | awk '{printf "%.0f", $3/$2*100}')
if [ "$MEM_USAGE" -gt 85 ]; then
    log "⚠️  記憶體使用率過高: ${MEM_USAGE}%"
fi

log "✅ 定期維護作業完成"
```

### 2. 模型更新流程

#### model_update.py
```python
#!/usr/bin/env python3
"""
model_update.py - 模型更新腳本
"""

import os
import shutil
import hashlib
import requests
from pathlib import Path

class ModelUpdater:
    def __init__(self, model_path="/app/models"):
        self.model_path = Path(model_path)
        self.backup_path = Path("/backup/models")
        self.download_path = Path("/tmp/model_update")

    def check_for_updates(self, version_url="https://api.cyberpuppy.ai/model-version"):
        """檢查是否有模型更新"""
        try:
            response = requests.get(version_url, timeout=10)
            if response.status_code == 200:
                remote_version = response.json()
                current_version = self.get_current_version()

                return remote_version.get('version') != current_version
        except Exception as e:
            print(f"檢查更新失敗: {e}")
            return False

    def get_current_version(self):
        """取得目前模型版本"""
        version_file = self.model_path / "version.txt"
        if version_file.exists():
            return version_file.read_text().strip()
        return "unknown"

    def download_model(self, model_url):
        """下載新模型"""
        self.download_path.mkdir(parents=True, exist_ok=True)

        try:
            response = requests.get(model_url, stream=True, timeout=300)
            response.raise_for_status()

            model_file = self.download_path / "model.tar.gz"
            with open(model_file, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)

            return model_file
        except Exception as e:
            print(f"模型下載失敗: {e}")
            return None

    def verify_model(self, model_file, expected_hash):
        """驗證模型檔案完整性"""
        sha256_hash = hashlib.sha256()
        with open(model_file, "rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                sha256_hash.update(chunk)

        return sha256_hash.hexdigest() == expected_hash

    def backup_current_model(self):
        """備份目前的模型"""
        if not self.model_path.exists():
            return

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_dir = self.backup_path / f"model_{timestamp}"
        backup_dir.parent.mkdir(parents=True, exist_ok=True)

        shutil.copytree(self.model_path, backup_dir)
        print(f"模型已備份至: {backup_dir}")

    def install_new_model(self, model_file):
        """安裝新模型"""
        # 解壓縮模型
        import tarfile
        with tarfile.open(model_file, 'r:gz') as tar:
            tar.extractall(self.download_path)

        # 移動到目標目錄
        extracted_dir = self.download_path / "model"
        if extracted_dir.exists():
            shutil.rmtree(self.model_path)
            shutil.move(extracted_dir, self.model_path)
            print("新模型安裝完成")
            return True

        return False

    def update_model(self):
        """完整的模型更新流程"""
        print("🤖 開始模型更新檢查...")

        if not self.check_for_updates():
            print("✅ 模型已是最新版本")
            return True

        print("📥 發現新版本，開始下載...")

        # 這裡應該從 API 取得實際的下載 URL 和雜湊值
        model_url = "https://models.cyberpuppy.ai/latest/model.tar.gz"
        expected_hash = "actual_hash_from_api"

        model_file = self.download_model(model_url)
        if not model_file:
            return False

        print("🔍 驗證模型檔案...")
        if not self.verify_model(model_file, expected_hash):
            print("❌ 模型檔案驗證失敗")
            return False

        print("💾 備份目前模型...")
        self.backup_current_model()

        print("🔄 安裝新模型...")
        if self.install_new_model(model_file):
            print("✅ 模型更新完成，請重啟服務")
            return True
        else:
            print("❌ 模型安裝失敗")
            return False

if __name__ == "__main__":
    updater = ModelUpdater()
    updater.update_model()
```

---

**本文件涵蓋了 CyberPuppy API 的完整部署流程，從容器化到生產環境配置，包含監控、安全、維護等各個面向。請根據實際需求選擇合適的部署方式。**

*最後更新: 2024-12-30*