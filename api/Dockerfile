# CyberPuppy API Dockerfile
# 基於 Python 3.11 官方精簡映像檔

FROM python:3.11-slim

# 設定工作目錄
WORKDIR /app

# 設定環境變數
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app \
    PORT=8000

# 建立非 root 使用者
RUN adduser --disabled-password --gecos '' --shell /bin/bash cyberpuppy

# 安裝系統依賴
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# 複製需求檔案
COPY requirements.txt .

# 升級 pip 並安裝 Python 依賴
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# 複製應用程式碼
COPY . .

# 變更檔案擁有者至非 root 使用者
RUN chown -R cyberpuppy:cyberpuppy /app

# 切換至非 root 使用者
USER cyberpuppy

# 暴露連接埠
EXPOSE $PORT

# 健康檢查
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:$PORT/healthz || exit 1

# 啟動指令
CMD ["sh", "-c", "uvicorn app:app --host 0.0.0.0 --port $PORT --workers 1"]