# CyberPuppy LINE Bot Dockerfile
FROM python:3.11-slim

# 設定工作目錄
WORKDIR /app

# 設定環境變數
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app \
    PORT=8080

# 建立非 root 使用者
RUN adduser --disabled-password --gecos '' --shell /bin/bash linebot

# 安裝系統依賴
RUN apt-get update && apt-get install -y \
    curl \
    && rm -rf /var/lib/apt/lists/*

# 複製需求檔案並安裝依賴
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# 複製應用程式碼
COPY . .

# 變更檔案擁有者
RUN chown -R linebot:linebot /app

# 切換至非 root 使用者
USER linebot

# 暴露連接埠
EXPOSE $PORT

# 健康檢查
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:$PORT/health || exit 1

# 啟動指令
CMD ["sh", "-c", "uvicorn line_bot:app --host 0.0.0.0 --port $PORT --workers 1"]