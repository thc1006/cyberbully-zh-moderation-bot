#!/bin/bash
# CyberPuppy LINE Bot 啟動腳本

set -e

# 顏色輸出
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${GREEN}🐕 CyberPuppy LINE Bot 啟動腳本${NC}"

# 檢查 Python 版本
if ! command -v python3 &> /dev/null; then
    echo -e "${RED}❌ Python 3 未安裝${NC}"
    exit 1
fi

PYTHON_VERSION=$(python3 -c 'import sys; print(".".join(map(str, sys.version_info[:2])))')
echo -e "${GREEN}✅ Python ${PYTHON_VERSION} 已安裝${NC}"

# 檢查環境變數檔案
if [ ! -f ".env" ]; then
    echo -e "${YELLOW}⚠️  .env 檔案不存在，請參考 .env.example 建立${NC}"
    if [ -f ".env.example" ]; then
        echo -e "${BLUE}📋 可執行: cp .env.example .env${NC}"
        exit 1
    fi
fi

# 檢查必要環境變數
source .env
if [ -z "$LINE_CHANNEL_ACCESS_TOKEN" ] || [ -z "$LINE_CHANNEL_SECRET" ]; then
    echo -e "${RED}❌ 遺失必要的 LINE Bot 設定${NC}"
    echo -e "${YELLOW}請在 .env 檔案中設定 LINE_CHANNEL_ACCESS_TOKEN 和 LINE_CHANNEL_SECRET${NC}"
    exit 1
fi

# 檢查並建立虛擬環境
if [ ! -d "venv" ]; then
    echo -e "${YELLOW}📦 建立虛擬環境...${NC}"
    python3 -m venv venv
fi

# 啟動虛擬環境
echo -e "${YELLOW}🔧 啟動虛擬環境...${NC}"
source venv/bin/activate

# 安裝依賴
echo -e "${YELLOW}📚 安裝依賴套件...${NC}"
pip install --upgrade pip
pip install -r requirements.txt

# 設定環境變數
export PYTHONPATH=$(pwd)
export PORT=${BOT_PORT:-8080}
export HOST=${BOT_HOST:-0.0.0.0}
export LOG_LEVEL=${LOG_LEVEL:-info}

# 檢查 CyberPuppy API 可用性
API_URL=${CYBERPUPPY_API_URL:-"http://localhost:8000"}
echo -e "${BLUE}🔍 檢查 CyberPuppy API (${API_URL})...${NC}"

if curl -f "${API_URL}/healthz" > /dev/null 2>&1; then
    echo -e "${GREEN}✅ CyberPuppy API 可用${NC}"
else
    echo -e "${YELLOW}⚠️  CyberPuppy API 無法連線，請確認 API 服務已啟動${NC}"
    echo -e "${BLUE}💡 可在另一個終端執行: cd ../api && ./start.sh${NC}"
fi

echo -e "${GREEN}🚀 啟動 CyberPuppy LINE Bot${NC}"
echo -e "   📍 URL: http://${HOST}:${PORT}"
echo -e "   🩺 健康檢查: http://${HOST}:${PORT}/health"
echo -e "   📊 統計資訊: http://${HOST}:${PORT}/stats"
echo -e "   🔗 Webhook: http://${HOST}:${PORT}/webhook"
echo ""
echo -e "${YELLOW}📝 記得在 LINE Developers Console 設定 Webhook URL${NC}"
echo ""

# 啟動服務
uvicorn line_bot:app \
    --host ${HOST} \
    --port ${PORT} \
    --log-level ${LOG_LEVEL} \
    --reload