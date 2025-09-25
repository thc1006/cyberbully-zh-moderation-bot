#!/bin/bash
# CyberPuppy API 啟動腳本

set -e

# 顏色輸出
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${GREEN}🐕 CyberPuppy API 啟動腳本${NC}"

# 檢查 Python 版本
if ! command -v python3 &> /dev/null; then
    echo -e "${RED}❌ Python 3 未安裝${NC}"
    exit 1
fi

PYTHON_VERSION=$(python3 -c 'import sys; print(".".join(map(str, sys.version_info[:2])))')
echo -e "${GREEN}✅ Python ${PYTHON_VERSION} 已安裝${NC}"

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
export PORT=${PORT:-8000}
export LOG_LEVEL=${LOG_LEVEL:-info}
export WORKERS=${WORKERS:-1}

echo -e "${GREEN}🚀 啟動 CyberPuppy API 服務${NC}"
echo -e "   📍 URL: http://localhost:${PORT}"
echo -e "   📖 API 文檔: http://localhost:${PORT}/docs"
echo -e "   🩺 健康檢查: http://localhost:${PORT}/healthz"
echo ""

# 啟動 uvicorn 服務器
uvicorn app:app \
    --host 0.0.0.0 \
    --port ${PORT} \
    --workers ${WORKERS} \
    --log-level ${LOG_LEVEL} \
    --reload