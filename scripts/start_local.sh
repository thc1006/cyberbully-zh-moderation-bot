#!/bin/bash
# CyberPuppy 本地快速啟動腳本

echo "======================================"
echo "CyberPuppy 本地服務啟動程式"
echo "======================================"
echo

# 檢查 Python
if ! command -v python3 &> /dev/null && ! command -v python &> /dev/null; then
    echo "[ERROR] Python 未安裝！"
    exit 1
fi

# 使用 python3 或 python
PYTHON_CMD="python3"
if ! command -v python3 &> /dev/null; then
    PYTHON_CMD="python"
fi

# 檢查必要檔案
echo "[INFO] 檢查必要檔案..."
$PYTHON_CMD scripts/check_requirements.py
if [ $? -ne 0 ]; then
    echo "[ERROR] 缺少必要檔案！請執行："
    echo "  $PYTHON_CMD scripts/download_datasets.py"
    exit 1
fi

# 啟動 API 服務
echo
echo "[INFO] 啟動 API 服務..."
echo "[INFO] 服務將在 http://localhost:8000 啟動"
echo
echo "按 Ctrl+C 停止服務"
echo

cd api
$PYTHON_CMD -m uvicorn app:app --host 0.0.0.0 --port 8000 --reload