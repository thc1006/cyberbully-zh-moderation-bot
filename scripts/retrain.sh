#!/bin/bash

# CyberPuppy 週期性再訓練腳本
# 用於主動學習循環的模型更新

set -e  # 遇錯即停

# ========================================
# 配置參數
# ========================================

# 基礎路徑
BASE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DATA_DIR="${BASE_DIR}/data"
MODELS_DIR="${BASE_DIR}/models"
LOGS_DIR="${BASE_DIR}/logs/retrain"
BACKUP_DIR="${BASE_DIR}/models/backups"

# 主動學習目錄
ACTIVE_LEARNING_DIR="${DATA_DIR}/active_learning"
RETRAIN_DATA_DIR="${ACTIVE_LEARNING_DIR}/retrain_data"
COMPLETED_ANNOTATIONS="${ACTIVE_LEARNING_DIR}/completed_*.csv"

# 訓練配置
CONFIG_FILE="${BASE_DIR}/configs/training.yaml"
MIN_SAMPLES_FOR_RETRAIN=500
VALIDATION_SPLIT=0.2

# 模型版本
CURRENT_MODEL_VERSION=$(cat "${MODELS_DIR}/current_version.txt" 2>/dev/null || echo "v1.0.0")
NEW_MODEL_VERSION=""

# 時間戳記
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
TODAY=$(date +"%Y-%m-%d")

# 顏色輸出
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# ========================================
# 輔助函數
# ========================================

log_info() {
    echo -e "${GREEN}[INFO]${NC} $(date '+%Y-%m-%d %H:%M:%S') - $1"
}

log_warn() {
    echo -e "${YELLOW}[WARN]${NC} $(date '+%Y-%m-%d %H:%M:%S') - $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $(date '+%Y-%m-%d %H:%M:%S') - $1"
}

log_section() {
    echo -e "\n${BLUE}========================================${NC}"
    echo -e "${BLUE}$1${NC}"
    echo -e "${BLUE}========================================${NC}\n"
}

check_requirements() {
    log_section "檢查環境需求"

    # 檢查 Python
    if ! command -v python3 &> /dev/null; then
        log_error "Python3 未安裝"
        exit 1
    fi

    # 檢查虛擬環境
    if [ ! -d "${BASE_DIR}/venv" ]; then
        log_warn "虛擬環境不存在，正在創建..."
        python3 -m venv "${BASE_DIR}/venv"
        "${BASE_DIR}/venv/bin/pip" install -r "${BASE_DIR}/requirements.txt"
    fi

    # 啟動虛擬環境
    source "${BASE_DIR}/venv/bin/activate"

    log_info "環境檢查完成"
}

backup_current_model() {
    log_section "備份當前模型"

    # 創建備份目錄
    mkdir -p "${BACKUP_DIR}"

    # 備份當前模型
    if [ -f "${MODELS_DIR}/best_model.pt" ]; then
        BACKUP_NAME="model_${CURRENT_MODEL_VERSION}_${TIMESTAMP}.pt"
        cp "${MODELS_DIR}/best_model.pt" "${BACKUP_DIR}/${BACKUP_NAME}"
        log_info "模型已備份至: ${BACKUP_DIR}/${BACKUP_NAME}"
    else
        log_warn "當前無模型可備份"
    fi

    # 備份配置
    if [ -f "${CONFIG_FILE}" ]; then
        cp "${CONFIG_FILE}" "${BACKUP_DIR}/config_${TIMESTAMP}.yaml"
    fi
}

collect_annotations() {
    log_section "收集標註數據"

    # 統計已完成的標註
    TOTAL_ANNOTATIONS=0
    for csv_file in ${COMPLETED_ANNOTATIONS}; do
        if [ -f "$csv_file" ]; then
            # 計算行數（排除標題）
            LINES=$(tail -n +2 "$csv_file" | wc -l)
            TOTAL_ANNOTATIONS=$((TOTAL_ANNOTATIONS + LINES))
            log_info "從 $(basename $csv_file) 收集 ${LINES} 筆標註"
        fi
    done

    log_info "總共收集 ${TOTAL_ANNOTATIONS} 筆標註"

    # 檢查是否足夠再訓練
    if [ ${TOTAL_ANNOTATIONS} -lt ${MIN_SAMPLES_FOR_RETRAIN} ]; then
        log_warn "標註數量不足 (${TOTAL_ANNOTATIONS}/${MIN_SAMPLES_FOR_RETRAIN})，跳過再訓練"
        exit 0
    fi

    return 0
}

prepare_training_data() {
    log_section "準備訓練數據"

    # 執行數據準備腳本
    python3 << EOF
import sys
sys.path.append('${BASE_DIR}/src')
from cyberpuppy.loop.active import ActiveLearningLoop, ActiveLearningConfig
import glob

# 初始化主動學習循環
config = ActiveLearningConfig(
    output_dir='${ACTIVE_LEARNING_DIR}',
    min_samples_for_retrain=${MIN_SAMPLES_FOR_RETRAIN}
)
active_loop = ActiveLearningLoop(config)

# 收集所有已完成的標註
all_annotations = []
for csv_file in glob.glob('${ACTIVE_LEARNING_DIR}/completed_*.csv'):
    annotations = active_loop.load_completed_annotations(csv_file)
    all_annotations.extend(annotations)

# 準備再訓練數據
output_files = active_loop.prepare_retraining_data(
    annotations=all_annotations,
    output_dir='${RETRAIN_DATA_DIR}'
)

print(f"Prepared {len(all_annotations)} samples for retraining")
print(f"Output files: {output_files}")
EOF

    if [ $? -ne 0 ]; then
        log_error "數據準備失敗"
        exit 1
    fi

    log_info "訓練數據準備完成"
}

merge_with_existing_data() {
    log_section "合併現有訓練數據"

    # 合併新標註數據與原始訓練數據
    python3 << EOF
import json
import random
from pathlib import Path

# 讀取新標註數據
new_toxicity_data = []
retrain_toxicity_file = Path('${RETRAIN_DATA_DIR}/toxicity_retrain.jsonl')
if retrain_toxicity_file.exists():
    with open(retrain_toxicity_file, 'r', encoding='utf-8') as f:
        for line in f:
            new_toxicity_data.append(json.loads(line))

new_emotion_data = []
retrain_emotion_file = Path('${RETRAIN_DATA_DIR}/emotion_retrain.jsonl')
if retrain_emotion_file.exists():
    with open(retrain_emotion_file, 'r', encoding='utf-8') as f:
        for line in f:
            new_emotion_data.append(json.loads(line))

# 讀取原始訓練數據（如果存在）
original_data = []
original_file = Path('${DATA_DIR}/processed/train.jsonl')
if original_file.exists():
    with open(original_file, 'r', encoding='utf-8') as f:
        for line in f:
            original_data.append(json.loads(line))

# 合併數據（新標註數據優先）
merged_data = new_toxicity_data + original_data

# 打亂並分割
random.shuffle(merged_data)
split_point = int(len(merged_data) * (1 - ${VALIDATION_SPLIT}))
train_data = merged_data[:split_point]
val_data = merged_data[split_point:]

# 儲存合併後的數據
merged_dir = Path('${DATA_DIR}/processed/retrain_${TIMESTAMP}')
merged_dir.mkdir(parents=True, exist_ok=True)

with open(merged_dir / 'train.jsonl', 'w', encoding='utf-8') as f:
    for item in train_data:
        f.write(json.dumps(item, ensure_ascii=False) + '\n')

with open(merged_dir / 'val.jsonl', 'w', encoding='utf-8') as f:
    for item in val_data:
        f.write(json.dumps(item, ensure_ascii=False) + '\n')

print(f"Merged data: {len(train_data)} train, {len(val_data)} validation")
print(f"Saved to: {merged_dir}")
EOF

    log_info "數據合併完成"
}

run_training() {
    log_section "執行模型訓練"

    # 創建日誌目錄
    mkdir -p "${LOGS_DIR}"

    # 設定新版本號
    MAJOR=$(echo $CURRENT_MODEL_VERSION | cut -d. -f1 | sed 's/v//')
    MINOR=$(echo $CURRENT_MODEL_VERSION | cut -d. -f2)
    PATCH=$(echo $CURRENT_MODEL_VERSION | cut -d. -f3)
    NEW_PATCH=$((PATCH + 1))
    NEW_MODEL_VERSION="v${MAJOR}.${MINOR}.${NEW_PATCH}"

    log_info "開始訓練新模型 ${NEW_MODEL_VERSION}"

    # 執行訓練
    python3 "${BASE_DIR}/train.py" \
        --config "${CONFIG_FILE}" \
        --data-dir "${DATA_DIR}/processed/retrain_${TIMESTAMP}" \
        --output-dir "${MODELS_DIR}/retrain_${TIMESTAMP}" \
        --experiment-name "retrain_${NEW_MODEL_VERSION}" \
        --use-amp \
        --early-stopping \
        --patience 5 \
        --max-epochs 20 \
        2>&1 | tee "${LOGS_DIR}/train_${TIMESTAMP}.log"

    if [ ${PIPESTATUS[0]} -ne 0 ]; then
        log_error "訓練失敗"
        exit 1
    fi

    log_info "訓練完成"
}

evaluate_new_model() {
    log_section "評估新模型"

    # 執行評估腳本
    python3 << EOF
import sys
sys.path.append('${BASE_DIR}/src')
from cyberpuppy.eval.evaluate import Evaluator
from pathlib import Path
import json

# 載入新模型
model_path = Path('${MODELS_DIR}/retrain_${TIMESTAMP}/best_model.pt')
val_data_path = Path('${DATA_DIR}/processed/retrain_${TIMESTAMP}/val.jsonl')

# 初始化評估器
evaluator = Evaluator(model_path=str(model_path))

# 載入驗證數據
val_data = []
with open(val_data_path, 'r', encoding='utf-8') as f:
    for line in f:
        val_data.append(json.loads(line))

# 執行評估
metrics = evaluator.evaluate_dataset(val_data)

# 儲存評估結果
metrics_file = Path('${MODELS_DIR}/retrain_${TIMESTAMP}/metrics.json')
with open(metrics_file, 'w', encoding='utf-8') as f:
    json.dump(metrics, f, ensure_ascii=False, indent=2)

# 檢查是否達到標準
toxicity_f1 = metrics.get('toxicity', {}).get('macro_f1', 0.0)
emotion_f1 = metrics.get('emotion', {}).get('macro_f1', 0.0)

print(f"Toxicity Macro F1: {toxicity_f1:.3f}")
print(f"Emotion Macro F1: {emotion_f1:.3f}")

# 判斷是否採用新模型
if toxicity_f1 >= 0.78 and emotion_f1 >= 0.85:
    print("✅ 新模型達到標準")
    exit(0)
else:
    print("❌ 新模型未達標準")
    exit(1)
EOF

    EVAL_RESULT=$?

    if [ ${EVAL_RESULT} -eq 0 ]; then
        log_info "新模型評估通過"
        return 0
    else
        log_warn "新模型未達標準，保留當前模型"
        return 1
    fi
}

deploy_new_model() {
    log_section "部署新模型"

    # 複製新模型到生產目錄
    cp "${MODELS_DIR}/retrain_${TIMESTAMP}/best_model.pt" "${MODELS_DIR}/best_model.pt"
    cp "${MODELS_DIR}/retrain_${TIMESTAMP}/config.yaml" "${MODELS_DIR}/config.yaml"

    # 更新版本號
    echo "${NEW_MODEL_VERSION}" > "${MODELS_DIR}/current_version.txt"

    # 創建版本標記
    echo "{
        \"version\": \"${NEW_MODEL_VERSION}\",
        \"trained_at\": \"${TIMESTAMP}\",
        \"deployed_at\": \"$(date +"%Y%m%d_%H%M%S")\",
        \"metrics\": $(cat "${MODELS_DIR}/retrain_${TIMESTAMP}/metrics.json")
    }" > "${MODELS_DIR}/version_info.json"

    log_info "新模型 ${NEW_MODEL_VERSION} 已部署"
}

restart_services() {
    log_section "重啟服務"

    # 檢查服務是否運行
    if pgrep -f "uvicorn api.app" > /dev/null; then
        log_info "重啟 API 服務..."
        pkill -f "uvicorn api.app"
        sleep 2
        cd "${BASE_DIR}/api" && nohup uvicorn app:app --reload --host 0.0.0.0 --port 8000 > /dev/null 2>&1 &
        log_info "API 服務已重啟"
    else
        log_warn "API 服務未運行"
    fi

    if pgrep -f "line_bot.py" > /dev/null; then
        log_info "重啟 LINE Bot..."
        pkill -f "line_bot.py"
        sleep 2
        cd "${BASE_DIR}/bot" && nohup python3 line_bot.py > /dev/null 2>&1 &
        log_info "LINE Bot 已重啟"
    else
        log_warn "LINE Bot 未運行"
    fi
}

cleanup_old_data() {
    log_section "清理舊數據"

    # 移動已處理的標註到歸檔
    ARCHIVE_DIR="${ACTIVE_LEARNING_DIR}/archive/${TODAY}"
    mkdir -p "${ARCHIVE_DIR}"

    for csv_file in ${COMPLETED_ANNOTATIONS}; do
        if [ -f "$csv_file" ]; then
            mv "$csv_file" "${ARCHIVE_DIR}/"
            log_info "歸檔: $(basename $csv_file)"
        fi
    done

    # 清理舊的再訓練數據（保留最近 3 次）
    find "${DATA_DIR}/processed" -name "retrain_*" -type d | sort -r | tail -n +4 | while read dir; do
        rm -rf "$dir"
        log_info "刪除舊訓練數據: $(basename $dir)"
    done

    # 清理舊模型備份（保留最近 5 個）
    ls -t "${BACKUP_DIR}"/model_*.pt 2>/dev/null | tail -n +6 | while read file; do
        rm "$file"
        log_info "刪除舊備份: $(basename $file)"
    done
}

send_notification() {
    local status=$1
    local message=$2

    log_section "發送通知"

    # 這裡可以整合各種通知服務
    # 例如: Slack, Email, LINE Notify 等

    # 寫入通知日誌
    echo "[${TIMESTAMP}] ${status}: ${message}" >> "${LOGS_DIR}/notifications.log"

    # 如果有設定 webhook
    if [ ! -z "${SLACK_WEBHOOK_URL}" ]; then
        curl -X POST -H 'Content-type: application/json' \
            --data "{\"text\":\"CyberPuppy Retrain ${status}: ${message}\"}" \
            "${SLACK_WEBHOOK_URL}" 2>/dev/null
    fi

    log_info "通知已發送: ${status}"
}

# ========================================
# 主流程
# ========================================

main() {
    log_section "CyberPuppy 再訓練流程開始"
    log_info "時間: ${TIMESTAMP}"
    log_info "當前模型版本: ${CURRENT_MODEL_VERSION}"

    # 1. 檢查環境
    check_requirements

    # 2. 收集標註數據
    collect_annotations

    # 3. 備份當前模型
    backup_current_model

    # 4. 準備訓練數據
    prepare_training_data

    # 5. 合併數據
    merge_with_existing_data

    # 6. 執行訓練
    run_training

    # 7. 評估新模型
    if evaluate_new_model; then
        # 8. 部署新模型
        deploy_new_model

        # 9. 重啟服務
        restart_services

        # 10. 清理舊數據
        cleanup_old_data

        # 發送成功通知
        send_notification "SUCCESS" "模型已更新至 ${NEW_MODEL_VERSION}"

        log_section "再訓練完成"
        log_info "新模型版本: ${NEW_MODEL_VERSION}"
        exit 0
    else
        # 發送失敗通知
        send_notification "FAILED" "新模型未達標準，保留 ${CURRENT_MODEL_VERSION}"

        log_section "再訓練結束"
        log_warn "新模型未達標準，繼續使用當前模型"
        exit 0
    fi
}

# ========================================
# 執行入口
# ========================================

# 處理參數
while [[ $# -gt 0 ]]; do
    case $1 in
        --dry-run)
            DRY_RUN=true
            log_info "乾跑模式（不執行實際訓練）"
            shift
            ;;
        --force)
            MIN_SAMPLES_FOR_RETRAIN=1
            log_warn "強制執行模式（忽略最小樣本數限制）"
            shift
            ;;
        --config)
            CONFIG_FILE="$2"
            shift 2
            ;;
        --help)
            echo "用法: $0 [選項]"
            echo "選項:"
            echo "  --dry-run    乾跑模式，不執行實際訓練"
            echo "  --force      強制執行，忽略最小樣本數限制"
            echo "  --config     指定訓練配置檔案"
            echo "  --help       顯示此說明"
            exit 0
            ;;
        *)
            log_error "未知參數: $1"
            exit 1
            ;;
    esac
done

# 檢查是否已有訓練在執行
LOCK_FILE="/tmp/cyberpuppy_retrain.lock"
if [ -f "${LOCK_FILE}" ]; then
    PID=$(cat "${LOCK_FILE}")
    if ps -p ${PID} > /dev/null 2>&1; then
        log_error "另一個再訓練流程正在執行 (PID: ${PID})"
        exit 1
    else
        log_warn "發現過期的鎖定檔案，移除中..."
        rm "${LOCK_FILE}"
    fi
fi

# 創建鎖定檔案
echo $$ > "${LOCK_FILE}"

# 設定清理函數
cleanup() {
    rm -f "${LOCK_FILE}"
}
trap cleanup EXIT

# 執行主流程
main

# 清理
cleanup