#!/bin/bash
# 霸凌偵測F1優化訓練執行腳本
# 使用改進的架構和RTX 3050優化配置

set -e  # 遇到錯誤立即退出

# 顏色輸出
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# 日誌函數
log_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

log_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# 專案根目錄
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"

log_info "專案根目錄: $PROJECT_ROOT"

# 檢查Python環境
if ! command -v python &> /dev/null; then
    log_error "Python 未安裝或不在PATH中"
    exit 1
fi

PYTHON_VERSION=$(python --version 2>&1 | awk '{print $2}')
log_info "Python 版本: $PYTHON_VERSION"

# 檢查GPU
if command -v nvidia-smi &> /dev/null; then
    log_info "GPU 資訊:"
    nvidia-smi --query-gpu=name,memory.total,memory.used,temperature.gpu --format=csv,noheader,nounits | head -1
else
    log_warning "未偵測到NVIDIA GPU或nvidia-smi不可用"
fi

# 設定預設參數
CONFIG_FILE="configs/training/bullying_f1_optimization.yaml"
EXPERIMENT_NAME="bullying_f1_075_$(date +%Y%m%d_%H%M%S)"
OUTPUT_DIR="experiments/bullying_f1_optimization"
DATA_DIR="data/processed/training_dataset"

# 解析命令列參數
while [[ $# -gt 0 ]]; do
    case $1 in
        --config)
            CONFIG_FILE="$2"
            shift 2
            ;;
        --experiment-name)
            EXPERIMENT_NAME="$2"
            shift 2
            ;;
        --output-dir)
            OUTPUT_DIR="$2"
            shift 2
            ;;
        --data-dir)
            DATA_DIR="$2"
            shift 2
            ;;
        --help)
            echo "使用方法: $0 [選項]"
            echo ""
            echo "選項:"
            echo "  --config FILE           配置檔案路徑 (預設: $CONFIG_FILE)"
            echo "  --experiment-name NAME  實驗名稱 (預設: 自動生成)"
            echo "  --output-dir DIR        輸出目錄 (預設: $OUTPUT_DIR)"
            echo "  --data-dir DIR          資料目錄 (預設: $DATA_DIR)"
            echo "  --help                  顯示此幫助訊息"
            exit 0
            ;;
        *)
            log_error "未知參數: $1"
            echo "使用 --help 查看可用選項"
            exit 1
            ;;
    esac
done

log_info "使用配置檔案: $CONFIG_FILE"
log_info "實驗名稱: $EXPERIMENT_NAME"
log_info "輸出目錄: $OUTPUT_DIR"
log_info "資料目錄: $DATA_DIR"

# 檢查必要檔案
log_info "檢查必要檔案..."

if [[ ! -f "$CONFIG_FILE" ]]; then
    log_error "配置檔案不存在: $CONFIG_FILE"
    exit 1
fi

if [[ ! -f "scripts/train_bullying_f1_optimizer.py" ]]; then
    log_error "訓練腳本不存在: scripts/train_bullying_f1_optimizer.py"
    exit 1
fi

# 檢查資料檔案
TRAIN_DATA="$DATA_DIR/train.json"
VAL_DATA="$DATA_DIR/val.json"
TEST_DATA="$DATA_DIR/test.json"

if [[ ! -f "$TRAIN_DATA" ]] && [[ ! -f "data/processed/cold/train.json" ]]; then
    log_error "找不到訓練資料檔案"
    log_error "請確認以下路徑之一存在:"
    log_error "  - $TRAIN_DATA"
    log_error "  - data/processed/cold/train.json"
    exit 1
fi

# 檢查Python依賴
log_info "檢查Python依賴..."

REQUIRED_PACKAGES=("torch" "transformers" "sklearn" "numpy" "pandas" "yaml" "tqdm")
MISSING_PACKAGES=()

for package in "${REQUIRED_PACKAGES[@]}"; do
    if ! python -c "import $package" &> /dev/null; then
        MISSING_PACKAGES+=("$package")
    fi
done

if [[ ${#MISSING_PACKAGES[@]} -gt 0 ]]; then
    log_error "缺少必要的Python套件: ${MISSING_PACKAGES[*]}"
    log_info "請執行: pip install ${MISSING_PACKAGES[*]}"
    exit 1
fi

# 建立輸出目錄
log_info "建立輸出目錄..."
mkdir -p "$OUTPUT_DIR"
mkdir -p "logs"

# 設定日誌檔案
LOG_FILE="logs/training_${EXPERIMENT_NAME}.log"

# 執行訓練
log_info "開始訓練霸凌偵測模型..."
log_info "目標: 霸凌F1≥0.75, 毒性F1≥0.78, 總體F1≥0.76"
log_info "日誌檔案: $LOG_FILE"

echo "==================== 訓練開始 ====================" | tee "$LOG_FILE"
echo "時間: $(date)" | tee -a "$LOG_FILE"
echo "配置: $CONFIG_FILE" | tee -a "$LOG_FILE"
echo "實驗: $EXPERIMENT_NAME" | tee -a "$LOG_FILE"
echo "=====================================================" | tee -a "$LOG_FILE"

# 執行Python訓練腳本
python scripts/train_bullying_f1_optimizer.py \
    --config "$CONFIG_FILE" \
    --experiment-name "$EXPERIMENT_NAME" \
    --output-dir "$OUTPUT_DIR" \
    --data-dir "$DATA_DIR" \
    2>&1 | tee -a "$LOG_FILE"

TRAINING_EXIT_CODE=${PIPESTATUS[0]}

echo "" | tee -a "$LOG_FILE"
echo "==================== 訓練結束 ====================" | tee -a "$LOG_FILE"
echo "時間: $(date)" | tee -a "$LOG_FILE"
echo "退出碼: $TRAINING_EXIT_CODE" | tee -a "$LOG_FILE"

if [[ $TRAINING_EXIT_CODE -eq 0 ]]; then
    log_success "訓練完成!"

    # 顯示結果摘要
    RESULTS_FILE="$OUTPUT_DIR/$EXPERIMENT_NAME/final_results.json"
    if [[ -f "$RESULTS_FILE" ]]; then
        log_info "結果摘要:"
        python -c "
import json
with open('$RESULTS_FILE', 'r', encoding='utf-8') as f:
    results = json.load(f)

test_metrics = results.get('test_metrics', {})
targets = results.get('target_achieved', {})

print(f\"霸凌F1: {test_metrics.get('bullying_f1', 0):.4f} ({'✅' if targets.get('bullying_f1_075', False) else '❌'})\")
print(f\"毒性F1: {test_metrics.get('toxicity_f1', 0):.4f} ({'✅' if targets.get('toxicity_f1_078', False) else '❌'})\")
print(f\"總體F1: {test_metrics.get('overall_macro_f1', 0):.4f} ({'✅' if targets.get('overall_macro_f1_076', False) else '❌'})\")
"
    fi

    # TensorBoard 資訊
    TENSORBOARD_DIR="$OUTPUT_DIR/$EXPERIMENT_NAME/tensorboard_logs"
    if [[ -d "$TENSORBOARD_DIR" ]]; then
        log_info "TensorBoard 日誌位置: $TENSORBOARD_DIR"
        log_info "啟動TensorBoard: tensorboard --logdir $TENSORBOARD_DIR"
    fi

    # 模型工件位置
    MODEL_DIR="$OUTPUT_DIR/$EXPERIMENT_NAME/model_artifacts"
    if [[ -d "$MODEL_DIR" ]]; then
        log_info "模型檔案位置: $MODEL_DIR"
    fi

else
    log_error "訓練失敗 (退出碼: $TRAINING_EXIT_CODE)"
    log_info "請查看日誌檔案: $LOG_FILE"
    exit $TRAINING_EXIT_CODE
fi

echo "=====================================================" | tee -a "$LOG_FILE"

# 可選: 自動啟動TensorBoard
read -p "是否要啟動TensorBoard? (y/N): " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    if [[ -d "$TENSORBOARD_DIR" ]]; then
        log_info "啟動TensorBoard..."
        echo "在瀏覽器中打開: http://localhost:6006"
        tensorboard --logdir "$TENSORBOARD_DIR" --port 6006
    else
        log_warning "TensorBoard日誌目錄不存在"
    fi
fi

log_success "腳本執行完成!"