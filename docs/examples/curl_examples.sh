#!/bin/bash
# CyberPuppy API cURL Examples
# 中文網路霸凌防治 API cURL 範例集合
#
# 使用方式:
# 1. 設定你的 API 密鑰: export CYBERPUPPY_API_KEY="cp_your_api_key_here"
# 2. 設定 API 端點: export CYBERPUPPY_API_URL="https://api.cyberpuppy.ai"
# 3. 執行腳本: bash curl_examples.sh
# 4. 或執行單一範例: bash curl_examples.sh basic_analysis

# 配置變數
API_KEY="${CYBERPUPPY_API_KEY:-cp_your_api_key_here}"
API_URL="${CYBERPUPPY_API_URL:-http://localhost:8000}"

# 顏色輸出
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[0;33m'
BLUE='\033[0;34m'
PURPLE='\033[0;35m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

# 輔助函數
print_header() {
    echo -e "\n${CYAN}═══════════════════════════════════════${NC}"
    echo -e "${CYAN}$1${NC}"
    echo -e "${CYAN}═══════════════════════════════════════${NC}\n"
}

print_step() {
    echo -e "${BLUE}🔹 $1${NC}"
}

print_success() {
    echo -e "${GREEN}✅ $1${NC}"
}

print_error() {
    echo -e "${RED}❌ $1${NC}"
}

print_warning() {
    echo -e "${YELLOW}⚠️  $1${NC}"
}

# 檢查依賴
check_dependencies() {
    print_step "檢查依賴項..."

    if ! command -v curl &> /dev/null; then
        print_error "curl 未安裝，請先安裝 curl"
        exit 1
    fi

    if ! command -v jq &> /dev/null; then
        print_warning "jq 未安裝，JSON 輸出將不會格式化"
        JQ_AVAILABLE=false
    else
        JQ_AVAILABLE=true
    fi

    print_success "依賴檢查完成"
}

# 格式化 JSON 輸出
format_json() {
    if [ "$JQ_AVAILABLE" = true ]; then
        echo "$1" | jq '.'
    else
        echo "$1"
    fi
}

# 執行 API 請求並處理回應
make_request() {
    local method="$1"
    local endpoint="$2"
    local data="$3"
    local description="$4"

    print_step "$description"

    # 建構 curl 命令
    local curl_cmd="curl -s -w \"\\n%{http_code}\\n\" -X $method"

    # 添加標頭
    if [ "$method" != "GET" ] || [ "$endpoint" = "/analyze" ]; then
        curl_cmd="$curl_cmd -H \"Authorization: Bearer $API_KEY\""
    fi

    if [ -n "$data" ]; then
        curl_cmd="$curl_cmd -H \"Content-Type: application/json\" -d '$data'"
    fi

    curl_cmd="$curl_cmd \"$API_URL$endpoint\""

    echo -e "${PURPLE}請求命令:${NC}"
    echo "$curl_cmd"
    echo

    # 執行請求
    local response
    response=$(eval "$curl_cmd")

    # 分離回應內容和狀態碼
    local http_code
    http_code=$(echo "$response" | tail -n1)
    local response_body
    response_body=$(echo "$response" | head -n -1)

    echo -e "${PURPLE}HTTP 狀態碼:${NC} $http_code"
    echo -e "${PURPLE}回應內容:${NC}"
    format_json "$response_body"

    # 檢查狀態碼
    if [[ "$http_code" =~ ^2[0-9][0-9]$ ]]; then
        print_success "請求成功"
    else
        print_error "請求失敗 (HTTP $http_code)"
    fi

    echo
    return 0
}

# 1. 系統健康檢查
health_check() {
    print_header "1. 系統健康檢查 (System Health Check)"
    make_request "GET" "/healthz" "" "檢查 API 服務健康狀態"
}

# 2. API 基本資訊
api_info() {
    print_header "2. API 基本資訊 (API Information)"
    make_request "GET" "/" "" "取得 API 基本資訊與功能列表"
}

# 3. 基本文本分析
basic_analysis() {
    print_header "3. 基本文本分析 (Basic Text Analysis)"

    local positive_text='{"text": "今天天氣真好，心情很棒！謝謝大家的關心。"}'
    make_request "POST" "/analyze" "$positive_text" "分析正面情緒文本"

    local neutral_text='{"text": "請問今天的會議是幾點開始？需要準備什麼資料嗎？"}'
    make_request "POST" "/analyze" "$neutral_text" "分析中性文本"
}

# 4. 毒性內容檢測
toxicity_detection() {
    print_header "4. 毒性內容檢測 (Toxicity Detection)"

    local toxic_text='{"text": "你這個笨蛋，滾開！不要再煩我了。"}'
    make_request "POST" "/analyze" "$toxic_text" "分析一般毒性文本"

    local severe_text='{"text": "你等著瞧，我會讓你後悔的！"}'
    make_request "POST" "/analyze" "$severe_text" "分析威脅性文本"

    local harassment_text='{"text": "你長得真醜，沒人會喜歡你的。"}'
    make_request "POST" "/analyze" "$harassment_text" "分析騷擾性文本"
}

# 5. 角色識別測試
role_identification() {
    print_header "5. 角色識別測試 (Role Identification)"

    local victim_text='{"text": "請大家幫幫我，我一直被欺負，不知道該怎麼辦..."}'
    make_request "POST" "/analyze" "$victim_text" "識別受害者角色"

    local perpetrator_text='{"text": "哈哈，就是要讓你難堪！看你能怎樣？"}'
    make_request "POST" "/analyze" "$perpetrator_text" "識別施暴者角色"

    local bystander_text='{"text": "我覺得你們這樣吵下去沒意思，不如都冷靜一下吧。"}'
    make_request "POST" "/analyze" "$bystander_text" "識別旁觀者角色"
}

# 6. 情緒分析測試
emotion_analysis() {
    print_header "6. 情緒分析測試 (Emotion Analysis)"

    local happy_text='{"text": "太開心了！終於考上理想的學校，感謝所有幫助過我的人！"}'
    make_request "POST" "/analyze" "$happy_text" "分析高興情緒 (強度測試)"

    local sad_text='{"text": "今天心情很低落，考試沒考好，感覺對不起父母的期望..."}'
    make_request "POST" "/analyze" "$sad_text" "分析悲傷情緒 (強度測試)"

    local angry_text='{"text": "真的很生氣！為什麼總是這樣對待我？太不公平了！"}'
    make_request "POST" "/analyze" "$angry_text" "分析憤怒情緒 (強度測試)"
}

# 7. 上下文分析
context_analysis() {
    print_header "7. 上下文分析 (Context Analysis)"

    local context_data='{
        "text": "我不同意你的看法",
        "context": "剛才討論的是關於教育政策的議題，大家都在理性討論不同的觀點",
        "thread_id": "edu_discussion_001"
    }'
    make_request "POST" "/analyze" "$context_data" "帶上下文的文本分析"

    local no_context_data='{"text": "我不同意你的看法"}'
    make_request "POST" "/analyze" "$no_context_data" "無上下文的同樣文本分析 (對比)"
}

# 8. 長文本處理
long_text_analysis() {
    print_header "8. 長文本處理測試 (Long Text Analysis)"

    local long_text='{
        "text": "這是一段較長的文本，用來測試 API 處理長文本的能力。在現代社會中，網路霸凌是一個嚴重的問題，它會對受害者造成心理創傷。我們需要建立一個更加友善和包容的網路環境。每個人都應該受到尊重，不論其背景、外表或觀點如何。透過教育和技術手段，我們可以減少網路上的有害內容，創造一個更好的數位世界。讓我們一起努力，讓網路成為一個充滿正能量的空間。"
    }'
    make_request "POST" "/analyze" "$long_text" "分析長文本 (測試處理能力與效能)"
}

# 9. 多語言混合測試
mixed_language_test() {
    print_header "9. 多語言混合測試 (Mixed Language Test)"

    local mixed_text='{"text": "今天天氣很好 nice weather 😊 とても良い天気ですね！"}'
    make_request "POST" "/analyze" "$mixed_text" "分析中英日混合文本"

    local emoji_text='{"text": "你好煩啊 😤 真的很討厭你！🤬"}'
    make_request "POST" "/analyze" "$emoji_text" "分析包含表情符號的文本"
}

# 10. 錯誤處理測試
error_handling_tests() {
    print_header "10. 錯誤處理測試 (Error Handling Tests)"

    # 空文本測試
    local empty_text='{"text": ""}'
    make_request "POST" "/analyze" "$empty_text" "測試空文本錯誤處理"

    # 文本過長測試
    local long_string=$(printf 'a%.0s' {1..1500})  # 1500 個字符
    local too_long_text="{\"text\": \"$long_string\"}"
    make_request "POST" "/analyze" "$too_long_text" "測試文本過長錯誤處理"

    # 無效 JSON 測試
    print_step "測試無效 JSON 格式"
    curl -s -w "\n%{http_code}\n" \
        -H "Authorization: Bearer $API_KEY" \
        -H "Content-Type: application/json" \
        -d '{"text": "測試", "invalid": }' \
        "$API_URL/analyze"
    echo

    # 無效 API 密鑰測試
    print_step "測試無效 API 密鑰"
    curl -s -w "\n%{http_code}\n" \
        -H "Authorization: Bearer invalid_api_key" \
        -H "Content-Type: application/json" \
        -d '{"text": "測試無效認證"}' \
        "$API_URL/analyze"
    echo
}

# 11. 效能測試
performance_test() {
    print_header "11. 效能測試 (Performance Test)"

    print_step "測試並行請求處理能力 (5 個並行請求)"

    local test_text='{"text": "效能測試文本 - 當前時間: '$(date +%s)'"}'

    # 並行發送 5 個請求
    for i in {1..5}; do
        (
            start_time=$(date +%s%N)
            response=$(curl -s -w "\n%{http_code}\n" \
                -H "Authorization: Bearer $API_KEY" \
                -H "Content-Type: application/json" \
                -d "$test_text" \
                "$API_URL/analyze")
            end_time=$(date +%s%N)

            duration=$(( (end_time - start_time) / 1000000 ))  # 轉換為毫秒
            http_code=$(echo "$response" | tail -n1)

            if [[ "$http_code" =~ ^2[0-9][0-9]$ ]]; then
                print_success "請求 $i 成功 - 耗時: ${duration}ms"
            else
                print_error "請求 $i 失敗 - HTTP $http_code"
            fi
        ) &
    done

    # 等待所有背景進程完成
    wait
    print_success "並行測試完成"
}

# 12. 監控端點測試
monitoring_endpoints() {
    print_header "12. 監控端點測試 (Monitoring Endpoints)"

    make_request "GET" "/metrics" "" "取得 API 效能指標"
    make_request "GET" "/model-info" "" "取得模型詳細資訊"
}

# 13. LINE Bot Webhook 測試
line_webhook_test() {
    print_header "13. LINE Bot Webhook 測試 (LINE Bot Webhook Test)"

    print_warning "注意：此測試需要正確的 LINE Channel Secret 進行簽名驗證"

    local webhook_data='{
        "events": [
            {
                "type": "message",
                "message": {
                    "type": "text",
                    "text": "你這個笨蛋！",
                    "id": "test_message_id"
                },
                "source": {
                    "type": "user",
                    "userId": "U1234567890abcdef"
                },
                "replyToken": "test_reply_token",
                "timestamp": '$(date +%s000)'
            }
        ]
    }'

    print_step "測試 LINE Webhook（無簽名）"
    curl -s -w "\n%{http_code}\n" \
        -H "Content-Type: application/json" \
        -H "User-Agent: LineBotWebhook/2.0" \
        -d "$webhook_data" \
        "$API_URL/webhook"
    echo
}

# 14. 批次分析模擬
batch_analysis_simulation() {
    print_header "14. 批次分析模擬 (Batch Analysis Simulation)"

    print_step "模擬批次分析 - 連續發送多個不同類型的文本"

    local texts=(
        '{"text": "你好，很高興認識你！"}'
        '{"text": "今天的會議取消了嗎？"}'
        '{"text": "你這個白痴！"}'
        '{"text": "心情好沮喪..."}'
        '{"text": "謝謝你的幫助！"}'
    )

    for i in "${!texts[@]}"; do
        echo -e "\n${BLUE}--- 批次項目 $((i+1)) ---${NC}"
        make_request "POST" "/analyze" "${texts[$i]}" "分析文本 $((i+1))"
        sleep 1  # 避免觸發限流
    done
}

# 15. 完整測試套件
run_complete_test_suite() {
    print_header "🧪 CyberPuppy API 完整測試套件"
    echo -e "${GREEN}開始執行完整的 API 測試...${NC}\n"

    # 檢查配置
    print_step "檢查配置..."
    echo "API URL: $API_URL"
    echo "API Key: ${API_KEY:0:10}..."
    echo

    # 執行所有測試
    health_check
    api_info
    basic_analysis
    toxicity_detection
    role_identification
    emotion_analysis
    context_analysis
    long_text_analysis
    mixed_language_test
    error_handling_tests
    performance_test
    monitoring_endpoints
    batch_analysis_simulation

    print_header "🎉 測試套件執行完成"
    print_success "所有測試已完成，請檢查上方輸出結果"
}

# 使用說明
show_usage() {
    echo -e "${CYAN}CyberPuppy API cURL 範例腳本${NC}"
    echo -e "${CYAN}==============================${NC}\n"
    echo "使用方式:"
    echo "  $0 [測試函數名稱]"
    echo
    echo "可用的測試函數:"
    echo "  health_check              - 系統健康檢查"
    echo "  api_info                 - API 基本資訊"
    echo "  basic_analysis           - 基本文本分析"
    echo "  toxicity_detection       - 毒性內容檢測"
    echo "  role_identification      - 角色識別測試"
    echo "  emotion_analysis         - 情緒分析測試"
    echo "  context_analysis         - 上下文分析"
    echo "  long_text_analysis       - 長文本處理"
    echo "  mixed_language_test      - 多語言混合測試"
    echo "  error_handling_tests     - 錯誤處理測試"
    echo "  performance_test         - 效能測試"
    echo "  monitoring_endpoints     - 監控端點測試"
    echo "  line_webhook_test        - LINE Bot Webhook 測試"
    echo "  batch_analysis_simulation - 批次分析模擬"
    echo "  run_complete_test_suite  - 完整測試套件"
    echo
    echo "範例:"
    echo "  $0 health_check"
    echo "  $0 basic_analysis"
    echo "  $0 run_complete_test_suite"
    echo
    echo "環境變數:"
    echo "  CYBERPUPPY_API_KEY - 設定 API 密鑰"
    echo "  CYBERPUPPY_API_URL - 設定 API 端點 (預設: http://localhost:8000)"
    echo
}

# 主程式
main() {
    # 檢查依賴
    check_dependencies

    # 檢查 API 密鑰
    if [ "$API_KEY" = "cp_your_api_key_here" ]; then
        print_warning "請設定正確的 API 密鑰："
        echo "export CYBERPUPPY_API_KEY=\"cp_your_actual_api_key\""
        echo
    fi

    # 根據參數執行對應測試
    case "${1:-help}" in
        "health_check")
            health_check
            ;;
        "api_info")
            api_info
            ;;
        "basic_analysis")
            basic_analysis
            ;;
        "toxicity_detection")
            toxicity_detection
            ;;
        "role_identification")
            role_identification
            ;;
        "emotion_analysis")
            emotion_analysis
            ;;
        "context_analysis")
            context_analysis
            ;;
        "long_text_analysis")
            long_text_analysis
            ;;
        "mixed_language_test")
            mixed_language_test
            ;;
        "error_handling_tests")
            error_handling_tests
            ;;
        "performance_test")
            performance_test
            ;;
        "monitoring_endpoints")
            monitoring_endpoints
            ;;
        "line_webhook_test")
            line_webhook_test
            ;;
        "batch_analysis_simulation")
            batch_analysis_simulation
            ;;
        "run_complete_test_suite")
            run_complete_test_suite
            ;;
        "help"|*)
            show_usage
            ;;
    esac
}

# 執行主程式
main "$@"