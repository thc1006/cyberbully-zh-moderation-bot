#!/bin/bash
# CyberPuppy Docker Deployment Script for Linux/macOS
# 中文網路霸凌防治系統 Docker 部署腳本

set -e  # Exit on any error

echo "================================"
echo "CyberPuppy Docker 部署測試"
echo "================================"
echo

# Function to print colored output
print_success() {
    echo -e "\033[32m✅ $1\033[0m"
}

print_error() {
    echo -e "\033[31m❌ $1\033[0m"
}

print_warning() {
    echo -e "\033[33m⚠️  $1\033[0m"
}

print_info() {
    echo -e "\033[34mℹ️  $1\033[0m"
}

# Check Docker
echo "[1/7] 檢查 Docker 服務狀態..."
if ! command -v docker &> /dev/null; then
    print_error "Docker 未安裝"
    exit 1
fi

if ! docker version &> /dev/null; then
    print_error "Docker 服務未運行，請先啟動 Docker 服務"
    exit 1
fi
print_success "Docker 服務正常運行"
echo

# Check Docker Compose
echo "[2/7] 檢查 Docker Compose..."
if ! command -v docker-compose &> /dev/null; then
    print_error "Docker Compose 不可用"
    exit 1
fi
print_success "Docker Compose 可用"
echo

# Check required files
echo "[3/7] 檢查必要檔案..."
required_files=("Dockerfile.api" "Dockerfile.bot" "docker-compose.yml" "requirements.txt")
for file in "${required_files[@]}"; do
    if [[ ! -f "$file" ]]; then
        print_error "缺少 $file"
        exit 1
    fi
done
print_success "所有必要檔案都存在"
echo

# Check environment file
echo "[4/7] 檢查環境變數設定..."
if [[ ! -f "configs/docker/.env" ]]; then
    print_warning ".env 檔案不存在，將複製範例檔案"
    if [[ -f "configs/docker/.env.example" ]]; then
        cp "configs/docker/.env.example" "configs/docker/.env"
        print_success "已建立 .env 檔案（請記得填入實際的 LINE Bot 設定）"
    else
        print_error ".env.example 檔案也不存在"
        exit 1
    fi
else
    print_success ".env 檔案已存在"
fi
echo

# Create necessary directories
echo "[5/7] 建立必要目錄..."
mkdir -p logs models data
print_success "目錄結構準備完成"
echo

# Validate Docker Compose file
echo "[6/7] 驗證 Docker Compose 檔案..."
if ! docker-compose config &> /dev/null; then
    print_error "docker-compose.yml 檔案格式錯誤"
    print_info "請檢查 YAML 語法"
    exit 1
fi
print_success "Docker Compose 檔案格式正確"
echo

# Build images
echo "[7/7] 建置 Docker 映像（不啟動服務）..."
echo "建置 API 映像..."
if ! docker-compose build --no-cache api; then
    print_error "API 映像建置失敗"
    exit 1
fi
print_success "API 映像建置成功"

echo "建置 Bot 映像..."
if ! docker-compose build --no-cache bot; then
    print_error "Bot 映像建置失敗"
    exit 1
fi
print_success "Bot 映像建置成功"
echo

echo "========================================"
echo "🎉 Docker 部署測試完成！"
echo "========================================"
echo
echo "下一步："
echo "1. 編輯 configs/docker/.env 填入正確的 LINE Bot 設定"
echo "2. 執行: docker-compose up -d"
echo "3. 測試健康檢查端點:"
echo "   - API: http://localhost:8000/healthz"
echo "   - Bot: http://localhost:8080/health"
echo

# Optional: Start services if requested
read -p "是否要立即啟動服務？ (y/N): " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    print_info "啟動 Docker 服務..."
    docker-compose up -d
    
    echo "等待服務啟動..."
    sleep 10
    
    echo "檢查服務狀態："
    docker-compose ps
    
    echo
    print_info "測試健康檢查端點..."
    
    # Test API health
    if curl -f http://localhost:8000/healthz &> /dev/null; then
        print_success "API 服務健康檢查通過"
    else
        print_warning "API 服務健康檢查失敗，請檢查日誌: docker-compose logs api"
    fi
    
    # Test Bot health
    if curl -f http://localhost:8080/health &> /dev/null; then
        print_success "Bot 服務健康檢查通過"
    else
        print_warning "Bot 服務健康檢查失敗，請檢查日誌: docker-compose logs bot"
    fi
    
    echo
    print_info "查看日誌: docker-compose logs -f"
    print_info "停止服務: docker-compose down"
fi