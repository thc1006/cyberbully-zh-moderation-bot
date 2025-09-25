#!/bin/bash
# CyberPuppy Docker Deployment Script

set -e

echo "🚀 CyberPuppy Docker Deployment Script"
echo "======================================"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Functions
log_info() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

log_warn() {
    echo -e "${YELLOW}[WARN]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Check prerequisites
check_prerequisites() {
    log_info "Checking prerequisites..."

    if ! command -v docker &> /dev/null; then
        log_error "Docker is not installed!"
        exit 1
    fi

    if ! command -v docker compose &> /dev/null && ! command -v docker-compose &> /dev/null; then
        log_error "Docker Compose is not installed!"
        exit 1
    fi

    log_info "Prerequisites check passed ✓"
}

# Setup environment
setup_environment() {
    log_info "Setting up environment..."

    if [ ! -f .env ]; then
        log_info "Creating .env file from template..."
        cp .env.example .env
        log_warn "Please edit .env file to set your configuration"
    else
        log_info ".env file already exists"
    fi
}

# Download models and data
download_resources() {
    log_info "Checking models and data..."

    # Check if models exist
    if [ ! -f "models/macbert_base_demo/best.ckpt" ] || [ ! -f "models/toxicity_only_demo/best.ckpt" ]; then
        log_warn "Models not found. Downloading..."
        python scripts/download_datasets.py
    else
        log_info "Models already exist ✓"
    fi

    # Check if data exists
    if [ ! -d "data/processed/unified" ] || [ ! -f "data/processed/unified/train_unified.json" ]; then
        log_warn "Processed data not found. Processing..."
        python scripts/create_unified_training_data_v2.py
    else
        log_info "Processed data already exists ✓"
    fi
}

# Build Docker images
build_images() {
    log_info "Building Docker images..."

    docker compose build --no-cache api

    # Optional: build bot if needed
    if [[ "$1" == "with-bot" ]]; then
        docker compose build bot
    fi

    log_info "Docker images built successfully ✓"
}

# Start services
start_services() {
    log_info "Starting services..."

    if [[ "$1" == "with-bot" ]]; then
        docker compose --profile with-bot up -d
    elif [[ "$1" == "with-cache" ]]; then
        docker compose --profile with-cache up -d
    elif [[ "$1" == "production" ]]; then
        docker compose --profile production --profile with-cache up -d
    else
        docker compose up -d api
    fi

    log_info "Services started ✓"
}

# Wait for services to be ready
wait_for_services() {
    log_info "Waiting for services to be ready..."

    # Wait for API to be healthy
    for i in {1..30}; do
        if curl -f http://localhost:8000/healthz &> /dev/null; then
            log_info "API service is ready ✓"
            break
        fi
        echo -n "."
        sleep 2
    done
}

# Show status
show_status() {
    log_info "Service Status:"
    docker compose ps

    echo ""
    log_info "Service URLs:"
    echo "  - API: http://localhost:8000"
    echo "  - API Docs: http://localhost:8000/docs"
    echo "  - Health Check: http://localhost:8000/healthz"

    if [[ "$1" == "with-bot" ]]; then
        echo "  - LINE Bot: http://localhost:5000"
    fi
}

# Main execution
main() {
    echo ""
    check_prerequisites
    setup_environment
    download_resources
    build_images "$1"
    start_services "$1"
    wait_for_services
    show_status "$1"

    echo ""
    log_info "Deployment completed successfully! 🎉"
    echo ""
    echo "Quick test:"
    echo '  curl -X POST http://localhost:8000/v1/analyze \'
    echo '    -H "Content-Type: application/json" \'
    echo '    -d '"'"'{"text": "測試文本"}'"'"
    echo ""
}

# Parse arguments
PROFILE="default"
if [ "$1" == "--with-bot" ]; then
    PROFILE="with-bot"
elif [ "$1" == "--with-cache" ]; then
    PROFILE="with-cache"
elif [ "$1" == "--production" ]; then
    PROFILE="production"
fi

# Run main function
main $PROFILE