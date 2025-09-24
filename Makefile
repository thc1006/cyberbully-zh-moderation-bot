# CyberPuppy Makefile
# 常用開發指令集合

.PHONY: help install test lint format type-check security clean api bot docker-up docker-down

# 預設 Python 與虛擬環境
PYTHON := python3
VENV := venv
PIP := $(VENV)/bin/pip
PYTHON_BIN := $(VENV)/bin/python

# 顏色輸出
RED := \033[0;31m
GREEN := \033[0;32m
YELLOW := \033[1;33m
BLUE := \033[0;34m
NC := \033[0m # No Color

#################################
# 說明
#################################
help: ## 顯示此說明訊息
	@echo "$(GREEN)CyberPuppy Development Commands$(NC)"
	@echo ""
	@echo "$(YELLOW)Available targets:$(NC)"
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | \
		awk 'BEGIN {FS = ":.*?## "}; {printf "  $(BLUE)%-20s$(NC) %s\n", $$1, $$2}'
	@echo ""
	@echo "$(YELLOW)Quick Start:$(NC)"
	@echo "  make install    - 安裝所有依賴"
	@echo "  make test       - 執行測試"
	@echo "  make api        - 啟動 API 服務"
	@echo "  make bot        - 啟動 LINE Bot"

#################################
# 環境設定
#################################
venv: ## 建立虛擬環境
	@echo "$(GREEN)建立虛擬環境...$(NC)"
	$(PYTHON) -m venv $(VENV)
	$(PIP) install --upgrade pip

install: venv ## 安裝所有依賴
	@echo "$(GREEN)安裝專案依賴...$(NC)"
	$(PIP) install -r requirements.txt
	@echo "$(GREEN)安裝開發依賴...$(NC)"
	$(PIP) install -r requirements-dev.txt
	@echo "$(BLUE)安裝完成！$(NC)"

install-dev: venv ## 僅安裝開發依賴
	@echo "$(GREEN)安裝開發依賴...$(NC)"
	$(PIP) install -r requirements-dev.txt

update: ## 更新所有依賴
	@echo "$(GREEN)更新依賴套件...$(NC)"
	$(PIP) install --upgrade -r requirements.txt
	$(PIP) install --upgrade -r requirements-dev.txt

freeze: ## 凍結當前依賴版本
	@echo "$(GREEN)凍結依賴版本...$(NC)"
	$(PIP) freeze > requirements-freeze.txt

#################################
# 程式碼品質
#################################
format: ## 格式化程式碼
	@echo "$(GREEN)格式化程式碼...$(NC)"
	@echo "$(YELLOW)執行 Black...$(NC)"
	$(VENV)/bin/black src/ tests/ api/ bot/
	@echo "$(YELLOW)執行 isort...$(NC)"
	$(VENV)/bin/isort src/ tests/ api/ bot/
	@echo "$(BLUE)格式化完成！$(NC)"

lint: ## 執行程式碼檢查
	@echo "$(GREEN)執行程式碼檢查...$(NC)"
	@echo "$(YELLOW)Flake8 檢查...$(NC)"
	-$(VENV)/bin/flake8 src/ tests/ api/ bot/ --config=.flake8
	@echo "$(YELLOW)Pylint 檢查...$(NC)"
	-$(VENV)/bin/pylint src/cyberpuppy --rcfile=.pylintrc
	@echo "$(YELLOW)Docstring 檢查...$(NC)"
	-$(VENV)/bin/pydocstyle src/ --config=.pydocstyle
	@echo "$(BLUE)檢查完成！$(NC)"

type-check: ## 執行型別檢查
	@echo "$(GREEN)執行型別檢查...$(NC)"
	$(VENV)/bin/mypy src/ api/ bot/ --config-file=mypy.ini
	@echo "$(BLUE)型別檢查完成！$(NC)"

security: ## 執行安全掃描
	@echo "$(GREEN)執行安全掃描...$(NC)"
	@echo "$(YELLOW)Bandit 掃描...$(NC)"
	-$(VENV)/bin/bandit -r src/ api/ bot/ -ll
	@echo "$(YELLOW)Safety 檢查...$(NC)"
	-$(VENV)/bin/safety check
	@echo "$(BLUE)安全掃描完成！$(NC)"

quality: lint type-check security ## 執行所有品質檢查

#################################
# 測試
#################################
test: ## 執行所有測試
	@echo "$(GREEN)執行測試套件...$(NC)"
	$(VENV)/bin/pytest tests/ -v \
		--cov=src/cyberpuppy \
		--cov-report=term-missing \
		--cov-report=html

test-unit: ## 僅執行單元測試
	@echo "$(GREEN)執行單元測試...$(NC)"
	$(VENV)/bin/pytest tests/ -v -m "not integration"

test-integration: ## 僅執行整合測試
	@echo "$(GREEN)執行整合測試...$(NC)"
	$(VENV)/bin/pytest tests/integration/ -v

test-coverage: ## 執行測試並生成覆蓋率報告
	@echo "$(GREEN)執行測試並生成覆蓋率報告...$(NC)"
	$(VENV)/bin/pytest tests/ \
		--cov=src/cyberpuppy \
		--cov-report=term-missing \
		--cov-report=html \
		--cov-report=xml
	@echo "$(BLUE)覆蓋率報告已生成於 htmlcov/index.html$(NC)"

test-watch: ## 監控模式執行測試
	@echo "$(GREEN)啟動測試監控模式...$(NC)"
	$(VENV)/bin/pytest-watch tests/ -- -v

#################################
# 服務啟動
#################################
api: ## 啟動 API 服務
	@echo "$(GREEN)啟動 CyberPuppy API...$(NC)"
	cd api && $(PYTHON_BIN) -m uvicorn app:app --reload --host 0.0.0.0 --port 8000

api-prod: ## 生產模式啟動 API
	@echo "$(GREEN)啟動 CyberPuppy API (生產模式)...$(NC)"
	cd api && $(PYTHON_BIN) -m uvicorn app:app --host 0.0.0.0 --port 8000 --workers 4

bot: ## 啟動 LINE Bot
	@echo "$(GREEN)啟動 CyberPuppy LINE Bot...$(NC)"
	cd bot && $(PYTHON_BIN) line_bot.py

bot-prod: ## 生產模式啟動 Bot
	@echo "$(GREEN)啟動 CyberPuppy LINE Bot (生產模式)...$(NC)"
	cd bot && $(PYTHON_BIN) -m uvicorn line_bot:app --host 0.0.0.0 --port 8080 --workers 2

dev: ## 同時啟動 API 和 Bot（開發模式）
	@echo "$(GREEN)啟動開發環境...$(NC)"
	make -j 2 api bot

#################################
# Docker
#################################
docker-build: ## 建構 Docker 映像
	@echo "$(GREEN)建構 Docker 映像...$(NC)"
	docker-compose build

docker-up: ## 啟動 Docker 容器
	@echo "$(GREEN)啟動 Docker 容器...$(NC)"
	docker-compose up -d
	@echo "$(BLUE)服務已啟動！$(NC)"
	@echo "  API: http://localhost:8000"
	@echo "  Bot: http://localhost:8080"

docker-down: ## 停止 Docker 容器
	@echo "$(YELLOW)停止 Docker 容器...$(NC)"
	docker-compose down

docker-logs: ## 查看 Docker 日誌
	docker-compose logs -f

docker-clean: ## 清理 Docker 資源
	@echo "$(YELLOW)清理 Docker 資源...$(NC)"
	docker-compose down -v
	docker system prune -f

#################################
# 資料處理
#################################
download-data: ## 下載訓練資料集
	@echo "$(GREEN)下載資料集...$(NC)"
	$(PYTHON_BIN) scripts/download_datasets.py

process-data: ## 處理與清理資料
	@echo "$(GREEN)處理資料...$(NC)"
	$(PYTHON_BIN) scripts/clean_normalize.py

train: ## 執行模型訓練
	@echo "$(GREEN)開始訓練模型...$(NC)"
	$(PYTHON_BIN) train.py --config configs/training_config.yaml

evaluate: ## 執行模型評估
	@echo "$(GREEN)評估模型...$(NC)"
	$(PYTHON_BIN) -m src.cyberpuppy.eval.evaluate

#################################
# 文檔
#################################
docs: ## 生成文檔
	@echo "$(GREEN)生成文檔...$(NC)"
	cd docs && make html
	@echo "$(BLUE)文檔已生成於 docs/_build/html/index.html$(NC)"

docs-serve: ## 啟動文檔伺服器
	@echo "$(GREEN)啟動文檔伺服器...$(NC)"
	cd docs/_build/html && python -m http.server 8888

notebook: ## 啟動 Jupyter Notebook
	@echo "$(GREEN)啟動 Jupyter Notebook...$(NC)"
	$(VENV)/bin/jupyter notebook notebooks/

#################################
# 部署
#################################
deploy-check: ## 部署前檢查
	@echo "$(GREEN)執行部署前檢查...$(NC)"
	make lint
	make type-check
	make security
	make test
	@echo "$(BLUE)部署檢查通過！$(NC)"

deploy-staging: ## 部署到測試環境
	@echo "$(GREEN)部署到測試環境...$(NC)"
	# 這裡添加部署到測試環境的指令
	@echo "$(BLUE)已部署到測試環境$(NC)"

deploy-prod: deploy-check ## 部署到生產環境
	@echo "$(RED)部署到生產環境...$(NC)"
	@read -p "確定要部署到生產環境嗎? [y/N] " -n 1 -r; \
	echo; \
	if [[ $$REPLY =~ ^[Yy]$$ ]]; then \
		echo "$(GREEN)開始部署...$(NC)"; \
		# 這裡添加部署到生產環境的指令 \
		echo "$(BLUE)已部署到生產環境$(NC)"; \
	else \
		echo "$(YELLOW)取消部署$(NC)"; \
	fi

#################################
# 清理
#################################
clean: ## 清理暫存檔案
	@echo "$(YELLOW)清理暫存檔案...$(NC)"
	find . -type f -name "*.pyc" -delete
	find . -type f -name "*.pyo" -delete
	find . -type f -name "*.pyd" -delete
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name "*.egg-info" -exec rm -rf {} + 2>/dev/null || true
	rm -rf .pytest_cache/
	rm -rf .coverage
	rm -rf htmlcov/
	rm -rf dist/
	rm -rf build/
	@echo "$(BLUE)清理完成！$(NC)"

clean-all: clean ## 完整清理（包含虛擬環境）
	@echo "$(RED)完整清理...$(NC)"
	rm -rf $(VENV)
	rm -rf .mypy_cache/
	rm -rf .ruff_cache/
	@echo "$(BLUE)完整清理完成！$(NC)"

#################################
# 工具安裝
#################################
install-hooks: ## 安裝 Git hooks
	@echo "$(GREEN)安裝 Git hooks...$(NC)"
	$(VENV)/bin/pre-commit install
	@echo "$(BLUE)Git hooks 已安裝$(NC)"

install-tools: ## 安裝開發工具
	@echo "$(GREEN)安裝開發工具...$(NC)"
	$(PIP) install black flake8 pylint mypy bandit safety
	$(PIP) install pytest pytest-cov pytest-watch pytest-benchmark
	$(PIP) install pre-commit pip-licenses
	@echo "$(BLUE)開發工具安裝完成！$(NC)"

#################################
# 快捷指令
#################################
all: install test lint ## 完整建構與測試

ci: quality test ## CI 流程

check: format lint type-check test ## 完整檢查

dev-setup: install install-hooks ## 開發環境設定

.DEFAULT_GOAL := help