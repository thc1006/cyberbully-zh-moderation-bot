# CyberPuppy API 文件目錄
# CyberPuppy API Documentation Index

## 📚 文件概覽

本目錄包含 CyberPuppy API 的完整技術文件，涵蓋 API 使用、安全政策、部署指南、錯誤處理等所有面向。

## 🗂️ 文件結構

### 核心文件 (Core Documentation)

| 檔案 | 描述 | 適用對象 |
|------|------|----------|
| **[API.md](./API.md)** | 完整 API 使用指南與範例 | 開發者 |
| **[openapi.yaml](./openapi.yaml)** | OpenAPI 3.0 規格檔案 | 開發者、工具 |
| **[SECURITY.md](./SECURITY.md)** | 安全政策與隱私保護指南 | 所有用戶 |
| **[ERROR_CODES.md](./ERROR_CODES.md)** | 錯誤代碼與故障排除指南 | 開發者、運維 |
| **[DEPLOYMENT.md](./DEPLOYMENT.md)** | 部署與配置完整指南 | 運維、DevOps |

### 程式碼範例 (Code Examples)

| 檔案 | 語言 | 描述 |
|------|------|------|
| **[examples/python_client.py](./examples/python_client.py)** | Python | 完整 Python SDK 與使用範例 |
| **[examples/javascript_client.js](./examples/javascript_client.js)** | JavaScript | Node.js 與瀏覽器客戶端 |
| **[examples/curl_examples.sh](./examples/curl_examples.sh)** | Bash | 完整 cURL 測試腳本 |

### 測試工具 (Testing Tools)

| 檔案 | 類型 | 描述 |
|------|------|------|
| **[CyberPuppy_API.postman_collection.json](./CyberPuppy_API.postman_collection.json)** | Postman | API 測試集合與自動化測試 |

## 🚀 快速開始指南

### 1. 基本設定

```bash
# 設定 API 密鑰
export CYBERPUPPY_API_KEY="cp_your_api_key_here"

# 設定 API 端點（可選，預設為 localhost）
export CYBERPUPPY_API_URL="https://api.cyberpuppy.ai"
```

### 2. 快速測試

```bash
# 健康檢查
curl https://api.cyberpuppy.ai/healthz

# 基本文本分析
curl -X POST "https://api.cyberpuppy.ai/analyze" \
  -H "Authorization: Bearer $CYBERPUPPY_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{"text": "你好，今天天氣很好！"}'
```

### 3. 使用客戶端 SDK

#### Python
```python
from cyberpuppy_client import CyberPuppyClient, ClientConfig

config = ClientConfig(api_key="cp_your_api_key_here")
client = CyberPuppyClient(config)

result = await client.analyze_text("你好，今天天氣很好！")
print(f"情緒: {result.emotion}, 毒性: {result.toxicity}")
```

#### JavaScript
```javascript
const client = new CyberPuppyClient({
  apiKey: 'cp_your_api_key_here'
});

const result = await client.analyzeText('你好，今天天氣很好！');
console.log(`情緒: ${result.emotion}, 毒性: ${result.toxicity}`);
```

## 📋 使用情境指南

### 🎯 開發者整合

1. **Web 應用整合**
   - 閱讀 [API.md](./API.md) 了解端點功能
   - 使用 [Python](./examples/python_client.py) 或 [JavaScript](./examples/javascript_client.js) SDK
   - 參考錯誤處理：[ERROR_CODES.md](./ERROR_CODES.md)

2. **聊天機器人整合**
   - LINE Bot 範例請參考 [API.md#LINE Bot Webhook](./API.md#line-bot-webhook-端點)
   - 使用 [Postman 集合](./CyberPuppy_API.postman_collection.json) 測試 Webhook

3. **內容管理系統**
   - 批次分析範例：[curl_examples.sh](./examples/curl_examples.sh)
   - 效能考量請參考 [DEPLOYMENT.md](./DEPLOYMENT.md)

### 🔒 安全與合規

1. **API 密鑰管理**
   - 詳細安全政策：[SECURITY.md](./SECURITY.md)
   - 認證與授權機制
   - 限流與 DDoS 防護

2. **隱私保護**
   - GDPR 合規說明
   - 資料處理透明度
   - PII 遮蔽機制

### 🚀 部署與運維

1. **生產部署**
   - Docker 容器化：[DEPLOYMENT.md#Docker](./DEPLOYMENT.md#docker-容器部署)
   - Kubernetes 部署：[DEPLOYMENT.md#Kubernetes](./DEPLOYMENT.md#kubernetes-部署)
   - 雲端平台：AWS、GCP、Azure

2. **監控與維護**
   - 健康檢查端點設定
   - 日誌與監控配置
   - 效能調優指南

## 🔍 API 功能特色

### 多任務分析能力
- **毒性偵測**: none | toxic | severe
- **霸凌識別**: none | harassment | threat
- **角色分析**: none | perpetrator | victim | bystander
- **情緒分類**: positive | neutral | negative (0-4 強度)

### 高可解釋性
- **詞彙重要性**: Integrated Gradients (IG) 分析
- **信心度評分**: 模型預測信心度
- **上下文理解**: 對話歷史與情境分析

### 企業級特性
- **隱私保護**: 不儲存原始文本，僅記錄雜湊值
- **高可用性**: 99.9% SLA 保證
- **彈性擴展**: 支援高並發與負載均衡
- **即時處理**: 低延遲回應 (<200ms 平均)

## 📊 效能指標

| 指標 | 數值 | 說明 |
|------|------|------|
| **平均回應時間** | <200ms | 一般文本分析 |
| **吞吐量** | 1000+ req/min | Premium 方案 |
| **準確率** | >95% | 中文毒性偵測 |
| **可用性** | 99.9% | SLA 保證 |

## 🛠️ 測試與除錯

### 自動化測試
```bash
# 執行完整測試套件
bash docs/examples/curl_examples.sh run_complete_test_suite

# 匯入 Postman 集合進行測試
# 1. 開啟 Postman
# 2. 匯入 docs/CyberPuppy_API.postman_collection.json
# 3. 設定環境變數 {{api_key}}
# 4. 執行測試集合
```

### 除錯工具
- **健康檢查**: `GET /healthz` - 檢查系統狀態
- **模型資訊**: `GET /model-info` - 檢查模型載入狀態
- **效能指標**: `GET /metrics` - 檢查 API 效能數據

### 常見問題排解
1. **認證失敗**: 檢查 API 密鑰格式與有效性
2. **限流錯誤**: 調整請求頻率或升級方案
3. **模型錯誤**: 檢查模型載入狀態與資源使用

## 📞 技術支援

### 聯絡方式
- **技術支援**: support@cyberpuppy.ai
- **問題回報**: 使用 [GitHub Issues](https://github.com/cyberpuppy/api-issues)
- **緊急問題**: 24小時內回應（企業用戶）

### 支援資源
- **API 狀態**: https://status.cyberpuppy.ai
- **社群討論**: https://community.cyberpuppy.ai
- **更新通知**: 訂閱 status-updates@cyberpuppy.ai

### 服務等級
| 方案 | 回應時間 | 支援管道 | SLA |
|------|----------|----------|-----|
| **Basic** | 48小時 | Email | 99% |
| **Premium** | 24小時 | Email + 即時聊天 | 99.5% |
| **Enterprise** | 2小時 | 專屬支援 + 電話 | 99.9% |

## 🔄 版本更新

### 目前版本: v1.0.0 (2024-12-30)

#### 新功能
- ✅ 中文毒性與霸凌偵測
- ✅ 多任務學習模型
- ✅ Integrated Gradients 可解釋性
- ✅ LINE Bot 整合支援
- ✅ 企業級安全與隱私保護

#### 即將推出
- 🔜 批次分析 API (v1.1.0)
- 🔜 自定義模型訓練 (v1.2.0)
- 🔜 多語言支援擴展 (v1.3.0)

### 更新通知
訂閱更新通知以獲得最新功能與安全更新：
```bash
curl -X POST "https://api.cyberpuppy.ai/subscribe" \
  -H "Content-Type: application/json" \
  -d '{"email": "your-email@example.com"}'
```

## 📜 授權與合規

### 開源授權
本專案採用 MIT 授權，詳見 [LICENSE](../LICENSE) 檔案。

### 合規認證
- **ISO 27001**: 資訊安全管理系統
- **SOC 2 Type II**: 安全性與可用性
- **GDPR**: 歐盟一般資料保護規則合規

### 使用條款
使用本 API 即表示同意我們的[服務條款](https://cyberpuppy.ai/terms)與[隱私政策](https://cyberpuppy.ai/privacy)。

---

## 🌟 貢獻與回饋

我們歡迎社群的回饋與貢獻！

### 如何貢獻
1. **回報問題**: 使用 GitHub Issues 回報 bug 或提出功能建議
2. **改善文件**: 提交 Pull Request 改善文件內容
3. **分享使用案例**: 與我們分享你的整合經驗

### 回饋管道
- **功能建議**: feature-requests@cyberpuppy.ai
- **使用案例分享**: community@cyberpuppy.ai
- **合作夥伴**: partnerships@cyberpuppy.ai

---

**最後更新**: 2024-12-30
**文件版本**: v1.0.0
**維護團隊**: CyberPuppy API Documentation Team

*如有任何問題或建議，請隨時聯絡我們！* 📧