# CyberPuppy API 安全與隱私政策
# Security & Privacy Policy

## 安全架構 (Security Architecture)

### 1. 認證與授權 (Authentication & Authorization)

#### API 密鑰認證
```http
Authorization: Bearer cp_1234567890abcdef
```

**密鑰格式**:
- 前綴: `cp_` (CyberPuppy 識別碼)
- 長度: 32 字符隨機字符串
- 編碼: Base64 安全字符集

**密鑰管理**:
- 使用 HTTPS 安全傳輸
- 定期輪換（建議每 90 天）
- 支援撤銷與重新生成
- 異常使用自動警報

#### 權限等級
```yaml
Basic:
  - 30 requests/minute
  - 基礎分析功能
  - 公開文檔存取

Premium:
  - 1000 requests/minute
  - 進階分析功能
  - 優先技術支援
  - 詳細使用統計

Enterprise:
  - 無限制請求
  - 自定義模型
  - 專屬技術支援
  - SLA 保證 99.9%
```

### 2. 傳輸安全 (Transport Security)

#### TLS/SSL 加密
- **最低版本**: TLS 1.2
- **推薦版本**: TLS 1.3
- **加密套件**: AES-256-GCM, ChaCha20-Poly1305
- **憑證**: Let's Encrypt 或 DigiCert EV SSL

#### HTTP 安全標頭
```http
Strict-Transport-Security: max-age=31536000; includeSubDomains
Content-Security-Policy: default-src 'self'
X-Frame-Options: DENY
X-Content-Type-Options: nosniff
X-XSS-Protection: 1; mode=block
Referrer-Policy: strict-origin-when-cross-origin
```

### 3. 輸入驗證與清理 (Input Validation & Sanitization)

#### 文本驗證
```python
# 長度限制
MAX_TEXT_LENGTH = 1000
MAX_CONTEXT_LENGTH = 2000

# 字符過濾
ALLOWED_CHARS = re.compile(r'^[\u4e00-\u9fff\w\s\.,!?;:()\-"\']+$')

# 惡意內容檢測
def validate_input(text: str) -> bool:
    # SQL 注入檢測
    sql_patterns = [r'union\s+select', r'drop\s+table', r'delete\s+from']

    # XSS 檢測
    xss_patterns = [r'<script', r'javascript:', r'on\w+\s*=']

    # 檢查惡意模式
    for pattern in sql_patterns + xss_patterns:
        if re.search(pattern, text, re.IGNORECASE):
            return False

    return True
```

#### PII 遮蔽 (Personal Information Protection)
```python
PII_PATTERNS = [
    r'\d{4}[-\s]?\d{4}[-\s]?\d{4}[-\s]?\d{4}',  # 信用卡號
    r'\b\d{10,11}\b',                            # 手機號碼
    r'\b[A-Z][0-9]{9}\b',                       # 身分證號
    r'\b\w+@\w+\.\w+\b',                        # 電子郵件
    r'\b\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}\b',  # IP 地址
]

def mask_pii(text: str) -> str:
    """自動遮蔽個人識別資訊"""
    for pattern in PII_PATTERNS:
        text = re.sub(pattern, "[MASKED]", text, flags=re.IGNORECASE)
    return text
```

### 4. 限流與 DDoS 防護 (Rate Limiting & DDoS Protection)

#### 多層限流策略
```yaml
Global Limit:
  - 10000 requests/hour per IP
  - 100 concurrent connections per IP

API Key Limit:
  - Basic: 30 requests/minute
  - Premium: 1000 requests/minute
  - Enterprise: 10000 requests/minute

Endpoint Specific:
  - /analyze: 嚴格限流
  - /healthz: 寬鬆限流
  - /webhook: 基於簽名驗證
```

#### 異常偵測
```python
class AnomalyDetector:
    def detect_suspicious_activity(self, request_data):
        """偵測可疑活動"""
        indicators = []

        # 高頻率請求
        if self.request_frequency > self.threshold:
            indicators.append("HIGH_FREQUENCY")

        # 異常請求模式
        if self.detect_bot_pattern(request_data):
            indicators.append("BOT_PATTERN")

        # 惡意載荷
        if self.detect_malicious_payload(request_data):
            indicators.append("MALICIOUS_PAYLOAD")

        return indicators

    def auto_ban(self, ip_address, duration_minutes=60):
        """自動封禁可疑 IP"""
        ban_record = {
            "ip": ip_address,
            "banned_at": datetime.utcnow(),
            "duration": duration_minutes,
            "reason": "Suspicious activity detected"
        }
        self.ban_list.add(ban_record)
```

## 隱私保護 (Privacy Protection)

### 1. 資料最小化原則

#### 不儲存原始文本
```python
def process_text(text: str) -> Dict:
    """處理文本但不儲存原始內容"""
    # 生成雜湊用於日誌
    text_hash = hashlib.sha256(text.encode()).hexdigest()[:16]

    # 進行分析
    result = analyze_text(text)

    # 記錄分析結果（不含原文）
    log_record = {
        "text_hash": text_hash,
        "text_length": len(text),
        "result": result,
        "timestamp": datetime.utcnow()
    }

    # 原始文本在此函數結束後自動清除
    return result
```

#### 資料保留政策
```yaml
Analysis Results:
  retention: 30 days
  purpose: 服務改進與除錯

Request Logs:
  retention: 7 days
  purpose: 安全監控與故障排除

Error Logs:
  retention: 90 days
  purpose: 系統維護與改進

User Sessions:
  retention: 24 hours
  purpose: 上下文分析
```

### 2. 資料處理透明度

#### 資料處理流程
1. **接收**: 透過 HTTPS 接收文本
2. **驗證**: 格式與安全性檢查
3. **遮蔽**: 自動遮蔽 PII 資訊
4. **分析**: AI 模型推理
5. **回應**: 回傳分析結果
6. **清除**: 立即清除原始文本
7. **記錄**: 僅記錄雜湊值與結果

#### 第三方服務整合
```yaml
Google Perspective API (可選):
  purpose: 輔助毒性檢測
  data_shared: 文本內容 (僅在用戶同意時)
  retention: 依 Google 政策

LINE Platform:
  purpose: Messaging Bot 功能
  data_shared: 用戶 ID、訊息內容
  retention: 24 小時

Monitoring Services:
  purpose: 系統監控
  data_shared: 系統指標、錯誤日誌
  retention: 30 天
```

### 3. 用戶權利保障

#### GDPR 合規
- **資料可攜性**: 提供資料匯出功能
- **被遺忘權**: 支援資料刪除請求
- **透明度**: 清楚說明資料處理目的
- **同意機制**: 明確的同意與撤回流程

#### 資料主體權利
```http
# 查詢個人資料
GET /api/user/data?user_id=xxx

# 刪除個人資料
DELETE /api/user/data?user_id=xxx

# 匯出個人資料
GET /api/user/export?user_id=xxx&format=json
```

## 安全監控 (Security Monitoring)

### 1. 威脅偵測

#### 異常請求偵測
```python
class ThreatDetector:
    def __init__(self):
        self.patterns = {
            'sql_injection': [
                r'union\s+select', r'drop\s+table',
                r'insert\s+into', r'update\s+set'
            ],
            'xss_attempt': [
                r'<script.*?>', r'javascript:',
                r'on\w+\s*=', r'eval\('
            ],
            'path_traversal': [
                r'\.\./', r'\.\.\\',
                r'/etc/passwd', r'windows\\system32'
            ]
        }

    def scan_request(self, request_data):
        threats = []
        for threat_type, patterns in self.patterns.items():
            for pattern in patterns:
                if re.search(pattern, str(request_data), re.IGNORECASE):
                    threats.append(threat_type)
        return threats
```

#### 自動回應機制
```python
def handle_security_incident(incident_type, severity, request_info):
    """處理安全事件"""
    if severity == "HIGH":
        # 立即封禁 IP
        ban_ip(request_info.remote_addr, duration_hours=24)

        # 發送警報
        send_alert(f"高風險安全事件: {incident_type}")

        # 記錄詳細資訊
        log_security_incident(incident_type, request_info)

    elif severity == "MEDIUM":
        # 增加監控
        increase_monitoring(request_info.remote_addr)

        # 降低限流閾值
        reduce_rate_limit(request_info.api_key)

    else:
        # 記錄警告
        log_warning(incident_type, request_info)
```

### 2. 日誌與稽核

#### 安全日誌格式
```json
{
  "timestamp": "2024-12-30T10:15:30Z",
  "event_type": "security_incident",
  "severity": "HIGH",
  "incident_type": "sql_injection_attempt",
  "source_ip": "192.168.1.100",
  "user_agent": "Mozilla/5.0...",
  "request_path": "/analyze",
  "request_method": "POST",
  "api_key_hash": "a1b2c3d4...",
  "action_taken": "ip_banned",
  "additional_info": {
    "detected_patterns": ["union select"],
    "request_size": 1024,
    "response_code": 400
  }
}
```

#### 稽核追蹤
```python
class AuditLogger:
    def log_api_access(self, api_key, endpoint, result):
        """記錄 API 存取"""
        audit_record = {
            "timestamp": datetime.utcnow(),
            "api_key_hash": hashlib.sha256(api_key.encode()).hexdigest()[:16],
            "endpoint": endpoint,
            "success": result.success,
            "processing_time": result.processing_time_ms,
            "error_code": result.error_code if not result.success else None
        }
        self.audit_db.insert(audit_record)

    def log_data_access(self, user_id, data_type, action):
        """記錄資料存取"""
        data_record = {
            "timestamp": datetime.utcnow(),
            "user_id_hash": hashlib.sha256(user_id.encode()).hexdigest()[:16],
            "data_type": data_type,
            "action": action,  # read, write, delete, export
            "compliance_check": self.check_gdpr_compliance(action)
        }
        self.audit_db.insert(data_record)
```

## 事件回應計劃 (Incident Response Plan)

### 1. 安全事件分類

#### 嚴重等級定義
```yaml
Critical (P0):
  - 資料洩露
  - 系統完全中斷
  - 大規模 DDoS 攻擊
  response_time: 15 分鐘內

High (P1):
  - API 服務異常
  - 認證系統故障
  - 可疑的大量存取
  response_time: 1 小時內

Medium (P2):
  - 個別 API 錯誤
  - 異常使用模式
  - 小規模攻擊嘗試
  response_time: 4 小時內

Low (P3):
  - 一般性能問題
  - 文檔或介面問題
  response_time: 24 小時內
```

### 2. 回應流程

#### 自動回應
```python
def automated_incident_response(incident):
    """自動事件回應"""
    if incident.severity == "CRITICAL":
        # 啟動緊急模式
        enable_emergency_mode()

        # 通知管理團隊
        notify_emergency_team(incident)

        # 隔離受影響系統
        isolate_affected_systems(incident.affected_components)

        # 啟用備份系統
        activate_backup_systems()

    elif incident.severity == "HIGH":
        # 增強監控
        increase_monitoring_level()

        # 通知技術團隊
        notify_technical_team(incident)

        # 收集詳細日誌
        collect_detailed_logs(incident.time_window)
```

#### 人工介入
```yaml
緊急聯絡人:
  - 安全長 (CISO): security-ciso@cyberpuppy.ai
  - 技術長 (CTO): tech-cto@cyberpuppy.ai
  - 運營經理: ops-manager@cyberpuppy.ai

24/7 待命電話:
  - 主要: +886-2-xxxx-xxxx
  - 備用: +886-9-xxxx-xxxx

外部支援:
  - 資安顧問: partner-security@example.com
  - 法務顧問: legal-counsel@example.com
  - 公關團隊: pr-team@cyberpuppy.ai
```

## 合規認證 (Compliance Certifications)

### 已獲得認證
- **ISO 27001**: 資訊安全管理系統
- **SOC 2 Type II**: 安全性與可用性
- **GDPR**: 歐盟一般資料保護規則合規

### 進行中認證
- **ISO 27018**: 雲端隱私保護
- **PCI DSS**: 支付卡行業資料安全標準
- **HIPAA**: 美國健康保險隱私法案

### 定期稽核
```yaml
內部稽核:
  frequency: 每季一次
  scope: 全系統安全檢查
  next_audit: 2024-03-15

外部稽核:
  frequency: 每年一次
  auditor: Third-party security firm
  next_audit: 2024-06-01

滲透測試:
  frequency: 每半年一次
  scope: API & Web Application
  next_test: 2024-04-01
```

## 安全最佳實踐 (Security Best Practices)

### 開發團隊
1. **安全編碼**: 遵循 OWASP 指南
2. **代碼審查**: 強制性安全審查
3. **依賴檢查**: 定期掃描第三方套件
4. **安全測試**: CI/CD 整合安全測試

### 運營團隊
1. **定期更新**: 及時安裝安全補丁
2. **監控告警**: 24/7 安全監控
3. **備份策略**: 定期備份與恢復測試
4. **存取控制**: 最小權限原則

### 用戶建議
1. **API 密鑰**: 妥善保管，定期更換
2. **HTTPS**: 始終使用 HTTPS 傳輸
3. **輸入驗證**: 客戶端也要進行驗證
4. **錯誤處理**: 不要暴露敏感資訊

---

**聯絡資訊**:
- 安全團隊: security@cyberpuppy.ai
- 隱私問題: privacy@cyberpuppy.ai
- 緊急事件: incident@cyberpuppy.ai

*最後更新: 2024-12-30*