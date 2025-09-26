# LINE Bot 憑證設定完整指南 (2025 最新版)

**文件更新日期**: 2025-09-27
**適用版本**: LINE Messaging API (2025)
**Python SDK 版本**: line-bot-sdk 3.19.0+
**作者**: CyberPuppy 開發團隊

---

## 📋 目錄

1. [前置準備](#前置準備)
2. [第一步：建立 LINE 開發者帳號](#第一步建立-line-開發者帳號)
3. [第二步：建立 Messaging API Channel](#第二步建立-messaging-api-channel)
4. [第三步：取得必要憑證](#第三步取得必要憑證)
5. [第四步：設定 Webhook URL](#第四步設定-webhook-url)
6. [第五步：配置專案環境變數](#第五步配置專案環境變數)
7. [第六步：測試連線](#第六步測試連線)
8. [常見問題與疑難排解](#常見問題與疑難排解)

---

## 🎯 前置準備

### 必要條件
- ✅ **LINE 帳號**（個人帳號即可）
- ✅ **LINE Official Account**（LINE 官方帳號）
- ✅ **可公開存取的 HTTPS 伺服器**（用於 Webhook）
  - 本地開發可使用 **ngrok** 或 **localtunnel**
  - 生產環境建議使用 **Heroku**、**AWS**、**GCP** 等雲端服務
- ✅ **Python 3.13.5** 環境（本專案使用版本）
- ✅ **line-bot-sdk 3.19.0+**

### 重要提醒 (2025 更新)
⚠️ **LINE Notify 將於 2025 年 3 月 31 日停止服務**
請確保使用 **Messaging API** 而非已棄用的 LINE Notify

---

## 第一步：建立 LINE 開發者帳號

### 1.1 註冊 LINE Developers
1. 前往 [LINE Developers Console](https://developers.line.biz/)
2. 點擊右上角 **「Log in」** 或 **「登入」**
3. 使用你的 LINE 帳號登入
4. 首次登入需同意開發者條款

### 1.2 建立 Provider (提供者)
1. 登入後，點擊 **「Create」** 或 **「創建」**
2. 選擇 **「Create a new provider」**
3. 輸入 Provider 名稱（例如：`CyberPuppy`）
   - 此名稱對外不可見，僅供內部管理使用
4. 點擊 **「Create」** 確認建立

---

## 第二步：建立 Messaging API Channel

### 2.1 建立新頻道
1. 在 Provider 頁面中，點擊 **「Create a Messaging API channel」**
2. 填寫頻道資訊：

   ```
   頻道類型：Messaging API
   頻道名稱：CyberPuppy Bot（或你想要的名稱）
   頻道說明：中文網路霸凌防治機器人
   類別：選擇適合的類別（例如：工具與生產力）
   子類別：根據需求選擇
   Email 地址：你的聯絡信箱
   ```

3. 閱讀並同意 **「LINE Official Account Terms of Use」** 和 **「LINE Official Account API Terms of Use」**
4. 點擊 **「Create」** 建立頻道

### 2.2 確認頻道建立成功
- 建立成功後會自動導向到該頻道的設定頁面
- 你會看到頻道的基本資訊和設定選項

---

## 第三步：取得必要憑證

CyberPuppy 需要三個關鍵憑證：
1. **Channel ID** (頻道 ID)
2. **Channel Secret** (頻道密鑰)
3. **Channel Access Token** (頻道存取權杖)

### 3.1 取得 Channel ID 和 Channel Secret

1. 在頻道設定頁面，點擊 **「Basic settings」** 分頁
2. 找到 **「Basic information」** 區塊
3. 複製以下資訊：
   ```
   Channel ID: 一串數字（例如：1234567890）
   Channel secret: 一串英數字組合（例如：a1b2c3d4e5f6g7h8i9j0）
   ```

> 💡 **提示**: 點擊 Channel secret 右側的眼睛圖示可以顯示/隱藏密鑰

### 3.2 簽發 Channel Access Token (長期有效)

#### 2025 年最新方式：使用長效 Token

1. 切換到 **「Messaging API」** 分頁
2. 滾動到 **「Channel access token (long-lived)」** 區塊
3. 點擊 **「Issue」** 按鈕

   ```
   ⚠️ 重要注意事項：
   - Token 僅會在簽發時顯示一次
   - 請立即複製並妥善保存
   - 遺失後需重新簽發（舊 Token 會失效）
   ```

4. 複製簽發的 Token（長度約 100+ 字元）
   ```
   範例格式：
   eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9...（後略）
   ```

5. 建議立即儲存到安全的地方（例如：密碼管理器）

#### Token 管理注意事項

- **長效 Token**: 永久有效，除非手動撤銷或重新簽發
- **短效 Token**: 已棄用，不建議使用
- **安全性**: Token 等同於帳號密碼，切勿公開或提交到 Git

---

## 第四步：設定 Webhook URL

### 4.1 準備 Webhook URL

Webhook URL 是 LINE 伺服器用來傳送訊息事件到你的應用程式的端點。

#### 本地開發環境（使用 ngrok）

1. **安裝 ngrok**:
   ```bash
   # Windows (使用 Chocolatey)
   choco install ngrok

   # 或直接下載：https://ngrok.com/download
   ```

2. **啟動 ngrok 隧道**:
   ```bash
   # 假設你的 Flask 應用運行在 port 8000
   ngrok http 8000
   ```

3. **取得 HTTPS URL**:
   ```
   ngrok 會顯示類似以下的 URL：
   Forwarding: https://abc123.ngrok.io -> http://localhost:8000

   你的 Webhook URL 就是：
   https://abc123.ngrok.io/callback
   ```

#### 生產環境

使用你的伺服器 HTTPS URL：
```
https://your-domain.com/callback
或
https://your-app.herokuapp.com/callback
```

### 4.2 在 LINE Developers Console 設定 Webhook

1. 在 **「Messaging API」** 分頁
2. 找到 **「Webhook settings」** 區塊
3. 點擊 **「Webhook URL」** 的 **「Edit」** 按鈕
4. 輸入你的 Webhook URL：
   ```
   https://abc123.ngrok.io/callback
   ```
5. 點擊 **「Update」** 儲存
6. 開啟 **「Use webhook」** 開關（設為啟用）
7. 點擊 **「Verify」** 按鈕測試連線
   - ✅ **成功**: 顯示 "Success"
   - ❌ **失敗**: 檢查伺服器是否運行、URL 是否正確

### 4.3 進階 Webhook 設定

#### 啟用 Webhook 重送機制 (Redelivery)
1. 在 **「Webhook settings」** 找到 **「Webhook redelivery」**
2. 將開關設為 **啟用**
3. 當 Webhook 失敗時，LINE 會自動重試

#### 設定統計與錯誤通知
- **Webhook statistics**: 可查看 Webhook 呼叫次數和失敗率
- **Error notifications**: 設定當 Webhook 持續失敗時接收通知

---

## 第五步：配置專案環境變數

### 5.1 建立 .env 檔案

在專案根目錄建立 `.env` 檔案（如果還沒有）：

```bash
# LINE Bot Configuration (2025 最新版)

# === LINE Messaging API 憑證 ===
LINE_CHANNEL_ID=你的_Channel_ID
LINE_CHANNEL_SECRET=你的_Channel_Secret
LINE_CHANNEL_ACCESS_TOKEN=你的_Channel_Access_Token

# === Webhook 設定 ===
LINE_WEBHOOK_URL=https://your-domain.com/callback

# === Bot 行為設定 ===
# 是否回覆所有訊息（false = 僅回覆 @提及）
LINE_REPLY_ALL=true

# 毒性檢測閾值 (0.0 - 1.0)
TOXICITY_THRESHOLD=0.7

# 霸凌檢測閾值 (0.0 - 1.0)
BULLYING_THRESHOLD=0.6

# === API 服務設定 ===
API_BASE_URL=http://localhost:8000
API_KEY=your_api_key_here

# === 環境設定 ===
ENVIRONMENT=development  # 或 production
DEBUG=true              # 生產環境設為 false

# === 模型設定 ===
MODEL_PATH=models/working_toxicity_model
DEVICE=cuda  # 或 cpu

# === 資料庫設定（如果需要） ===
REDIS_URL=redis://localhost:6379/0
POSTGRES_URL=postgresql://user:password@localhost/cyberpuppy
```

### 5.2 填入實際憑證

將第三步取得的憑證填入：

```env
LINE_CHANNEL_ID=1234567890
LINE_CHANNEL_SECRET=a1b2c3d4e5f6g7h8i9j0k1l2m3n4o5p6
LINE_CHANNEL_ACCESS_TOKEN=eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJzdWIiOiIxMjM0NTY3ODkwIiwibmFtZSI6IkpvaG4gRG9lIiwiaWF0IjoxNTE2MjM5MDIyfQ.SflKxwRJSMeKKF2QT4fwpMeJf36POk6yJV_adQssw5c
```

### 5.3 確保 .env 在 .gitignore 中

**⚠️ 安全性關鍵步驟**：確保 `.env` 檔案不會被提交到 Git

檢查 `.gitignore` 檔案包含：
```gitignore
# 環境變數
.env
.env.*
!.env.example

# 敏感資料
*.key
*.pem
secrets/
```

### 5.4 建立 .env.example 範本

建立一個範本檔案供團隊成員參考：

```bash
cp .env .env.example
```

編輯 `.env.example`，將實際值替換為佔位符：
```env
LINE_CHANNEL_ID=your_channel_id_here
LINE_CHANNEL_SECRET=your_channel_secret_here
LINE_CHANNEL_ACCESS_TOKEN=your_channel_access_token_here
```

---

## 第六步：測試連線

### 6.1 安裝 Python 相依套件

```bash
# 確保使用最新版本的 line-bot-sdk (3.19.0+)
pip install line-bot-sdk==3.19.0
pip install flask
pip install python-dotenv
```

### 6.2 建立簡單的測試腳本

建立 `test_line_bot.py`：

```python
#!/usr/bin/env python3
"""
LINE Bot 連線測試腳本
測試 Channel Access Token 和 Webhook 設定是否正確
"""

import os
from dotenv import load_dotenv
from linebot.v3 import WebhookHandler
from linebot.v3.messaging import (
    Configuration,
    ApiClient,
    MessagingApi,
    TextMessage,
    ReplyMessageRequest
)
from linebot.v3.exceptions import InvalidSignatureError
from linebot.v3.webhooks import MessageEvent, TextMessageContent

# 載入環境變數
load_dotenv()

# 取得 LINE Bot 憑證
CHANNEL_ACCESS_TOKEN = os.getenv('LINE_CHANNEL_ACCESS_TOKEN')
CHANNEL_SECRET = os.getenv('LINE_CHANNEL_SECRET')

if not CHANNEL_ACCESS_TOKEN or not CHANNEL_SECRET:
    print("❌ 錯誤：找不到 LINE Bot 憑證")
    print("請檢查 .env 檔案是否正確設定")
    exit(1)

# 初始化 LINE Bot SDK (v3)
configuration = Configuration(access_token=CHANNEL_ACCESS_TOKEN)
handler = WebhookHandler(CHANNEL_SECRET)

def test_bot_info():
    """測試 Bot 資訊取得（驗證 Access Token）"""
    print("🔍 測試 1: 驗證 Channel Access Token...")

    try:
        with ApiClient(configuration) as api_client:
            line_bot_api = MessagingApi(api_client)
            bot_info = line_bot_api.get_bot_info()

            print("✅ Channel Access Token 有效！")
            print(f"   Bot 名稱: {bot_info.display_name}")
            print(f"   Bot ID: {bot_info.user_id}")
            print(f"   圖片 URL: {bot_info.picture_url}")
            return True

    except Exception as e:
        print(f"❌ Token 驗證失敗: {e}")
        print("請檢查 Channel Access Token 是否正確")
        return False

def test_webhook_signature():
    """測試 Webhook Signature 驗證"""
    print("\n🔍 測試 2: 驗證 Channel Secret...")

    # 模擬 LINE 伺服器傳送的 Webhook 資料
    test_body = '{"events":[]}'
    test_signature = 'invalid_signature'  # 刻意使用無效簽章

    try:
        handler.handle(test_body, test_signature)
        print("⚠️ 簽章驗證未如預期失敗")
        return False
    except InvalidSignatureError:
        print("✅ Channel Secret 正確設定！")
        print("   （簽章驗證機制正常運作）")
        return True
    except Exception as e:
        print(f"❌ 未預期的錯誤: {e}")
        return False

def print_configuration_summary():
    """顯示當前配置摘要"""
    print("\n📋 當前配置摘要")
    print("=" * 60)
    print(f"Channel Secret: {'*' * 10}{CHANNEL_SECRET[-10:]}")
    print(f"Access Token: {'*' * 20}{CHANNEL_ACCESS_TOKEN[-20:]}")
    print(f"Webhook URL: {os.getenv('LINE_WEBHOOK_URL', '未設定')}")
    print(f"環境: {os.getenv('ENVIRONMENT', 'development')}")
    print("=" * 60)

def main():
    """主測試流程"""
    print("🚀 LINE Bot 連線測試")
    print("=" * 60)

    # 顯示配置
    print_configuration_summary()

    # 執行測試
    test1_passed = test_bot_info()
    test2_passed = test_webhook_signature()

    # 總結
    print("\n📊 測試結果總結")
    print("=" * 60)
    print(f"Channel Access Token: {'✅ 通過' if test1_passed else '❌ 失敗'}")
    print(f"Channel Secret: {'✅ 通過' if test2_passed else '❌ 失敗'}")

    if test1_passed and test2_passed:
        print("\n🎉 所有測試通過！LINE Bot 已準備就緒")
        print("\n📝 下一步：")
        print("1. 啟動 Flask 伺服器")
        print("2. 使用 ngrok 建立 HTTPS 隧道")
        print("3. 在 LINE Developers Console 設定 Webhook URL")
        print("4. 使用 LINE 手機應用加 Bot 為好友並測試")
    else:
        print("\n⚠️ 部分測試失敗，請檢查配置")

if __name__ == "__main__":
    main()
```

### 6.3 執行測試

```bash
python test_line_bot.py
```

**預期輸出**：
```
🚀 LINE Bot 連線測試
============================================================

📋 當前配置摘要
============================================================
Channel Secret: **********h8i9j0k1l2
Access Token: ********************wRJSMeKKF2QT4fwpMeJf
Webhook URL: https://abc123.ngrok.io/callback
環境: development
============================================================

🔍 測試 1: 驗證 Channel Access Token...
✅ Channel Access Token 有效！
   Bot 名稱: CyberPuppy Bot
   Bot ID: U1234567890abcdef
   圖片 URL: https://profile.line-scdn.net/...

🔍 測試 2: 驗證 Channel Secret...
✅ Channel Secret 正確設定！
   （簽章驗證機制正常運作）

📊 測試結果總結
============================================================
Channel Access Token: ✅ 通過
Channel Secret: ✅ 通過

🎉 所有測試通過！LINE Bot 已準備就緒
```

### 6.4 啟動完整 Bot 服務

```bash
# 方法 1: 使用專案提供的啟動腳本
./start_local.sh    # Linux/Mac
.\start_local.bat   # Windows

# 方法 2: 直接啟動 Flask
python bot/line_bot.py

# 方法 3: 使用 Docker
docker-compose up line-bot
```

### 6.5 測試 Webhook（實際對話）

1. **在手機上加 Bot 為好友**：
   - 開啟 LINE Developers Console
   - 在 **「Messaging API」** 分頁找到 **「Bot information」**
   - 掃描 QR Code 加入好友

2. **傳送測試訊息**：
   ```
   你：你好
   Bot：[CyberPuppy] 訊息已收到，正在分析...

   你：這個垃圾
   Bot：⚠️ 偵測到潛在的負面內容
       毒性等級: 中等 (0.72)
       建議: 請使用更友善的表達方式
   ```

3. **查看伺服器日誌**：
   ```
   [INFO] 收到訊息事件: type=message, userId=U1234...
   [INFO] 分析文字: "你好"
   [INFO] 毒性分數: 0.03, 霸凌分數: 0.01
   [INFO] 回覆訊息已送出
   ```

---

## 🔧 常見問題與疑難排解

### Q1: Token 驗證失敗 (401 Unauthorized)

**錯誤訊息**：
```
linebot.v3.exceptions.ApiException: (401)
Reason: Unauthorized
```

**解決方案**：
1. 檢查 Channel Access Token 是否正確複製（完整字串）
2. 確認沒有多餘的空格或換行
3. 嘗試重新簽發 Token
4. 確認使用的是 **Long-lived token**，而非短效 token

### Q2: Webhook 驗證失敗

**錯誤訊息**：
```
The webhook returned an error: HTTP Status Code: 404
```

**檢查清單**：
- [ ] Flask 應用是否正在運行？
- [ ] ngrok 隧道是否已啟動？
- [ ] Webhook URL 是否包含 `/callback` 路徑？
- [ ] 伺服器防火牆是否阻擋外部連線？
- [ ] HTTPS 憑證是否有效？（LINE 只接受有效的 HTTPS）

**測試指令**：
```bash
# 測試 Webhook 端點是否可存取
curl -X POST https://your-domain.com/callback \
  -H "Content-Type: application/json" \
  -d '{"events":[]}'

# 預期回應：200 OK
```

### Q3: Signature 驗證錯誤

**錯誤訊息**：
```python
linebot.v3.exceptions.InvalidSignatureError: Invalid signature
```

**解決方案**：
1. 確認 Channel Secret 正確無誤
2. 檢查 Webhook 處理程式碼：
   ```python
   # ❌ 錯誤寫法
   handler.handle(body, request.headers['X-Line-Signature'])

   # ✅ 正確寫法
   signature = request.headers.get('X-Line-Signature', '')
   handler.handle(body, signature)
   ```
3. 確保 request body 沒有被修改或重複讀取

### Q4: Bot 無法回覆訊息

**可能原因**：
1. **Bot 未啟用自動回覆**：
   - LINE Developers Console → Messaging API
   - 關閉 **「Auto-reply messages」**
   - 關閉 **「Greeting messages」**

2. **Reply Token 已過期**：
   - Reply Token 僅能使用一次
   - 有效期限 30 秒
   - 超過時限改用 Push Message

3. **用戶已封鎖 Bot**：
   - 無法主動傳送訊息給封鎖的用戶
   - 檢查用戶狀態：`line_bot_api.get_profile(user_id)`

### Q5: SDK 版本相容性問題

**錯誤訊息**：
```python
ImportError: cannot import name 'LineBotApi' from 'linebot'
```

**原因**：SDK 3.0+ 大幅改版，API 介面不同

**解決方案**：
```bash
# 方案 1: 升級到 SDK 3.x (建議)
pip uninstall line-bot-sdk
pip install line-bot-sdk==3.19.0

# 方案 2: 使用舊版 SDK 2.x (不建議)
pip install line-bot-sdk==2.4.2
```

**SDK 3.x 主要變更**：
```python
# SDK 2.x (舊版)
from linebot import LineBotApi, WebhookHandler
from linebot.models import TextSendMessage

line_bot_api = LineBotApi(CHANNEL_ACCESS_TOKEN)
line_bot_api.reply_message(reply_token, TextSendMessage(text='Hello'))

# SDK 3.x (新版)
from linebot.v3.messaging import (
    Configuration, ApiClient, MessagingApi,
    TextMessage, ReplyMessageRequest
)

configuration = Configuration(access_token=CHANNEL_ACCESS_TOKEN)
with ApiClient(configuration) as api_client:
    line_bot_api = MessagingApi(api_client)
    line_bot_api.reply_message(
        ReplyMessageRequest(
            reply_token=reply_token,
            messages=[TextMessage(text='Hello')]
        )
    )
```

### Q6: ngrok 免費版限制

**問題**：
- 免費版每次啟動 URL 都會變更
- 需要頻繁更新 Webhook URL

**解決方案**：
1. **註冊 ngrok 帳號獲得固定 subdomain**：
   ```bash
   ngrok config add-authtoken YOUR_AUTH_TOKEN
   ngrok http 8000 --subdomain=cyberpuppy
   # URL: https://cyberpuppy.ngrok.io
   ```

2. **使用替代服務**：
   - **localtunnel**: `lt --port 8000 --subdomain cyberpuppy`
   - **serveo**: `ssh -R 80:localhost:8000 serveo.net`

3. **部署到雲端平台**（推薦生產環境）：
   - Heroku (免費額度)
   - Railway (免費額度)
   - Render (免費額度)
   - Google Cloud Run

### Q7: Rate Limiting (請求頻率限制)

**錯誤訊息**：
```
Status code: 429, Error: Rate limit exceeded
```

**LINE Messaging API 限制**：
- Push Message: 500 次/秒
- Reply Message: 無限制（但有 Reply Token 時效）
- Multicast: 100 次/秒

**解決方案**：
```python
import time
from functools import wraps

def rate_limit(calls_per_second=10):
    """簡單的速率限制裝飾器"""
    min_interval = 1.0 / calls_per_second
    last_called = [0.0]

    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            elapsed = time.time() - last_called[0]
            left_to_wait = min_interval - elapsed
            if left_to_wait > 0:
                time.sleep(left_to_wait)
            result = func(*args, **kwargs)
            last_called[0] = time.time()
            return result
        return wrapper
    return decorator

@rate_limit(calls_per_second=5)
def send_push_message(user_id, text):
    line_bot_api.push_message(user_id, TextMessage(text=text))
```

---

## 📚 參考資源

### 官方文件
- [LINE Developers 官方網站](https://developers.line.biz/)
- [Messaging API 參考文件](https://developers.line.biz/zh-hant/docs/messaging-api/)
- [Python SDK GitHub](https://github.com/line/line-bot-sdk-python)
- [Python SDK API 文件](https://line-bot-sdk-python.readthedocs.io/)

### 教學資源
- [LINE Bot 開發實戰教學（繁體中文）](https://oberonlai.blog/line-messaging-api-setting/)
- [Python LINE Bot 開發指南](https://yual.in/posts/python-line-bot-tutorial)
- [LINE Messaging API 快速開始](https://developers.line.biz/zh-hant/docs/messaging-api/getting-started/)

### 社群與支援
- [LINE Developers Community](https://www.line-community.me/)
- [Stack Overflow - LINE Bot](https://stackoverflow.com/questions/tagged/line-messaging-api)

---

## 🔒 安全性最佳實踐

### 憑證管理
1. ✅ **絕不提交憑證到 Git**
   - 使用 `.env` 檔案
   - 加入 `.gitignore`
   - 使用環境變數或密鑰管理服務

2. ✅ **定期輪換 Access Token**
   - 建議每 90 天更換一次
   - 發現外洩立即撤銷並重新簽發

3. ✅ **限制伺服器存取**
   - 使用防火牆限制來源 IP
   - 啟用 HTTPS 加密傳輸
   - 實施請求頻率限制

### Webhook 安全
```python
# 驗證請求來源確實是 LINE 伺服器
@app.route("/callback", methods=['POST'])
def callback():
    signature = request.headers.get('X-Line-Signature', '')
    body = request.get_data(as_text=True)

    try:
        # SDK 會自動驗證簽章
        handler.handle(body, signature)
    except InvalidSignatureError:
        # 拒絕無效的請求
        abort(400)

    return 'OK'
```

### 資料隱私
- 🔒 不記錄用戶的原始訊息內容
- 🔒 僅儲存必要的統計資料（雜湊後）
- 🔒 遵守 GDPR 和個資法規定

---

## 🎉 完成！

恭喜你完成 LINE Bot 的憑證設定！現在 CyberPuppy 可以開始提供中文網路霸凌防治服務了。

### 下一步行動
- [ ] 完成 API 服務部署（`api/app.py`）
- [ ] 啟動 LINE Bot 服務（`bot/line_bot.py`）
- [ ] 進行完整的整合測試
- [ ] 監控 Bot 運行狀況
- [ ] 收集用戶反饋並持續改進

有任何問題，請參考本文件的疑難排解章節，或聯繫開發團隊。

---

**文件維護者**: CyberPuppy 開發團隊
**最後更新**: 2025-09-27
**文件版本**: v2.0 (2025 Edition)