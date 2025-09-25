# 貢獻指南

感謝您對 CyberPuppy 專案的興趣！我們歡迎所有形式的貢獻。

## 🤝 如何貢獻

### 1. 報告問題
- 使用 [GitHub Issues](https://github.com/thc1006/cyberbully-zh-moderation-bot/issues)
- 提供詳細的問題描述和重現步驟
- 包含錯誤訊息和系統環境資訊

### 2. 提交程式碼
1. Fork 專案
2. 建立功能分支 (`git checkout -b feature/AmazingFeature`)
3. 提交變更 (`git commit -m 'Add some AmazingFeature'`)
4. 推送到分支 (`git push origin feature/AmazingFeature`)
5. 建立 Pull Request

### 3. 程式碼規範
- 遵循 PEP 8 規範
- 撰寫單元測試
- 更新相關文件
- 確保所有測試通過

## 📝 開發流程

1. **設置開發環境**
   ```bash
   pip install -r requirements-dev.txt
   ```

2. **執行測試**
   ```bash
   pytest tests/
   ```

3. **程式碼品質檢查**
   ```bash
   flake8 src/
   black src/ --check
   mypy src/
   ```

## 📜 授權

提交貢獻即表示您同意以 Apache License 2.0 授權您的程式碼。

## 📞 聯絡

- Email: hctsai@linux.com
- GitHub Issues: [問題追蹤](https://github.com/thc1006/cyberbully-zh-moderation-bot/issues)