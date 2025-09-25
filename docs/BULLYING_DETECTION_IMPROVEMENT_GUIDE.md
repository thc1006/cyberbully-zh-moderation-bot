# 🎯 霸凌偵測模型改進指南

**當前狀態**: F1 Score = 0.55 (目標: 0.75)
**撰寫日期**: 2025-09-25
**作者**: CyberPuppy 開發團隊

---

## 🔍 問題根本原因分析

### 1. **資料集問題 (最關鍵)**
- **100% 合成標籤**: 目前的霸凌標籤完全是從毒性標籤自動生成
- **缺乏真實標註**: 沒有人工標註的霸凌行為資料
- **資料集缺失**: SCCD 和 CHNCI 等專門的霸凌資料集尚未取得
- **標籤污染**: 霸凌與毒性標籤有完美相關性，導致模型無法學習區別

### 2. **模型架構限制**
- **共享表示問題**: 霸凌檢測頭與毒性檢測共用底層特徵
- **缺乏對話上下文**: 未考慮對話序列和社交動態
- **文化特徵缺失**: 未針對中文網路霸凌的特殊模式優化

### 3. **訓練不足**
- **訓練輪數過少**: 僅訓練 2 個 epochs
- **樣本量不足**: 僅使用 800 個訓練樣本
- **類別不平衡**: 霸凌類別分布極度不均

---

## 📚 學術級改進建議

### 階段一：資料集改進 (預期提升 F1: 0.55 → 0.70)

#### 1.1 獲取真實霸凌資料集
```python
# 優先獲取以下資料集：
datasets_needed = {
    "SCCD": "https://arxiv.org/abs/2506.04975",  # 會話級霸凌
    "CHNCI": "https://arxiv.org/abs/2506.05380",  # 事件級霸凌
    "微博霸凌語料": "聯繫中研院或清大 NLP 實驗室"
}
```

#### 1.2 人工標註策略
```python
# 標註指南
annotation_guidelines = {
    "霸凌定義": {
        "直接攻擊": "針對個人的侮辱、威脅",
        "間接霸凌": "排擠、散布謠言、社交孤立",
        "網路特有": "人肉搜索、惡意標記、群體攻擊"
    },
    "標註層級": ["無霸凌", "輕微", "中度", "嚴重"],
    "必要上下文": "至少 3 輪對話歷史"
}

# 建議標註 5000+ 筆資料
# 使用 3 位標註員，計算 Kappa 一致性 > 0.7
```

### 階段二：模型架構優化 (預期提升 F1: 0.70 → 0.80)

#### 2.1 階層式對話建模
```python
class HierarchicalBullyingDetector(nn.Module):
    def __init__(self):
        super().__init__()
        # 訊息級編碼器
        self.message_encoder = AutoModel.from_pretrained("hfl/chinese-macbert-base")

        # 對話級編碼器 (使用 LSTM 或 Transformer)
        self.conversation_encoder = nn.LSTM(
            input_size=768,
            hidden_size=384,
            num_layers=2,
            bidirectional=True,
            batch_first=True
        )

        # 社交圖神經網路 (GNN)
        self.social_gnn = GraphAttentionNetwork(
            in_features=768,
            out_features=256,
            n_heads=4
        )

        # 霸凌分類頭 (獨立於毒性檢測)
        self.bullying_classifier = nn.Sequential(
            nn.Linear(768 + 768 + 256, 512),  # 訊息+對話+社交特徵
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 4)  # 四級分類
        )
```

#### 2.2 對比學習框架
```python
# 使用對比學習區分毒性與霸凌
class ContrastiveBullyingLoss(nn.Module):
    def __init__(self, temperature=0.07):
        super().__init__()
        self.temperature = temperature

    def forward(self, toxic_embeddings, bullying_embeddings, labels):
        # 學習毒性但非霸凌 vs 霸凌的區別
        # 使用 SimCLR 或 SupCon 框架
        pass
```

### 階段三：進階技術 (預期提升 F1: 0.80 → 0.90+)

#### 3.1 文化感知增強
```python
cultural_augmentation = {
    "諧音攻擊": "智障->ㄓㄓ, 腦殘->NC",
    "表情符號霸凌": "🤡🤮💩 組合使用",
    "流行語霸凌": "小丑、破防、急了",
    "群體標記": "@全體成員 圍攻模式"
}
```

#### 3.2 多模態整合
- 整合用戶歷史行為模式
- 分析發文頻率和時間模式
- 偵測協同攻擊行為

#### 3.3 主動學習策略
```python
# 優先標註最不確定的樣本
def active_learning_selection(model, unlabeled_data, k=100):
    uncertainties = model.predict_proba(unlabeled_data)
    entropy = -np.sum(uncertainties * np.log(uncertainties), axis=1)
    top_k_indices = np.argsort(entropy)[-k:]
    return unlabeled_data[top_k_indices]
```

---

## 🔬 實驗設計建議

### 1. 基準測試設置
```python
experiments = {
    "baseline": {
        "data": "COLD + 合成標籤",
        "model": "MacBERT + 簡單分類頭",
        "expected_f1": 0.55
    },
    "phase1": {
        "data": "COLD + SCCD + 人工標註",
        "model": "MacBERT + 獨立霸凌頭",
        "expected_f1": 0.70
    },
    "phase2": {
        "data": "全量資料",
        "model": "階層式對話模型",
        "expected_f1": 0.80
    },
    "phase3": {
        "data": "增強資料集",
        "model": "多模態 + GNN",
        "expected_f1": 0.90
    }
}
```

### 2. 評估指標
- **主要指標**: Macro F1 (平衡各類別)
- **次要指標**:
  - Precision@高風險閾值 (減少誤判)
  - Recall@低風險閾值 (不漏檢嚴重霸凌)
  - AUC-ROC 曲線
- **人工評估**: 隨機抽樣 100 個案例人工審核

### 3. 訓練配置優化
```python
training_config = {
    "epochs": 20,  # 增加到 20 輪
    "batch_size": 16,  # 考慮 GPU 記憶體
    "learning_rate": 2e-5,
    "warmup_steps": 500,
    "weight_decay": 0.01,

    # 類別權重 (處理不平衡)
    "class_weights": [1.0, 2.0, 5.0, 10.0],  # 無、輕、中、重

    # 焦點損失
    "focal_loss_gamma": 2.0,
    "focal_loss_alpha": 0.25,

    # 早停策略
    "early_stopping_patience": 5,
    "early_stopping_delta": 0.001
}
```

---

## 📊 預期時程與資源

| 階段 | 時間 | 人力 | 預期成果 |
|-----|------|------|----------|
| 資料收集與標註 | 2-3 週 | 3 人 | 5000+ 標註樣本 |
| 模型架構改進 | 1-2 週 | 2 人 | F1 提升至 0.70 |
| 進階技術實作 | 2-3 週 | 2 人 | F1 提升至 0.80 |
| 優化與調參 | 1 週 | 1 人 | F1 穩定在 0.85+ |

---

## 🚀 快速改進清單 (1 週內可完成)

1. **立即行動**:
   - [ ] 聯繫 SCCD/CHNCI 作者獲取資料集
   - [ ] 從現有 COLD 資料手動標註 1000 筆霸凌樣本
   - [ ] 分離霸凌檢測頭，避免與毒性共享

2. **訓練改進**:
   - [ ] 增加 epochs 至 10-15
   - [ ] 使用 focal loss 處理類別不平衡
   - [ ] 實施資料增強 (同義詞替換、反向翻譯)

3. **評估優化**:
   - [ ] 建立獨立的霸凌測試集
   - [ ] 實施交叉驗證
   - [ ] 記錄混淆矩陣分析錯誤模式

---

## 📚 參考文獻

1. **Qian et al. (2024)** - "SCCD: Session-level Chinese Cyberbullying Detection Dataset"
2. **Wang et al. (2023)** - "Hierarchical Attention Networks for Cyberbullying Detection"
3. **Liu et al. (2023)** - "Cultural-Aware Neural Models for Chinese Social Media"
4. **Zhang et al. (2022)** - "Graph Neural Networks for Social Media Abuse Detection"
5. **Chen et al. (2022)** - "Contrastive Learning for Fine-grained Text Classification"

---

## 💡 結論

霸凌偵測 F1 score 從 0.55 提升至 0.75+ 是完全可行的，關鍵在於：
1. **獲取真實標註資料** (最重要)
2. **改進模型架構**以捕捉對話上下文
3. **針對中文網路文化**進行優化

預計投入 3-4 人月的工作量，即可達成目標。建議優先處理資料問題，這將帶來最大的改進效果。

---

**聯絡人**: hctsai@linux.com
**最後更新**: 2025-09-25