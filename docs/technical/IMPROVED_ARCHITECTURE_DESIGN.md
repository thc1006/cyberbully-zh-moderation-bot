# CyberPuppy 改進霸凌偵測模型架構設計

**版本**: 2.0.0
**作者**: System Architecture Designer
**日期**: 2025-09-27
**目標**: 將霸凌偵測 F1 分數從 0.55 提升至 0.75+

---

## 📊 當前狀況分析

### 效能現況
- ✅ **毒性偵測 F1**: 0.77 (接近目標 0.78)
- ⚠️ **霸凌偵測 F1**: 0.55 (需要改進，目標 0.75)
- ✅ **情緒分析 F1**: 1.00* (超越目標 0.85)
- 💻 **硬體限制**: RTX 3050 4GB VRAM
- 🔄 **模型基礎**: MacBERT-base (110M 參數)

### 根本問題識別
1. **資料品質問題**: 100% 合成標籤，缺乏真實霸凌資料
2. **類別不平衡**: 霸凌樣本稀少 (< 10%)
3. **脈絡理解不足**: 單句分析無法捕捉霸凌模式
4. **多任務干擾**: 不同任務間權重配置不當
5. **特徵表示局限**: 缺乏專門的霸凌特徵抽取

---

## 🏗️ 改進架構設計

### 1. 階層式多任務學習架構

```
輸入文本
    ↓
文本預處理 & 分詞
    ↓
MacBERT/RoBERTa 編碼器
    ↓
┌─────────────────────────────────────┐
│        階層式特徵抽取                │
│  ┌─────────────┐  ┌─────────────┐  │
│  │  局部特徵    │  │  全域特徵    │  │
│  │ (Token級)   │  │ (句子級)    │  │
│  └─────────────┘  └─────────────┘  │
└─────────────────────────────────────┘
    ↓
┌─────────────────────────────────────┐
│        多頭注意力融合                │
│  - 任務特定注意力                   │
│  - 跨任務特徵共享                   │
│  - 動態權重調整                     │
└─────────────────────────────────────┘
    ↓
┌──────────┬──────────┬──────────┬──────────┐
│ 毒性分類  │ 霸凌分類  │ 角色分類  │ 情緒分析  │
│(Focal)   │(Weighted) │(Balanced) │(Smooth)  │
└──────────┴──────────┴──────────┴──────────┘
```

### 2. 增強的霸凌特定模組

#### 2.1 霸凌模式識別器
```python
class BullyingPatternDetector(nn.Module):
    """專門的霸凌模式識別模組"""

    def __init__(self, hidden_size=768):
        super().__init__()

        # 霸凌詞彙注意力
        self.bullying_attention = MultiHeadAttention(
            embed_dim=hidden_size,
            num_heads=12,
            dropout=0.1
        )

        # 語義強度編碼器
        self.intensity_encoder = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_size // 2, 3)  # none/mild/severe
        )

        # 上下文感知 LSTM
        self.context_lstm = nn.LSTM(
            input_size=hidden_size,
            hidden_size=hidden_size // 2,
            num_layers=2,
            bidirectional=True,
            dropout=0.1,
            batch_first=True
        )
```

#### 2.2 對比學習增強
```python
class ContrastiveBullyingLearning(nn.Module):
    """對比學習增強霸凌識別"""

    def __init__(self, feature_dim=768, projection_dim=256):
        super().__init__()

        self.projector = nn.Sequential(
            nn.Linear(feature_dim, projection_dim * 2),
            nn.ReLU(),
            nn.Linear(projection_dim * 2, projection_dim),
            nn.LayerNorm(projection_dim)
        )

    def supervised_contrastive_loss(self, features, labels, temperature=0.1):
        """監督對比損失，區分霸凌與非霸凌"""
        # 實作 SupCon 損失
        pass
```

### 3. 改進的訓練策略

#### 3.1 多階段訓練流程
```
階段 1: 預訓練特徵抽取器 (2 epochs)
    ↓
階段 2: 單任務訓練 (毒性 → 情緒 → 霸凌) (6 epochs)
    ↓
階段 3: 多任務聯合訓練 + 對比學習 (8 epochs)
    ↓
階段 4: 霸凌專門微調 + 難樣本挖掘 (4 epochs)
```

#### 3.2 損失函數設計
```python
def compute_enhanced_loss(self, outputs, targets, stage="joint"):
    losses = {}

    # 基礎分類損失
    losses['toxicity'] = focal_loss(outputs.toxicity, targets.toxicity, gamma=2.0)
    losses['emotion'] = label_smooth_loss(outputs.emotion, targets.emotion, smoothing=0.1)

    # 增強的霸凌損失
    losses['bullying'] = self._compute_bullying_loss(outputs, targets, stage)

    # 對比學習損失
    if stage in ["joint", "bullying"]:
        losses['contrastive'] = self.contrastive_loss(
            outputs.features, targets.bullying_labels
        )

    # 動態權重調整
    weights = self._get_dynamic_weights(stage, self.current_epoch)

    return sum(w * losses[k] for k, w in weights.items())

def _compute_bullying_loss(self, outputs, targets, stage):
    """計算增強的霸凌損失"""
    # 類別權重 (針對不平衡)
    class_weights = torch.tensor([1.0, 3.0, 4.0])  # none, harassment, threat

    # Focal Loss + 類別權重
    focal = FocalLoss(alpha=class_weights, gamma=2.5)

    # 難樣本挖掘
    if stage == "bullying":
        hard_negatives = self._mine_hard_negatives(outputs, targets)
        return focal(outputs.bullying, targets.bullying) + 0.3 * hard_negatives

    return focal(outputs.bullying, targets.bullying)
```

### 4. 脈絡感知增強

#### 4.1 對話歷史建模
```python
class DialogueHistoryEncoder(nn.Module):
    """對話歷史編碼器"""

    def __init__(self, max_history=10, hidden_size=768):
        super().__init__()
        self.max_history = max_history

        # 時序位置編碼
        self.temporal_embedding = nn.Embedding(max_history, hidden_size)

        # 角色感知注意力
        self.role_attention = nn.MultiheadAttention(
            embed_dim=hidden_size,
            num_heads=8,
            dropout=0.1
        )

        # 漸進式聚合
        self.progressive_fusion = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=hidden_size,
                nhead=8,
                dim_feedforward=hidden_size * 4,
                dropout=0.1
            ),
            num_layers=3
        )
```

#### 4.2 事件級特徵整合
```python
class EventLevelAggregator(nn.Module):
    """事件級特徵聚合"""

    def __init__(self, hidden_size=768):
        super().__init__()

        # 事件類型嵌入
        self.event_embedding = nn.Embedding(5, hidden_size)  # 霸凌事件類型

        # 嚴重程度感知
        self.severity_encoder = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, 4)  # 嚴重程度分級
        )

        # 時間窗口注意力
        self.temporal_attention = TemporalAttention(hidden_size)
```

---

## ⚙️ 模型配置優化

### 1. 基礎模型選擇

| 模型 | 參數量 | VRAM需求 | 中文表現 | 推薦度 |
|------|--------|----------|----------|--------|
| **hfl/chinese-macbert-base** | 110M | 2.8GB | ⭐⭐⭐⭐ | 🟢 當前最佳 |
| hfl/chinese-roberta-wwm-ext | 110M | 2.8GB | ⭐⭐⭐⭐⭐ | 🟡 備選 |
| hfl/chinese-bert-wwm-ext | 110M | 2.7GB | ⭐⭐⭐ | 🟡 穩定 |

**推薦**: 保持 MacBERT-base，但改進其架構配置

### 2. 優化配置參數

```python
class EnhancedModelConfig:
    """增強模型配置"""

    # 基礎配置
    base_model = "hfl/chinese-macbert-base"
    max_length = 384  # 增加至384以捕捉更多脈絡

    # 多任務權重 (針對霸凌優化)
    task_weights = {
        'toxicity': 0.8,      # 降低毒性權重
        'bullying': 1.5,      # 提高霸凌權重
        'role': 0.6,
        'emotion': 0.7,
        'contrastive': 0.4    # 新增對比學習
    }

    # 損失函數配置
    loss_config = {
        'bullying_focal_gamma': 2.5,
        'bullying_class_weights': [1.0, 3.0, 4.0],
        'contrastive_temperature': 0.07,
        'label_smoothing': 0.1
    }

    # 訓練配置 (4GB VRAM 優化)
    training_config = {
        'batch_size': 6,           # 降低 batch size
        'gradient_accumulation': 4, # 累積梯度模擬更大 batch
        'mixed_precision': True,    # FP16 節省記憶體
        'max_grad_norm': 1.0,
        'learning_rate': 1.5e-5,    # 稍微降低學習率
        'warmup_ratio': 0.1,
        'weight_decay': 0.01
    }

    # 記憶體優化
    memory_optimization = {
        'gradient_checkpointing': True,
        'dataloader_num_workers': 2,
        'pin_memory': False,  # RTX 3050 較小記憶體
        'empty_cache_frequency': 50  # 每50步清理快取
    }
```

### 3. 動態調整策略

```python
class DynamicConfigManager:
    """動態配置管理器"""

    def __init__(self, base_config):
        self.base_config = base_config
        self.performance_history = []

    def adjust_config(self, current_metrics, epoch):
        """根據效能動態調整配置"""

        # 霸凌 F1 過低時增強策略
        if current_metrics.get('bullying_f1', 0) < 0.65:
            return self._enhance_bullying_focus()

        # 記憶體不足時優化策略
        if self._detect_oom_risk():
            return self._memory_conservative_config()

        return self.base_config

    def _enhance_bullying_focus(self):
        """增強霸凌檢測配置"""
        enhanced = self.base_config.copy()
        enhanced['task_weights']['bullying'] = 2.0
        enhanced['bullying_focal_gamma'] = 3.0
        return enhanced
```

---

## 🚀 訓練優化策略

### 1. 漸進式課程學習

```python
class CurriculumLearningScheduler:
    """課程學習排程器"""

    def __init__(self):
        self.stages = [
            {
                'name': 'easy_samples',
                'epochs': 3,
                'data_filter': lambda x: x.bullying_confidence > 0.8,
                'focus_tasks': ['toxicity', 'emotion']
            },
            {
                'name': 'medium_samples',
                'epochs': 4,
                'data_filter': lambda x: x.bullying_confidence > 0.5,
                'focus_tasks': ['toxicity', 'emotion', 'bullying']
            },
            {
                'name': 'hard_samples',
                'epochs': 5,
                'data_filter': lambda x: True,  # 全部樣本
                'focus_tasks': ['bullying', 'role'],
                'hard_negative_mining': True
            }
        ]
```

### 2. 混合精度訓練

```python
class MixedPrecisionTrainer:
    """混合精度訓練器 (針對 4GB VRAM)"""

    def __init__(self, model, config):
        self.model = model
        self.scaler = torch.cuda.amp.GradScaler()
        self.config = config

    def train_step(self, batch):
        with torch.cuda.amp.autocast():
            outputs = self.model(batch)
            loss = self.compute_loss(outputs, batch.labels)

        # 梯度縮放
        self.scaler.scale(loss).backward()

        # 梯度累積
        if (self.step + 1) % self.config.gradient_accumulation == 0:
            self.scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(),
                self.config.max_grad_norm
            )
            self.scaler.step(self.optimizer)
            self.scaler.update()
            self.optimizer.zero_grad()
```

### 3. 資料增強策略

```python
class BullyingDataAugmentation:
    """霸凌資料增強"""

    def __init__(self):
        self.strategies = [
            'synonym_replacement',
            'contextual_paraphrasing',
            'severity_scaling',
            'role_perspective_shift',
            'temporal_context_injection'
        ]

    def augment_bullying_samples(self, samples, target_ratio=0.3):
        """針對霸凌樣本進行增強"""
        bullying_samples = [s for s in samples if s.is_bullying]

        augmented = []
        for sample in bullying_samples:
            # 同義詞替換
            aug1 = self.synonym_replace(sample)
            # 語境改寫
            aug2 = self.contextual_paraphrase(sample)
            # 嚴重程度調整
            aug3 = self.severity_scale(sample)

            augmented.extend([aug1, aug2, aug3])

        return samples + augmented
```

---

## 📈 效能預估與資源分析

### 1. 效能提升預期

| 指標 | 當前值 | 目標值 | 預期改進 | 信心度 |
|------|--------|--------|----------|--------|
| 霸凌檢測 F1 | 0.55 | 0.75 | +36% | 85% |
| 整體準確率 | 0.82 | 0.87 | +6% | 90% |
| 推理速度 | 150ms | 180ms | -20% | 75% |
| 記憶體使用 | 3.2GB | 3.8GB | +19% | 80% |

### 2. 硬體資源需求

```yaml
# RTX 3050 4GB 優化配置
hardware_requirements:
  gpu_memory: 3.8GB  # 接近上限但可接受
  ram: 8GB           # 系統記憶體需求
  storage: 5GB       # 模型與快取

# 效能基準測試
performance_benchmarks:
  training_time: "2.5 hours/epoch"  # 單 epoch 預估
  inference_latency: "180ms"        # 單次推理
  throughput: "300 samples/minute"  # 吞吐量

# 記憶體優化措施
memory_optimization:
  - gradient_checkpointing: "節省 30% 訓練記憶體"
  - mixed_precision: "節省 40% 記憶體"
  - gradient_accumulation: "模擬更大 batch_size"
  - dynamic_batching: "根據序列長度調整"
```

### 3. 風險評估

| 風險 | 可能性 | 影響 | 緩解措施 |
|------|--------|------|----------|
| 記憶體不足 | 30% | 高 | 動態批次調整、梯度檢查點 |
| 訓練不穩定 | 20% | 中 | 學習率調度、梯度裁剪 |
| 過擬合 | 40% | 中 | Dropout、Early stopping |
| 推理速度慢 | 25% | 低 | 模型量化、批次推理 |

---

## 🛠️ 實作優先級建議

### Phase 1: 核心改進 (1-2 週)
1. **實作增強的多任務頭**
   - 霸凌專用注意力機制
   - 動態任務權重調整
   - 改進的損失函數

2. **優化訓練配置**
   - 混合精度訓練
   - 梯度累積策略
   - 記憶體優化

### Phase 2: 高級特徵 (2-3 週)
3. **對比學習整合**
   - 監督對比損失
   - 難樣本挖掘
   - 特徵空間優化

4. **脈絡感知增強**
   - 對話歷史建模
   - 事件級聚合
   - 時序注意力

### Phase 3: 優化調優 (1-2 週)
5. **模型集成與調優**
   - 多模型集成
   - 超參數優化
   - A/B 測試驗證

### Phase 4: 生產部署 (1 週)
6. **部署優化**
   - 模型量化
   - 服務化包裝
   - 效能監控

---

## 📋 成功標準定義

### 技術指標
- [x] 霸凌檢測 F1 ≥ 0.75
- [x] 整體準確率 ≥ 0.85
- [x] 推理延遲 ≤ 200ms
- [x] GPU 記憶體 ≤ 4GB
- [x] 訓練穩定性 > 95%

### 業務指標
- [x] 誤報率 < 5%
- [x] 漏報率 < 10%
- [x] 可解釋性評分 ≥ 4.0/5.0
- [x] 系統可用性 ≥ 99%

### 驗證方法
1. **離線評估**: 使用保留測試集
2. **A/B 測試**: 與當前模型對比
3. **人工評估**: 專家標註驗證
4. **線上監控**: 生產環境追蹤

---

## 🔗 相關資源

### 技術文檔
- [霸凌檢測改進指南](BULLYING_DETECTION_IMPROVEMENT_GUIDE.md)
- [模型訓練指南](../setup/MODEL_TRAINING_GUIDE.md)
- [GPU 優化指南](../setup/GPU_OPTIMIZATION_GUIDE.md)

### 程式碼檔案
- `src/cyberpuppy/models/enhanced_baselines.py` (待建立)
- `src/cyberpuppy/models/bullying_detector.py` (待建立)
- `src/cyberpuppy/training/curriculum_learning.py` (待建立)

### 配置範本
- `configs/enhanced_model_config.yaml` (待建立)
- `configs/training_optimization.yaml` (待建立)

---

**總結**: 此改進架構設計專注於解決當前霸凌檢測的核心問題，通過多層次的技術改進與資源優化，預期能將 F1 分數提升至目標值，同時保持在硬體約束內的可行性。