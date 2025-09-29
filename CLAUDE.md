# CLAUDE.md

## 專案宗旨
CyberPuppy 是一個以「網路霸凌防治」為核心，結合 **毒性偵測** 與 **情緒分析** 的中文聊天機器人。它需在私訊與群組對話中，以**高可解釋性**、**低誤傷**的方式提供即時提醒。

## 專案規模 (2025-09-29 深度分析)
- **總檔案數**: 9,185 個檔案 (~3.7 GB)
- **程式碼**: 254 個 Python 檔案 (79 個核心模組 in src/cyberpuppy/)
- **測試**: 70 個測試檔案，179+ 測試用例
- **資料總量**: **4,749,491 行** (總計 666 MB)
- **模型權重**: 10 個模型檔案 (~1.6 GB，但無可用模型)
- **套件依賴**: 160 個 (64 生產 + 96 開發)

## 目錄大小分布
| 目錄 | 大小 | 內容說明 |
|------|------|----------|
| `models/` | **2.4 GB** | 模型權重檔案（但無可用模型） |
| `data/` | **666 MB** | 訓練與測試資料 |
| `htmlcov/` | 11 MB | 測試覆蓋率報告 |
| `tests/` | 3.8 MB | 測試程式碼 |
| `src/` | 2.8 MB | 源程式碼 |
| `scripts/` | 1.4 MB | 腳本工具 |

## 任務與資料
### 資料集規模詳細統計
- **COLD Dataset**: 37,483 樣本 (主要訓練資料)
  - train.csv: 25,727 樣本
  - dev.csv: 6,432 樣本
  - test.csv: 5,324 樣本
- **DMSC**: 2,131,887 行 (情緒分析，387 MB)
- **ChnSentiCorp**: 15,534 樣本 (情感分析)
  - 注意: 2 個檔案為空檔案 (0 樣本)
- **訓練集最終分割**:
  - 訓練: 25,659 (toxic: 49.4%, 平均長度: 47.8 字)
  - 開發: 6,430 (toxic: 49.9%, 平均長度: 47.3 字)
  - 測試: 5,320 (toxic: 39.6%, 平均長度: 48.3 字)

### 標籤體系
- 主任務：毒性/霸凌偵測（COLD 為主，SCCD/CHNCI 脈絡加強）
- 副任務：情緒分類／強度（ChnSentiCorp、DMSC v2、NTUSD）
- 統一標籤：`toxicity{none,toxic,severe}`、`bullying{none,harassment,threat}`、`role{none,perpetrator,victim,bystander}`、`emotion{pos,neu,neg}` + `emotion_strength{0..4}`
- **⚠️ 問題**: 當前訓練資料 role 全為 "none"，emotion 全為 "neu"

## 技術與工具
- 模型：HuggingFace（`hfl/chinese-macbert-base`、`hfl/chinese-roberta-wwm-ext`）
- 可解釋性：Captum（IG）、SHAP（text plots）
- 文字處理：CKIPTagger / ckip-transformers、OpenCC
- Bot：LINE Messaging API（Webhook + X-Line-Signature 驗證）
- 可選仲裁：Perspective API（僅作輔助，不直接決策）

## 風格與原則
- **明確輸出**：每次改動請列出「新增檔案」「修改檔案」「刪除檔案」清單。
- **可測試**：所有模組附最小單元測試；CI 必跑 `pytest -q`。
- **可回溯**：資料清理、標籤映射皆落地成腳本，輸入→輸出可重現。
- **隱私優先**：不寫入原文內容到日誌；僅保存雜湊摘要與分數。

## 目錄規範

s```
data/{raw,processed,external}
models/
src/cyberpuppy/{config.py, labeling/, models/, explain/, safety/, eval/, loop/, arbiter/}
api/
bot/
scripts/
tests/
docs/
notebooks/
```

## 常用工作流（給 Claude Code）
### 初始化
- 產生專案骨架與 `pyproject.toml` / `requirements.txt`
- 建立 `docs/DATA_CONTRACT.md` 與 `docs/POLICY.md`

### 資料
- `scripts/download_datasets.py`：下載 COLD、ChnSentiCorp、DMSC v2、NTUSD；SCCD/CHNCI 依作者提供方式導入
- `scripts/clean_normalize.py`：正規化、繁簡轉換、去識別

### 建模
- `src/cyberpuppy/models/baselines.py`：多任務頭
- `src/cyberpuppy/models/contextual.py`：會話/事件脈絡融合
- `train.py`：CLI 訓練入口，支援 early stopping、AMP

### 解釋
- `src/cyberpuppy/explain/ig.py`、`notebooks/explain_ig.ipynb`
- `notebooks/explain_shap.ipynb`

### 服務
- `api/app.py`（FastAPI）
- `bot/line_bot.py`（LINE SDK，驗簽）

### 安全
- `src/cyberpuppy/safety/rules.py`（分級回覆策略）
- 可選 `src/cyberpuppy/arbiter/perspective.py`

## 模型現況（2025-09-29 深度分析）

### 📊 完整模型清單與問題

| 模型 | 檔案大小 | 格式 | 實測 F1 | 狀態 | 描述 |
|------|----------|------|---------|------|------|
| `bullying_a100_best/pytorch_model.bin` | **391 MB** | bin | **0.826** | ✅ **生產級** | L4 GPU 訓練，達標模型 |
| `working_toxicity_model/pytorch_model.bin` | **397 MB** | bin | 無法載入 | ❌ 不可用 | config.json 格式錯誤 |
| `macbert_base_demo/best.ckpt` | **397 MB** | ckpt | 未測試 | ⚠️ 未驗證 | 需要自定義載入程式碼 |
| `toxicity_only_demo/best.ckpt` | **397 MB** | ckpt | 未測試 | ⚠️ 未驗證 | 需要自定義載入程式碼 |
| `local_training/macbert_aggressive/*.pt` | **16.5 MB** | pt (7檔案) | 0.34 | ❌ 訓練失敗 | 過擬合，性能差 |
| **總計** | **~1.6 GB** | | | | **1 個可用生產模型** |

### ✅ 生產級模型：bullying_a100_best 性能分析

**測試結果**: 5,320 個樣本
```
Class_0 (none):       Precision: 0.902, Recall: 0.796, F1: 0.845  ← 優秀平衡
Class_1 (toxic):      Precision: 0.736, Recall: 0.868, F1: 0.796  ← 高召回率
Class_2 (severe):     Precision: 0.000, Recall: 0.000, F1: 0.000  ← 測試集無樣本
```

**結論**: ✅ **模型達到 Weighted F1=0.826，超越生產標準 0.75**

### 🎯 生產部署準備

**已完成**: ✅ L4 GPU 訓練成功，F1=0.826

1. ✅ **模型訓練**: L4 GPU 訓練完成
   - 模型權重已保存到 `models/bullying_a100_best/`
   - 檔案大小: 391MB (符合預期)
   - 已推送到 GitHub (Git LFS)

2. ✅ **性能驗證**: 實際測試通過
   - Weighted F1: 0.826 (超越目標 0.75)
   - 準確率: 82.4%
   - 使用真實模型權重進行推論測試

3. 🔄 **待執行**: 生產環境部署
   - API 服務部署測試
   - LINE Bot 整合測試
   - 跨領域測試（PTT, Dcard, 微博）

4. 🎯 **後續**: 模型監控與持續改進

## ✅ 專案狀態總結

1. **✅ 生產級模型可用**: bullying_a100_best 達到 F1=0.826，超越目標 0.75
2. **✅ 模型權重完整**: L4 GPU 訓練結果包含完整權重檔案 (391MB)
3. **⚠️ 標籤系統不完整**: role/emotion 標籤在訓練資料中未使用
4. **⚠️ 資料不平衡**: 測試集毒性比例 (39.6%) 與訓練集 (49.4%) 不一致
5. **✅ 模型性能優秀**: 平衡的精確率和召回率，適合生產使用

## 📋 後續發展建議

1. **✅ 已完成**: L4 GPU 訓練，模型達到生產標準 (F1=0.826)
2. **✅ 已完成**: 模型權重保存與驗證
3. **🔄 進行中**: 生產環境部署準備
4. **📋 計劃中**: 整合 role/emotion 多任務學習
5. **🎯 可開始**: API 服務與 LINE Bot 部署

## 資料庫與儲存

### SQLite 資料庫
| 資料庫 | 位置 | 大小 | 用途 |
|--------|------|------|------|
| `hive.db` | `.hive-mind/` | **124 KB** | Hive-mind 協調系統 |
| `memory.db` | `.hive-mind/` | **16 KB** | Hive-mind 記憶體儲存 |
| `memory.db` | `.swarm/` | **112 KB** | Swarm 系統記憶體 |

### Git LFS 大檔案
- 訓練資料: `data/processed/training_dataset/*.json`
- 處理資料: `data/processed/cold/*.csv`
- 情緒資料: `data/processed/chnsenticorp/*.json`

## API 與服務
- **API 端點**: 17 個 RESTful endpoints
- **LINE Bot**: 2 個主要檔案 (25.6 KB)
- **Docker**: docker-compose.yml 部署配置
- **Notebooks**: 5 個 Jupyter notebooks (114 KB)
- **套件依賴**: 160 個 (64 生產 + 96 開發)

## 完成定義（DoD）

- ✅ 單元測試通過（70-90% 核心模組覆蓋，詳見 docs/TEST_COVERAGE_IMPROVEMENTS.md）
- ❌ **離線評估**: 實測 F1=0.28 (目標：毒性≥0.78，霸凌≥0.75) - **嚴重未達標**
- 🔄 情緒 F1=1.00 需大規模驗證；SCCD 會話級報告待生成
- ⚠️ 提供 IG/SHAP 可視化範例與誤判分析（部分完成）
- ⚠️ Docker 化的 API 與可用的 LINE Bot Webhook（待驗證）

**深度分析報告**: 詳見 `docs/PROJECT_DEEP_ANALYSIS.md`

---

# Claude Code Configuration - SPARC Development Environment

## 🚨 CRITICAL: CONCURRENT EXECUTION & FILE MANAGEMENT

**ABSOLUTE RULES**:
1. ALL operations MUST be concurrent/parallel in a single message
2. **NEVER save working files, text/mds and tests to the root folder**
3. ALWAYS organize files in appropriate subdirectories
4. **USE CLAUDE CODE'S TASK TOOL** for spawning agents concurrently, not just MCP

### ⚡ GOLDEN RULE: "1 MESSAGE = ALL RELATED OPERATIONS"

**MANDATORY PATTERNS:**
- **TodoWrite**: ALWAYS batch ALL todos in ONE call (5-10+ todos minimum)
- **Task tool (Claude Code)**: ALWAYS spawn ALL agents in ONE message with full instructions
- **File operations**: ALWAYS batch ALL reads/writes/edits in ONE message
- **Bash commands**: ALWAYS batch ALL terminal operations in ONE message
- **Memory operations**: ALWAYS batch ALL memory store/retrieve in ONE message

### 🎯 CRITICAL: Claude Code Task Tool for Agent Execution

**Claude Code's Task tool is the PRIMARY way to spawn agents:**
```javascript
// ✅ CORRECT: Use Claude Code's Task tool for parallel agent execution
[Single Message]:
  Task("Research agent", "Analyze requirements and patterns...", "researcher")
  Task("Coder agent", "Implement core features...", "coder")
  Task("Tester agent", "Create comprehensive tests...", "tester")
  Task("Reviewer agent", "Review code quality...", "reviewer")
  Task("Architect agent", "Design system architecture...", "system-architect")
```

**MCP tools are ONLY for coordination setup:**
- `mcp__claude-flow__swarm_init` - Initialize coordination topology
- `mcp__claude-flow__agent_spawn` - Define agent types for coordination
- `mcp__claude-flow__task_orchestrate` - Orchestrate high-level workflows

### 📁 File Organization Rules

**NEVER save to root folder. Use these directories:**
- `/src` - Source code files
- `/tests` - Test files
- `/docs` - Documentation and markdown files
- `/config` - Configuration files
- `/scripts` - Utility scripts
- `/examples` - Example code

## Project Overview

This project uses SPARC (Specification, Pseudocode, Architecture, Refinement, Completion) methodology with Claude-Flow orchestration for systematic Test-Driven Development.

## SPARC Commands

### Core Commands
- `npx claude-flow sparc modes` - List available modes
- `npx claude-flow sparc run <mode> "<task>"` - Execute specific mode
- `npx claude-flow sparc tdd "<feature>"` - Run complete TDD workflow
- `npx claude-flow sparc info <mode>` - Get mode details

### Batchtools Commands
- `npx claude-flow sparc batch <modes> "<task>"` - Parallel execution
- `npx claude-flow sparc pipeline "<task>"` - Full pipeline processing
- `npx claude-flow sparc concurrent <mode> "<tasks-file>"` - Multi-task processing

### Build Commands
- `npm run build` - Build project
- `npm run test` - Run tests
- `npm run lint` - Linting
- `npm run typecheck` - Type checking

## SPARC Workflow Phases

1. **Specification** - Requirements analysis (`sparc run spec-pseudocode`)
2. **Pseudocode** - Algorithm design (`sparc run spec-pseudocode`)
3. **Architecture** - System design (`sparc run architect`)
4. **Refinement** - TDD implementation (`sparc tdd`)
5. **Completion** - Integration (`sparc run integration`)

## Code Style & Best Practices

- **Modular Design**: Files under 500 lines
- **Environment Safety**: Never hardcode secrets
- **Test-First**: Write tests before implementation
- **Clean Architecture**: Separate concerns
- **Documentation**: Keep updated

## 🚀 Available Agents (54 Total)

### Core Development
`coder`, `reviewer`, `tester`, `planner`, `researcher`

### Swarm Coordination
`hierarchical-coordinator`, `mesh-coordinator`, `adaptive-coordinator`, `collective-intelligence-coordinator`, `swarm-memory-manager`

### Consensus & Distributed
`byzantine-coordinator`, `raft-manager`, `gossip-coordinator`, `consensus-builder`, `crdt-synchronizer`, `quorum-manager`, `security-manager`

### Performance & Optimization
`perf-analyzer`, `performance-benchmarker`, `task-orchestrator`, `memory-coordinator`, `smart-agent`

### GitHub & Repository
`github-modes`, `pr-manager`, `code-review-swarm`, `issue-tracker`, `release-manager`, `workflow-automation`, `project-board-sync`, `repo-architect`, `multi-repo-swarm`

### SPARC Methodology
`sparc-coord`, `sparc-coder`, `specification`, `pseudocode`, `architecture`, `refinement`

### Specialized Development
`backend-dev`, `mobile-dev`, `ml-developer`, `cicd-engineer`, `api-docs`, `system-architect`, `code-analyzer`, `base-template-generator`

### Testing & Validation
`tdd-london-swarm`, `production-validator`

### Migration & Planning
`migration-planner`, `swarm-init`

## 🎯 Claude Code vs MCP Tools

### Claude Code Handles ALL EXECUTION:
- **Task tool**: Spawn and run agents concurrently for actual work
- File operations (Read, Write, Edit, MultiEdit, Glob, Grep)
- Code generation and programming
- Bash commands and system operations
- Implementation work
- Project navigation and analysis
- TodoWrite and task management
- Git operations
- Package management
- Testing and debugging

### MCP Tools ONLY COORDINATE:
- Swarm initialization (topology setup)
- Agent type definitions (coordination patterns)
- Task orchestration (high-level planning)
- Memory management
- Neural features
- Performance tracking
- GitHub integration

**KEY**: MCP coordinates the strategy, Claude Code's Task tool executes with real agents.

## 🚀 Quick Setup

```bash
# Add MCP servers (Claude Flow required, others optional)
claude mcp add claude-flow npx claude-flow@alpha mcp start
claude mcp add ruv-swarm npx ruv-swarm mcp start  # Optional: Enhanced coordination
claude mcp add flow-nexus npx flow-nexus@latest mcp start  # Optional: Cloud features
```

## MCP Tool Categories

### Coordination
`swarm_init`, `agent_spawn`, `task_orchestrate`

### Monitoring
`swarm_status`, `agent_list`, `agent_metrics`, `task_status`, `task_results`

### Memory & Neural
`memory_usage`, `neural_status`, `neural_train`, `neural_patterns`

### GitHub Integration
`github_swarm`, `repo_analyze`, `pr_enhance`, `issue_triage`, `code_review`

### System
`benchmark_run`, `features_detect`, `swarm_monitor`

### Flow-Nexus MCP Tools (Optional Advanced Features)
Flow-Nexus extends MCP capabilities with 70+ cloud-based orchestration tools:

**Key MCP Tool Categories:**
- **Swarm & Agents**: `swarm_init`, `swarm_scale`, `agent_spawn`, `task_orchestrate`
- **Sandboxes**: `sandbox_create`, `sandbox_execute`, `sandbox_upload` (cloud execution)
- **Templates**: `template_list`, `template_deploy` (pre-built project templates)
- **Neural AI**: `neural_train`, `neural_patterns`, `seraphina_chat` (AI assistant)
- **GitHub**: `github_repo_analyze`, `github_pr_manage` (repository management)
- **Real-time**: `execution_stream_subscribe`, `realtime_subscribe` (live monitoring)
- **Storage**: `storage_upload`, `storage_list` (cloud file management)

**Authentication Required:**
- Register: `mcp__flow-nexus__user_register` or `npx flow-nexus@latest register`
- Login: `mcp__flow-nexus__user_login` or `npx flow-nexus@latest login`
- Access 70+ specialized MCP tools for advanced orchestration

## 🚀 Agent Execution Flow with Claude Code

### The Correct Pattern:

1. **Optional**: Use MCP tools to set up coordination topology
2. **REQUIRED**: Use Claude Code's Task tool to spawn agents that do actual work
3. **REQUIRED**: Each agent runs hooks for coordination
4. **REQUIRED**: Batch all operations in single messages

### Example Full-Stack Development:

```javascript
// Single message with all agent spawning via Claude Code's Task tool
[Parallel Agent Execution]:
  Task("Backend Developer", "Build REST API with Express. Use hooks for coordination.", "backend-dev")
  Task("Frontend Developer", "Create React UI. Coordinate with backend via memory.", "coder")
  Task("Database Architect", "Design PostgreSQL schema. Store schema in memory.", "code-analyzer")
  Task("Test Engineer", "Write Jest tests. Check memory for API contracts.", "tester")
  Task("DevOps Engineer", "Setup Docker and CI/CD. Document in memory.", "cicd-engineer")
  Task("Security Auditor", "Review authentication. Report findings via hooks.", "reviewer")
  
  // All todos batched together
  TodoWrite { todos: [...8-10 todos...] }
  
  // All file operations together
  Write "backend/server.js"
  Write "frontend/App.jsx"
  Write "database/schema.sql"
```

## 📋 Agent Coordination Protocol

### Every Agent Spawned via Task Tool MUST:

**1️⃣ BEFORE Work:**
```bash
npx claude-flow@alpha hooks pre-task --description "[task]"
npx claude-flow@alpha hooks session-restore --session-id "swarm-[id]"
```

**2️⃣ DURING Work:**
```bash
npx claude-flow@alpha hooks post-edit --file "[file]" --memory-key "swarm/[agent]/[step]"
npx claude-flow@alpha hooks notify --message "[what was done]"
```

**3️⃣ AFTER Work:**
```bash
npx claude-flow@alpha hooks post-task --task-id "[task]"
npx claude-flow@alpha hooks session-end --export-metrics true
```

## 🎯 Concurrent Execution Examples

### ✅ CORRECT WORKFLOW: MCP Coordinates, Claude Code Executes

```javascript
// Step 1: MCP tools set up coordination (optional, for complex tasks)
[Single Message - Coordination Setup]:
  mcp__claude-flow__swarm_init { topology: "mesh", maxAgents: 6 }
  mcp__claude-flow__agent_spawn { type: "researcher" }
  mcp__claude-flow__agent_spawn { type: "coder" }
  mcp__claude-flow__agent_spawn { type: "tester" }

// Step 2: Claude Code Task tool spawns ACTUAL agents that do the work
[Single Message - Parallel Agent Execution]:
  // Claude Code's Task tool spawns real agents concurrently
  Task("Research agent", "Analyze API requirements and best practices. Check memory for prior decisions.", "researcher")
  Task("Coder agent", "Implement REST endpoints with authentication. Coordinate via hooks.", "coder")
  Task("Database agent", "Design and implement database schema. Store decisions in memory.", "code-analyzer")
  Task("Tester agent", "Create comprehensive test suite with 90% coverage.", "tester")
  Task("Reviewer agent", "Review code quality and security. Document findings.", "reviewer")
  
  // Batch ALL todos in ONE call
  TodoWrite { todos: [
    {id: "1", content: "Research API patterns", status: "in_progress", priority: "high"},
    {id: "2", content: "Design database schema", status: "in_progress", priority: "high"},
    {id: "3", content: "Implement authentication", status: "pending", priority: "high"},
    {id: "4", content: "Build REST endpoints", status: "pending", priority: "high"},
    {id: "5", content: "Write unit tests", status: "pending", priority: "medium"},
    {id: "6", content: "Integration tests", status: "pending", priority: "medium"},
    {id: "7", content: "API documentation", status: "pending", priority: "low"},
    {id: "8", content: "Performance optimization", status: "pending", priority: "low"}
  ]}
  
  // Parallel file operations
  Bash "mkdir -p app/{src,tests,docs,config}"
  Write "app/package.json"
  Write "app/src/server.js"
  Write "app/tests/server.test.js"
  Write "app/docs/API.md"
```

### ❌ WRONG (Multiple Messages):
```javascript
Message 1: mcp__claude-flow__swarm_init
Message 2: Task("agent 1")
Message 3: TodoWrite { todos: [single todo] }
Message 4: Write "file.js"
// This breaks parallel coordination!
```

## Performance Benefits

- **84.8% SWE-Bench solve rate**
- **32.3% token reduction**
- **2.8-4.4x speed improvement**
- **27+ neural models**

## Hooks Integration

### Pre-Operation
- Auto-assign agents by file type
- Validate commands for safety
- Prepare resources automatically
- Optimize topology by complexity
- Cache searches

### Post-Operation
- Auto-format code
- Train neural patterns
- Update memory
- Analyze performance
- Track token usage

### Session Management
- Generate summaries
- Persist state
- Track metrics
- Restore context
- Export workflows

## Advanced Features (v2.0.0)

- 🚀 Automatic Topology Selection
- ⚡ Parallel Execution (2.8-4.4x speed)
- 🧠 Neural Training
- 📊 Bottleneck Analysis
- 🤖 Smart Auto-Spawning
- 🛡️ Self-Healing Workflows
- 💾 Cross-Session Memory
- 🔗 GitHub Integration

## Integration Tips

1. Start with basic swarm init
2. Scale agents gradually
3. Use memory for context
4. Monitor progress regularly
5. Train patterns from success
6. Enable hooks automation
7. Use GitHub tools first

## Support

- Documentation: https://github.com/ruvnet/claude-flow
- Issues: https://github.com/ruvnet/claude-flow/issues
- Flow-Nexus Platform: https://flow-nexus.ruv.io (registration required for cloud features)

---

Remember: **Claude Flow coordinates, Claude Code creates!**

# important-instruction-reminders
Do what has been asked; nothing more, nothing less.
NEVER create files unless they're absolutely necessary for achieving your goal.
ALWAYS prefer editing an existing file to creating a new one.
NEVER proactively create documentation files (*.md) or README files. Only create documentation files if explicitly requested by the User.
Never save working files, text/mds and tests to the root folder.
