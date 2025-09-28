# CLAUDE.md

## 專案宗旨
CyberPuppy 是一個以「網路霸凌防治」為核心，結合 **毒性偵測** 與 **情緒分析** 的中文聊天機器人。它需在私訊與群組對話中，以**高可解釋性**、**低誤傷**的方式提供即時提醒。

## 專案規模 (2025-09-29 深度分析)
- **總檔案數**: 9,185 個檔案 (~3.7 GB)
- **程式碼**: 254 個 Python 檔案 (79 個核心模組)
- **測試**: 70 個測試檔案，179+ 測試用例
- **資料**: 4.75M 行資料 (666 MB)
- **模型**: 10 個模型檔案 (1.6 GB)

## 任務與資料
### 資料集規模
- **COLD**: 37,483 樣本 (主要訓練資料)
- **DMSC**: 2,131,887 樣本 (情緒分析，387 MB)
- **ChnSentiCorp**: 15,534 樣本 (情感分析)
- **訓練集分割**: train(25,659) / dev(6,430) / test(5,320)

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

## 模型現況（2025-09-29 更新）

### ⚠️ A100 訓練結果（需重新訓練）

**評估結果**: `models/bullying_a100_best/` (2025-09-27)

| 任務 | 測試集 F1 | 目標 | 狀態 |
|------|-----------|------|------|
| **毒性偵測** | **0.8206** | 0.78 | ✅ **超越目標** (+5.3%) |
| **霸凌偵測** | **0.8207** | 0.75 | ✅ **超越目標** (+9.4%) |
| **情緒分析** | 1.00* | 0.85 | ✅ 達標 |

**⚠️ 重要**: 此目錄僅包含評估結果 JSON 檔案，**不包含模型權重檔案** (`.safetensors`/`.bin`)。
- 目錄大小: 595KB (應為 ~400MB)
- 內容: tokenizer + JSON 結果
- **狀態**: ❌ 無法直接部署使用

**來源**: `models/bullying_a100_best/final_results.json`

### ⚠️ 模型性能驗證結果（2025-09-29）

**驗證報告**: `docs/MODEL_VERIFICATION_REPORT.md`

| 模型 | 位置 | 聲稱 F1 | **實際 F1** | 狀態 | 問題 |
|------|------|---------|-------------|------|------|
| `gpu_trained_model` | models/gpu_trained_model/ | 0.77 | **0.28** | ❌ **不可用** | 實際性能僅 28%，幾乎無法檢測毒性內容 |
| `working_toxicity_model` | models/working_toxicity_model/ | - | **無法載入** | ❌ 不可用 | config.json 格式錯誤 |
| `bullying_a100_best` | models/bullying_a100_best/ | 0.82 | **未驗證** | ⚠️ 缺少權重 | 只有評估結果，無 .safetensors/.bin |
| `local_training/macbert_aggressive` | models/local_training/macbert_aggressive/ | - | **0.34** | ❌ 不可用 | 訓練未收斂，過擬合 |

**結論**: ❌ **目前沒有可用的生產級模型**（所有測試模型均未達 F1≥0.75 目標）

### 🔄 下一步行動

**優先**: 重新在 A100 上訓練並正確保存模型權重

1. ✅ **已更新** `notebooks/train_on_colab_a100.ipynb`:
   - 添加模型權重檔案驗證
   - 確保 Git LFS 正確追蹤大檔案
   - 可選 Google Drive 備份

2. 🔄 **待執行**: 在 Colab A100 上重新訓練
   - 確認模型權重正確保存到 `models/bullying_a100_best/`
   - 驗證檔案大小 (~400MB)
   - 推送到 GitHub (Git LFS)

3. 📊 **驗證**: 重現 F1=0.82 的成績
   - 使用實際模型權重進行推論測試
   - 確認不是僅基於 JSON 檔案的聲明

4. 🎯 **後續**: 模型穩定性與跨領域測試（PTT, Dcard, 微博）

## 資料庫與儲存

### SQLite 資料庫
- **hive.db**: 124 KB (`.hive-mind/` - Hive-mind 協調系統)
- **memory.db**: 16 KB + 112 KB (記憶體儲存)

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
