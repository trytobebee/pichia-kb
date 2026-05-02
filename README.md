# kb-core

通用的、AI 辅助的"陌生领域知识构建"框架。喂给它若干篇领域论文,得到一个**可问答的结构化知识库** + **可复用的实验工艺单** + **可对比的实验数据** + **可对话编辑的 schema**。

首个项目实例: **pichia-collagen** — 毕赤酵母 (*Pichia pastoris* / *Komagataella phaffii*) 发酵生产重组人源胶原蛋白,7 篇论文。框架代码与领域内容彻底解耦 — 任意新领域 (法律 / 临床试验 / 综述等) 创建一个新 project 即可复用。

---

## 快速开始

```bash
# 1. 配置 API Key
cp .env.example .env
# 编辑 .env,填入 GEMINI_API_KEY=your_key

# 2. 启动 web UI(自动 uv sync,打开 http://localhost:8501)
./scripts/start.sh

# 或手动:
uv sync
source .venv/bin/activate
export $(grep -v '^#' .env | xargs)
streamlit run web/Home.py
```

打开浏览器后:
1. 侧边栏选择 `pichia-collagen`(或点 `+ New project` 新建一个)
2. 进 **📄 论文管理** 页拖入 PDF + 触发抽取
3. 进 **💬 问答** 页问问题
4. 进 **🛠️ Schema Curator** 页跟 LLM agent 对话调整 schema

---

## 核心概念

### Project — 顶层隔离单元

每个研究方向是一个独立项目,数据完全隔离在 `data/projects/<slug>/` 下。框架代码与领域内容彻底分离:**所有 domain-specific 的东西**(prompt、schema、关键词、QA 角色描述)**只存在于 `data/projects/<slug>/`**,框架代码 grep 不到任何领域词。

```
data/projects/<slug>/
├── papers/                # 原始 PDF(每篇 .pdf;不入 git)
├── db/                    # ChromaDB 向量库(自动生成)
├── cache/                 # PDF 文本缓存
├── figures/               # 图表 PNG
├── structured/            # 抽取产出(实体 / 实验 / 图表 / 综述 JSON)
├── config.yaml            # 领域 context(expert_field / keywords / qa_role / ...)
├── schema/
│   ├── knowledge.json     # 实体类型 schema(Strain / Promoter / ...)
│   ├── experiments.json   # ExperimentRun + 嵌套子类型
│   └── data.json          # FigureData + 嵌套
├── schema_audit.jsonl     # schema 修改流水(curator agent 写)
└── tasks.jsonl            # 抽取任务历史(orchestrator 写)
```

### 三系统 — 知识 / 实验 / 数据

知识体系是**主体**(领域是怎么回事),实验和数据是它在不同维度的具体化(如何获取/验证 + 实际证据)。三个 schema 文件分别承载:

| 文件 | 内容 | 谁产生 |
|---|---|---|
| `schema/knowledge.json` | 实体类型 + 概念关系(Strain, Promoter, TargetProduct, ...) | `kb add` / `kb ingest` |
| `schema/experiments.json` | ExperimentRun + 嵌套(StrainConstruct, FermentationSetup, PhaseParams, ExperimentOutcome) | `kb extract-experiments` |
| `schema/data.json` | FigureData + DataPoint + NotablePoint + ExperimentalVariable | `kb extract-figures` |

### Schema-as-data + 动态 Pydantic

Schema 不是写死在 Python 代码里,而是 JSON 文件。`kb_core/schema_engine/` 在运行时用 Pydantic v2 `create_model()` 动态构建实际的 Pydantic 类。换领域只需要换 JSON。

### Schema Curator Agent — 对话式 schema 维护

🛠️ Schema Curator 页是一个 Gemini function-calling agent,它有 8 个工具:
- **inspection**: `find_papers_with_field` / `compute_field_completeness` / `query_schema_provenance`
- **mutations**: `add_field` / `remove_field` / `rename_field` / `add_entity_type` / `record_rejection`

工作模式:你跟它说自然语言,它先用 inspection 工具查证,再决定是否动 schema。每次修改自动写入 `schema_audit.jsonl`(可在 sidebar 实时看到)。

### Orchestrator + tasks.jsonl

📄 论文管理页底层是 `kb_core/orchestrator/`:每次抽取任务用 `task()` context manager 自动计时 + 写 `tasks.jsonl`。状态矩阵直接从 jsonl 派生("most recent entry per `(paper, stage)`")。

---

## 项目结构

```
pichia-kb/                       (this repo, also the code root)
├── .env                         # API 密钥(不入 git)
├── pyproject.toml
├── scripts/
│   └── start.sh                 # 一键启动 web UI
│
├── src/kb_core/                 # 100% 通用框架代码,grep 不到任何 Pichia 字串
│   ├── schema_engine/           # 元 schema + 动态 Pydantic + audit
│   ├── ingestion/               # 通用抽取流水线(prompt 从 schema 动态生成)
│   ├── qa/                      # 通用 RAG (Assistant + DomainContext)
│   ├── knowledge_base/          # KB facade + ChromaDB + structured store
│   ├── curator/                 # schema curator agent + tools
│   ├── orchestrator/            # task runner + tasks.jsonl
│   ├── templates/               # 领域种子模板(数据,非代码)
│   │   └── fermentation/        # config.yaml + schema/ 三件套,可 fork
│   ├── config.py                # ProjectConfig / DomainContext loader
│   └── cli.py                   # CLI(`kb` 命令)
│
├── data/projects/<slug>/        # 项目实例(纯数据,大部分不入 git)
│   ├── papers/  cache/  figures/  db/  structured/
│   ├── config.yaml  schema/  schema_audit.jsonl  tasks.jsonl
│
├── web/                         # Streamlit UI
│   ├── Home.py                  # 首页(项目概览)
│   ├── _project.py              # 项目选择器 + 共享 helpers
│   └── pages/
│       ├── 1_💬_问答.py
│       ├── 2_🔬_控制原则.py
│       ├── 3_⚖️_辩证综合.py
│       ├── 4_🔍_语义搜索.py
│       ├── 5_🧪_实验抽取.py
│       ├── 6_📊_跨论文对比.py
│       ├── 7_🛠️_Schema_Curator.py    # 对话式 schema 编辑
│       └── 8_📄_论文管理.py            # PDF 上传 + 触发抽取
│
└── tests/                       # 30+ tests, no API calls required
```

---

## CLI 参考

所有命令都需要 `--project <slug>` (简写 `-p`):

### 项目管理

```bash
kb list-projects                                # 列出所有项目
kb list-templates                               # 列出可用 seed 模板
kb new-project my-domain --template fermentation --name "My Domain KB"
                                                # 从模板创建新项目
```

### 知识摄入

```bash
# 单篇:vector store + 实体抽取
kb ingest data/projects/pichia-collagen/papers/新论文.pdf --project pichia-collagen

# 一键 ingest + synthesize 控制原则
kb add data/projects/pichia-collagen/papers/新论文.pdf --project pichia-collagen

# 跨论文辩证综合(批量新增后跑一次)
kb review --project pichia-collagen

# 图表抽取 / 实验抽取(独立 stage)
kb extract-figures data/projects/pichia-collagen/papers/ --project pichia-collagen
kb extract-experiments data/projects/pichia-collagen/papers/ --project pichia-collagen
```

### 查看与查询

```bash
kb status --project pichia-collagen             # 全项目统计
kb principles --project pichia-collagen         # 控制原则
kb show-review --project pichia-collagen        # 辩证评审
kb show-domain --project pichia-collagen        # 领域全景
kb entities target_products --project pichia-collagen
kb search "胶原蛋白羟基化" --project pichia-collagen
```

### 问答

```bash
kb ask "如何选择 P4H 共表达策略?" --project pichia-collagen
kb chat --project pichia-collagen               # 多轮交互
kb ask "..." --model gemini-2.5-pro --project pichia-collagen
kb ask "..." --n-chunks 10 --project pichia-collagen
```

---

## 新项目完整流程

```bash
# 1. 创建项目骨架(把 fermentation 模板复制到 data/projects/<slug>/)
kb new-project ecoli-protease --template fermentation --name "E. coli Protease KB"

# 2. 编辑 config.yaml,把 expert_field / paper_topic / keywords / entity_hints 改成你自己的
$EDITOR data/projects/ecoli-protease/config.yaml

# 3. 把 PDF 放进 data/projects/ecoli-protease/papers/(或者 web 上拖)

# 4. 跑抽取(任选,或全跑)
kb add data/projects/ecoli-protease/papers/ --project ecoli-protease
kb extract-experiments data/projects/ecoli-protease/papers/ --project ecoli-protease
kb extract-figures data/projects/ecoli-protease/papers/ --project ecoli-protease
kb review --project ecoli-protease

# 5. 问答
kb ask "..." --project ecoli-protease

# 6. 用 web 端 🛠️ Schema Curator 边用边演化 schema
```

---

## 依赖

| 库 | 版本 | 用途 |
|---|---|---|
| google-genai | ≥1.73 | Gemini API(抽取/综合/问答/curator) |
| chromadb | ≥1.5 | 本地向量库 |
| pdfplumber + pymupdf | — | PDF 文本/图像提取 |
| pydantic | ≥2.13 | schema 验证(含 dynamic create_model) |
| typer + rich | — | CLI |
| streamlit | ≥1.56 | Web UI |

**API 依赖:** 至少配一个 LLM provider 的 key。多 provider 共存,model 字符串决定走哪家:

| Provider | Key (.env) | Model 前缀 | 适合场景 |
|---|---|---|---|
| Google Gemini | `GEMINI_API_KEY` | `gemini-*` | 默认;图表抽取必须用(vision);海外网络 |
| DeepSeek | `DEEPSEEK_API_KEY` | `deepseek-*` | OpenAI-兼容;**国内服务器友好**(Gemini 出海受限时) |
| OpenAI | `OPENAI_API_KEY` | `gpt-*` / `o1-*` / `o3-*` | OpenAI 直连 |

CLI 用 `--model deepseek-chat` 切到 DeepSeek 例如:
```
kb ask "..." --project pichia-collagen --model deepseek-chat
```

**仍只支持 Gemini 的两个能力**:
- `kb extract-figures`(vision API,DeepSeek 没有 vision 模型)
- 🛠️ Schema Curator 页(用 Gemini 的 function-calling 协议;OpenAI-compat 的 tool calling 后续可加)

**运行环境:** Python ≥ 3.12,包管理用 [uv](https://docs.astral.sh/uv/)。

---

## 测试

```bash
uv run pytest tests/ -v
# 30+ tests,不需要 API key,验证 schema_engine + KB roundtrip + curator 工具
```
