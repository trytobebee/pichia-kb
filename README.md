# pichia-kb

基于论文的毕赤酵母（*Pichia pastoris* / *Komagataella phaffii*）发酵知识库与问答系统。

目标：通过整合多篇领域论文，构建一个可驱动实验设计的权威问答系统，帮助研究人员可控地生产具有特定结构、修饰和活性的目标产物（当前聚焦：重组人源胶原蛋白）。

---

## 目录

- [项目结构](#项目结构)
- [数据与知识组织](#数据与知识组织)
- [三层知识架构](#三层知识架构)
- [依赖说明](#依赖说明)
- [快速开始](#快速开始)
- [命令参考](#命令参考)
- [新增论文的完整流程](#新增论文的完整流程)

---

## 项目结构

```
pichia-kb/
├── .env                        # API 密钥（不入 git）
├── .env.example                # 密钥模板
├── pyproject.toml              # 项目配置与依赖
│
├── src/pichia_kb/              # 核心源码包
│   ├── schema/                 # 知识实体定义（Pydantic 模型）
│   │   ├── entities.py         # 基础领域实体（菌株、载体、培养基等）
│   │   ├── process_knowledge.py # 过程控制知识（控制原则、发酵方案等）
│   │   └── dialectical.py      # 辩证综合实体（共识、冲突、置信度）
│   │
│   ├── ingestion/              # 知识摄入流水线
│   │   ├── pdf_processor.py    # PDF 解析 → 文本分块
│   │   ├── extractor.py        # 逐块 LLM 抽取结构化实体
│   │   ├── synthesizer.py      # 全文 LLM 合成发酵控制原则
│   │   └── dialectical_reviewer.py  # 跨论文辩证综合
│   │
│   ├── knowledge_base/         # 知识存储与检索
│   │   ├── vector_store.py     # ChromaDB 向量库（语义检索）
│   │   ├── structured_store.py # JSON 结构化存储（实体 + 控制原则 + 辩证综合）
│   │   └── kb.py               # 统一入口，构建 RAG 上下文
│   │
│   ├── qa/
│   │   └── assistant.py        # RAG + Gemini 问答引擎
│   └── cli.py                  # 命令行界面
│
├── data/
│   └── projects/<slug>/        # 每个项目独立隔离的数据
│       ├── papers/             # 原始 PDF 论文（放这里）
│       ├── db/                 # ChromaDB 向量库（自动生成，不入 git）
│       ├── cache/              # PDF 文本缓存
│       ├── figures/            # 图表 PNG（自动生成）
│       └── structured/         # JSON 知识文件（自动生成）
│
└── tests/
    └── test_schema.py          # 单元测试（无需 API）
```

---

## 数据与知识组织

### 项目模型

每个研究方向是一个独立的**项目** (project),数据完全隔离在 `data/projects/<slug>/` 下。所有 CLI 命令都需要 `--project <slug>` (简写 `-p`) 显式指定操作哪个项目;web 端有项目选择器。当前仓库内置一个项目:

| 项目 slug | 内容 |
|---|---|
| `pichia-collagen` | 毕赤酵母重组人源胶原蛋白发酵 7 篇论文 |

未来其他领域(法律、临床试验、综述等)以新 slug 创建,共享框架代码,数据互不干扰。

### `data/projects/<slug>/papers/`

存放原始 PDF 论文。`pichia-collagen` 项目当前收录 7 篇:

| 文件 | 内容 |
|------|------|
| 毕赤酵母重组蛋白高效表达策略及应用_耿坤.pdf | 综述：启动子、拷贝数、低温、抑制蛋白酶等高效表达策略 |
| 甲醇在巴氏毕赤酵母胶原蛋白生产中的流加控制及风险管理_成中山.pdf | 甲醇浓度在线检测与闭环反馈控制 |
| Ⅲ型类人胶原蛋白在毕赤酵母中的表达优化及其生物活性初探_王婧.pdf | III 型胶原蛋白发酵优化（温度、pH、DO） |
| 具有羟基化功能的人源胶原蛋白毕赤酵母工程菌的构建_林雪媛.pdf | P4H 双质粒共转化实现羟基化 |
| 表达重组人源化Ⅲ型胶原蛋白毕赤酵母工程菌的构建_钱娜娜.pdf | 2A 肽共表达 P4H 策略 |
| 重组Ⅲ型人源化胶原蛋白表达体系构建与性质探究_刘伟兴.pdf | COL3 表达体系构建与性质 |
| 重组人Ⅰ型、Ⅱ型胶原蛋白α1链的全长表达优化及应用_李佳佳.pdf | I/II 型胶原蛋白全长表达 |

### `data/projects/<slug>/db/`

ChromaDB 向量库，存储所有论文的文本 chunks，支持语义相似度检索。由 `ingest` 命令自动写入，**不需要手动操作**。

### `data/projects/<slug>/cache/`

PDF 原文文本缓存（`<论文名>.txt`）。所有摄入命令首次解析 PDF 时写入，之后所有阶段（ingest / synthesize / review / domain-knowledge / extract-figures）直接读缓存，避免重复 pdfplumber 解析；也方便排查"LLM 看到了什么"。删除该目录会触发重新解析，不影响其他数据。

### `data/projects/<slug>/structured/`

JSON 格式的结构化知识，包含 3 类文件：

| 文件 | 产生命令 | 内容 |
|------|----------|------|
| `{论文名}.json` | `ingest` | 从该论文抽取的结构化实体（菌株、启动子、发酵条件、目标产物等） |
| `process_knowledge.json` | `synthesize` | 每篇论文的控制原则、工艺阶段、故障排查、产品质量因子（29 条控制原则、10 个工艺阶段、9 条故障排查、16 条质量因子） |
| `dialectical_review.json` | `review` | 跨论文辩证综合（8 个主题的共识与冲突分析） |

---

## 三层知识架构

问答系统在回答问题时，会同时调用三层上下文：

```
Layer 3  辩证综合层        dialectical_review.json
         ─────────────────────────────────────────
         跨论文比较：哪些结论互相印证（高置信度）？
         哪些存在冲突（需要实验验证）？
         → 注入：高置信发现 + 冲突警告 + 相关主题的完整分析

Layer 2  控制原则层        process_knowledge.json
         ─────────────────────────────────────────
         每篇论文提炼的发酵控制逻辑：
         控制原则（参数 → 机理 → 建议 → 目标值）
         产品质量因子（工艺参数 → 结构/修饰/活性）
         故障排查（问题 → 根因 → 解决方案）
         → 注入：与问题相关的原则列表

Layer 1  实体 + 语义检索层  向量库 + {论文}.json
         ─────────────────────────────────────────
         论文原文语义检索（最相关的 6 段原文）
         结构化实体（菌株、启动子、目标产物等）
         → 注入：Top-N 相关文本段落
```

---

## 依赖说明

| 库 | 版本要求 | 用途 |
|----|----------|------|
| **google-genai** | ≥1.73 | Gemini API：LLM 抽取、合成、问答 |
| **chromadb** | ≥1.5 | 本地向量数据库，存储并检索论文 chunks |
| **pdfplumber** | ≥0.11 | 从 PDF 中提取文本（支持中文） |
| **pydantic** | ≥2.13 | 知识实体的 schema 定义与数据验证 |
| **typer** | ≥0.24 | CLI 框架，提供所有命令行命令 |
| **rich** | ≥15.0 | 终端格式化输出（表格、面板、彩色） |
| **python-dotenv** | ≥1.2 | 从 `.env` 文件加载 API Key |
| **pyyaml** | ≥6.0 | 预留：配置文件支持 |
| **hatchling** | ≥1.29 | Python 包构建后端 |
| pytest（dev） | ≥9.0 | 单元测试 |

**外部 API 依赖：**

- **Gemini API**（Google AI Studio）：需要在 `.env` 中配置 `GEMINI_API_KEY`
  - `ingest`、`synthesize` 使用 `gemini-2.5-flash`（速度快、成本低）
  - `review` 使用 `gemini-2.5-pro`（质量更高，用于跨论文综合）
  - `chat` / `ask` 默认使用 `gemini-2.5-flash`，可用 `--model` 切换

**运行环境：**
- Python ≥ 3.12
- 包管理：[uv](https://docs.astral.sh/uv/)

---

## 快速开始

```bash
# 1. 进入项目目录
cd ~/code/pichia-kb

# 2. 配置 API Key
cp .env.example .env
# 编辑 .env，填入 GEMINI_API_KEY=your_key

# 3. 安装依赖（已完成则跳过）
uv sync

# 4. 直接提问（知识库已构建）
uv run pichia-kb ask "甲醇诱导阶段的最佳温度和pH是多少？" --project pichia-collagen

# 5. 交互式问答
uv run pichia-kb chat --project pichia-collagen

# 6. 启动 Web UI(浏览 + 多轮问答 + 跨论文对比 + 实验/控制原则浏览)
uv run streamlit run web/Home.py
# 浏览器自动打开 http://localhost:8501
# 侧边栏顶部有 "📁 Project" 选择器
```

---

## 命令参考

### 知识摄入（添加新论文时使用）

```bash
# 摄入单篇论文（文本分块 → 向量库 + 结构化实体抽取）
uv run pichia-kb ingest data/projects/pichia-collagen/papers/新论文.pdf --project pichia-collagen

# 摄入整个目录所有 PDF
uv run pichia-kb ingest data/projects/pichia-collagen/papers/ --project pichia-collagen

# 从论文中提炼发酵控制原则（建议每次 ingest 后运行）
uv run pichia-kb synthesize data/projects/pichia-collagen/papers/新论文.pdf --project pichia-collagen

# 跨论文辩证综合（所有论文都摄入后运行，约 1-2 分钟）
uv run pichia-kb review --project pichia-collagen
```

### 知识查看

```bash
# 查看知识库整体统计
uv run pichia-kb status --project pichia-collagen

# 查看控制原则
uv run pichia-kb principles --project pichia-collagen

# 查看故障排查
uv run pichia-kb principles --category troubleshooting --project pichia-collagen

# 查看发酵工艺阶段
uv run pichia-kb principles --category process_stages --project pichia-collagen

# 查看产品质量因子
uv run pichia-kb principles --category product_quality_factors --project pichia-collagen

# 查看辩证评审（所有主题）
uv run pichia-kb show-review --project pichia-collagen

# 查看辩证评审（按关键词过滤）
uv run pichia-kb show-review --topic "Temperature" --project pichia-collagen
uv run pichia-kb show-review --topic "P4H" --project pichia-collagen
uv run pichia-kb show-review --topic "Methanol" --project pichia-collagen

# 查看抽取到的结构化实体
uv run pichia-kb entities target_products --project pichia-collagen
uv run pichia-kb entities promoters --project pichia-collagen
uv run pichia-kb entities fermentation_conditions --project pichia-collagen

# 语义搜索（返回原文段落）
uv run pichia-kb search "甲醇流加控制" --project pichia-collagen
uv run pichia-kb search "胶原蛋白羟基化" --project pichia-collagen
```

### 问答

```bash
# 单次提问（流式输出）
uv run pichia-kb ask "如何选择P4H共表达策略？" --project pichia-collagen

# 交互式对话（支持多轮，/reset 清除历史，/quit 退出）
uv run pichia-kb chat --project pichia-collagen

# 使用更强的模型回答
uv run pichia-kb ask "..." --model gemini-2.5-pro --project pichia-collagen

# 调整检索上下文数量（默认 6）
uv run pichia-kb ask "..." --n-chunks 10 --project pichia-collagen
```

---

## 新增论文的完整流程

```bash
# Step 1: 将 PDF 放入 data/projects/pichia-collagen/papers/

# Step 2: 一键添加（ingest + synthesize 合并）【必须】
uv run pichia-kb add data/projects/pichia-collagen/papers/新论文.pdf --project pichia-collagen

# Step 3: 更新跨论文辩证综合【可选，建议批量新增后做一次】
uv run pichia-kb review --project pichia-collagen
```

**什么时候需要跑 `review`？**
- 新增了 2 篇以上论文之后
- 想重新获得包含新论文的共识/冲突分析时
- 不是每次 `add` 都必须立刻跑，可以积累几篇再统一更新

> `review` 会覆盖之前的辩证综合结果，总是读取所有已有论文重新综合（调用 gemini-2.5-pro，约 1-2 分钟）。

---

## 运行测试

```bash
# 不需要 API Key，测试 schema 和向量库本地逻辑
uv run pytest tests/ -v
```
