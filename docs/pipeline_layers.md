# Pipeline Layers — kb-core 数据加工流水线

整个框架把"一堆 PDF 论文"变成"可问答的知识库"分成 **7 层**。每层有自己的输入、输出、存储位置、CLI 命令。本表是项目内的权威结构图,改任何一层先来这里对齐。

```
PDF papers
   │
   ▼
[L1] PDF Ingestion           ──→ 文本块(chunks)
   │
   ├──→ [L2] Vector Store    ──→ ChromaDB(语义检索用)
   │
   └──→ [L3] Entity Extraction       ──→ 每篇 .pdf.json
              │
              ├──→ [L3.5] Cross-paper Registry  ──→ entity_registry.json
              │
              ├──→ [L4a] Process Knowledge      ──→ process_knowledge.json
              │           │
              │           └──→ [L4b] Dialectical Review  ──→ dialectical_review.json
              │
              ├──→ [L4c] Domain Knowledge       ──→ domain_knowledge.json
              │
              ├──→ [L5] Figure Extraction       ──→ figures/*.json (vision)
              │
              └──→ [L6] Experiment Extraction   ──→ *.experiments.json
                          │
                          └──→ Lineage edges    (写在 *.experiments.json 内)

   所有 L2~L6 的产物
   ─────────────────────────────► [L7] QA Context Assembly  ──→ 用户问答
```

---

## Layer 总表

| Layer | 名称 | 输入 | 输出文件 | 模块 | CLI 命令 | 默认模型 |
|---|---|---|---|---|---|---|
| **L1** | PDF Ingestion | PDF 文件 | 内存中的 chunks(不持久化) | `kb_core/ingestion/pdf_processor.py` | (内嵌于 L2/L3) | — |
| **L2** | Vector Store | chunks | `data/projects/<slug>/db/`(ChromaDB) | `kb_core/knowledge_base/vector_store.py` | `kb add` | bge-m3(本地 CPU 推理) |
| **L3** | Entity Extraction | chunks | `structured/<paper>.pdf.json` | `ingestion/extractor.py` | `kb add` / `kb ingest` | gemini-2.5-flash |
| **L3.5** | Cross-paper Registry | 所有 L3 输出 | `structured/entity_registry.json` | `ingestion/cross_normalizer.py` | `kb build-registry` | gemini-2.5-flash(同义词聚类)|
| **L4a** | Process Knowledge | PDF 文本 | `structured/process_knowledge.json`(列表,每篇一条) | `ingestion/synthesizer.py` | `kb synthesize` | gemini-2.5-flash |
| **L4b** | Dialectical Review | L4a 输出 | `structured/dialectical_review.json` | `ingestion/dialectical_reviewer.py` | `kb review` | gemini-2.5-pro(高质量推理) |
| **L4c** | Domain Knowledge | PDF 文本(全部 7 篇)| `structured/domain_knowledge.json` | `ingestion/domain_synthesizer.py` | `kb domain-knowledge` | gemini-2.5-flash |
| **L5** | Figure Extraction | PDF 图像(pymupdf 抽出)+ 视觉模型 | `structured/figures/<paper>__<fid>.json` | `ingestion/figure_extractor.py` | `kb extract-figures` | gemini-2.5-flash(vision) |
| **L6** | Experiment Extraction | PDF 文本 | `structured/<paper>.experiments.json` | `ingestion/experiment_extractor.py` + `lineage_extractor.py` | `kb extract-experiments` + `kb extract-lineage` | gemini-2.5-flash |
| **L7** | QA Context Assembly | L2~L6 全部 | 一段拼装好的 prompt | `knowledge_base/kb.py` 的 `context_for_query()` | `kb ask` / Web 1_💬_问答 | 任何 LLM(默认按 `KB_DEFAULT_MODEL`) |

---

## 各层产出的内容是什么?

### L1 — PDF Ingestion(切块)
- 用 pymupdf / pdfplumber 把 PDF 切成 ~500 字符的文本块
- 内存中,不持久化
- 给 L2 做向量化、给 L3 做实体抽取

### L2 — Vector Store(语义检索)
- 用 BAAI/bge-m3(1024 维多语言 embedding)把每个 chunk 编成向量
- 存到 ChromaDB 的 `pichia_chunks` collection
- 问答时按 cosine 相似度召回 top-N

### L3 — Entity Extraction(每篇,实体抽取)
- 把每篇论文的 chunks 喂给 LLM,按项目 `schema/knowledge.json` 定义的实体类型抽出来
- 类型由 schema 决定,例如 pichia 项目:strains / promoters / expressionvectors / culturemediums / fermentationconditionfacts / targetproducts / processparameters / analyticalmethods / glycosylationpatterns
- 一篇论文一个 `<paper>.pdf.json`

### L3.5 — Cross-paper Registry(跨论文实体合并)
- 读所有 L3 的 .pdf.json,把同一个实体在不同论文里的不同写法聚合
- 规则匹配 + LLM 同义词聚类(target_products / analytical_methods / promoters 走 LLM)
- 输出一份 `entity_registry.json`,每个实体有 canonical_name + aliases + 出现的论文

### L4a — Process Knowledge(每篇,工艺控制原则)
- 抽取每篇论文里的:
  - control_principles(工艺控制原则)
  - process_stages(工艺阶段)
  - fermentation_protocols(发酵协议)
  - troubleshooting(故障排查)
  - product_quality_factors(产物质量因素)
- 一个 JSON 列表,每篇论文一个条目

### L4b — Dialectical Review(跨论文辩证综合)
- 拿 L4a 的所有内容,跨论文分析:
  - 哪些主题论文之间共识?(consensus_points)
  - 哪些主题论文有分歧?(conflict_points)
  - 高置信度结论(highest_confidence_findings)
  - 不确定的领域(most_uncertain_areas)
- 用 Gemini 2.5 Pro(高推理) 一次跑全部 7 篇

### L4c — Domain Knowledge(跨论文领域综合)
- 直接把 7 篇论文的 abstract + methods + results 喂进去,产出领域级综合:
  - target_proteins(目标蛋白)
  - production_substrates(生产底物)
  - fermentation_conditions(发酵条件)
  - yield_benchmarks(产量基准)
  - technical_challenges(技术挑战)
  - innovations(创新点)
  - field_maturity / industrialization_readiness(领域成熟度评估)
  - key_open_questions(开放问题)

### L5 — Figure Extraction(图表抽取,视觉)
- 用 pymupdf 把 PDF 里嵌入的图像扣出来存成 PNG
- 每张图喂给 Gemini Vision,按多 panel schema 抽:
  - panels[]:每个 sub-panel 独立的 x_axis/y_axis/data_points/notable_points/fitted_equation
  - fixed_conditions(整张图的固定条件)
  - author_conclusion(作者结论)
  - industrial_note(工业建议)
- 一张图一个 JSON

### L6 — Experiment Extraction(实验工艺单)
- 抽取每篇论文的所有 fermentation runs,每个 run 有:
  - strain_construct(菌株构建:host / vector / promoter / signal_peptide / tag / selection)
  - setup(发酵设置:scale / medium / 各阶段参数 / 补料策略)
  - outcome(结果:max_yield / max_yield_time / max_wet_cell_weight)
  - varied_parameters(变量轴:这个实验改了什么)
  - figure_links(关联到 L5 的哪些图)
- lineage_extractor 再分析这些 runs 之间的"父子关系"(parameter A optimized → parameter B tuned 之类)
- 一篇论文一个 `<paper>.experiments.json`

### L7 — QA Context Assembly(问答上下文拼装)
- 拿到用户的问题后,从 L2~L6 各拉一段:
  1. `## Paper Excerpts` ← L2 向量召回 top-N chunks
  2. `## Per-paper Control Principles` ← L4a
  3. `## Dialectical Cross-Paper Review` ← L4b
  4. `## Relevant Figure Data` ← L5(按关键词过滤)
  5. `## Cross-Paper Entity Registry` ← L3.5
  6. `## Experiment Protocols` ← L6
  7. `## Domain Knowledge Overview` ← L4c
- 拼成一个 prompt 给 LLM,LLM 按这些 context + 自己的领域知识答题

---

## 各层之间的依赖关系

```
L1 ──→ L2
L1 ──→ L3 ──→ L3.5
L1 ──→ L4a ──→ L4b
L1 ──→ L4c
L1 ──→ L5
L1 ──→ L6 (extract-experiments)
        └──→ L6 (extract-lineage)
```

实际跑批的标准顺序:
1. `kb add <pdf>` —— 一次性跑 L1+L2+L3(往向量库 + 实体抽取)
2. `kb synthesize <pdf>` —— L4a
3. (重复 1-2 直到所有论文都进了)
4. `kb build-registry` —— L3.5(基于全部 L3)
5. `kb review` —— L4b(基于全部 L4a)
6. `kb domain-knowledge` —— L4c
7. `kb extract-figures <pdf>` —— L5
8. `kb extract-experiments <pdf>` —— L6
9. `kb extract-lineage` —— L6 lineage

或者用 web 的 8 📄 论文管理页面图形化跑(Step 6 实现的 orchestrator)。

---

## 当前 pichia-collagen 项目各层状态(2026-05-09)

| Layer | 状态 |
|---|---|
| L1 / L2 | ✅ 326 chunks 已索引(bge-m3) |
| L3 | ✅ 7 篇论文实体已抽,2026-05-03 用 gemini-2.5-flash 重跑过 |
| L3.5 | ✅ 7 类别全部填充(strains/promoters/vectors/media/target_products/analytical_methods/process_parameters)|
| L4a | ✅ 7 篇论文 process_knowledge 都齐 |
| L4b | ✅ dialectical_review.json 5 个主题综合 |
| L4c | ✅ domain_knowledge.json 完整(2026-05-04 重跑)|
| L5 | 🔄 王婧 35 张图 2026-05-09 用新 prompt(multi-panel + 数柱 + 中文输出)重抽中 / 其他 6 篇是 2026-05-02 老版本 |
| L6 | ⚠️ 7 篇都有(2026-04-30 抽),但 2026-05-09 加了"中文输出"指令后还没重跑,UI 显示英文居多 |
| L7 | ✅ 在跑(本地 + 阿里云 docker 容器都接通了 7 sections)|

---

## 跟知识架构表的关系

`data/projects/pichia-collagen/knowledge_architecture.md` 里的:
- **25 个 setup 字段** → 应被 L6 的 `Experiment.setup / strain_construct / outcome` 完整填充
- **8 类时间序列曲线** → 应被 L5 的 `panels[].curve_subtype`(暂未实现)分类标注

L5 → L6 的反向回填(从已分类的曲线推导动态 setup 字段)是图表精度工程的关键步骤,见 `knowledge_architecture.md` 第四节。
