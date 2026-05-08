# Pichia-Collagen 实验知识架构

领域专家提供的实验完整描述应当包含 **25 个静态字段**(setup) + **8 类时间序列曲线**(dynamic),两者合起来才能完整重现一个发酵实验。

本文件是项目的**抽取基准**:
- Layer 6(experiments)抽取时,setup 字段应至少尝试填满下表中的 25 个
- Layer 5(figures)抽取时,line_curve 类型的曲线应进一步分类到下面 8 类之一
- 问答 / 综合(Layer 4b/4c)的"完整度评分"应以这两张表为锚点

---

## 一、25 个静态字段(Experiment.setup baseline)

每篇论文的每个 fermentation run 都应记录这 25 项;空缺要明确标注 unknown,不要默认丢弃。

| 字段 | 示例填写 | 在 schema 中的位置 |
|---|---|---|
| 产物 | recombinant type III human-like collagen, hlCOLIII | `target_product.name` |
| 产物版本 | III 型胶原 α1 链片段,Gly-X-Y 框架,部分疏水氨基酸替换 | `target_product.subtype` / `notes` |
| 宿主 | P. pastoris GS115 | `strain_construct.host` |
| 表达载体 | pPIC9K | `strain_construct.vector` |
| 启动子 | AOX1 + DAS2 | `strain_construct.promoter` |
| 信号肽 | (文献记录) | `strain_construct.signal_peptide` |
| 标签 | 6×His | `strain_construct.tag` |
| 筛选 | G418 高拷贝筛选 | `strain_construct.selection` |
| 发酵规模 | 5 L bioreactor | `setup.scale` |
| 初始培养基 | BSM | `setup.medium` |
| 接种量 | 4% | `setup.inoculation_pct` |
| 初始温度 | 30°C | `setup.initial_temperature_celsius` |
| 初始 pH | 5.0 | `setup.initial_ph` |
| 初始搅拌 | 600 rpm | `setup.initial_stirring_rpm` |
| 初始通气 | 1 vvm | `setup.initial_aeration_vvm` |
| 甘油补料 | 50% 甘油 + 1.2% PTM1, 18 mL/h/L | `setup.glycerol_feed` |
| 饥饿 | 2 h | `setup.starvation_duration_hours` |
| 诱导 pH | 6.0 | `setup.induction_ph` |
| 甲醇补料 | 甲醇 + 1.2% PTM1, 前 4h 适应性,之后 DO-stat | `setup.methanol_feed` |
| 诱导搅拌 | 1000 rpm | `setup.induction_stirring_rpm` |
| 诱导通气 | 2-3 vvm | `setup.induction_aeration_vvm` |
| DO 控制 | 全诱导阶段 >20%, DO 回升到 40% 时补料 | `setup.do_control_strategy` |
| 诱导时长 | 96 h | `setup.induction_duration_hours` |
| 最高产量 | 1.05 g/L | `outcome.max_yield` |
| 最高产量时间 | 66 h | `outcome.max_yield_time_hours` |
| 最高湿菌重 | 270 g/L | `outcome.max_wet_cell_weight` |

---

## 二、8 类时间序列曲线(Figure.panels[].curve_subtype)

`figure_type=line_curve` 的 panel 必须进一步标注属于哪一种;模型读不出来时填 `other`。

| 曲线类型 | 典型 y 轴 / 关键词 | 可推导出的静态字段 |
|---|---|---|
| `biomass` 生物量 | OD600, DCW, wet cell weight, 菌体浓度 | 最高湿菌重 / 最高产量时间 |
| `carbon_source` 碳源 | glycerol, methanol concentration, residual carbon, 甘油浓度 | 甘油/甲醇补料策略 |
| `dissolved_oxygen` DO | DO%, dissolved oxygen, 溶氧 | DO 控制 |
| `ph` pH | pH | 初始/诱导 pH |
| `aeration_stirring` 通气搅拌 | aeration rate (vvm), stirring (rpm), agitation | 初始/诱导通气、搅拌 |
| `product` 产物 | yield, titer, hlCOLIII production, 产物浓度, mAU | 最高产量 / 最高产量时间 |
| `feed` 补料 | feed rate, methanol feed, 补料速率 | 甘油/甲醇补料 |
| `metabolite_byproduct` 副产物 | acetate, ethanol, lactate, 副产物 | 辅助判断代谢状态 |
| `other` | 不属于以上任何一类 | — |

---

## 三、完整实验描述 = 25 字段 + 全套曲线

一个"完整描述"的发酵实验应满足:

1. 25 个 setup 字段中,**至少 20 个有值**(剩余 5 个明确标 unknown)
2. 至少包含 **biomass 曲线 + product 曲线**(否则不知道何时收料、产量趋势)
3. 工业级实验额外包含 **DO + pH + carbon_source + feed** 四条曲线(过程控制可重现性)

完整度评分公式(待实现):
```
score = (filled_setup_fields / 25) * 0.5
      + (covered_curve_types / 8) * 0.5
```

## 四、当前差距(2026-05-09 测量)

7 篇论文 × 45 个实验:
- 静态字段:**11/25 填得好**(宿主/标签/筛选/规模/培养基/信号肽/最高产量等),**14/25 几乎全空**(初始温度/pH/搅拌/通气/补料/诱导参数/产量时间)
- 曲线分类:**0/8** —— Layer 5 现在没有 `curve_subtype` 字段,所有曲线混在一起

补齐路径:
1. 给 `data.json` 的 panel schema 加 `curve_subtype` 枚举字段
2. 改 `figure_extractor.py` prompt:line_curve 必须分类
3. Layer 6 抽取时,从已标注 `curve_subtype` 的图反推 setup 字段(比如 DO 曲线最低点 → DO 控制设定值)
